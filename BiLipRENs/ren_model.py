import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Union, Callable, Any, Tuple, Optional

from flax import linen as nn
from flax.linen import initializers as init
from flax.typing import Dtype

from BiLipRENs.utils import l2_norm, identity_init

jax.config.update("jax_default_matmul_precision", "highest")

# ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
Array = Union[jax.Array, Any]

class BiLipREN(nn.Module):
    input_size: int
    state_size: int
    features: int
    lower_bound: float
    upper_bound: float
    # activation: ActivationFn = nn.relu
    kernel_init: Initializer = init.glorot_normal()
    recurrent_kernel_init: Initializer = init.lecun_normal()
    bias_init: Initializer = init.zeros_init()
    carry_init: Initializer = init.zeros_init()
    # Initialization mode:
    #   "random"   – default; X uses recurrent_kernel_init, Y1 uses kernel_init
    #   "identity" – X initialised to I (so X^T X = I, recurrent H ≈ p² · I),
    #                 Y1 initialised to zero (skew-symmetric part of E vanishes).
    init_mode: str = "random"
    param_dtype: Dtype = jnp.float32
    eps = jnp.finfo(jnp.float32).eps
    abar = jnp.float32(1)

    def setup(self):
        
        nu = self.input_size
        nx = self.state_size
        nv = self.features
        ny = nu

        # self.beta = (self.upper_bound+self.lower_bound)/2
        # gamma = (self.upper_bound-self.lower_bound)/2

        # alpha_1 = 1/self.beta*(self.beta**2-gamma**2)
        # alpha_2 = 1/self.beta

        mu_ = self.lower_bound
        nv_= self.upper_bound
        alpha_1 = 2*(mu_*nv_)/(mu_+nv_)
        alpha_2 = 2/(mu_+nv_)
        
        # Define IQC parameters
        self.Q = -alpha_2*jnp.eye(nu)
        self.S = jnp.eye(nu)
        self.R = -alpha_1*jnp.eye(nu)

        # Define direct params for REN
        if self.init_mode == "identity":
            x_init  = identity_init()
            y1_init = init.zeros_init()
        elif self.init_mode == "random":
            x_init  = self.recurrent_kernel_init
            y1_init = self.kernel_init
        else:
            raise ValueError(f"Unknown init_mode={self.init_mode!r}; "
                             "expected 'random' or 'identity'.")

        B2 = self.param("B2", self.kernel_init, (nx, nu), self.param_dtype)
        D12 = self.param("D12", self.kernel_init, (nv, nu), self.param_dtype)
        X = self.param("X", x_init,
                       (2 * nx + nv, 2 * nx + nv), self.param_dtype)
        p = self.param("polar", init.constant(l2_norm(X, eps=self.eps)),
                       (1,), self.param_dtype)        
        Y1 = self.param("Y1", y1_init, (nx, nx), self.param_dtype)
        bx = self.param("bx", self.bias_init, (nx,), self.param_dtype)
        bv = self.param("bv", self.bias_init, (nv,), self.param_dtype)
        
        # Output layer params
        out_kernel_init = self.kernel_init
        out_bias_init = self.bias_init
            
        by = self.param("by", out_bias_init, (ny,), self.param_dtype)
        C2 = self.param("C2", out_kernel_init, (ny, nx), self.param_dtype)
        D21 = self.param("D21", out_kernel_init, (ny, nv), self.param_dtype)
           
        X3 = self.param("X3", identity_init(), (nu, nu), self.param_dtype)
        Y3 = self.param("Y3", init.zeros_init(), (nu, nu), self.param_dtype)
         
        self.direct = {
        "p": p,
        "X": X,
        "B2": B2,
        "D12": D12,
        "Y1": Y1,
        "C2": C2,
        "D21": D21,
        "X3": X3,
        "Y3": Y3,
        "bx": bx,
        "bv": bv,
        "by": by
    }

    def __call__(self, state: Array, inputs: Array, inv: bool = False, Jac: bool = False) -> Tuple[Array, Array]:

        # Direct parameterisation mapping
        explicit = self.direct_to_explicit(self.direct)
        # Invert the explicit model if required
        if inv:
            explicit = self.explicit_inverse(explicit)      

        # Call the explicit REN form and return
        if Jac:
            prev_state = state
            state, out = self.explicit_call(state, inputs, explicit)
            jacobian = solve_input_output_jacobian(nn.relu, prev_state, inputs, explicit)
            return state, out, jacobian
        else:
            state, out = self.explicit_call(state, inputs, explicit)
            return state, out

    def direct_to_explicit(self, ps):
        nu = self.input_size
        nx = self.state_size
        ny = nu
        # Q, S, R = self._adjust_iqc_params()
        Q = self.Q
        S = self.S
        R = self.R
        # mu_ = self.lower_bound
        # nv_ = self.upper_bound
        
        I = jnp.identity(ny, self.param_dtype)
                
        # Compute useful decompositions
        R_temp = R - S @ jnp.linalg.solve(Q, S.T)
        LQ = jnp.linalg.cholesky(-Q, upper=True)
        LR = jnp.linalg.cholesky(R_temp, upper=True)
        
        # # Construct D22 (Eqns 31-33 of Revay et al. (2023))

        M = ps["X3"].T @ ps["X3"] + ps["Y3"] - ps["Y3"].T + self.eps*I
        N = jnp.linalg.solve((I + M).T, (I - M).T).T
            
        D22 = jnp.linalg.solve(-Q, S.T) + jnp.linalg.solve(LQ, N) @ LR
        
        # D22 = -(mu_*nv_)/2*I + (-mu_+nv_)/2* N
        # Implicit params
        B2_imp = ps["B2"]
        D12_imp = ps["D12"]
        
        # Construct H (Eqn. 28 of Revay et al. (2023))
        C2_imp = (D22.T @ Q + S) @ ps["C2"]
        D21_imp = (D22.T @ Q + S) @ ps["D21"] - D12_imp.T
        
        R1 = R + S @ D22 + D22.T @ S.T + D22.T @ Q @ D22
        mul_Q = jnp.hstack((ps["C2"], ps["D21"], jnp.zeros((ny, nx), self.param_dtype)))
        mul_R = jnp.hstack((C2_imp, D21_imp, B2_imp.T))
        Gamma_Q = mul_Q.T @ Q @ mul_Q
        Gamma_R = mul_R.T @ jnp.linalg.solve(R1, mul_R)
        
        H = self.x_to_h(ps["X"], ps["p"]) + Gamma_R - Gamma_Q
        explicit = self.hmatrix_to_explicit(ps, H, D22)
        return explicit

    # def _adjust_iqc_params(self):
    #     """Small delta to help numerical conditioning with cholesky decomposition."""
    #     Q = self.Q - self.eps * jnp.identity(self.Q.shape[0], self.param_dtype)
    #     R = self.R + self.eps * jnp.identity(self.R.shape[0], self.param_dtype)
    #     return Q, self.S, R
    
    def explicit_call(
        self, x: Array, u: Array, e) -> Tuple[Array, Array]:
        """
        Evaluate a REN given its explicit parameterization.
        """
        b = x @ e["C1"].T + u @ e["D12"].T + e["bv"]
        w = solve_equlibrium_layer(nn.relu, e["D11"], b)
        x1 = x @ e["A"].T + w @ e["B1"].T + u @ e["B2"].T + e["bx"]
        y = x @ e["C2"].T + w @ e["D21"].T + u @ e["D22"].T + e["by"]
        return x1, y

    
    def x_to_h(self, X: Array, p: Array) -> Array:
        """Convert REN X matrix to H matrix using polar parameterization."""
        H = p**2 * (X.T @ X) / (l2_norm(X)**2) + self.eps * jnp.identity(jnp.shape(X)[0])
        return H
    
    def hmatrix_to_explicit(self, ps, H: Array, D22: Array):
        """Convert REN H matrix to explict form given direct params."""
        
        nx = self.state_size
        nv = self.features
        
        # Extract sections of the H matrix
        H11 = H[:nx, :nx]
        H22 = H[nx:(nx + nv), nx:(nx + nv)]
        H33 = H[(nx + nv):(2*nx + nv), (nx + nv):(2*nx + nv)]
        H21 = H[nx:(nx + nv), :nx]
        H31 = H[(nx + nv):(2*nx + nv), :nx]
        H32 = H[(nx + nv):(2*nx + nv), nx:(nx + nv)]
                
        # Construct implicit model parameters
        P_imp = H33
        F = H31
        E = (H11 + P_imp / (self.abar**2) + ps["Y1"] - ps["Y1"].T) / 2
        
        # Equilibrium network params (imp for "implicit")
        B1_imp = H32
        C1_imp = -H21
        Lambda_inv = 2 / jnp.diag(H22)
        D11_imp = -jnp.tril(H22, k=-1)
        
        # Construct the explicit model (e for "explicit")
        A_e = jnp.linalg.solve(E, F)
        B1_e = jnp.linalg.solve(E, B1_imp)
        B2_e = jnp.linalg.solve(E, ps["B2"])
        
        # Equilibrium layer matrices
        C1_e = (Lambda_inv * C1_imp.T).T
        D11_e = (Lambda_inv * D11_imp.T).T
        D12_e = (Lambda_inv * ps["D12"].T).T
        # Remaining explicit params are biases/in the output layer (unchanged)
        explicit = {
            "A": A_e,
            "B1": B1_e,
            "B2": B2_e,
            "C1": C1_e,
            "C2": ps["C2"],
            "D11": D11_e,
            "D12": D12_e,
            "D21": ps["D21"],
            "D22": D22,
            "bx": ps["bx"],
            "bv": ps["bv"],
            "by": ps["by"]
            }
        return explicit
    
    def explicit_inverse(self, explicit):
        A = explicit["A"]
        B1 = explicit["B1"]
        B2 = explicit["B2"]
        C1 = explicit["C1"]
        C2 = explicit["C2"]
        D11 = explicit["D11"]
        D12 = explicit["D12"]
        D21 = explicit["D21"]
        D22 = explicit["D22"]
        bx = explicit["bx"]
        bv = explicit["bv"]
        by = explicit["by"]

        A_inv = A - B2 @ jnp.linalg.inv(D22) @ C2
        B1_inv = B1 - B2 @ jnp.linalg.inv(D22) @ D21
        B2_inv =B2 @ jnp.linalg.inv(D22)
        C1_inv = C1 - D12 @ jnp.linalg.inv(D22) @ C2
        C2_inv = -jnp.linalg.inv(D22) @ C2
        D11_inv = D11 - D12 @ jnp.linalg.inv(D22) @ D21
        D12_inv = D12 @ jnp.linalg.inv(D22)
        D21_inv = -jnp.linalg.inv(D22) @ D21
        D22_inv = jnp.linalg.inv(D22)
        bx_inv = bx-B2 @ jnp.linalg.inv(D22) @ by
        bv_inv = bv - D12 @ jnp.linalg.inv(D22) @ by
        by_inv = -jnp.linalg.inv(D22) @ by
        explicit_inv = {
            "A": A_inv,
            "B1": B1_inv,
            "B2": B2_inv,
            "C1": C1_inv,
            "C2": C2_inv,
            "D11": D11_inv,
            "D12": D12_inv,
            "D21": D21_inv,
            "D22": D22_inv,
            "bx": bx_inv,
            "bv": bv_inv,
            "by": by_inv
            }
        return explicit_inv

    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> Array:
        """Initialize the REN state (carry).
        
        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the network.
        
        Returns:
        An initialized state (carry) vector for the REN network.
        """
        batch_dims = input_shape[:-1]
        rng, _ = jax.random.split(rng)
        mem_shape = batch_dims + (self.state_size,)
        return self.carry_init(rng, mem_shape, self.param_dtype)
    

@partial(jax.jit, static_argnums=(0,))
def solve_equlibrium_layer(activation, D11, b):
    """
    Solve `w = activation(D11 @ w + b)` for full D11.
    
    Activation must be monotone with slope restricted to `[0,1]`.
    """
    w_eq = jnp.zeros_like(b)
    D11_T = D11.T
    is_tril = jnp.all(jnp.isclose(D11, jnp.tril(D11)))
    # Forward pass (compute custom grads below)
    def tril_branch():
        w_eq = jnp.zeros_like(b)
        D11_T = D11.T
        for i in range(D11.shape[0]):
            Di_T = D11_T[:i, i]
            wi = w_eq[..., :i]
            bi = b[..., i]
            Di_wi = wi @ Di_T
            w_eq = w_eq.at[..., i].set(activation(Di_wi + bi))
        return w_eq
    
    def full_branch():
        w_eq = jax.lax.stop_gradient(solve_full_layer(activation, D11, b))
        v = w_eq @ D11.T + b
        w_eq = activation(v)
        return tril_layer_do_grad(activation, D11, v, w_eq)
    
    w_eq = jax.lax.cond(is_tril, tril_branch, full_branch)
    # Re-evaluate the equilibrium layer so autodiff can track grads
    # through these two operations, then customise for grad of w_eq
    return w_eq

# --- Inverse-REN equilibrium solver iteration budget -------------------------
# solve_full_layer (Douglas-Rachford) is only used by the INVERSE REN.  Its
# iteration count trades speed against round-trip accuracy.  Use the switch
# below to pick a budget:
#   fast  -> 200    iters  (quick plotting / evaluation)
#   slow  -> 20000  iters  (high-accuracy G^-1(G(u)) round-trip checks)
_INV_SOLVER_FAST_ITERS = 200
_INV_SOLVER_SLOW_ITERS = 20000
_INV_SOLVER_FAST = True


def solve_full_layer(activation, D11, b):
    """
    Solve `w = activation(D11 @ w + b)` using operator splitting.
    
    Only valid for the forward pass (not backprop with auto-diff).
    """
    w_eq = jnp.zeros_like(b)

    # """Forward-backward spliting algorithm for solving the equilibrium layer."""
    # tolerance = 1e-5
    # alpha = 0.8

    # def body_fun(w_eq):
    #     u = w_eq @ ((1 - alpha) * jnp.eye(D11.shape[0]) + alpha * D11).T  + alpha * (b)
    #     w_eq_new = activation(u)
    #     return w_eq_new

    # def cond_fun(w_eq):
    #     return jnp.linalg.norm(w_eq - activation(w_eq), 2) >= tolerance

    # w_eq_final= lax.while_loop(cond_fun, body_fun, w_eq)
    
    # return w_eq_final
    # '''PeacemanRachford splitting algorithm for solving the equilibrium layer.'''
    # tol = 1e-3
    # alpha = 0.8
    # uk = jnp.zeros_like(b)
    
    # def body_fun(carry):
    #     w_eq , uk = carry
    #     uh = 2*w_eq - uk
    #     zh = jnp.linalg.solve((jnp.eye(D11.shape[0])+alpha*(jnp.eye(D11.shape[0])-D11)),(uh+alpha*b).T)
    #     uk_new = 2*zh.T - uh
    #     w_eq_new = activation(uk_new)
    #     return (w_eq_new, uk_new)
 
    # def cond_fun(carry):
    #     w_eq , uk = carry
    #     (w_eq_new ,uk_new) = body_fun(carry)
    #     return l2_norm(w_eq - w_eq_new) >= tol
    
    # (w_eq_final,_)= lax.while_loop(cond_fun, body_fun, (w_eq, uk))
    # return w_eq_final
    '''DouglasRachford splitting algorithm for solving the equilibrium layer.

    NOTE: this branch is ONLY taken when D11 is NOT lower-triangular, i.e. for the
    INVERSE REN (explicit_inverse makes D11_inv = D11 - D12 @ inv(D22) @ D21 full).
    The forward REN keeps D11 strictly lower-triangular and uses the exact tril solve,
    so tightening the tolerance here improves G^-1 accuracy WITHOUT affecting the
    forward pass / training hot path.  A loose tol (1e-3) left a ~1e-2 y-space residual
    that the 1/L inverse-Lipschitz gain amplified into a ~0.15 u-space round-trip error
    (errA = ||G^-1(G(u)) - u||); 1e-9 collapses it to ~1e-5.
    '''
    tol = 1e-9
    alpha = 0.6
    max_iter = _INV_SOLVER_FAST_ITERS if _INV_SOLVER_FAST else _INV_SOLVER_SLOW_ITERS
    uk = jnp.zeros_like(b)
    
    def body_fun(carry):
        w_eq, uk, _, k = carry
        uh = 2*w_eq - uk
        zh = jnp.linalg.solve((jnp.eye(D11.shape[0])+alpha*(jnp.eye(D11.shape[0])-D11)),(uh+alpha*b).T)
        uk_new = uk - w_eq + zh.T
        w_eq_new = activation(uk_new)

        error = l2_norm(w_eq - w_eq_new)
        return (w_eq_new, uk_new, error, k + 1)
 
    def cond_fun(carry):
        _, _, error, k = carry
        # (w_eq_new ,_) = body_fun(carry)
        # return l2_norm(w_eq - w_eq_new) >= tol
        return jnp.logical_and(error >= tol, k < max_iter)
    init_carry = (w_eq, uk, jnp.inf, 0)
    (w_eq_final, _, _, _) = lax.while_loop(cond_fun, body_fun, init_carry)
    return w_eq_final
       
@partial(jax.custom_vjp, nondiff_argnums=(0,))
def tril_layer_do_grad(activation, D11, v, w_eq):
    return w_eq
def tril_layer_do_grad_fwd(activation, D11, v, w_eq):
    I = jnp.identity(v.shape[-1])
    return w_eq, (D11, v, I)
def tril_layer_do_grad_bwd(activation, res, y_bar):
    """
    Compute backwards pass with implicit function theorem.
    See Equation 13 of Revay et al. (2023).
    """
    D11, v, I = res
  
    # Ignore grads for D11, v
    D11_bar = jnp.zeros_like(D11)
    v_bar = jnp.zeros_like(v)
    
    # Get Jacobian of activation(v) evaluated at v
    # Scalar activation ==> diagonal Jacobian, so get
    # diagonal elements for each batch. j_diag has
    # dimensions (batches, nv)
    _, vjp_act_v = jax.vjp(activation, v)
    j_diag, = vjp_act_v(jnp.ones_like(v))
    
    # Compute gradient with implicit function theorem (per batch)
    w_eq_bar = jnp.zeros_like(v)
    for i in range(w_eq_bar.shape[0]):
        ji = j_diag[i, ...]
        y_bar_i = y_bar[i, ...]
        w_grad = jnp.linalg.solve(I - (ji * D11.T), y_bar_i.T).T
        w_eq_bar = w_eq_bar.at[i, ...].set(w_grad)
    
    return (D11_bar, v_bar, w_eq_bar)

tril_layer_do_grad.defvjp(tril_layer_do_grad_fwd, tril_layer_do_grad_bwd)

def solve_input_output_jacobian_analytic(activation, state, inputs, e):
    """
    Compute dy/du analytically via the implicit function theorem (IFT).

    For the REN with equilibrium w = sigma(D11 @ w + b), b = C1 x + D12 u + bv
    and output y = C2 x + D21 w + D22 u + by, the input-output Jacobian is:

        dy/du = D22 + D21 @ (I - Lambda @ D11)^{-1} @ Lambda @ D12

    where Lambda = diag(sigma'(v)), v = D11 @ w + b.
    This avoids autodiff and requires only one nv-by-nv linear solve per sample.

    Returns shape (batch, nu, nu).
    """
    # Recompute equilibrium (shares computation with the forward pass)
    b = state @ e["C1"].T + inputs @ e["D12"].T + e["bv"]   # (batch, nv)
    w_eq = solve_equlibrium_layer(activation, e["D11"], b)    # (batch, nv)
    v = w_eq @ e["D11"].T + b                                 # (batch, nv)

    # Diagonal of Lambda per sample: relu'(v) = 1_{v>0}
    lam = (v > 0).astype(b.dtype)                             # (batch, nv)

    def per_sample(lam_i):
        nv = e["D11"].shape[0]
        I = jnp.eye(nv, dtype=lam_i.dtype)
        M = I - lam_i[:, None] * e["D11"]    # (I - Lambda D11), shape (nv, nv)
        LamD12 = lam_i[:, None] * e["D12"]   # Lambda D12,      shape (nv, nu)
        dw_du = jnp.linalg.solve(M, LamD12)  # (nv, nu)
        return e["D22"] + e["D21"] @ dw_du   # (nu, nu)

    return jax.vmap(per_sample)(lam)          # (batch, nu, nu)


def solve_input_output_jacobian(activation, state, inputs, e):
    """Analytic IFT-based Jacobian (replaces the former jacrev implementation)."""
    return solve_input_output_jacobian_analytic(activation, state, inputs, e)