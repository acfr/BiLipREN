import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Sequence, Callable, Any, Tuple

from BiLipRENs.utils import cayley

class Orthogonal(nn.Module):
    units: int = 0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.array, inv: bool= False) -> jnp.array:
        n = jnp.shape(x)[-1]
        m = n if self.units == 0 else self.units
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (m, n),
                       jnp.float32)
        a = self.param('a', 
                       nn.initializers.constant(jnp.linalg.norm(W)), 
                       (1,),
                       jnp.float32)

        R = cayley((a / jnp.linalg.norm(W)) * W)
        # R = cayley(W)
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros_init(), (m,), jnp.float32)
        else:
            b = None

        if inv:
            if b is not None:
                x = x - b
            z = x @ R
        else:
            z = x @ R.T
            if b is not None:
                z += b

        return z 
    
class DynOrthogonal(nn.Module):
    layer_sizes: Sequence[int]
    inv:bool = False
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, state: jnp.array, inputs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        nx = self.layer_sizes[0]
        ny = self.layer_sizes[-1]

        # Set up parameters
        X = self.param("X", nn.initializers.glorot_normal(), 
                        (nx+ny, nx+ny), jnp.float32)
        bx = self.param("bx", nn.initializers.zeros, (nx,), jnp.float32)
        by = self.param("by", nn.initializers.zeros, (ny,), jnp.float32)
        if not self.use_bias:
            bx = jnp.zeros_like(bx)
            by = jnp.zeros_like(by)
        G = cayley(X)

        A = G[:nx, :nx]
        B = G[:nx, nx:]
        C = G[nx:, :nx]
        D = G[nx:, nx:]
        
        explicit = {'A': A, 'B': B, 'C': C, 'D': D, 'bx': bx, 'by': by}
                
        if self.inv:
             state, out = self.explicit_call_inverse(state, inputs, explicit)
        else:
             state, out = self.explicit_call(state, inputs, explicit)
        return state, out
    def explicit_call(self, x:jnp.array, u:jnp.array, e) -> Tuple[jnp.array, jnp.array]:
        x1 = x @ e["A"].T + u @ e["B"].T + e["bx"]
        y =  x @ e["C"].T + u @ e["D"].T + e["by"]
        return x1 , y
    
    def explicit_call_inverse(self, x:jnp.array, u:jnp.array, e) -> Tuple[jnp.array, jnp.array]:
        x1 = (x-e["bx"]) @ e["A"] + (u-e["by"]) @ e["C"]
        y =  (x-e["bx"]) @ e["B"] + (u-e["by"]) @ e["D"]
        return x1 , y