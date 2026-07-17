"""
surrogate_cost/surrogate_gd.py
==============================
Gradient-descent experiment for the two NON-injective cost surrogates
(CREN ablation and LSTM ablation).  Treats the trained network as a
differentiable cost surrogate

        F(u) = 0.5 * ||G(u)||^2 + c

and minimises it by Adam gradient descent,  u* = argmin_u F(u).  This is a
DATA-ONLY experiment script (no plotting).  A single multi-restart search
(62 restarts: zeros, dataset-mean, 30 lowest-cost + 30 random SUCCESSFUL
trajectories) is run per model and ALL searched trajectories are saved, along
with the lowest TRUE cost (oracle upper bound) and the lowest SURROGATE F (what
the deploy step picks).

            -> data/{MODEL}_search.pkl

Trained surrogates are read from data/{CREN,LSTM}_best.pkl.

Run from the project root:
    python surrogate_cost/surrogate_gd.py cren
    python surrogate_cost/surrogate_gd.py lstm
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as fnn

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"

MODEL = (sys.argv[1].lower() if len(sys.argv) > 1 else "cren")
if MODEL not in ("cren", "lstm"):
    print(f"ERROR: unknown model '{MODEL}'. Use 'cren' or 'lstm'.")
    sys.exit(1)

# ── Architecture (must match surrogate_training_ablation.py) ──────────────────
nu, nx, T = 2, 8, 55
NV_CREN   = 166
H_LSTM    = 91
STATE_DIM = nx if MODEL == "cren" else 2 * H_LSTM

# ── Geometry / dynamics constants (identical to the dataset generator) ───────
START   = np.array([-1.0, -2.0])   # (y_world, x_world)
TARGET  = np.array([ 3.0,  5.0])   # (y_world, x_world)
MAX_STEP, EPS = 0.4, 1e-8
DW, AW  = 10.0, 3.0                 # true-cost distance / action weights


# ── Data + true cost ─────────────────────────────────────────────────────────
data = np.load(str(_DATA / "dataset.npz"))
cost = data["costs"]
act  = data["actions"]                 # (N, T, 2) (y, x)
traj = data["trajectories"]            # (N, T, 2) (y, x) rolled-out positions
obs  = data["obstacle_x_y_radius"]     # (M, 3) cx(x), cy(y), r
dm_idx = int(np.argmin(cost))
u_dm   = act[dm_idx].astype(np.float32)

# ── Successful trajectories: collision-free AND reach the target (<0.5) ───────
#   warm-start restarts are drawn from these so the search starts inside the
#   feasible/goal-reaching region instead of from arbitrary random sequences.
_fin      = traj[:, -1, :]
_dist_fin = np.hypot(_fin[:, 0] - TARGET[0], _fin[:, 1] - TARGET[1])
_sdf      = np.min(np.hypot(traj[:, :, 1][:, :, None] - obs[:, 0],
                            traj[:, :, 0][:, :, None] - obs[:, 1]) - obs[:, 2], axis=2)
_collfree = ~(_sdf <= 0).any(axis=1)
SUCCESS_IDX = np.where(_collfree & (_dist_fin < 0.5))[0]
print(f"[success] {len(SUCCESS_IDX)} collision-free & reach<0.5 trajectories "
      f"(of {len(cost)})")


with open(_DATA / "ipopt.pkl", "rb") as f:
    IPOPT_COST = float(pickle.load(f)["cost"])


def _rollout_np(u_seq):
    pos = START.copy().astype(float); pts = []
    for u in u_seq:
        n = np.hypot(u[0], u[1]) + EPS
        pos = pos + 1.0 / (1.0 + np.exp(-n / 10.0)) * MAX_STEP * (u / n)
        pts.append(pos.copy())
    return np.array(pts)


def true_cost(u_seq):
    P = _rollout_np(u_seq); Tn = len(u_seq); tot = 0.0
    for k in range(Tn - 1):
        p, u = P[k], u_seq[k]
        dist = np.hypot(p[0] - TARGET[0], p[1] - TARGET[1]); ac = np.hypot(u[0], u[1])
        sdf = np.min(np.hypot(p[1] - obs[:, 0], p[0] - obs[:, 1]) - obs[:, 2])
        tot += DW * dist + AW * ac + (10.0 if sdf <= 0 else 0.0)
    pT = P[-1]; distT = np.hypot(pT[0] - TARGET[0], pT[1] - TARGET[1])
    sdfT = np.min(np.hypot(pT[1] - obs[:, 0], pT[0] - obs[:, 1]) - obs[:, 2])
    tot += 20.0 * (distT + (10.0 if sdfT <= 0 else 0.0))
    return float(tot)


# ── Trained surrogate F(u) = 0.5||G(u)||^2 + c ───────────────────────────────
class LSTMSurrogate(fnn.Module):
    hidden_size: int
    output_size: int

    @fnn.compact
    def __call__(self, state, inputs):
        Hh = self.hidden_size
        c, h = state[:, :Hh], state[:, Hh:]
        new_carry, y = fnn.LSTMCell(features=Hh)((c, h), inputs)
        out = fnn.Dense(self.output_size)(y)
        return jnp.concatenate(new_carry, axis=-1), out


if MODEL == "cren":
    from robustnn import ren_jax as ren
    model = ren.ContractingREN(input_size=nu, state_size=nx,
                               features=NV_CREN, output_size=nu)
else:
    model = LSTMSurrogate(hidden_size=H_LSTM, output_size=nu)

with open(_DATA / f"{MODEL.upper()}_best.pkl", "rb") as f:
    params = pickle.load(f)
c_val = float(jax.nn.softplus(params["c_raw"]))
s0 = jnp.zeros((1, STATE_DIM), dtype=jnp.float32)

print(f"[ref] model={MODEL.upper()}  surrogate c={c_val:.1f}  "
      f"dataset-min cost={cost[dm_idx]:.1f}  IPOPT cost={IPOPT_COST:.1f}")


def scan_fn(carry, inp):
    st, p = carry
    ns, no = model.apply({"params": p["params"]}, st, inp)
    return (ns, p), no


@jax.jit
def F_of_u(u_seq):                       # u_seq (T,1,nu) -> scalar surrogate cost
    _, ro = jax.lax.scan(scan_fn, (s0, params), u_seq)
    return 0.5 * jnp.sum(ro ** 2) + jax.nn.softplus(params["c_raw"])


# =============================================================================
# Run 1 — broad multi-restart search  ->  data/{MODEL}_search.pkl
# =============================================================================
def run_search():
    opt = optax.adam(3e-3); STEPS = 4000

    @jax.jit
    def F_batch(U):                      # (R,T,nu) -> sum of per-restart 0.5||G||^2
        def one(u):
            _, ro = jax.lax.scan(scan_fn, (s0, params), u[:, None, :])
            return 0.5 * jnp.sum(ro ** 2)
        return jnp.sum(jax.vmap(one)(U))

    @jax.jit
    def step(U, st):
        val, g = jax.value_and_grad(F_batch)(U)
        upd, st2 = opt.update(g, st)
        return optax.apply_updates(U, upd), st2, val, optax.global_norm(g)


    # restart pool: zeros, dataset-mean, 30 low-cost + 30 random SUCCESSFUL warm
    #   starts.  Random Gaussian restarts are dropped; every data warm start is a
    #   real collision-free, goal-reaching trajectory (SUCCESS_IDX).
    rng = np.random.default_rng(0)
    pool = [np.zeros((T, nu), np.float32), act.mean(0).astype(np.float32)]
    low_idx  = np.argsort(cost)[:30]                       # 30 lowest-cost (all successful)
    n_rand   = min(30, len(SUCCESS_IDX))
    rand_idx = rng.choice(SUCCESS_IDX, n_rand, replace=False)  # 30 random successful
    warm_idx = np.concatenate([low_idx, rand_idx])
    for i in warm_idx:
        pool.append(act[int(i)].astype(np.float32))
    U0 = np.stack(pool, 0)               # (R,T,nu)

    R = U0.shape[0]
    print(f"[search] {R} restarts x {STEPS} Adam steps ...")

    U = jnp.asarray(U0); st = opt.init(U)
    # convergence trace: surrogate objective 0.5*sum_r||G(u_r)||^2 and grad norm
    loss_steps, loss_hist, gnorm_hist = [], [], []
    for k in range(STEPS):
        U, st, val, gn = step(U, st)
        if k % 20 == 0 or k == STEPS - 1:
            loss_steps.append(k)
            loss_hist.append(float(val))
            gnorm_hist.append(float(gn))
    U = np.array(U)
    loss_steps  = np.asarray(loss_steps)
    loss_hist   = np.asarray(loss_hist)          # F_sum over R restarts
    gnorm_hist  = np.asarray(gnorm_hist)
    print(f"  [converge] F_sum {loss_hist[0]:.1f} -> {loss_hist[-1]:.3f} "
          f"({100*(loss_hist[0]-loss_hist[-1])/max(loss_hist[0],1e-9):.2f}% drop)  "
          f"final grad_norm={gnorm_hist[-1]:.2e}")


    Fs    = np.array([float(F_of_u(jnp.asarray(U[r])[:, None, :])) for r in range(R)])
    TCs   = np.array([true_cost(U[r]) for r in range(R)])
    drift = np.array([float(np.linalg.norm(U[r] - u_dm)) for r in range(R)])
    best_tc   = int(np.argmin(TCs))      # oracle: best achievable true cost
    best_surr = int(np.argmin(Fs))       # deploy: what the surrogate would pick

    print(f"  [best TRUE] #{best_tc:3d}: TRUE={TCs[best_tc]:.1f} F={Fs[best_tc]:.1f} "
          f"drift={drift[best_tc]:.2f}   (oracle upper bound)")
    print(f"  [best SURR] #{best_surr:3d}: F={Fs[best_surr]:.1f} TRUE={TCs[best_surr]:.1f} "
          f"drift={drift[best_surr]:.2f}   (deploy-time pick)")
    print(f"  true-cost over restarts: min={TCs.min():.0f} median={np.median(TCs):.0f} "
          f"max={TCs.max():.0f}")

    out = _DATA / f"{MODEL.upper()}_search.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "all_u": U,                        # (R, T, nu) every searched trajectory
            "u_best_truecost":  U[best_tc],   "cost_best_truecost":  float(TCs[best_tc]),
            "u_best_surrogate": U[best_surr], "cost_best_surrogate": float(TCs[best_surr]),
            "F_best_truecost":  float(Fs[best_tc]), "F_best_surrogate": float(Fs[best_surr]),
            "all_true_costs": TCs, "all_F": Fs, "all_drift": drift,
            "u_dm": u_dm, "dm_cost": float(cost[dm_idx]), "ipopt_cost": IPOPT_COST,
            "obstacle": obs, "start": START, "target": TARGET, "c": c_val,
            # GD convergence trace (surrogate objective F_sum = 0.5*sum_r||G||^2)
            "conv_steps": loss_steps, "conv_Fsum": loss_hist,
            "conv_Fmean": loss_hist / R, "conv_gnorm": gnorm_hist,
            "n_restarts": R,
        }, f)

    print(f"[saved] {out}")


if __name__ == "__main__":
    run_search()
