"""
IPOPT trajectory optimisation over 60 initial guesses (single reference solver).

Builds the direct-shooting NLP (hard obstacle-clearance constraints,
limited-memory Ipopt) and solves it from 60 diverse initial guesses U0:

    * zeros                                   (1)
    * straight-line toward the goal           (9 magnitudes)
    * top-25 lowest-cost dataset trajectories (25, data-driven warm start)
    * random Gaussian actions                 (25, 5 scales x 5 seeds)

For every start it records the converged true cost (hard-indicator, matching
the dataset cost_code), minimum obstacle clearance, feasibility and trajectory.
The single output data file keeps the BEST (lowest-cost feasible) and WORST
(highest converged cost over all starts) trajectories plus all 60 results:

    -> data/ipopt.pkl   (u_opt/trajectory/cost + u_worst/trajectory_worst/cost_worst + all_results)

Run:
    python surrogate_cost/surrogate_ipopt.py
"""

import time
import pickle
from pathlib import Path as _P

import casadi as ca
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import sigmoid as jax_sigmoid

# -- Environment ---------------------------------------------------------------
_HERE = _P(__file__).resolve().parent
_DATA = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)

# Dataset: the one whose recorded costs lie in the 1000+ range
# (data/dataset.npz, cost range ~1863-5054).
_DATASET = _HERE / "data" / "dataset.npz"

data    = np.load(str(_DATASET))
print(f"Dataset        : {_DATASET.name}  "
      f"(N={data['actions'].shape[0]}, cost {data['costs'].min():.0f}-{data['costs'].max():.0f})")
obs_np  = data["obstacle_x_y_radius"]   # (N_OBS, 3): cx, cy, r  [plot (x,y)]
costs_np = data["costs"]                # (N,)
acts_np = data["actions"]              # (N, T, 2)  col0=a_y, col1=a_x

# -- Problem constants --------------------------------------------------------
START_NP  = np.array([-2.0, -1.0], dtype=np.float64)   # (x, y)
GOAL_NP   = np.array([ 5.0,  3.0], dtype=np.float64)
T         = int(acts_np.shape[1])      # 55
MAX_STEP  = 0.4
U_MAX     = 20.0

OBS_XY_NP = obs_np[:, :2].astype(np.float64)
OBS_R_NP  = obs_np[:,  2].astype(np.float64)
N_OBS     = len(OBS_R_NP)

DIST_W       = 10.0   # stage distance-to-goal weight (matches dataset cost)
AC_W         = 3.0    # stage action weight           (matches dataset cost)
COLL_PENALTY = 10.0
W_TERM       = 20.0
_EPS         = 1e-6


def _ca_sigmoid(x):
    return 1.0 / (1.0 + ca.exp(-x))


# -- Build the NLP once --------------------------------------------------------
print("Building CasADi / Ipopt NLP ...")
opti = ca.Opti()
U = opti.variable(T, 2)


def _ca_step(p, uk):
    norm = ca.norm_2(uk) + _EPS
    d    = uk / norm
    return p + _ca_sigmoid(norm / 10.0) * MAX_STEP * d


P = [None] * (T + 1)
P[0] = ca.DM(START_NP)
for k in range(T):
    P[k + 1] = _ca_step(P[k], U[k, :].T)

goal_dm = ca.DM(GOAL_NP)
obj = 0
for k in range(1, T):                       # stage over p_1..p_{T-1}, u_0..u_{T-2}
    obj += DIST_W * ca.norm_2(P[k] - goal_dm)
    obj += AC_W   * ca.norm_2(U[k - 1, :].T)
obj += W_TERM * ca.norm_2(P[T] - goal_dm)    # terminal p_T
opti.minimize(obj)

for k in range(1, T + 1):
    for j in range(N_OBS):
        opti.subject_to(ca.norm_2(P[k] - ca.DM(OBS_XY_NP[j])) >= float(OBS_R_NP[j]))
opti.subject_to(opti.bounded(-U_MAX, U, U_MAX))

ipopt_opts = {
    "hessian_approximation": "limited-memory",
    "max_iter"          : 500,
    "tol"               : 1e-5,
    "constr_viol_tol"   : 1e-3,
    "acceptable_tol"    : 1e-3,
    "acceptable_iter"   : 15,
    "mu_strategy"       : "adaptive",
    "nlp_scaling_method": "gradient-based",
    "print_level"       : 0,
}
opti.solver("ipopt", {"print_time": False}, ipopt_opts)
print(f"  vars={T * 2}  ineq={T * N_OBS}  bounds={T * 2}\n")


# -- JAX true-cost evaluator (hard indicator, matches dataset) -----------------
OBS_XY_JAX = jnp.array(OBS_XY_NP, dtype=jnp.float32)
OBS_R_JAX  = jnp.array(OBS_R_NP,  dtype=jnp.float32)
START_JAX  = jnp.array(START_NP,  dtype=jnp.float32)


def _jax_snorm(x):
    return jnp.sqrt(jnp.sum(x ** 2) + _EPS ** 2)


def _jax_snorm_ax(x, axis=-1):
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + _EPS ** 2)


@jax.jit
def _jax_rollout(u_xy):
    def step(pos, uk):
        d   = uk / _jax_snorm(uk)
        nxt = pos + jax_sigmoid(_jax_snorm(uk) / 10.0) * MAX_STEP * d
        return nxt, nxt
    _, pts = jax.lax.scan(step, START_JAX, u_xy)
    return jnp.vstack([START_JAX[None], pts])


@jax.jit
def _jax_true_cost(u_xy):
    traj  = _jax_rollout(u_xy)
    goal  = jnp.array(GOAL_NP, dtype=jnp.float32)
    p_chk = traj[1:]
    diffs = p_chk[:, None, :] - OBS_XY_JAX[None, :, :]
    dists = _jax_snorm_ax(diffs, axis=-1)
    sdf   = (dists - OBS_R_JAX[None, :]).min(axis=-1)
    dist_g = DIST_W * _jax_snorm_ax(traj[1:-1] - goal, axis=1)   # p_1..p_{T-1}
    act_c  = AC_W   * _jax_snorm_ax(u_xy[:-1], axis=1)            # u_0..u_{T-2}
    coll   = jnp.where(sdf[:-1] <= 0, COLL_PENALTY, 0.0)
    stage  = jnp.sum(dist_g + act_c + coll)
    coll_T   = jnp.where(sdf[-1] <= 0, COLL_PENALTY, 0.0)
    terminal = W_TERM * (_jax_snorm(traj[-1] - goal) + coll_T)
    return stage + terminal


_jax_true_cost(jnp.zeros((T, 2), dtype=jnp.float32))   # warm-up


# -- Initial-guess generators --------------------------------------------------
def _dataset_warmstart(ds_idx):
    a  = acts_np[ds_idx]
    u0 = np.zeros((T, 2), dtype=np.float64)
    u0[:, 0] = a[:, 1]      # a_x -> x
    u0[:, 1] = a[:, 0]      # a_y -> y
    return u0


def _straight_line(mag):
    """Constant action of magnitude ``mag`` pointing start->goal."""
    d = (GOAL_NP - START_NP)
    d = d / np.linalg.norm(d)
    return np.tile(mag * d, (T, 1)).astype(np.float64)


def _random(seed, scale):
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((T, 2))).astype(np.float64)


guesses = []
guesses.append(("zeros", np.zeros((T, 2))))
for mag in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0):          # 9
    guesses.append((f"straight_mag{mag:g}", _straight_line(mag)))
top_idx = np.argsort(costs_np)[:25]                                      # 25
for r, i in enumerate(top_idx):
    guesses.append((f"ds_top{r + 1:02d}(idx={i})", _dataset_warmstart(int(i))))
for scale in (5.0, 10.0, 15.0, 20.0, 25.0):                             # 5 x 5 = 25
    for seed in (0, 1, 2, 3, 4):
        guesses.append((f"rand_sc{scale:g}_s{seed}", _random(seed, scale)))
assert len(guesses) == 60, f"expected 60 initial guesses, got {len(guesses)}"


# -- Sweep ---------------------------------------------------------------------
def _eval(u_opt):
    uj     = jnp.array(u_opt, dtype=jnp.float32)
    traj   = np.array(_jax_rollout(uj))
    c_true = float(_jax_true_cost(uj))
    p_chk  = traj[1:]
    dists  = np.linalg.norm(p_chk[:, None, :] - OBS_XY_NP[None, :, :], axis=-1)
    sdf    = float((dists - OBS_R_NP[None, :]).min())
    return traj, c_true, sdf


results = []
print(f"Sweeping {len(guesses)} initial guesses ...\n")
for name, u0 in guesses:
    opti.set_initial(U, u0)
    t0 = time.time()
    try:
        sol    = opti.solve()
        u_opt  = np.array(sol.value(U))
        status = "ok"
    except RuntimeError as e:
        u_opt  = np.array(opti.debug.value(U))
        status = str(e).splitlines()[0][:24]
    dt = time.time() - t0
    try:
        n_iter = int(opti.stats().get("iter_count", -1))
    except Exception:
        n_iter = -1

    traj, c_true, sdf = _eval(u0)            # cost AT the initial guess
    traj_opt, c_opt, sdf_opt = _eval(u_opt)  # cost + trajectory at the solution
    feas = sdf_opt >= -1e-4
    end  = traj_opt[-1]

    results.append(dict(name=name, c0=c_true, cost=c_opt, sdf=sdf_opt,
                        feasible=feas, n_iter=n_iter, dt=dt,
                        status=status, end=end,
                        traj=traj_opt, u_opt=u_opt))
    print(f"  {name:<22s}  c0={c_true:8.2f} -> cost={c_opt:8.2f}  "
          f"sdf={sdf_opt:8.4f}  {'OK ' if feas else 'INF'}  "
          f"iter={n_iter:>3d}  {dt:5.1f}s  [{status}]")


# -- Summary -------------------------------------------------------------------
feas_costs = np.array([r["cost"] for r in results if r["feasible"]])
print("\n" + "=" * 78)
print("Sorted by converged true cost:")
print(f"  {'init guess':<22s}  {'cost0':>8s}  {'cost*':>8s}  {'sdf':>8s}  "
      f"{'feas':>4s}  {'iter':>4s}")
print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*4}  {'-'*4}")
for r in sorted(results, key=lambda r: r["cost"]):
    print(f"  {r['name']:<22s}  {r['c0']:8.2f}  {r['cost']:8.2f}  "
          f"{r['sdf']:8.4f}  {'yes' if r['feasible'] else 'no ':>4s}  "
          f"{r['n_iter']:>4d}")

print("\nSpread across initial guesses:")
all_costs = np.array([r["cost"] for r in results])
print(f"  all  : min={all_costs.min():.2f}  max={all_costs.max():.2f}  "
      f"mean={all_costs.mean():.2f}  std={all_costs.std():.2f}  "
      f"range={all_costs.max() - all_costs.min():.2f}")
if feas_costs.size:
    print(f"  feas : n={feas_costs.size}/{len(results)}  "
          f"min={feas_costs.min():.2f}  max={feas_costs.max():.2f}  "
          f"mean={feas_costs.mean():.2f}  std={feas_costs.std():.2f}  "
          f"range={feas_costs.max() - feas_costs.min():.2f}")
_feas = [r for r in results if r["feasible"]]
_best_pool  = _feas if _feas else results
best  = min(_best_pool,  key=lambda r: r["cost"])
worst = max(results, key=lambda r: r["cost"])
print(f"  best  feasible start: '{best['name']}'   cost={best['cost']:.2f}")
print(f"  worst start (any)  : '{worst['name']}'  cost={worst['cost']:.2f}")

# -- Save single data file (best AND worst trajectories + all 60 results) -----
_DATA.mkdir(parents=True, exist_ok=True)
out = str(_DATA / "ipopt.pkl")
with open(out, "wb") as f:
    pickle.dump({
        "u_opt"            : best["u_opt"],
        "trajectory"       : best["traj"],
        "cost"             : best["cost"],
        "min_clearance"    : best["sdf"],
        "u_worst"          : worst["u_opt"],
        "trajectory_worst" : worst["traj"],
        "cost_worst"       : worst["cost"],
        "all_results"      : [
            dict(name=r["name"], cost=r["cost"], sdf=r["sdf"],
                 feasible=r["feasible"], u=r["u_opt"], traj=r["traj"])
            for r in results
        ],
    }, f)
print(f"\nSaved -> {out}")
