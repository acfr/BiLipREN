"""
robust_inv/msd_lip_sweep_table.py
=================================
Adversarial-robustness sweep table for the MSD experiment (reproduces the
paper's Table I):

  * the C-REN baseline is evaluated on the FORWARD map only;
  * the adversarial attack is the same per-step (L-inf) PGD used by
    msd_final_plot.py, so both report NSE against the identical adversary;
  * results are reported with the NSE metric only.

Metrics (washout-50 NSE, averaged over the whole validation set):

    Forward NSE : clean = NSE(f(u),      y_meas)
                  adv   = NSE(f(u+delta), y_meas)   delta from PGD, eps=FWD_EPS_FRAC*||u||
    Inverse NSE : clean = NSE(g(y_clean),      u_true)
                  adv   = NSE(g(y_clean+delta), u_true)  delta from PGD, eps=BWD_EPS_FRAC*||y||

Configurations (Table I):
    C-REN                          (forward only)
    BiLipREN  mu=0.1,  nu in {4, 8, 16, 32}
    BiLipREN  nu=8,    mu in {0.01, 0.05, 0.1, 0.5}

Run from anywhere:
    python robust_inv/msd_lip_sweep_table.py

Output:
    console table + robust_inv/results/msd_lip_sweep_table.txt
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# Put the workspace root on sys.path BEFORE importing BiLipRENs, so the local
# package wins over any editable install pointing elsewhere.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
from BiLipRENs.utils import configure_device
configure_device("cpu")
import jax.numpy as jnp

from BiLipRENs.ren_composition import CompREN
from BiLipRENs.ren_composition_inverse import CompRENinv
from robustnn import ren_jax as ren

jax.config.update("jax_default_matmul_precision", "highest")

# ── Architecture (identical for all sweep models) ────────────────────────────
NU, NX, NV, NUM_LAYERS = 1, 16, 64, 4
NX_CREN, NV_CREN = 8, 180

# ── Attack settings (per-step L-inf PGD, identical to msd_final_plot.py) ──────
FWD_EPS_FRAC = 0.1     # forward budget as fraction of ||u||_2
BWD_EPS_FRAC = 0.03    # inverse budget as fraction of ||y||_2
PGD_STEPS    = 150
STEP_FRAC    = 0.10
WASHOUT      = 50      # leading steps masked from every reported NSE
SEED         = 0

# ── Table I configurations ───────────────────────────────────────────────────
#   group A: fixed mu=0.1, sweep nu ; group B: fixed nu=8, sweep mu
GROUP_A = [(0.1, 4.0), (0.1, 8.0), (0.1, 16.0), (0.1, 32.0)]
GROUP_B = [(0.01, 8.0), (0.05, 8.0), (0.1, 8.0), (0.5, 8.0)]

_MODELS  = _HERE / "models" / "lip_sweep"
_CREN    = _HERE / "models" / "cren" / "msd_cren_fwd_params.pkl"
_DATA    = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)


def _model_path(lower, upper):
    return _MODELS / f"msd_params_l{lower}_u{upper}.pkl"


# ── Data (measured + clean validation sets) ──────────────────────────────────
with open(_DATA / "msd_data_val.pkl", "rb") as f:
    _u, _y = pickle.load(f)
with open(_DATA / "msd_data_val_clean.pkl", "rb") as f:
    _uc, _yc = pickle.load(f)
U_VAL   = jnp.asarray(np.asarray(_u,  np.float32))   # (n, T, 1)
Y_MEAS  = jnp.asarray(np.asarray(_y,  np.float32))   # measured (noisy) y
Y_CLEAN = jnp.asarray(np.asarray(_yc, np.float32))   # clean reference y
print(f"val: u={U_VAL.shape}  y_meas={Y_MEAS.shape}  y_clean={Y_CLEAN.shape}")


# ── Rollouts ─────────────────────────────────────────────────────────────────
def _make_forward(lower, upper):
    model = CompREN(NU, NX, NV, NUM_LAYERS, lower, upper, dyn_orth=False)

    def fwd(params, u_bm):
        n   = u_bm.shape[0]
        u_T = jnp.transpose(u_bm, (1, 0, 2))
        s0  = [[jnp.zeros((n, NX))] for _ in range(NUM_LAYERS)]

        def scan_fn(carry, inp):
            s, p = carry
            ns, out = model.apply(p, s, inp)
            return (ns, p), out

        _, y_T = jax.lax.scan(scan_fn, (s0, params), u_T)
        return jnp.transpose(y_T, (1, 0, 2))

    return fwd


def _make_inverse(lower, upper):
    inv_model = CompRENinv(NU, NX, NV, NUM_LAYERS, lower, upper, dyn_orth=False)

    def inv(inv_params, y_bm):
        n   = y_bm.shape[0]
        y_T = jnp.transpose(y_bm, (1, 0, 2))
        s0  = [[jnp.zeros((n, NX))] for _ in range(NUM_LAYERS)]

        def scan_fn(carry, inp):
            s, p = carry
            ns, out = inv_model.apply(p, s, inp)
            return (ns, p), out

        _, u_T = jax.lax.scan(scan_fn, (s0, inv_params), y_T)
        return jnp.transpose(u_T, (1, 0, 2))

    return inv


def _make_cren_forward():
    cren = ren.ContractingREN(input_size=1, state_size=NX_CREN,
                              features=NV_CREN, output_size=1)

    def fwd(params, u_bm):
        n   = u_bm.shape[0]
        u_T = jnp.transpose(u_bm, (1, 0, 2))
        x0  = jnp.zeros((n, NX_CREN))
        _, y_T = cren.simulate_sequence(params, x0, u_T)
        return jnp.transpose(y_T, (1, 0, 2))

    return fwd


# ── NSE (per trajectory); _nse_w masks the leading washout window ────────────
def _nse(pred, target):
    num = jnp.sqrt(((pred - target) ** 2).sum(axis=(1, 2)))
    den = jnp.sqrt((target ** 2).sum(axis=(1, 2))) + 1e-12
    return num / den


def _nse_w(pred, target):
    return _nse(pred[:, WASHOUT:, :], target[:, WASHOUT:, :])


def pgd_attack(rollout_fn, params, x_base, target, eps_frac, key, metric=_nse):
    """Per-step (L-inf) PGD attack — identical to msd_final_plot.py so the
    sweep table computes its NSE values against the same adversary.

    A per-element L-inf budget ``eps_step`` is used so a fully saturated
    perturbation has the same L2 size as the eps_frac*||x||_2 ball.
    """
    x_base = jnp.asarray(x_base)
    target = jnp.asarray(target)
    n_elem = x_base.shape[1] * x_base.shape[2]
    eps_step = (eps_frac * jnp.sqrt((x_base ** 2).sum(axis=(1, 2)))
                / jnp.sqrt(n_elem))[:, None, None]
    step = STEP_FRAC * eps_step

    def loss(delta):
        return metric(rollout_fn(params, x_base + delta), target).sum()

    grad_fn = jax.jit(jax.grad(loss))

    def project(delta):
        return jnp.clip(delta, -eps_step, eps_step)

    delta = project(1e-3 * eps_step * jax.random.normal(key, x_base.shape))
    for _ in range(PGD_STEPS):
        g = grad_fn(delta)
        delta = project(delta + step * jnp.sign(g))
    return x_base + delta


def _mean_nse(pred, target):
    return float(jnp.mean(_nse_w(pred, target)))


# ── Evaluate C-REN (forward only) ────────────────────────────────────────────
print("\n=== C-REN forward baseline ===")
with open(_CREN, "rb") as f:
    cren_params = pickle.load(f)
cren_fwd = _make_cren_forward()
cren_clean = _mean_nse(cren_fwd(cren_params, U_VAL), Y_MEAS)
u_adv = pgd_attack(cren_fwd, cren_params, U_VAL, Y_MEAS,
                   FWD_EPS_FRAC, jax.random.PRNGKey(SEED + 999))
cren_adv = _mean_nse(cren_fwd(cren_params, u_adv), Y_MEAS)
print(f"  forward: clean NSE={cren_clean:.4f}  adv NSE={cren_adv:.4f}")


# ── Evaluate one BiLipREN config (forward + inverse) ─────────────────────────
def eval_bilipren(lower, upper):
    with open(_model_path(lower, upper), "rb") as f:
        params = pickle.load(f)
    inv_params = CompRENinv.reverse_params(params, NUM_LAYERS, dyn_orth=False)
    fwd_fn = _make_forward(lower, upper)
    inv_fn = _make_inverse(lower, upper)
    key = jax.random.PRNGKey(SEED)

    # Forward: attack u toward the measured y
    fwd_clean = _mean_nse(fwd_fn(params, U_VAL), Y_MEAS)
    key, sub = jax.random.split(key)
    u_adv = pgd_attack(fwd_fn, params, U_VAL, Y_MEAS, FWD_EPS_FRAC, sub)
    fwd_adv = _mean_nse(fwd_fn(params, u_adv), Y_MEAS)

    # Inverse: attack the clean y toward the true u
    inv_clean = _mean_nse(inv_fn(inv_params, Y_CLEAN), U_VAL)
    key, sub = jax.random.split(key)
    y_adv = pgd_attack(inv_fn, inv_params, Y_CLEAN, U_VAL, BWD_EPS_FRAC, sub,
                       metric=_nse_w)
    inv_adv = _mean_nse(inv_fn(inv_params, y_adv), U_VAL)
    return fwd_clean, fwd_adv, inv_clean, inv_adv


rows = []   # (arch, mu, nu, fwd_clean, fwd_adv, inv_clean, inv_adv)
rows.append(("C-REN", None, None, cren_clean, cren_adv, None, None))

for lo, up in GROUP_A:
    print(f"=== BiLipREN mu={lo} nu={up} ===")
    fc, fa, ic, ia = eval_bilipren(lo, up)
    print(f"  forward: {fc:.4f}/{fa:.4f}   inverse: {ic:.4f}/{ia:.4f}")
    rows.append(("BiLipREN", lo, up, fc, fa, ic, ia))

for lo, up in GROUP_B:
    print(f"=== BiLipREN mu={lo} nu={up} ===")
    fc, fa, ic, ia = eval_bilipren(lo, up)
    print(f"  forward: {fc:.4f}/{fa:.4f}   inverse: {ic:.4f}/{ia:.4f}")
    rows.append(("BiLipREN", lo, up, fc, fa, ic, ia))


# ── Render Table I ───────────────────────────────────────────────────────────
def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


lines = []
lines.append("Table I — MSD adversarial robustness (per-step L-inf PGD, NSE only)")
lines.append(f"  forward eps={FWD_EPS_FRAC}*||u||   inverse eps={BWD_EPS_FRAC}*||y||   "
             f"washout={WASHOUT}   mean over {U_VAL.shape[0]} val trajectories")
lines.append("")
lines.append(f"{'Arch':<9} {'mu':>5} {'nu':>4} | "
             f"{'Fwd Clean':>10} {'Fwd Adv':>9} | {'Inv Clean':>10} {'Inv Adv':>9}")
lines.append("-" * 66)
for arch, mu, nu, fc, fa, ic, ia in rows:
    mu_s = "  -  " if mu is None else f"{mu:g}"
    nu_s = " - " if nu is None else f"{nu:g}"
    lines.append(f"{arch:<9} {mu_s:>5} {nu_s:>4} | "
                 f"{_fmt(fc):>10} {_fmt(fa):>9} | {_fmt(ic):>10} {_fmt(ia):>9}")

table = "\n".join(lines)
print("\n" + table)

out = _RESULTS / "msd_lip_sweep_table.txt"
out.write_text(table + "\n", encoding="utf-8")
print(f"\nSaved: {out}")
