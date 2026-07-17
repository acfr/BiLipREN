"""
robust_inv/msd_final_plot.py
============================
Self-contained reproduction of the paper's forward/inverse adversarial figure
(`results/msd_validation.pdf`). For two BiLipREN configurations

    column 1 : mu = 0.1,  nu = 8
    column 2 : mu = 0.01, nu = 8

it runs a white-box PGD attack on a single shared validation trajectory and
draws a 2x2 grid:

    top    row : forward  y = G(u)        (ground truth / clean / adversarial)
    bottom row : inverse  u = G^{-1}(y)   (ground truth / clean / adversarial)

The shown trajectory is auto-selected as the one with the smallest BiLipREN
inverse adversarial NSE for the reference config (mu=0.1, nu=8), so both
columns display the same, cleanly-invertible trajectory. No intermediate cache
is written; only the final image is saved.

Run from project root:
    python robust_inv/msd_final_plot.py
"""

from __future__ import annotations
import os, sys, pickle
from pathlib import Path

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BiLipRENs.ren_composition import CompREN
from BiLipRENs.ren_composition_inverse import CompRENinv
from plot.style import (apply_tac_style, FIG_W1, LW_THIN, savefig,
                        FS_TICK, FS_LABEL, FS_TITLE, FS_LEG,
                        PAD_W, PAD_H, WSPACE, HSPACE)

_HERE       = Path(__file__).resolve().parent
DATA_VAL    = _HERE / "data" / "msd_data_val.pkl"
DATA_CLEAN  = _HERE / "data" / "msd_data_val_clean.pkl"
MODELS_DIR  = _HERE / "models" / "lip_sweep"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Architecture (identical for all sweep models)
NU, NX, NV, NUM_LAYERS = 1, 16, 64, 4

# Attack hyper-parameters (match the published figure)
FWD_EPS_FRAC = 0.1
BWD_EPS_FRAC = 0.03
PGD_STEPS    = 150
STEP_FRAC    = 0.10
ATTACK_SEED  = 0

# Window / display
T_PLOT      = 300
START       = 0
WASHOUT_SEL = 50     # leading steps masked from the trajectory-selection NSE
BURN_INV    = 50     # leading steps cropped from the DISPLAY (cold-start transient)

# Reference config used to pick the shown trajectory.
SELECT_LOWER, SELECT_UPPER = 0.1, 8.0

# Columns: (lower, upper, column title)
CONFIGS = [
    (0.1,  8.0, r"$\mu = 0.1$"),
    (0.01, 8.0, r"$\mu = 0.01$"),
]

COLOR_GT, COLOR_CLEAN, COLOR_ATK = "black", "#1f77b4", "#FF7F0E"


def _mp(lo, up):
    return MODELS_DIR / f"msd_params_l{lo}_u{up}.pkl"


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


def _nse(pred, target):
    num = jnp.sqrt(((pred - target) ** 2).sum(axis=(1, 2)))
    den = jnp.sqrt((target ** 2).sum(axis=(1, 2))) + 1e-12
    return num / den


def _nse_w(pred, target):
    return _nse(pred[:, WASHOUT_SEL:, :], target[:, WASHOUT_SEL:, :])


def pgd_attack(rollout_fn, params, x_base, target, eps_frac, key, metric=_nse):
    """Per-step (L-inf) PGD: each timestep is perturbed by at most eps_step,
    scaled so a fully-saturated perturbation matches an L2 ball of eps_frac*||x||."""
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


# ── Load data ────────────────────────────────────────────────────────────────
with open(DATA_VAL, "rb") as f:
    _u_all, _y_all = pickle.load(f)
with open(DATA_CLEAN, "rb") as f:
    _u_c, _y_c = pickle.load(f)
_u_all = np.asarray(_u_all, np.float32)
_y_all = np.asarray(_y_all, np.float32)
_y_cln = np.asarray(_y_c, np.float32)


# ── Select the trajectory (smallest inverse adv NSE for the reference cfg) ────
print("=== selecting trajectory (mu=0.1, nu=8 inverse adv NSE) ===")
with open(_mp(SELECT_LOWER, SELECT_UPPER), "rb") as f:
    _sel_params = pickle.load(f)
_sel_inv_params = CompRENinv.reverse_params(_sel_params, NUM_LAYERS, dyn_orth=False)
_sel_inv = _make_inverse(SELECT_LOWER, SELECT_UPPER)
_adv_scores = []
for _i in range(_u_all.shape[0]):
    _ut = _u_all[_i, START:START + T_PLOT, :]
    _yc = _y_cln[_i, START:START + T_PLOT, :]
    _ub = jnp.asarray(_ut[None])
    _yb = jnp.asarray(_yc[None])
    _ya = pgd_attack(_sel_inv, _sel_inv_params, _yb, _ub, BWD_EPS_FRAC,
                     jax.random.PRNGKey(ATTACK_SEED), metric=_nse_w)
    _adv = float(_nse_w(_sel_inv(_sel_inv_params, _ya), _ub)[0])
    _adv_scores.append(_adv)
    print(f"  traj #{_i}: inverse adv NSE={_adv:.4f}")
TRAJ_IDX = int(np.argmin(_adv_scores))
print(f"  -> selected traj #{TRAJ_IDX} (adv NSE={_adv_scores[TRAJ_IDX]:.4f})")

u_true  = _u_all[TRAJ_IDX, START:START + T_PLOT, :]
y_meas  = _y_all[TRAJ_IDX, START:START + T_PLOT, :]
y_clean = _y_cln[TRAJ_IDX, START:START + T_PLOT, :]
_u_batch   = jnp.asarray(u_true[None])
_ymeas_bm  = jnp.asarray(y_meas[None])
_yclean_bm = jnp.asarray(y_clean[None])


# ── Compute forward / inverse clean + adv for each config ────────────────────
_sl = slice(BURN_INV, None)
t = np.arange(T_PLOT - BURN_INV)
panels = []

for lower, upper, title in CONFIGS:
    print(f"=== config mu={lower} nu={upper} ===")
    with open(_mp(lower, upper), "rb") as f:
        params = pickle.load(f)
    inv_params = CompRENinv.reverse_params(params, NUM_LAYERS, dyn_orth=False)
    fwd_fn = _make_forward(lower, upper)
    inv_fn = _make_inverse(lower, upper)
    key = jax.random.PRNGKey(ATTACK_SEED)

    # Forward attack (u -> y), eps = FWD_EPS_FRAC * ||u||
    y_clean_pred = np.asarray(fwd_fn(params, _u_batch))[0, :, 0]
    key, sub = jax.random.split(key)
    u_adv = pgd_attack(fwd_fn, params, _u_batch, _ymeas_bm, FWD_EPS_FRAC, sub)
    y_adv_pred = np.asarray(fwd_fn(params, u_adv))[0, :, 0]

    # Inverse attack (y -> u), eps = BWD_EPS_FRAC * ||y||
    u_clean_rec = np.asarray(inv_fn(inv_params, _yclean_bm))[0, :, 0]
    key, sub = jax.random.split(key)
    y_adv = pgd_attack(inv_fn, inv_params, _yclean_bm, _u_batch, BWD_EPS_FRAC,
                       sub, metric=_nse_w)
    u_adv_rec = np.asarray(inv_fn(inv_params, y_adv))[0, :, 0]

    panels.append(dict(
        title=title,
        y_meas=y_meas[_sl, 0], y_clean_pred=y_clean_pred[_sl],
        y_adv_pred=y_adv_pred[_sl],
        u_true=u_true[_sl, 0], u_clean_rec=u_clean_rec[_sl],
        u_adv_rec=u_adv_rec[_sl],
    ))


# ── Draw ─────────────────────────────────────────────────────────────────────
apply_tac_style()
plt.rcParams.update({
    "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEG, "legend.title_fontsize": FS_LEG,
})

fig, axes = plt.subplots(
    2, 2, sharex=True, sharey="row",
    figsize=(FIG_W1, FIG_W1 * 0.756),
    constrained_layout={"w_pad": PAD_W, "h_pad": PAD_H,
                        "wspace": WSPACE, "hspace": HSPACE},
)

l_gt = l_cl = l_atk = None
for c, d in enumerate(panels):
    # Forward panel (top row)
    ax = axes[0, c]
    _g, = ax.plot(t, d["y_meas"], color=COLOR_GT, linewidth=LW_THIN,
                  linestyle="--", zorder=5)
    _a, = ax.plot(t, d["y_adv_pred"], color=COLOR_ATK, linewidth=LW_THIN,
                  alpha=0.9, zorder=3)
    _c, = ax.plot(t, d["y_clean_pred"], color=COLOR_CLEAN, linewidth=LW_THIN,
                  alpha=0.9, zorder=4)
    l_gt, l_atk, l_cl = _g, _a, _c
    ax.set_title(d["title"], fontsize=FS_TITLE)
    if c == 0:
        ax.set_ylabel(r"$\boldsymbol{y} = \mathcal{G}(\boldsymbol{u})$",
                      fontsize=FS_LABEL)
    ax.set_xlim(0, float(t[-1]))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    # Inverse panel (bottom row)
    ax = axes[1, c]
    ax.plot(t, d["u_true"], color=COLOR_GT, linewidth=LW_THIN, linestyle="--",
            zorder=5)
    ax.plot(t, d["u_adv_rec"], color=COLOR_ATK, linewidth=LW_THIN, alpha=0.9,
            zorder=3)
    ax.plot(t, d["u_clean_rec"], color=COLOR_CLEAN, linewidth=LW_THIN, alpha=0.9,
            zorder=4)
    if c == 0:
        ax.set_ylabel(r"$\boldsymbol{u} = \mathcal{G}^{-1}(\boldsymbol{y})$",
                      fontsize=FS_LABEL)
    ax.set_xlim(0, float(t[-1]))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, linewidth=0.4, alpha=0.6)

for ch in range(2):
    axes[1, ch].set_xlabel("Time step", fontsize=FS_LABEL)

axes[0, 0].legend(
    [l_gt, l_cl, l_atk],
    ["Ground truth", "BiLipREN (clean)", "BiLipREN (adv.)"],
    loc="upper left", fontsize=FS_LEG, handlelength=1.2,
    handletextpad=0.4, labelspacing=0.3, borderpad=0.3, frameon=True,
    framealpha=0.9, edgecolor="0.5")

fig.align_ylabels(axes[:, 0])

out = RESULTS_DIR / "msd_validation.pdf"
savefig(fig, out, pad_inches=0.03)
plt.close(fig)
print(f"Saved: {out}")
