"""
io_fact/io_plot.py
==================
Decompose the trained inner-outer ORTHREN model and draw
``results/io_fact.pdf``:

    top    panel : impulse response of the inner factor O(z) (dynamic orthogonal)
    bottom panel : composite system M vs. outer factor G_min vs. true plant,
                   on a single test trajectory.

All network sizes are read from the model's ``meta`` (written by io_train.py),
so the only knob is ``TRAJ`` (which test trajectory to display).

Run from the project root:
    python io_fact/io_plot.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# --- Which test trajectory to plot (the only knob) -------------------------
TRAJ = 504

# --- Paths -----------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[2]
INNER_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(INNER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot import (apply_tac_style, COLORS, FIG_W1, LW_MAIN, LW_REF, LW_THIN,
                  savefig, FS_TICK, FS_LABEL, FS_TITLE, FS_LEG,
                  PAD_W, PAD_H, WSPACE, HSPACE)
from BiLipRENs.orthogonal_layer import DynOrthogonal
from BiLipRENs.ren_composition import ORTHREN
from BiLipRENs.utils import cayley

RESULTS_DIR = INNER_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data and trained model -------------------------------------------
with open(INNER_DIR / "data" / "inner_outer_data.pkl", "rb") as f:
    inputs_data, outputs_data = pickle.load(f)

with open(INNER_DIR / "models" / "inner_outer_params.pkl", "rb") as f:
    blob = pickle.load(f)

params_tree = blob["params"]
params      = {"params": params_tree}
meta        = blob["meta"]

# Core (network) parameters, taken straight from training meta (paper notation).
num_layers = meta["num_layers"]
n_in       = meta["n_in"]          # input dimension
nx, nv     = meta["nx"], meta["nv"]
mu, nu     = meta["mu"], meta["nu"]
dyn_mult   = meta["dyn_mult"]

T = min(50, inputs_data.shape[1])
time = np.arange(T)

# Single test trajectory, shaped (T, batch=1, n_in) for lax.scan over time.
u_traj = jnp.asarray(inputs_data[TRAJ, :T, :])[:, None, :]

# True plant y[k] = alpha*tanh(y[k-1]) + u[k-delay] (alpha=0.9, delay=3),
# simulated on the fly with no warmup discard so the reference matches the
# paper figure exactly (the stored dataset drops the first
# DELAY+1 samples, which shifts the transient by a few steps).
_ALPHA_IO, _DELAY_IO = 0.9, 3


def _sim_io_plant(u_arr, delay=_DELAY_IO, alpha=_ALPHA_IO):
    Tn = u_arr.shape[0]
    out = np.zeros(Tn, dtype=np.float32)
    y_prev = 0.0
    buf = [0.0] * delay if delay > 0 else []
    for k in range(Tn):
        if delay > 0:
            buf.append(float(u_arr[k, 0]))
            u_d = buf.pop(0)
        else:
            u_d = float(u_arr[k, 0])
        y_new = alpha * np.tanh(y_prev) + u_d
        out[k] = y_new
        y_prev = y_new
    return out


y_true = _sim_io_plant(np.asarray(inputs_data[TRAJ, :T, :]))


def run(model, model_params, states):
    """Run ``model`` over the single trajectory; return the (T,) output."""
    def step(carry, x):
        st, p = carry
        new_st, y = model.apply(p, st, x)
        return (new_st, p), y
    _, out = jax.lax.scan(step, (states, model_params), u_traj)
    return np.asarray(out[:, 0, 0])


# --- Composite system M (full model, with output-side DynOrth) -------------
comp_model = ORTHREN(n_in, nx, nv, num_layers, mu, nu,
                     dyn_orth=False, dyn_orth_at_output=True,
                     dyn_orth_state_multiplier=dyn_mult)
comp_states = [[jnp.zeros((1, nx))] for _ in range(num_layers - 1)]
comp_states.append([jnp.zeros((1, nx)), jnp.zeros((1, dyn_mult * nx))])
comp_out = run(comp_model, params, comp_states)

# --- Outer factor G_min (same weights, DynOrth stripped) -------------------
outer_model = ORTHREN(n_in, nx, nv, num_layers, mu, nu,
                      dyn_orth=False, dyn_orth_at_output=False)
outer_keys = (["models_0"]
              + [f"models_{i}_0" for i in range(1, num_layers + 1)]
              + [f"models_{i}_1" for i in range(1, num_layers + 1)]
              + [f"models_{num_layers + 1}"])
outer_params = {"params": {k: params_tree[k] for k in outer_keys}}
outer_states = [[jnp.zeros((1, nx))] for _ in range(num_layers)]
gmin_out = run(outer_model, outer_params, outer_states)

# --- Impulse response of the inner factor (OutOrth rotation -> DynOrth) -----
# Apply only OutOrth's rotation R^T (skip its bias) so a pure unit impulse
# stays inside [-1, 1] through the non-expansive dynamic block.
out_orth = params_tree[f"models_{num_layers + 1}"]
R = cayley((out_orth["a"] / jnp.linalg.norm(out_orth["W"])) * out_orth["W"])
impulse = jnp.zeros((T, 1, n_in)).at[0].set(jnp.ones((1, n_in)))
imp_in = impulse @ R.T

dynorth = DynOrthogonal((dyn_mult * nx, n_in))
dyn_params = {"params": params_tree[f"models_{num_layers + 2}"]}


def run_dyn(model, model_params, states, seq):
    def step(carry, x):
        st, p = carry
        new_st, y = model.apply(p, st, x)
        return (new_st, p), y
    _, out = jax.lax.scan(step, (states, model_params), seq)
    return np.asarray(out[:, 0, 0])


imp_resp = run_dyn(dynorth, dyn_params, jnp.zeros((1, dyn_mult * nx)), imp_in)
imp_sig  = np.asarray(impulse[:, 0, 0])

# --- Figure (TAC single-column style) ---------------------------------------
_STEM_LW = 1.0
_MKR_SZ  = 2.2

apply_tac_style()
plt.rcParams.update({
    "font.size":             FS_LABEL,
    "axes.titlesize":        FS_TITLE,
    "axes.labelsize":        FS_LABEL,
    "xtick.labelsize":       FS_TICK,
    "ytick.labelsize":       FS_TICK,
    "legend.fontsize":       FS_LEG,
    "legend.title_fontsize": FS_LEG,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(FIG_W1, FIG_W1 * 0.705),
    constrained_layout={"w_pad": PAD_W, "h_pad": PAD_H,
                        "wspace": WSPACE, "hspace": HSPACE},
)


def draw_stem(ax, y, color, label):
    markers, stems, _ = ax.stem(time, y, linefmt="-", markerfmt="o",
                                basefmt=" ", label=label)
    plt.setp(markers, color=color, markersize=_MKR_SZ, zorder=4)
    plt.setp(stems, color=color, linewidth=_STEM_LW, zorder=3)


# Top: impulse response of the inner factor.
draw_stem(ax1, imp_sig, "black", "Impulse")
draw_stem(ax1, imp_resp, "#FF7F0E", "Inner system")
ax1.set_ylabel("Amplitude", fontsize=FS_LABEL)
ax1.tick_params(axis="both", labelsize=FS_TICK)
ax1.tick_params(axis="x", labelbottom=False, bottom=False)
ax1.legend(loc="best", fontsize=FS_LEG)

# Bottom: composite vs outer (G_min) vs true system.
ax2.plot(time, comp_out, color="#FF7F0E", linewidth=LW_THIN, linestyle="-",
         label="Composite system", zorder=2)
ax2.plot(time, gmin_out, color=COLORS["output_1"], linewidth=LW_THIN,
         label="Outer system", zorder=3)
ax2.plot(time, y_true, color=COLORS["reference"], linewidth=LW_THIN,
         linestyle="--", label="True system", zorder=4)
ax2.set_xlabel("Time step", fontsize=FS_LABEL)
ax2.set_ylabel("Amplitude", fontsize=FS_LABEL)
ax2.tick_params(axis="both", labelsize=FS_TICK)
ax2.legend(loc="best", fontsize=FS_LEG)

fig.align_ylabels([ax1, ax2])

out_pdf = RESULTS_DIR / "io_fact.pdf"
savefig(fig, out_pdf, pad_inches=0.03)
plt.close(fig)
print(f"Saved: {out_pdf}")
