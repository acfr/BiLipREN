"""
flow/flow_plot.py
=================
Figures for the trained CompREN flow. Merges the former flow_acf_qq.py and
flow_inverse.py so the model loading, `build_states`, `infer_arch` and the
plotting-style setup are shared instead of duplicated:

  * plot_acf_qq()     -> results/mbd_acf_qq.pdf
        Latent Gaussianity diagnostic (ACF + QQ of z = G(x)).
  * plot_generation() -> results/mbd_generation.pdf
        Generative sampling z ~ N(0, I) -> x_gen = G^-1(z); dataset vs generated.

Run from the project root:
    python flow/flow_plot.py
"""
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from BiLipRENs.utils import configure_device
configure_device("cpu")
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from plot.style import (apply_tac_style, FIG_W1, LW_THIN, DPI,
                        FS_TICK, FS_LABEL, FS_TITLE, FS_LEG,
                        PAD_W, PAD_H, WSPACE, HSPACE)
from BiLipRENs.ren_composition import CompREN
from BiLipRENs.ren_composition_inverse import CompRENinv

# ── Paths / constants ────────────────────────────────────────────────────────
_DATA    = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)
PARAMS_PATH  = _DATA / "mbd_flow_params.pkl"
DATASET      = _DATA / "dataset.npz"
LOWER, UPPER = 0.5, 2.0     # Lipschitz bounds of the flow model

# ACF / QQ diagnostic
ACF_SAMPLES = 200
MAX_LAG     = 54
DIM         = 2             # latent component to diagnose
# Generative sampling
GEN_SAMPLES = 500
N_BG        = 500
SEED        = 0


# ── Shared helpers ───────────────────────────────────────────────────────────
def build_states(batch_size, nx, num_layers):
    return [[jnp.zeros((batch_size, nx)) + 0.1] for _ in range(num_layers)]


def infer_arch(params):
    p = params["params"]
    block_indices = set(
        int(k.split("_")[1]) for k in p.keys()
        if k.startswith("models_") and len(k.split("_")) == 3 and k.endswith("_0")
    )
    num_layers = len(block_indices)
    ren0 = p["models_1_0"]
    nx, nu = ren0["B2"].shape
    nv = ren0["D12"].shape[0]
    return int(nu), int(nx), int(nv), int(num_layers)


def _apply_style():
    apply_tac_style()
    plt.rcParams.update({
        "font.size":             FS_LABEL,
        "axes.labelsize":        FS_LABEL,
        "axes.titlesize":        FS_TITLE,
        "xtick.labelsize":       FS_TICK,
        "ytick.labelsize":       FS_TICK,
        "legend.fontsize":       FS_LEG,
        "legend.title_fontsize": FS_LEG,
    })


def _acf_series(x, max_lag):
    """Normalized ACF for 1-D series x, lags 0..max_lag."""
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    return (r / var)[: max_lag + 1]


# ── Load model + dataset (shared, state_action: trajectories ++ actions) ─────
with open(PARAMS_PATH, "rb") as f:
    PARAMS = pickle.load(f)
NU, NX, NV, NUM_LAYERS = infer_arch(PARAMS)
MODEL = CompREN(NU, NX, NV, NUM_LAYERS, LOWER, UPPER, dyn_orth=False)

_DS        = np.load(str(DATASET))
TRAJ_YX    = np.asarray(_DS["trajectories"], dtype=np.float32)       # (N, T, 2) col0=y col1=x
INPUTS_ALL = np.concatenate(
    [TRAJ_YX, np.asarray(_DS["actions"], dtype=np.float32)], axis=-1)   # (N, T, 4)
T_STEPS    = int(INPUTS_ALL.shape[1])


def plot_acf_qq():
    """Latent Gaussianity diagnostic (ACF + QQ) -> results/mbd_acf_qq.pdf."""
    n = min(ACF_SAMPLES, INPUTS_ALL.shape[0])
    inp = INPUTS_ALL[:n]
    feat_mean = inp.reshape(-1, NU).mean(0)
    feat_std  = inp.reshape(-1, NU).std(0) + 1e-8
    inputs_norm = ((inp - feat_mean) / feat_std).astype(np.float32)
    x_seq = jnp.transpose(jnp.asarray(inputs_norm), (1, 0, 2))
    T = int(x_seq.shape[0])
    print(f"[ACF/QQ] nu={NU} nx={NX} nv={NV} L={NUM_LAYERS}  n={n} T={T}")

    @jax.jit
    def fwd_step(carry, x_t):
        states, p = carry
        new_states, z_t, _ = MODEL.apply(p, states, x_t, return_jacobians=True)
        return (new_states, p), z_t

    (_, _), z_seq = jax.lax.scan(
        fwd_step, (build_states(n, NX, NUM_LAYERS), PARAMS), x_seq)
    z_np = np.asarray(z_seq)
    print(f"  z shape: {z_np.shape}  mean={z_np.mean():.4f}  std={z_np.std():.4f}")

    max_lag = min(MAX_LAG, T - 1)
    lags = np.arange(max_lag + 1)
    acf_mean = np.stack(
        [_acf_series(z_np[:, i, DIM], max_lag) for i in range(n)], axis=0).mean(axis=0)
    conf = 1.96 / np.sqrt(T)

    z_flat = z_np[:, :, DIM].reshape(-1)
    (osm, osr), (slope, intercept, r_val) = stats.probplot(z_flat, dist="norm")

    _apply_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_W1, FIG_W1 * 0.399),
        constrained_layout={"w_pad": PAD_W, "h_pad": PAD_H,
                            "wspace": WSPACE, "hspace": HSPACE},
    )

    # Left: ACF
    ax_left.vlines(lags, 0, acf_mean, colors="black", linewidth=LW_THIN)
    ax_left.plot(lags, acf_mean, "o", color="#1f77b4", markersize=1.5, zorder=3)
    ax_left.axhline(0, color="black", linewidth=LW_THIN)
    ax_left.axhline( conf, color="black", linestyle="--", linewidth=LW_THIN,
                     label=f"95% CI ($\\pm${conf:.3f})")
    ax_left.axhline(-conf, color="black", linestyle="--", linewidth=LW_THIN)
    ax_left.set_ylim(min(-conf * 1.5, float(acf_mean[1:].min()) - 0.05), 1.1)
    ax_left.set_xlim(-0.5, max_lag + 0.5)
    ax_left.set_xlabel("Lag", fontsize=FS_LABEL)
    ax_left.set_ylabel("Autocorrelation", fontsize=FS_LABEL)
    ax_left.tick_params(axis="both", labelsize=FS_TICK)
    ax_left.legend(loc="upper right", fontsize=FS_LEG, handlelength=1.4)
    ax_left.grid(True, linewidth=0.4, alpha=0.6)

    # Right: QQ
    ax_right.scatter(osm, osr, s=2.25, color="#1f77b4", alpha=0.5, zorder=3,
                     label="$z$ quantiles")
    x_line = np.array([osm[0], osm[-1]])
    ax_right.plot(x_line, slope * x_line + intercept, color="black",
                  linewidth=LW_THIN, linestyle="--", zorder=4,
                  label=f"$R^2$={r_val**2:.4f}")
    ax_right.set_xlabel("Theoretical", fontsize=FS_LABEL)
    ax_right.set_ylabel("Sample", fontsize=FS_LABEL)
    ax_right.tick_params(axis="both", labelsize=FS_TICK)
    _qq_ticks = np.array([-4, -2, 0, 2, 4])
    ax_right.set_xticks(_qq_ticks)
    ax_right.set_yticks(_qq_ticks)
    ax_right.set_xlim(-4.5, 4.5)
    ax_right.set_ylim(-4.5, 4.5)
    ax_right.legend(loc="upper left", fontsize=FS_LEG, handlelength=1.4)
    ax_right.grid(True, linewidth=0.4, alpha=0.6)

    out_path = _RESULTS / "mbd_acf_qq.pdf"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_generation():
    """Generative sampling figure -> results/mbd_generation.pdf."""
    obstacle = np.asarray(_DS["obstacle_x_y_radius"], dtype=np.float32)
    goal_xy  = tuple(np.asarray(_DS["target_position"],   dtype=np.float32)[::-1])
    start_xy = tuple(np.asarray(_DS["starting_position"], dtype=np.float32)[::-1])

    n = max(1, min(GEN_SAMPLES, int(INPUTS_ALL.shape[0])))
    feat_mean = INPUTS_ALL.reshape(-1, NU).mean(0)
    feat_std  = INPUTS_ALL.reshape(-1, NU).std(0) + 1e-8
    T = T_STEPS
    print(f"[Generation] nu={NU} nx={NX} nv={NV} L={NUM_LAYERS}  "
          f"bounds=({LOWER},{UPPER})  generating n={n} trajectories, T={T}")

    _bg_rng = np.random.default_rng(42)
    bg_idx  = _bg_rng.choice(TRAJ_YX.shape[0], size=min(N_BG, TRAJ_YX.shape[0]), replace=False)
    bg_xy   = np.stack([TRAJ_YX[bg_idx, :, 1], TRAJ_YX[bg_idx, :, 0]], axis=-1)   # (N_BG, T, 2)

    inv_model  = CompRENinv(NU, NX, NV, NUM_LAYERS, LOWER, UPPER, dyn_orth=False)
    inv_params = CompRENinv.reverse_params(PARAMS, NUM_LAYERS, dyn_orth=False)

    with open(_DATA / "mbd_latents.pkl", "rb") as f:
        z_sample = jnp.asarray(pickle.load(f)["latent"])
    n = int(z_sample.shape[1])

    @jax.jit
    def inv_scan(carry, z_t):
        states, p = carry
        rev_states = [[states[NUM_LAYERS - 1 - i][0]] for i in range(NUM_LAYERS)]
        _, x_t = inv_model.apply(inv_params, rev_states, z_t)
        new_states, _ = MODEL.apply(p, states, x_t)
        return (new_states, p), x_t

    (_, _), x_gen_seq = jax.lax.scan(
        inv_scan, (build_states(n, NX, NUM_LAYERS), PARAMS), z_sample)
    x_gen_phys = np.asarray(x_gen_seq) * feat_std + feat_mean          # (T, n, nu) col0=y col1=x
    gen_xy = np.stack([x_gen_phys[..., 1], x_gen_phys[..., 0]], axis=-1)   # (T, n, 2)
    print(f"[Generated] gen_xy shape={tuple(gen_xy.shape)}")

    _apply_style()
    fig, (ax_data, ax_gen) = plt.subplots(
        1, 2, figsize=(FIG_W1, FIG_W1 * 0.416),
        constrained_layout={"w_pad": PAD_W, "h_pad": PAD_H,
                            "wspace": WSPACE, "hspace": HSPACE},
    )

    _XLIM = (-3, 6)
    _YLIM = (-2.243, 4.243)
    _BLUE = "#4488CC"

    def _add_scene(ax):
        for (cx, cy, r) in obstacle:
            ax.add_patch(Circle((cx, cy), r, color="#92c5de", ec="#555555",
                                linewidth=0.6, alpha=0.55, zorder=0))
        ax.plot(start_xy[0], start_xy[1], marker="*", markersize=9,
                linestyle="none", color="#0d6b2c", zorder=5)
        ax.plot(goal_xy[0], goal_xy[1], marker="*", markersize=9,
                linestyle="none", color="#8b0a0d", zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*_XLIM)
        ax.set_ylim(*_YLIM)
        ax.tick_params(axis="both", labelbottom=False, labelleft=False)
        ax.grid(True, linewidth=0.4, alpha=0.6)

    for i in range(bg_xy.shape[0]):
        ax_data.plot(bg_xy[i, :, 0], bg_xy[i, :, 1], color=_BLUE,
                     linewidth=LW_THIN, alpha=0.15, zorder=1)
    _add_scene(ax_data)
    ax_data.set_title("Dataset samples", fontsize=FS_TITLE)

    for k in range(gen_xy.shape[1]):
        ax_gen.plot(gen_xy[:, k, 0], gen_xy[:, k, 1], color=_BLUE,
                    linewidth=LW_THIN, alpha=0.15, zorder=3)
    _add_scene(ax_gen)
    ax_gen.set_title("Generated samples", fontsize=FS_TITLE)

    out_path = _RESULTS / "mbd_generation.pdf"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    plot_acf_qq()
    plot_generation()
