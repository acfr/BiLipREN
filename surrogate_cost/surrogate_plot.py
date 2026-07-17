"""
surrogate_cost/surrogate_plot.py
================================
Rebuild the canonical combined 3x3 figure using clean-only inputs.

Rows    : trajectory | distance-to-goal | control norm
Columns : LSTM | C-REN | BiLipREN

Overlays in every panel:
    - dataset background samples (gray)
    - sample best / sample worst (red solid / red dashed)
    - IPOPT best / IPOPT worst (black solid / black dashed)
    - method best / method worst (method color solid / dashed)

Inputs:
    data/dataset.npz
    data/ipopt.pkl
    data/LSTM_search.pkl
    data/CREN_search.pkl
    data/bilipren.pkl

Outputs:
    results/mbd_pl.pdf
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from plot.style import (
        apply_tac_style,
        FIG_W2,
        LW_THIN,
        DPI,
        FS_TICK,
        FS_LABEL,
        FS_TITLE,
        FS_LEG,
        savefig_legend_margin,
    )
    apply_tac_style()
except Exception:
    FIG_W2, LW_THIN, DPI = 7.16, 1.0, 300
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEG = 9, 9, 9, 7

    def savefig_legend_margin(fig, path, legend, fixed_width=7.16, dpi=300):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)

START_YX = np.array([-1.0, -2.0], dtype=float)
GOAL_YX = np.array([3.0, 5.0], dtype=float)
GOAL_XY = np.array([5.0, 3.0], dtype=float)
MAX_STEP = 0.4

_COL_DATA = "#999999"
_COL_BEST = "#d62728"
_COL_IPOPT = "black"
_COL_LSTM = "#ff7f0e"
_COL_CREN = "#984ea3"
_COL_BILIP = "#1f77b4"


def _rollout_yx(u_yx, start_yx=START_YX):
    pos = np.asarray(start_yx, float).copy()
    pts = []
    for u in np.asarray(u_yx, float):
        n = np.hypot(u[0], u[1]) + 1e-8
        pos = pos + 1.0 / (1.0 + np.exp(-n / 10.0)) * MAX_STEP * (u / n)
        pts.append(pos.copy())
    return np.asarray(pts)


def _dist_xy(traj_xy):
    return np.linalg.norm(traj_xy - GOAL_XY[None, :], axis=1)


def _norm(a):
    return np.linalg.norm(np.asarray(a, float), axis=-1)


data = np.load(str(_DATA / "dataset.npz"))
state_data = np.asarray(data["trajectories"], dtype=float)
actions_data = np.asarray(data["actions"], dtype=float)
obstacle = np.asarray(data["obstacle_x_y_radius"], dtype=float)
costs = np.asarray(data["costs"], dtype=float)
T = int(actions_data.shape[1])
steps = np.arange(1, T + 1)

rng = np.random.default_rng(0)
bg_idx = rng.choice(state_data.shape[0], size=min(400, state_data.shape[0]), replace=False)
bg_yx = state_data[bg_idx]
bg_xy = np.stack([bg_yx[..., 1], bg_yx[..., 0]], axis=-1)
d_rand = np.linalg.norm(bg_xy - GOAL_XY[None, None, :], axis=-1)
n_rand = _norm(actions_data[bg_idx])

ib = int(np.argmin(costs))
iw = int(np.argmax(costs))
u_ds_best_yx = actions_data[ib]
u_ds_worst_yx = actions_data[iw]
traj_ds_best_yx = _rollout_yx(u_ds_best_yx)
traj_ds_worst_yx = _rollout_yx(u_ds_worst_yx)
traj_ds_best_xy = np.vstack([np.array([[-2.0, -1.0]]), traj_ds_best_yx[:, [1, 0]]])
traj_ds_worst_xy = np.vstack([np.array([[-2.0, -1.0]]), traj_ds_worst_yx[:, [1, 0]]])
d_ds_best = _dist_xy(traj_ds_best_xy[1:])
d_ds_worst = _dist_xy(traj_ds_worst_xy[1:])
n_ds_best = _norm(u_ds_best_yx)
n_ds_worst = _norm(u_ds_worst_yx)

with open(_DATA / "ipopt.pkl", "rb") as f:
    ipopt = pickle.load(f)
u_ip_best_xy = np.asarray(ipopt["u_opt"], dtype=float)
traj_ip_best_xy = np.asarray(ipopt["trajectory"], dtype=float)
d_ip_best = _dist_xy(traj_ip_best_xy[1:])
n_ip_best = _norm(u_ip_best_xy)

# Worst-case IPOPT run from the initial-guess sensitivity sweep (highest
# converged cost) — stored alongside the best run in data/ipopt.pkl.
traj_ip_worst_xy = np.asarray(ipopt["trajectory_worst"], dtype=float)
u_ip_worst_xy = np.asarray(ipopt["u_worst"], dtype=float)
d_ip_worst = _dist_xy(traj_ip_worst_xy[1:])
n_ip_worst = _norm(u_ip_worst_xy)

with open(_DATA / "LSTM_search.pkl", "rb") as f:
    lstm = pickle.load(f)
with open(_DATA / "CREN_search.pkl", "rb") as f:
    cren = pickle.load(f)
with open(_DATA / "bilipren.pkl", "rb") as f:
    bilip = pickle.load(f)


def _load_model_bundle(name, search, color, worst_override=None):
    all_u = np.asarray(search["all_u"], dtype=float)
    start = np.asarray(search["start"], dtype=float)
    target = np.asarray(search["target"], dtype=float)
    all_roll = np.asarray([_rollout_yx(u, start) for u in all_u])
    all_final_d = np.linalg.norm(all_roll[:, -1, :] - target[None, :], axis=1)
    i_worst = int(worst_override) if worst_override is not None else int(np.argmax(all_final_d))
    u_best = np.asarray(search["u_best_surrogate"], dtype=float)
    u_worst = np.asarray(all_u[i_worst], dtype=float)
    traj_best_yx = _rollout_yx(u_best, start)
    traj_worst_yx = _rollout_yx(u_worst, start)
    return {
        "name": name,
        "color": color,
        "all_roll": all_roll,
        "traj_best_yx": traj_best_yx,
        "traj_worst_yx": traj_worst_yx,
        "d_best": np.linalg.norm(traj_best_yx - GOAL_YX[None, :], axis=1),
        "d_worst": np.linalg.norm(traj_worst_yx - GOAL_YX[None, :], axis=1),
        "n_best": _norm(u_best),
        "n_worst": _norm(u_worst),
    }


def _load_bilip_bundle():
    u_best = np.asarray(bilip["actions_inv"], dtype=float)
    traj_best_yx = _rollout_yx(u_best, START_YX)
    return {
        "name": "BiLipREN",
        "color": _COL_BILIP,
        "all_roll": None,
        "traj_best_yx": traj_best_yx,
        "traj_worst_yx": None,
        "d_best": np.linalg.norm(traj_best_yx - GOAL_YX[None, :], axis=1),
        "d_worst": None,
        "n_best": _norm(u_best),
        "n_worst": None,
    }


models = [
    _load_model_bundle("LSTM", lstm, _COL_LSTM),
    _load_model_bundle("C-REN", cren, _COL_CREN, worst_override=42),
    _load_bilip_bundle(),
]

plt.rcParams.update(
    {
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
        "savefig.pad_inches": 0.03,
    }
)

fig = plt.figure(figsize=(FIG_W2, FIG_W2 * 0.68))
outer = fig.add_gridspec(2, 3, height_ratios=[1.45, 2.05], hspace=0.22, wspace=0.06)

for j, m in enumerate(models):
    ax_t = fig.add_subplot(outer[0, j])
    sub = outer[1, j].subgridspec(2, 1, hspace=0.12)
    ax_d = fig.add_subplot(sub[0])
    ax_u = fig.add_subplot(sub[1], sharex=ax_d)

    for (cx, cy, r) in obstacle:
        ax_t.add_patch(Circle((cx, cy), r, color="#92c5de", ec="#555555", linewidth=0.6, alpha=0.55, zorder=0))
    for tr in bg_xy:
        ax_t.plot(tr[:, 0], tr[:, 1], color=_COL_DATA, linewidth=LW_THIN, alpha=0.10, zorder=1)

    ax_t.plot(traj_ds_best_xy[:, 0], traj_ds_best_xy[:, 1], color=_COL_BEST, linewidth=LW_THIN, ls="-", zorder=2)
    ax_t.plot(traj_ds_worst_xy[:, 0], traj_ds_worst_xy[:, 1], color=_COL_BEST, linewidth=LW_THIN, ls="--", zorder=2)
    ax_t.plot(traj_ip_best_xy[:, 0], traj_ip_best_xy[:, 1], color=_COL_IPOPT, linewidth=LW_THIN, ls="-", zorder=4)
    ax_t.plot(traj_ip_worst_xy[:, 0], traj_ip_worst_xy[:, 1], color=_COL_IPOPT, linewidth=LW_THIN, ls="--", zorder=4)

    if m["traj_worst_yx"] is not None:
        ax_t.plot(m["traj_worst_yx"][:, 1], m["traj_worst_yx"][:, 0], color=m["color"], linewidth=LW_THIN, ls="--", alpha=0.95, zorder=3)
    ax_t.plot(m["traj_best_yx"][:, 1], m["traj_best_yx"][:, 0], color=m["color"], linewidth=LW_THIN, ls="-", alpha=0.95, zorder=4)

    ax_t.plot(-2.0, -1.0, marker="*", markersize=9, linestyle="none", color="#0d6b2c", zorder=5)
    ax_t.plot(5.0, 3.0, marker="*", markersize=9, linestyle="none", color="#8b0a0d", zorder=5)
    ax_t.set_aspect("equal", adjustable="datalim")
    ax_t.set_xlim(-3, 6)
    ax_t.set_ylim(-2, 4)
    ax_t.set_title(m["name"], fontsize=FS_TITLE)

    for i in range(d_rand.shape[0]):
        ax_d.plot(steps, d_rand[i], color=_COL_DATA, linewidth=LW_THIN, alpha=0.12, zorder=1)
        ax_u.plot(steps, n_rand[i], color=_COL_DATA, linewidth=LW_THIN, alpha=0.12, zorder=1)

    ax_d.plot(steps, d_ds_best, color=_COL_BEST, linewidth=LW_THIN, ls="-", zorder=4)
    ax_d.plot(steps, d_ds_worst, color=_COL_BEST, linewidth=LW_THIN, ls="--", zorder=4)
    ax_d.plot(steps, d_ip_best, color=_COL_IPOPT, linewidth=LW_THIN, ls="-", zorder=5)
    ax_d.plot(steps, d_ip_worst, color=_COL_IPOPT, linewidth=LW_THIN, ls="--", zorder=5)
    if m["d_worst"] is not None:
        ax_d.plot(np.arange(1, len(m["d_worst"]) + 1), m["d_worst"], color=m["color"], linewidth=LW_THIN, ls="--", alpha=0.95, zorder=3)
    ax_d.plot(np.arange(1, len(m["d_best"]) + 1), m["d_best"], color=m["color"], linewidth=LW_THIN, ls="-", alpha=0.95, zorder=4)
    ax_d.set_xlim(0, 55)
    ax_d.set_ylim(0, 9.5)
    ax_d.set_yticks([0, 3, 6, 9])
    ax_d.set_ylabel(r"Norm $|\xi_t - \xi_r|$", fontsize=FS_LABEL)
    ax_d.grid(True, linewidth=0.4, alpha=0.6)

    ax_u.plot(steps, n_ds_best, color=_COL_BEST, linewidth=LW_THIN, ls="-", zorder=4)
    ax_u.plot(steps, n_ds_worst, color=_COL_BEST, linewidth=LW_THIN, ls="--", zorder=4)
    ax_u.plot(steps, n_ip_best, color=_COL_IPOPT, linewidth=LW_THIN, ls="-", zorder=5)
    ax_u.plot(steps, n_ip_worst, color=_COL_IPOPT, linewidth=LW_THIN, ls="--", zorder=5)
    if m["n_worst"] is not None:
        ax_u.plot(np.arange(1, len(m["n_worst"]) + 1), m["n_worst"], color=m["color"], linewidth=LW_THIN, ls="--", alpha=0.95, zorder=3)
    ax_u.plot(np.arange(1, len(m["n_best"]) + 1), m["n_best"], color=m["color"], linewidth=LW_THIN, ls="-", alpha=0.95, zorder=4)
    ax_u.set_xlim(0, 55)
    ax_u.set_xticks([0, 20, 40, 55])
    ax_u.set_ylim(0, 9.5)
    ax_u.set_yticks([0, 3, 6, 9])
    ax_u.set_ylabel(r"Norm $|u_t|$", fontsize=FS_LABEL)
    ax_u.set_xlabel("Time step", fontsize=FS_LABEL)
    ax_u.grid(True, linewidth=0.4, alpha=0.6)

    plt.setp(ax_d.get_xticklabels(), visible=False)
    if j != 0:
        ax_d.set_ylabel("")
        ax_u.set_ylabel("")
        plt.setp(ax_t.get_yticklabels(), visible=False)
        plt.setp(ax_d.get_yticklabels(), visible=False)
        plt.setp(ax_u.get_yticklabels(), visible=False)

handles = [
    Line2D([0], [0], color=_COL_BEST, lw=LW_THIN, ls="-"),
    Line2D([0], [0], color=_COL_BEST, lw=LW_THIN, ls="--"),
    Line2D([0], [0], color=_COL_IPOPT, lw=LW_THIN, ls="-"),
    Line2D([0], [0], color=_COL_IPOPT, lw=LW_THIN, ls="--"),
    Line2D([0], [0], color=_COL_LSTM, lw=LW_THIN, ls="-"),
    Line2D([0], [0], color=_COL_LSTM, lw=LW_THIN, ls="--"),
    Line2D([0], [0], color=_COL_CREN, lw=LW_THIN, ls="-"),
    Line2D([0], [0], color=_COL_CREN, lw=LW_THIN, ls="--"),
    Line2D([0], [0], color=_COL_BILIP, lw=LW_THIN, ls="-"),
    Line2D([0], [0], color=_COL_DATA, lw=LW_THIN, ls="-", alpha=0.6),
]
labels = [
    "Sample (Best)",
    "Sample (Worst)",
    "IPOPT (Best)",
    "IPOPT (Worst)",
    "LSTM (Best)",
    "LSTM (Worst)",
    "C-REN (Best)",
    "C-REN (Worst)",
    "BiLipREN",
    "Data",
]
legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    fontsize=FS_LEG,
    handlelength=1.6,
    framealpha=0.8,
    borderpad=0.3,
    columnspacing=1.2,
    bbox_to_anchor=(0.5, 0.05),
)

fig.subplots_adjust(left=0.062, right=0.974, top=0.945, bottom=0.165)
pdf = _RESULTS / "mbd_pl.pdf"
savefig_legend_margin(fig, pdf, legend, fixed_width=FIG_W2)
plt.close(fig)

print(f"Saved: {pdf}")
