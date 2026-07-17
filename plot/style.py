"""
plot/style.py
=============
Centralised style configuration for all TAC-submission figures.

Usage
-----
from plot.style import apply_tac_style, COLORS, FIG_W1, FIG_W2

apply_tac_style()          # call once at top of every plot script
fig, ax = plt.subplots(figsize=(FIG_W1, FIG_W1 * 0.75))
ax.plot(t, y, color=COLORS["output_1"], linewidth=LW_MAIN)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Figure width presets (inches) — IEEE TAC is a double-column journal
# ---------------------------------------------------------------------------
FIG_W1 = 3.5        # single-column figure
FIG_W2 = 7.16       # double-column (full-width) figure

# Default figure heights (use as FIG_W1 * ASPECT_4_3 etc.)
ASPECT_4_3  = 0.85   # 4:3
ASPECT_WIDE = 0.65   # wide/cinema

# Convenience full-size presets (width, height)
FIG_1COL_SQ    = (FIG_W1, FIG_W1)
FIG_1COL_43    = (FIG_W1, FIG_W1 * ASPECT_4_3)
FIG_2COL_43    = (FIG_W2, FIG_W2 * ASPECT_4_3)
FIG_2COL_WIDE  = (FIG_W2, FIG_W2 * ASPECT_WIDE)
FIG_2COL_HALF  = (FIG_W2, FIG_W2 * 0.45)   # for stacked 2-row subplots

# ---------------------------------------------------------------------------
# Line & marker weights
# ---------------------------------------------------------------------------
LW_MAIN   = 1.6    # primary data series
LW_REF    = 1.2    # reference / setpoint (usually dashed)
LW_THIN   = 1.0    # secondary / background trajectories
LW_THICK  = 2.0    # highlighted series

MARKER_SZ = 3.5    # default marker size
DPI        = 300   # save DPI for all figures

# ---------------------------------------------------------------------------
# Unified font sizes (points) — shared across ALL final figures so that,
# whether a figure is single- or double-column, the in-paper rendered text is
# identical (every figure is placed at scale ~1.0). All axis labels, tick
# numbers and subplot titles use ONE common size (= combined_3x3.pdf's label
# size, i.e. body 10 pt). Inset/thumbnail labels are set locally and excluded.
# ---------------------------------------------------------------------------
FS_TICK  = 9    # tick numbers
FS_LABEL = 9    # axis labels (x & y)
FS_TITLE = 9    # subplot titles
FS_LEG   = 7    # legend text

# ---------------------------------------------------------------------------
# Unified subplot spacing — shared across ALL final figures so the inter-panel
# gaps look identical everywhere. `PAD_*` are constrained_layout padding
# (inches) around axes; `WSPACE`/`HSPACE` are the inter-column / inter-row
# gaps (fractions). HSPACE applies to label-less row separations (rows that
# share an x-axis, i.e. only the bottom row carries the x-label).
# ---------------------------------------------------------------------------
PAD_W  = 0.04   # constrained_layout w_pad (inches)
PAD_H  = 0.04   # constrained_layout h_pad (inches)
WSPACE = 0.03   # gap between columns (multi-column figures)
HSPACE = 0.06   # gap between label-less rows (multi-row figures)

# ---------------------------------------------------------------------------
# Unified colour palette
# Designed to:
#   * distinguish series in colour printing
#   * remain readable when converted to greyscale
#   * follow a consistent role-based naming convention
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    # --- control / tracking roles ---
    "reference":  "#000000",   # black dashed   — reference / setpoint
    "output_1":   "#2166ac",   # dark blue       — primary plant output y1
    "output_2":   "#d6604d",   # brick red       — secondary plant output y2
    "output_3":   "#4dac26",   # green           — third output / model output
    "output_4":   "#8073ac",   # purple          — fourth output
    "model":      "#4dac26",   # green           — model prediction / y_m
    "input_1":    "#4575b4",   # blue            — control input u1
    "input_2":    "#d73027",   # red             — control input u2
    "disturbance":"#737373",   # dark grey       — disturbance estimate

    # --- flow / generative roles ---
    "data":       "#4393c3",   # medium blue     — training data / reference
    "latent":     "#e08214",   # amber           — latent / mapped variable z
    "generated":  "#f4a582",   # salmon          — generated / reconstructed
    "forward":    "#d6604d",   # brick red       — forward-pass output
    "inverse":    "#4dac26",   # green           — inverse-pass output

    # --- trajectory / spatial roles ---
    "traj_ref":   "#003f5c",   # navy            — reference trajectory
    "traj_gen":   "#ff7f0e",   # orange          — generated trajectory
    "traj_alt":   "#2ca02c",   # green           — alternative trajectory
    "obstacle":   "#92c5de",   # sky blue        — obstacle fill
    "start":      "#1a9641",   # dark green      — start marker
    "goal":       "#d7191c",   # dark red        — goal marker

    # --- generic extras (index by number if needed) ---
    "c0": "#2166ac",
    "c1": "#d6604d",
    "c2": "#4dac26",
    "c3": "#8073ac",
    "c4": "#e08214",
    "c5": "#737373",
}

# Ordered list for cycling over N series
COLOR_CYCLE = [
    COLORS["c0"], COLORS["c1"], COLORS["c2"],
    COLORS["c3"], COLORS["c4"], COLORS["c5"],
]

# ---------------------------------------------------------------------------
# rcParams — call apply_tac_style() to activate
# ---------------------------------------------------------------------------
_TAC_RC = {
    # Font
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "stix",
    "font.size":          14,
    "axes.titlesize":     14,
    "axes.labelsize":     14,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    13,
    "legend.title_fontsize": 13,

    # Lines
    "lines.linewidth":    LW_MAIN,
    "lines.markersize":   MARKER_SZ,
    "patch.linewidth":    0.6,

    # Axes
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "axes.prop_cycle":    mpl.cycler(color=COLOR_CYCLE),
    "axes.spines.top":    True,
    "axes.spines.right":  True,

    # Ticks
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",

    # Legend
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.7",
    "legend.handlelength": 1.8,

    # Saving
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.format":     "pdf",
    "pdf.fonttype":       42,   # embed fonts as TrueType (required by IEEE)
    "ps.fonttype":        42,
}


def apply_tac_style() -> None:
    """Apply IEEE TAC-compatible rcParams globally."""
    plt.rcParams.update(_TAC_RC)


def savefig(fig: "plt.Figure", path: str, *, dpi: int = DPI, **kwargs) -> None:
    """Save *fig* to *path* with TAC defaults (tight layout + 300 dpi)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)


# Fixed whitespace (inches) kept below the legend so that figures sharing a
# legend at the bottom have an identical bottom margin, independent of the
# figure's height, legend size, or any later layout changes.
LEGEND_BOTTOM_MARGIN_IN = 0.05

# Fixed vertical gap (inches) between the legend's top edge and the bottom of
# the nearest axes content (x-label / tick labels). Shared so the legend sits
# the same distance below the plots in every figure.
LEGEND_AXES_GAP_IN = 0.10


def savefig_legend_margin(
    fig: "plt.Figure",
    path: str,
    legend,
    *,
    dpi: int = DPI,
    margin_in: float = LEGEND_BOTTOM_MARGIN_IN,
    gap_in: float | None = LEGEND_AXES_GAP_IN,
    side_pad_in: float = 0.03,
    fixed_width: float | None = None,
    **kwargs,
) -> None:
    """Save *fig* with deterministic spacing around a bottom *legend*.

    Two distances are pinned to absolute inches, independent of figure size or
    layout, so figures that share a bottom legend line up identically:

    * ``gap_in``    – vertical gap between the legend's top edge and the bottom
                      of the nearest axes content (x-label/ticks). If ``None``
                      the legend is left where it was placed.
    * ``margin_in`` – whitespace kept below the legend's lower edge.

    ``fixed_width`` forces the saved crop to that exact width (inches), e.g. to
    keep a double-column figure at the text width so fonts are not rescaled.
    Otherwise the crop uses the tight bounding box of all artists.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    _fig_h = fig.get_size_inches()[1]

    if gap_in is not None:
        # Bottom of the lowest axes content, ignoring the figure legend.
        content_bottom = min(
            ax.get_tightbbox(renderer).transformed(inv).y0 for ax in fig.axes
        )
        legend.set_bbox_to_anchor(
            (0.5, (content_bottom - gap_in) / _fig_h), transform=fig.transFigure
        )
        fig.canvas.draw()

    legbb = legend.get_window_extent(renderer).transformed(inv)
    if fixed_width is not None:
        x0, x1 = 0.0, fixed_width
        y1 = _fig_h
    else:
        tight = fig.get_tightbbox(renderer).padded(side_pad_in)
        x0, x1, y1 = tight.x0, tight.x1, tight.y1
    crop = Bbox.from_extents(x0, legbb.y0 - margin_in, x1, y1)
    fig.savefig(path, dpi=dpi, bbox_inches=crop, pad_inches=0, **kwargs)
