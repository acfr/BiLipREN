"""
plot package
============
Centralised plotting-style helpers shared by every figure-producing script in
this repository. Scripts import it either as ``from plot import ...`` or
``from plot.style import ...`` after adding the project root to ``sys.path``.

All public names are re-exported from :mod:`plot.style`.
"""

from __future__ import annotations

from .style import *          # noqa: F401,F403  (re-export public style API)
from .style import (          # noqa: F401  (explicit re-export of names used by scripts)
    FIG_W1, FIG_W2,
    FIG_1COL_SQ, FIG_1COL_43, FIG_2COL_43, FIG_2COL_WIDE, FIG_2COL_HALF,
    ASPECT_4_3, ASPECT_WIDE,
    LW_MAIN, LW_REF, LW_THIN, LW_THICK, MARKER_SZ, DPI,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEG,
    PAD_W, PAD_H, WSPACE, HSPACE,
    COLORS, COLOR_CYCLE,
    apply_tac_style, savefig, savefig_legend_margin,
)
