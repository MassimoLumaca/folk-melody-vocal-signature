#!/usr/bin/env python3
"""Fig. S4. Rendered from the committed tables in results/.

Run directly, or let notebooks/recreate_all_paper_figures.ipynb call it. Set
RESULTS_DIR / FIGURE_DIR to override the input and output locations.
"""
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from _common import apply_style, figure_dir, load_tables, save_figure

apply_style()
SI_DIR = MAIN_DIR = figure_dir("SI")
_t = load_tables()
compact_vs_full = _t["compact_vs_full"]

s4 = compact_vs_full[
    (compact_vs_full["model"] == "compact") &
    (compact_vs_full["reference"] == "combined") &
    (compact_vs_full["metric"] == "d_roc_auc")
].set_index("window").loc[["le15", "le25", "le50", "le75", "le100", "unrestricted"]].reset_index()

labels = ["≤15", "≤25", "≤50", "≤75", "≤100", "All"]
x = np.arange(len(s4))
est = s4["estimate"].to_numpy()
yerr = np.vstack([
    est - s4["ci95_lo"].to_numpy(),
    s4["ci95_hi"].to_numpy() - est,
])

fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.errorbar(x, est, yerr=yerr, fmt="o", capsize=4, linewidth=2, markersize=8)
ax.axhline(0, linewidth=1.3)
ax.axhline(-0.02, linestyle="--", linewidth=1.3)
ax.annotate("Prespecified model-reduction tolerance (−0.02)", (0.015, -0.02), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points", fontsize=10, color="0.35")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("AUC difference (compact − full)")
ax.set_xlabel("Maximum vocal-instrumental year difference")
ax.set_title("Compact and full models have similar average AUCs", fontsize=19)

save_figure(fig, SI_DIR, "FigS4_compact_vs_full")
