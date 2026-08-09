#!/usr/bin/env python3
"""Fig. S6. Rendered from the committed tables in results/.

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
core_validation = _t["core_validation"]

def parse_loco_string(text):
    result = {}
    for item in str(text).split(";"):
        if not item.strip():
            continue
        key, value = item.rsplit("=", 1)
        result[key] = float(value)
    return result

core_row = core_validation[core_validation["scheme"] == "finnish_blind"].iloc[0]
core_loco = parse_loco_string(core_row["loco_auc_by_left_out"])

collection_order = [
    "HS1", "LS1", "LS2", "LS3", "LS4", "RS1", "RS2",
    "Jouhikkosävelmiä", "KT1", "Kantelesävelmiä"
]
core_values = [core_loco[c] for c in collection_order]

x = np.arange(len(collection_order))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 7.1))
# AUC has a meaningful 0.5 baseline; bars from 0.55 would exaggerate the contrast.
ax.axhline(0.5, ls=":", lw=1.2, color="0.45")
ax.annotate("chance", (len(collection_order) - 0.5, 0.5), (len(collection_order) - 0.5, 0.515),
            fontsize=9, color="0.45", ha="right")
ax.plot(x, core_values, "o", ms=11, color="tab:blue", label="Pitch-core")
ax.vlines(x, 0.5, core_values, color="tab:blue", lw=1.6, alpha=0.55)
ax.set_ylim(0.45, 1.01)          # AUC cannot exceed 1
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Omitted collection")
ax.set_ylabel("AUC after omitting each collection")
ax.set_title("Pitch-core score: leave-one-collection-out AUC", fontsize=17, pad=26)
ax.set_xticks(x)
ax.set_xticklabels(collection_order, rotation=52, ha="right")
ax.legend(loc="lower left", frameon=False)

save_figure(fig, SI_DIR, "FigS6_leave_one_collection_out")
