#!/usr/bin/env python3
"""Fig. S1. Rendered from the committed tables in results/.

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
melody_models = _t["melody_models"]

s1_order = [
    "baseline_n_notes",
    "continuity",
    "pitch_movement",
    "combined",
    "baseline_year",
    "baseline_n_notes_year",
]
s1_labels = {
    "baseline_n_notes": "Melody length",
    "continuity": "Continuity",
    "pitch_movement": "Pitch",
    "combined": "Combined",
    "baseline_year": "Year",
    "baseline_n_notes_year": "Year + melody length",
}

# Use the precomputed cross-validation mean rows only (fold == "mean");
# averaging over all rows would wrongly include the per-fold and "sd" rows.
s1 = (
    melody_models[
        (melody_models["model"].isin(s1_order)) &
        (melody_models["fold"] == "mean")
    ]
    .set_index("model").loc[s1_order, ["roc_auc"]].reset_index()
)

fig, ax = plt.subplots(figsize=(10.5, 7))
x = np.arange(len(s1))
ax.bar(x, s1["roc_auc"])
# Black and above the bars: the original line took the default blue and so
# vanished wherever a bar crossed it. Black reads against both the bars and
# the background, so the line stays continuous across the whole axis.
ax.axhline(0.5, linestyle="--", linewidth=1.4, color="black", zorder=3)
ax.set_ylim(0.45, 0.86)
ax.set_ylabel("Vocal-instrumental separation (mean AUC)")
ax.set_title("At melody level, documentation year outperforms melodic features", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels([s1_labels[m] for m in s1_order], rotation=30, ha="right")

save_figure(fig, SI_DIR, "FigS1_melody_level_classification")
