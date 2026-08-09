#!/usr/bin/env python3
"""Fig. S3. Rendered from the committed tables in results/.

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
window_performance = _t["window_performance"]

s3_windows = ["le15", "le25", "le50", "le75", "le100", "unrestricted"]
s3_window_labels = ["≤15", "≤25", "≤50", "≤75", "≤100", "All"]
s3_models = [
    "year_only",
    "pitch_movement",
    "combined",
    "compact",
    "year_plus_pitch_movement",
    "year_plus_combined",
    "year_plus_compact",
]
s3_labels = {
    "year_only": "Year only",
    "pitch_movement": "Pitch-movement model",
    "combined": "Full seven-feature model",
    "compact": "Compact four-feature model",
    "year_plus_pitch_movement": "Year + pitch movement",
    "year_plus_combined": "Year + full model",
    "year_plus_compact": "Year + compact model",
}

s3 = window_performance[
    (window_performance["metric"] == "roc_auc") &
    (window_performance["reference"].isna()) &
    (window_performance["model"].isin(s3_models))
].copy()

fig, ax = plt.subplots(figsize=(10.5, 7))
for model in s3_models:
    tmp = s3[s3["model"] == model].set_index("window").loc[s3_windows]
    ax.plot(s3_window_labels, tmp["estimate"], marker="o", linewidth=2,
            label=s3_labels[model])
ax.axhline(0.5, linestyle="--", linewidth=1.3)
ax.set_ylim(0.45, 0.77)
ax.set_xlabel("Maximum vocal-instrumental year difference")
ax.set_ylabel("Vocal-instrumental separation (AUC)")
ax.set_title("Melodic models remain informative across documentation-year gap restrictions",
             fontsize=19)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)

save_figure(fig, SI_DIR, "FigS3_all_pairwise_models")
