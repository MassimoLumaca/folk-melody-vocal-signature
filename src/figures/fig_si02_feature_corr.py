#!/usr/bin/env python3
"""Fig. S2. Rendered from the committed tables in results/.

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
dutch_features = _t["dutch_features"]

if dutch_features is None:
    print("Fig. S2 skipped: DATA_DIR/02_dutch_features.csv is not committed.\n"
          "Regenerate it with:  python src/extract_features.py")
else:
    corr_cols = [
        "prop_gap_transitions",
        "mean_notes_between_gaps",
        "max_notes_between_gaps",
        "mean_dur_between_gaps",
        "max_dur_between_gaps",
        "note_density",
        "pitch_range",
        "max_abs_interval",
        "mean_abs_interval_nonrep",
        "prop_repeated_pitch",
        "prop_large_interval_ge5",
        "n_notes",
    ]
    corr_labels = [
        "Gap rate",
        "Mean notes between gaps",
        "Maximum notes between gaps",
        "Mean duration between gaps",
        "Maximum duration between gaps",
        "Note density",
        "Pitch range",
        "Maximum interval",
        "Mean nonrepeated interval",
        "Repeated-pitch proportion",
        "Large-interval proportion",
        "Melody length",
    ]

    corr = dutch_features[corr_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(len(corr_labels)))
    ax.set_yticks(np.arange(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=55, ha="right")
    ax.set_yticklabels(corr_labels)
    ax.set_title("Continuity measures are strongly correlated in the Dutch corpus")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Pearson r")

    save_figure(fig, SI_DIR, "FigS2_feature_correlations")
