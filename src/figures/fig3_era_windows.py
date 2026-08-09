#!/usr/bin/env python3
"""Fig. 3. Rendered from the committed tables in results/.

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
SI_DIR = MAIN_DIR = figure_dir("main")
_t = load_tables()
final_model_comparison = _t["final_model_comparison"]

window_order = ["le15", "le25", "le50", "le75", "le100", "unrestricted"]
window_labels = ["≤15", "≤25", "≤50", "≤75", "≤100", "All"]

# Panel A models are taken from the final model-comparison stage.
panel_a_models = [
    "year_only",
    "minimal_pitch_core",
    "continuity_augmented",
    "year_plus_minimal_pitch_core",
]
panel_a_labels = {
    "year_only": "Year only",
    "minimal_pitch_core": "Minimal pitch-core model",
    "continuity_augmented": "Continuity-augmented model",
    "year_plus_minimal_pitch_core": "Year + minimal pitch-core",
}

p3a = final_model_comparison[
    (final_model_comparison["record_type"] == "performance") &
    (final_model_comparison["metric"] == "roc_auc") &
    (final_model_comparison["reference_or_term"].isna()) &
    (final_model_comparison["model"].isin(panel_a_models))
].copy()

# Panel B: the jointly fitted year-adjusted continuity-augmented candidate.
feature_terms = [
    "d_prop_repeated_pitch",
    "d_pitch_range",
    "d_prop_gap_transitions",
]
feature_labels = {
    "d_prop_repeated_pitch": "Repeated-pitch proportion",
    "d_pitch_range": "Pitch range",
    "d_prop_gap_transitions": "Gap rate",
}
p3b = final_model_comparison[
    (final_model_comparison["record_type"] == "coefficient") &
    (final_model_comparison["model"] == "year_plus_continuity_augmented") &
    (final_model_comparison["reference_or_term"].isin(feature_terms))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), gridspec_kw={"wspace": 0.25})
xs = np.arange(len(window_order))

# A -- shaded 95% family-bootstrap bands; the windows rest on as few as
# 125 families, so the estimates are imprecise.
ax = axes[0]
for model in panel_a_models:
    tmp = p3a[p3a["model"] == model].set_index("window").loc[window_order]
    line, = ax.plot(xs, tmp["estimate"], marker="o", linewidth=2,
                    label=panel_a_labels[model])
    ax.fill_between(xs, tmp["ci95_lo"], tmp["ci95_hi"], alpha=0.13,
                    color=line.get_color(), linewidth=0)
ax.axhline(0.5, linestyle="--", linewidth=1.3, color="0.45", zorder=0)
ax.set_ylim(0.38, 0.88)
ax.set_xticks(xs)
ax.set_xticklabels(window_labels)
ax.set_xlabel("Maximum vocal-instrumental year difference")
ax.set_ylabel("Vocal-instrumental separation (AUC)")
ax.set_title("Melodic models remain informative across year-gap restrictions")
ax.legend(loc="lower right", frameon=True, framealpha=1.0,
          facecolor="white", edgecolor="0.8", ncol=2)
ax.text(-0.12, 1.01, "A", transform=ax.transAxes, fontsize=18, fontweight="bold")

# B
ax = axes[1]
for k, term in enumerate(feature_terms):
    tmp = p3b[p3b["reference_or_term"] == term].set_index("window").loc[window_order]
    e = tmp["estimate"].to_numpy()
    err = np.vstack([e - tmp["ci95_lo"].to_numpy(), tmp["ci95_hi"].to_numpy() - e])
    ax.errorbar(xs + (k - 1) * 0.12, e, yerr=err, marker="o", capsize=3,
                linewidth=1.8, markersize=6, elinewidth=1.2,
                label=feature_labels[term])
ax.axhline(0, linewidth=1.2, color="0.45", zorder=0)
ax.set_ylim(-0.85, 0.95)
ax.set_xticks(xs)
ax.set_xticklabels(window_labels)
ax.set_xlabel("Maximum vocal-instrumental year difference")
ax.set_ylabel("Association with the vocal side (standardized coefficient)")
ax.set_title("Repetition and range are the most stable features")
ax.legend(loc="lower left", frameon=True, framealpha=1.0,
          facecolor="white", edgecolor="0.8")
ax.text(-0.12, 1.01, "B", transform=ax.transAxes, fontsize=18, fontweight="bold")

save_figure(fig, MAIN_DIR, "Fig3_era_window_sensitivity")
