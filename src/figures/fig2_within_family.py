#!/usr/bin/env python3
"""Fig. 2. Rendered from the committed tables in results/.

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
pairwise_results = _t["pairwise_results"]
pairwise_coefficients = _t["pairwise_coefficients"]
source_type_results = _t["source_type_results"]

# Panel A: pooled out-of-fold AUCs
model_order = [
    "year_only",
    "continuity",
    "pitch_movement",
    "combined",
    "year_plus_continuity",
    "year_plus_pitch_movement",
    "year_plus_combined",
]
model_labels = {
    "year_only": "Year",
    "continuity": "Continuity",
    "pitch_movement": "Pitch",
    "combined": "Combined",
    "year_plus_continuity": "Year + continuity",
    "year_plus_pitch_movement": "Year + pitch",
    "year_plus_combined": "Year + combined",
}

auc = pairwise_results[
    (pairwise_results["metric"] == "roc_auc") &
    (pairwise_results["model"].isin(model_order))
].set_index("model").loc[model_order].reset_index()

# Panel B: year-adjusted combined-model coefficients
coef_order = [
    "d_year",
    "d_prop_gap_transitions",
    "d_mean_notes_between_gaps",
    "d_note_density",
    "d_prop_repeated_pitch",
    "d_mean_abs_interval_nonrep",
    "d_pitch_range",
    "d_max_abs_interval",
]
coef_labels = {
    "d_year": "Documentation year",
    "d_prop_gap_transitions": "Gap rate",
    "d_mean_notes_between_gaps": "Mean notes between gaps",
    "d_note_density": "Note density",
    "d_prop_repeated_pitch": "Repeated-pitch proportion",
    "d_mean_abs_interval_nonrep": "Mean nonrepeated interval",
    "d_pitch_range": "Pitch range",
    "d_max_abs_interval": "Maximum absolute interval",
}
coef = pairwise_coefficients[
    (pairwise_coefficients["model"] == "year_plus_combined") &
    (pairwise_coefficients["term"].isin(coef_order))
].set_index("term").loc[coef_order].reset_index()

fig, axes = plt.subplots(1, 3, figsize=(20, 6.1), gridspec_kw={"wspace": 0.55})

# A -- points rather than bars: with filled bars the lower half of each
# confidence interval is hidden inside the bar.
ax = axes[0]
x = np.arange(len(auc))
y = auc["estimate"].to_numpy()
yerr = np.vstack([
    y - auc["ci95_lo_boot_families"].to_numpy(),
    auc["ci95_hi_boot_families"].to_numpy() - y,
])
ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, linewidth=1.6, markersize=7)
ax.axhline(0.5, linestyle="--", linewidth=1.3, color="0.45")
ax.set_ylim(0.44, 0.81)
ax.set_xlim(-0.6, len(auc) - 0.4)
ax.set_ylabel("Vocal-instrumental separation (AUC)")
ax.set_title("Melody outperforms year within tune families", loc="left", fontsize=12.5)
ax.set_xticks(x)
ax.set_xticklabels([model_labels[m] for m in model_order], rotation=35, ha="right")
ax.text(-0.12, 1.01, "A", transform=ax.transAxes, fontsize=18, fontweight="bold")

# B
ax = axes[1]
ypos = np.arange(len(coef))[::-1]
est = coef["coefficient_std_scale"].to_numpy()
lo = coef["ci95_lo_boot_families"].to_numpy()
hi = coef["ci95_hi_boot_families"].to_numpy()
xerr = np.vstack([est - lo, hi - est])

ax.errorbar(est, ypos, xerr=xerr, fmt="o", capsize=4, linewidth=1.6, markersize=7)
ax.axvline(0, linewidth=1.2, color="0.45")
ax.set_yticks(ypos)
ax.set_yticklabels([coef_labels[t] for t in coef_order])
ax.set_xlabel("Association with the vocal side (standardized coefficient)")
ax.set_title("Predictors associated with the vocal side of a family", loc="left", fontsize=12.5)
ax.text(-0.12, 1.01, "B", transform=ax.transAxes, fontsize=18, fontweight="bold")

# C -- source-type sensitivity
ax = axes[2]
samples = ["full", "notated", "strict"]
st = source_type_results[source_type_results["metric"] == "roc_auc"]
nfam = {s: int(st[st["sample"] == s]["n_families"].iloc[0]) for s in samples}
xs = np.arange(len(samples))

for model, label, colour, dx in [
    ("year_only", "Year only", "tab:orange", -0.06),
    ("combined", "Combined melodic", "tab:blue", 0.06),
]:
    sub = st[st["model"] == model].set_index("sample").loc[samples]
    e = sub["estimate"].to_numpy()
    err = np.vstack([
        e - sub["ci95_lo_boot_families"].to_numpy(),
        sub["ci95_hi_boot_families"].to_numpy() - e,
    ])
    ax.errorbar(xs + dx, e, yerr=err, fmt="o-", capsize=4, linewidth=1.6,
                markersize=7, color=colour, label=label)

ax.axhline(0.5, linestyle="--", linewidth=1.3, color="0.45")
ax.set_ylim(0.38, 0.83)
ax.set_xlim(-0.45, len(samples) - 0.55)
ax.set_xticks(xs)
sample_labels = {"full": "All families", "notated": "Notated\nvariants",
                 "strict": "Notated-only\nfamilies"}
ax.set_xticklabels([f"{sample_labels.get(s, s.capitalize())}\n({nfam[s]} fam.)" for s in samples])
ax.set_ylabel("Vocal-instrumental separation (AUC)")
ax.set_title("The melodic result survives source matching", loc="left", fontsize=12.5)
ax.legend(loc="lower left", frameon=False)
ax.text(-0.12, 1.01, "C", transform=ax.transAxes, fontsize=18, fontweight="bold")

save_figure(fig, MAIN_DIR, "Fig2_within_family_prediction")
