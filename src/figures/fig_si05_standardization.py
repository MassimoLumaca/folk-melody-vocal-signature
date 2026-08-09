#!/usr/bin/env python3
"""Fig. S5. Rendered from the committed tables in results/.

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
finnish_collections = _t["finnish_collections"]

blind = finnish_collections[
    finnish_collections["scheme"] == "finnish_blind"
][["collection", "medium", "mean_augmented"]].rename(columns={"mean_augmented": "blind"})

dutch_ref = finnish_collections[
    finnish_collections["scheme"] == "dutch_ref"
][["collection", "mean_augmented"]].rename(columns={"mean_augmented": "dutch_ref"})

s5 = blind.merge(dutch_ref, on="collection", validate="one_to_one")
s5 = s5.sort_values("dutch_ref", ascending=False).reset_index(drop=True)

# The seven vocal collections plus Jouhikkosavelmia form a tight cluster whose
# labels collided when drawn inline; they get leader lines to a spread column,
# while the two isolated instrumental points keep inline labels.
cluster = set(s5.nlargest(8, "dutch_ref")["collection"])

fig, ax = plt.subplots(figsize=(8.6, 7.4))
vocal = s5["medium"].eq("vocal").to_numpy()
ax.scatter(s5.loc[vocal, "blind"], s5.loc[vocal, "dutch_ref"], s=75,
           marker="o", color="tab:blue", zorder=3)
ax.scatter(s5.loc[~vocal, "blind"], s5.loc[~vocal, "dutch_ref"], s=75,
           marker="s", color="tab:orange", zorder=3)

lo = min(s5["blind"].min(), s5["dutch_ref"].min()) - 0.18
hi = max(s5["blind"].max(), s5["dutch_ref"].max()) + 0.95
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="0.55", zorder=0)

cl = s5[s5["collection"].isin(cluster)]
ys = cl["dutch_ref"].to_numpy().astype(float)
xs = cl["blind"].to_numpy().astype(float)
sep = (hi - lo) * 0.052
lab = ys.copy()
for i in range(1, len(lab)):
    if lab[i - 1] - lab[i] < sep:
        lab[i] = lab[i - 1] - sep
lab += (ys.max() - lab.max())
label_x = xs.max() + 0.42
for x, y, ly, name in zip(xs, ys, lab, cl["collection"]):
    ax.annotate(name, xy=(x, y), xytext=(label_x, ly), textcoords="data",
                fontsize=10, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", linewidth=0.7, color="0.6",
                                shrinkA=0, shrinkB=4))
for _, row in s5[~s5["collection"].isin(cluster)].iterrows():
    ax.annotate(row["collection"], xy=(row["blind"], row["dutch_ref"]),
                xytext=(9, -3), textcoords="offset points", fontsize=10,
                va="center")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Augmented score using Finnish standardization")
ax.set_ylabel("Augmented score using Dutch-reference standardization")
ax.set_title("Finnish collection ordering is unchanged by standardization", fontsize=15)
ax.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", color="tab:blue",
           markersize=8, label="Vocal"),
    Line2D([], [], marker="s", linestyle="none", color="tab:orange",
           markersize=8, label="Instrumental")],
    loc="lower right", frameon=True, framealpha=1.0,
    facecolor="white", edgecolor="0.8")

save_figure(fig, SI_DIR, "FigS5_standardization_sensitivity")
