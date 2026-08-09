#!/usr/bin/env python3
"""Fig. 1. Rendered from the committed tables in results/.

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


# Three-corpus study design: identify the pattern in Dutch tune families, then transfer it.
fig, ax = plt.subplots(figsize=(19.0, 7.6))
ax.set_xlim(0, 19.0); ax.set_ylim(0, 7.7); ax.axis("off")

TOP_Y, TOP_H = 4.55, 2.20
LOW_Y, LOW_H = 1.45, 1.85

def add_box(x, y, width, height, title, body, title_size=15, body_size=12.5, fc="white", ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.8, edgecolor=ec, facecolor=fc))
    n_title = title.count("\n") + 1
    ax.text(x + width/2, y + height - 0.30 - 0.16*(n_title-1), title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", linespacing=1.15)
    ax.text(x + width/2, y + height*0.32, body, ha="center", va="center",
            fontsize=body_size, linespacing=1.20)

def add_arrow(x1, y1, x2, y2, lw=1.7, color="black"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=lw, color=color))

add_box(0.55, TOP_Y, 3.2, TOP_H, "Dutch corpus", "18,109 total melodies\n2,820 melodies in\n433 mixed families")
add_box(4.20, TOP_Y, 3.2, TOP_H, "Within-family\ncomparison", "Vocal mean −\ninstrumental mean\n433 tune families")
add_box(7.85, TOP_Y, 3.2, TOP_H, "Selected melodic\npattern", "More repeated pitches\nNarrower pitch range\nMore notated gaps")
add_box(11.50, TOP_Y, 2.8, TOP_H, "Transferred\nscores", "Pitch-core\nGap-rate\nAugmented")
add_box(3.35, LOW_Y, 3.0, LOW_H, "Era stress test", "15, 25, 50, 75, 100 y\nand unrestricted")
add_box(6.95, LOW_Y, 3.0, LOW_H, "Source-type test", "Notated sources only\n358 of 433 families")

VX, VW = 14.80, 4.10
add_box(VX, 4.55, VW, 2.05, "Finnish folk corpus",
        "8,613 melodies\n10 source collections\nseparately constructed folk archive",
        title_size=12.5, body_size=11.0, fc="#eef4fb")
add_box(VX, 1.45, VW, 2.05, "Turkish makam corpus",
        "1,878 analyzed pieces\n1,749 medium-labelled\n1,711 in 20 tested forms\ndifferent musical and pitch-notation system",
        title_size=12.5, body_size=9.9, fc="#fbeeee")

ax.text(6.4, TOP_Y + TOP_H + 0.42, "Identify the pattern in Dutch tune families",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="0.2")
ax.text(VX + VW/2, 7.15, "Apply fixed score definitions without target-label fitting",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="0.2")

mid = TOP_Y + TOP_H/2
add_arrow(3.75, mid, 4.20, mid); add_arrow(7.40, mid, 7.85, mid); add_arrow(11.05, mid, 11.50, mid)
add_arrow(5.30, TOP_Y, 4.85, LOW_Y + LOW_H); add_arrow(6.30, TOP_Y, 8.45, LOW_Y + LOW_H)
add_arrow(14.30, 5.575, VX, 5.575)
add_arrow(14.30, 5.575, VX, 2.475)
ax.text(7.2, 0.55, "Does the Dutch vocal-associated pattern\nappear in other corpora?",
        ha="center", va="center", fontsize=14.5, fontweight="bold", linespacing=1.25)

save_figure(fig, MAIN_DIR, "Fig1_study_design")
