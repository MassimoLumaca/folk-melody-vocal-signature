"""SI Figures S3 and S6.

Source recovered from 03_outputs/recreate_all_paper_figures.ipynb (cells 17 and 23).
Two corrections relative to that notebook:
  S3: the notebook cell predates the <=15-year window; the deployed figure has it,
      so "le15" is restored here. y-limits set to (0.45, 0.77) to match the
      deployed figure (the notebook's (0.47, 0.755) clips the year-only trough).
  Titles de-ranked to match the revised manuscript wording:
      S3 "All predefined pairwise models ..." -> "All pairwise models ..."
      S6 "External validation robustness"     -> "Cross-corpus transfer robustness"
S6 reproduces the previously deployed figure pixel-for-pixel apart from the title.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/Users/massimolumaca/Documents/music_memorability_project/03_outputs")
OUT = DATA / "paper/figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print("wrote", stem)

# ----------------------------- Fig. S3 -----------------------------
window_performance = pd.read_csv(DATA / "04_window_model_performance.csv")
windows = ["le15", "le25", "le50", "le75", "le100", "unrestricted"]
wlabels = ["≤15", "≤25", "≤50", "≤75", "≤100", "All"]
models = ["year_only", "pitch_movement", "combined", "compact",
          "year_plus_pitch_movement", "year_plus_combined", "year_plus_compact"]
labels = {"year_only": "year only", "pitch_movement": "pitch movement",
          "combined": "combined", "compact": "compact",
          "year_plus_pitch_movement": "year plus pitch movement",
          "year_plus_combined": "year plus combined",
          "year_plus_compact": "year plus compact"}

s3 = window_performance[(window_performance["metric"] == "roc_auc") &
                        (window_performance["reference"].isna()) &
                        (window_performance["model"].isin(models))].copy()
fig, ax = plt.subplots(figsize=(10.5, 7))
for m in models:
    tmp = s3[s3["model"] == m].set_index("window").loc[windows]
    ax.plot(wlabels, tmp["estimate"], marker="o", linewidth=2, label=labels[m])
ax.axhline(0.5, linestyle="--", linewidth=1.3)
ax.set_ylim(0.45, 0.77)
ax.set_xlabel("Era window"); ax.set_ylabel("AUC")
ax.set_title("All pairwise models across era windows", fontsize=19)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
save(fig, "figureSI03"); plt.close(fig)

# ----------------------------- Fig. S6 -----------------------------
core_validation = pd.read_csv(DATA / "06_primary_core_validation.csv")

def parse_loco(text):
    out = {}
    for item in str(text).split(";"):
        if item.strip():
            k, v = item.rsplit("=", 1)
            out[k] = float(v)
    return out

core = parse_loco(core_validation[core_validation["scheme"] == "finnish_blind"]
                  .iloc[0]["loco_auc_by_left_out"])
order = ["HS1", "LS1", "LS2", "LS3", "LS4", "RS1", "RS2",
         "Jouhikkosävelmiä", "KT1", "Kantelesävelmiä"]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 7.1))
ax.axhline(0.5, ls=":", lw=1.2, color="0.45")
ax.annotate("chance", (len(order) - 0.5, 0.5), (len(order) - 0.5, 0.515),
            fontsize=9, color="0.45", ha="right")
ax.plot(x - 0.09, [core[c] for c in order], "o", ms=11, color="tab:blue", label="Core")
ax.plot(x + 0.09, [1.0] * len(order), "s", ms=11, color="tab:orange", label="Augmented")
ax.vlines(x - 0.09, 0.5, [core[c] for c in order], color="tab:blue", lw=1.6, alpha=0.55)
ax.vlines(x + 0.09, 0.5, [1.0] * len(order), color="tab:orange", lw=1.6, alpha=0.55)
ax.set_ylim(0.45, 1.01)          # AUC cannot exceed 1
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("Leave-one-collection-out AUC")
ax.set_title("Cross-corpus transfer robustness", fontsize=17, pad=26)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=52, ha="right")
ax.legend(loc="lower left", ncol=2, frameon=False)
save(fig, "figureSI06"); plt.close(fig)
