#!/usr/bin/env python3
"""Fig 4 redesign: single cross-corpus validation figure (A Finnish, B Turkish, C components)."""
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- portable paths -------------------------------------------------------
# Inputs come from the committed results/ directory so the script runs from a
# fresh clone. Override with RESULTS_DIR / FIGURE_DIR if your layout differs.
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("RESULTS_DIR", _ROOT / "results"))
FIGDIR = Path(os.environ.get("FIGURE_DIR", Path.cwd() / "paper_figures" / "main"))
FIGDIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------


# Human-readable Turkish form names, matching SI Table S16 and the main text.
FORM_DISPLAY = {
    "nefes": "nefes", "turku": "türkü", "oyunhavasi": "oyun havası",
    "murabba": "murabba", "nakis": "nakış", "ilahi": "ilahi", "kar": "kâr",
    "beste": "beste", "yuruksemai": "yürük semai", "rumeliturkusu": "Rumeli türküsü",
    "fantezi": "fantezi", "selam": "selam", "sarki": "şarkı",
    "agirsemai": "ağır semai", "sirto": "sırto", "sazsemaisi": "saz semaisi",
    "pesrev": "peşrev", "longa": "longa", "kanto": "kanto", "sazeseri": "saz eseri",
    "aranagme": "aranağme",
}
form_label = lambda f: FORM_DISPLAY.get(f, f)

# Start from matplotlib defaults so the figure renders identically whether this
# script is run on its own or executed from the notebook, which sets its own
# rcParams (spines, title sizes) that would otherwise leak in.
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
VOC = "#1f77b4"; INS = "#d62728"

# ---------- data ----------
fin = [r for r in csv.DictReader(open(RESULTS / "06_finnish_collection_scores.csv"))
       if r["scheme"] == "finnish_blind"]
sym = json.load(open(RESULTS / "09_symbtr_results.json"))
ft = sym["form_table"]; aran = sym["aranagme_sensitivity"]["mean_aug"]

fig = plt.figure(figsize=(14.0, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.35, 1.05], wspace=0.5)

# ===== Panel A: Finnish collections (augmented) =====
axA = fig.add_subplot(gs[0])
rng = np.random.default_rng(0)
for r in fin:
    y = float(r["mean_augmented"]); voc = r["medium"] == "vocal"
    x = 1 + (rng.random()-0.5)*0.28
    axA.scatter(x, y, s=90, marker='o' if voc else 's',
                facecolor=VOC if voc else INS, edgecolor='k', linewidth=0.6, zorder=3)
vmin = min(float(r["mean_augmented"]) for r in fin if r["medium"]=="vocal")
imax = max(float(r["mean_augmented"]) for r in fin if r["medium"]=="instrumental")
axA.axhline((vmin+imax)/2, ls=':', c='0.4', lw=1.2)
axA.annotate("separation\nboundary", (1.32, (vmin+imax)/2), fontsize=8.5, va='center', color='0.3')
axA.set_xlim(0.4, 1.7); axA.set_xticks([])
# headroom below the lowest collection (KT1, -2.26) so the legend cannot cover it
axA.set_ylim(-2.95, 0.80)
axA.set_ylabel("Augmented score (collection mean)")
axA.set_title("A   Finnish collections:\ncomplete separation", loc='left', fontsize=12, fontweight='bold')

# ===== Panel B: Turkish forms, sized by n =====
axB = fig.add_subplot(gs[1])
def size(n): return float(np.clip(n*3.2, 16, 620))
rng2 = np.random.default_rng(1)
for r in ft:
    y = r["aug"]; voc = r["group"] == "vocal"; n = r["n"]
    x = 1 + (rng2.random()-0.5)*0.7
    axB.scatter(x, y, s=size(n), marker='o' if voc else 's',
                facecolor=VOC if voc else INS, edgecolor='k', linewidth=0.6, alpha=0.85, zorder=3)
    if r["form"] in ("oyunhavasi", "kanto"):
        axB.annotate(f"{form_label(r['form'])}\n(n={n})", (x, y), (x+0.28, y),
                     fontsize=8, va='center', color='0.2',
                     arrowprops=dict(arrowstyle='-', lw=0.6, color='0.5'))
axB.scatter(0.5, aran, s=70, marker='D', facecolor='none', edgecolor='0.35', linewidth=1.3, zorder=4)
axB.annotate("aranağme\n(n = 73; not tested)", (0.5, aran), (0.14, aran-1.00),
             fontsize=8, color='0.35', va='center',
             arrowprops=dict(arrowstyle='-', lw=0.6, color='0.5'))
axB.axhline(0, ls=':', c='0.6', lw=1.0)
axB.set_xlim(0.1, 2.0); axB.set_xticks([])
axB.set_ylabel("Augmented score (form mean)")
axB.set_title("B   Turkish forms: vocal forms\ngenerally score higher",
              loc='left', fontsize=12, fontweight='bold')
# size legend
# Size key sits below the axes: inside the panel it reads as three more data points,
# and the header label was clipped by the axes frame.
axB.annotate("pieces per form", (0.02, -0.055), xycoords='axes fraction', fontsize=7.5,
             ha='left', va='center', color='0.4', annotation_clip=False)
for n_leg, xx in [(4, 0.46), (90, 0.62), (300, 0.82)]:
    axB.scatter(xx, -0.055, s=size(n_leg), facecolor='0.7', edgecolor='k', lw=0.5,
                transform=axB.transAxes, clip_on=False, zorder=5)
    axB.annotate(str(n_leg), (xx, -0.125), xycoords='axes fraction', fontsize=7.5,
                 ha='center', va='top', color='0.4', annotation_clip=False)

# ===== Panel C: component AUC by corpus =====
axC = fig.add_subplot(gs[2])
comps = ["Pitch-core", "Gap-rate", "Augmented"]
finnish = [0.762, 0.857, 1.000]
turkish = [sym["form_test_core"]["auc"], sym["form_test_continuity"]["auc"], sym["form_test_augmented"]["auc"]]
xx = np.arange(3)
axC.plot(xx, finnish, 'o', linestyle='none', color="#2ca02c", ms=9, label="Finnish (10 collections)")
axC.plot(xx, turkish, 's', linestyle='none', color="#9467bd", ms=9, label="Turkish (20 forms)")
for i,(f,t) in enumerate(zip(finnish, turkish)):
    axC.annotate(f"{f:.3f}", (i, f-0.035), fontsize=8, ha='center', va='top', color="#2ca02c")
    axC.annotate(f"{t:.3f}", (i, t+0.022), fontsize=8, ha='center', va='bottom', color="#9467bd")
axC.axhline(1.0, ls='--', c='0.75', lw=1)
axC.text(0.03, 0.955, "Complete separation (AUC = 1.00)", transform=axC.transAxes, fontsize=7.3, color='0.55')
axC.set_xticks(xx); axC.set_xticklabels(comps)
axC.set_ylim(0.5, 1.06); axC.set_ylabel("Vocal-instrumental separation (AUC)")
axC.set_title("C   Score performance\ndiffers between corpora", loc='left', fontsize=12, fontweight='bold')
axC.legend(fontsize=8.5, loc='lower right', frameon=True)

# shared legend for medium
leg = [Line2D([],[],marker='o',color='w',markerfacecolor=VOC,markeredgecolor='k',ms=9,label='vocal'),
       Line2D([],[],marker='s',color='w',markerfacecolor=INS,markeredgecolor='k',ms=9,label='instrumental')]
axA.legend(handles=leg, fontsize=8.5, loc='lower left', frameon=True)

fig.savefig(FIGDIR / "Fig4_crosscorpus_transfer.png", dpi=600, bbox_inches='tight', pad_inches=0.2, facecolor='white')
fig.savefig(FIGDIR / "Fig4_crosscorpus_transfer.pdf", bbox_inches='tight', pad_inches=0.2, facecolor='white')
print("saved", FIGDIR / "Fig4_crosscorpus_transfer.png")
