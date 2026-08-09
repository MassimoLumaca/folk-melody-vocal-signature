#!/usr/bin/env python3
"""SI diagnostic: SymbTr gap rate by makam form (piece level) + piece-vs-form AUC."""
import csv, json, bisect
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
FIGDIR = Path(os.environ.get("FIGURE_DIR", Path.cwd() / "paper_figures" / "SI"))
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
VOC="#1f77b4"; INS="#d62728"

pieces=list(csv.DictReader(open(RESULTS / "09_symbtr_piece_scores.csv")))
sym=json.load(open(RESULTS / "09_symbtr_results.json"))
ft=sym["form_table"]                      # 20 test forms (>=4 pieces), with gap_raw + group
test_forms={r["form"]:r for r in ft}

# order forms by form-mean gap rate (ascending -> instrumental low, vocal high)
order=sorted(ft, key=lambda r: r["gap_raw"])
form_names=[r["form"] for r in order]
ypos={f:i for i,f in enumerate(form_names)}

fig=plt.figure(figsize=(12.5, 7.2))
gs=fig.add_gridspec(1,2,width_ratios=[1.7,1.0],wspace=0.42)

# ---- Panel A: per-piece gap rate by form ----
axA=fig.add_subplot(gs[0])
rng=np.random.default_rng(0)
for x in pieces:
    f=x["form"]
    if f not in ypos: continue
    g=test_forms[f]["group"]
    y=ypos[f]+(rng.random()-0.5)*0.55
    axA.scatter(float(x["gap"]), y, s=10, color=VOC if g=="vocal" else INS,
                alpha=0.28, edgecolor='none', zorder=2)
for r in order:
    y=ypos[r["form"]]
    axA.scatter(r["gap_raw"], y, marker='D', s=70,
                facecolor=VOC if r["group"]=="vocal" else INS,
                edgecolor='k', linewidth=0.8, zorder=4)
axA.set_yticks(range(len(form_names)))
axA.set_yticklabels([f"{form_label(f)}  (n={test_forms[f]['n']})" for f in form_names], fontsize=8.5)
for t,f in zip(axA.get_yticklabels(),form_names):
    t.set_color(VOC if test_forms[f]["group"]=="vocal" else INS)
axA.set_xlabel("Gap rate: proportion of transitions separated by ≥0.5 beats of silence")
axA.set_title("A   Gap rate varies widely among pieces within the same form",
              loc='left', fontsize=12, fontweight='bold')
axA.set_ylim(-0.8, len(form_names)-0.2)
axA.legend(handles=[
    Line2D([],[],marker='o',color='w',markerfacecolor=VOC,markersize=8,label='vocal pieces',alpha=0.5),
    Line2D([],[],marker='o',color='w',markerfacecolor=INS,markersize=8,label='instrumental pieces',alpha=0.5),
    Line2D([],[],marker='D',color='w',markerfacecolor='0.5',markeredgecolor='k',markersize=8,label='form mean')],
    fontsize=10, loc='lower right', frameon=True)

# ---- Panel B: piece-level vs form-level AUC per component ----
axB=fig.add_subplot(gs[1])
comps=["Pitch-core","Gap-rate","Augmented"]
# Panel B compares like with like: the piece-level AUCs are computed on the same
# 1,711 pieces that make up the 20 forms in panel A, so the contrast is purely
# the effect of averaging within forms. (The 1,749 medium-labelled values are in
# Table S15.)
def _auc(pos, neg):
    neg = sorted(neg); n = len(pos)*len(neg); acc = 0.0
    for v in pos:
        lo = bisect.bisect_left(neg, v); hi = bisect.bisect_right(neg, v)
        acc += lo + 0.5*(hi-lo)
    return acc/n

_sub = [p for p in pieces if p["form"] in test_forms and p["group"] in ("vocal", "instrumental")]
_v = [p for p in _sub if p["group"] == "vocal"]
_i = [p for p in _sub if p["group"] == "instrumental"]
N_MATCHED = len(_sub)
piece_auc = [_auc([float(p[c]) for p in _v], [float(p[c]) for p in _i])
             for c in ("core", "gap", "aug")]
form_auc=[sym["form_test_core"]["auc"], sym["form_test_continuity"]["auc"], sym["form_test_augmented"]["auc"]]
xx=np.arange(3); w=0.36
b1=axB.bar(xx-w/2, piece_auc, w, color='0.6', label=f'Individual pieces (n = {N_MATCHED:,})')
b2=axB.bar(xx+w/2, form_auc, w, color='#9467bd', label='Form means (20 forms)')
for b,v in zip(b1,piece_auc): axB.annotate(f"{v:.3f}",(b.get_x()+b.get_width()/2,v),(0,2),
    textcoords='offset points',ha='center',fontsize=9.5)
for b,v in zip(b2,form_auc): axB.annotate(f"{v:.3f}",(b.get_x()+b.get_width()/2,v),(0,2),
    textcoords='offset points',ha='center',fontsize=9.5)
axB.axhline(0.5, ls=':', c='0.5', lw=1); axB.annotate("chance",(2.35,0.5),(0,2),
    textcoords='offset points',fontsize=7.5,color='0.5')
axB.set_xticks(xx); axB.set_xticklabels(comps, fontsize=9.5)
axB.set_ylim(0.45,1.0); axB.set_ylabel("Vocal-instrumental separation (AUC)")
axB.set_title("B   Averaging within forms reveals the gap-rate pattern",
              loc='left', fontsize=12, fontweight='bold')
axB.legend(fontsize=9.5, loc='upper center', frameon=True)
axB.annotate("gap rate separates media\nonly after form averaging",
    (1,piece_auc[1]),(1.15,0.60), fontsize=9.5, color='0.25',
    arrowprops=dict(arrowstyle='->',lw=0.8,color='0.4'))

fig.savefig(FIGDIR / "FigS8_symbtr_gap_rate_by_form.png", dpi=600, bbox_inches='tight', pad_inches=0.2, facecolor='white')
fig.savefig(FIGDIR / "FigS8_symbtr_gap_rate_by_form.pdf", bbox_inches='tight', pad_inches=0.2, facecolor='white')
print("saved SI gap-rate diagnostic")
