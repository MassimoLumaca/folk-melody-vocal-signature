#!/usr/bin/env python3
"""SI: Finnish component decomposition — pitch core and gap-rate continuity make complementary errors."""
import csv
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


# Start from matplotlib defaults so the figure renders identically whether this
# script is run on its own or executed from the notebook, which sets its own
# rcParams (spines, title sizes) that would otherwise leak in.
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
VOC="#1f77b4"; INS="#d62728"

rows=[r for r in csv.DictReader(open(RESULTS / "06_finnish_collection_scores.csv"))
      if r["scheme"]=="finnish_blind"]
def val(r,k): return float(r[k])
def voc(r): return r["medium"]=="vocal"

fig,(axA,axB)=plt.subplots(1,2,figsize=(12.5,5.6))

def panel(ax,key,title,notes):
    order=sorted(rows,key=lambda r:val(r,key))          # low at bottom
    for i,r in enumerate(order):
        ax.scatter(val(r,key), i, s=95, marker='o' if voc(r) else 's',
                   facecolor=VOC if voc(r) else INS, edgecolor='k', linewidth=0.7, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r["collection"] for r in order], fontsize=9)
    for t,r in zip(ax.get_yticklabels(),order):
        t.set_color(VOC if voc(r) else INS)
    ax.axvline(0, ls=':', c='0.6', lw=1)
    ax.set_xlabel("Mean collection score")
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.7, len(order)-0.3)
    for coll,txt,dx in notes:
        i=[j for j,r in enumerate(order) if r["collection"]==coll][0]
        ax.annotate(txt,(val([r for r in order if r["collection"]==coll][0],key),i),
                    (dx,0),textcoords='offset points',fontsize=9.5,va='center',color='0.2')

panel(axA,"mean_core","A   Pitch-core score: repetition and range",
      [("Jouhikkosävelmiä","← instrumental; ranks among vocal collections",10),
       ("HS1","← vocal; lowest of the vocal collections",10)])
panel(axB,"mean_continuity","B   Gap-rate score: notated gaps",
      [("HS1","← vocal; ranks highest here",-140),
       ("RS2","← runo-song (vocal);\n   ranks among instrumental collections",10),
       ("RS1","",10)])

fig.legend(handles=[
    Line2D([],[],marker='o',color='w',markerfacecolor=VOC,markeredgecolor='k',ms=9,label='vocal collection'),
    Line2D([],[],marker='s',color='w',markerfacecolor=INS,markeredgecolor='k',ms=9,label='instrumental collection')],
    fontsize=10.5, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.02))
fig.suptitle("The component scores rank different Finnish collections incorrectly",
             fontsize=12.5, y=0.99, color='0.3')
fig.tight_layout(rect=(0,0.04,1,0.96))
fig.savefig(FIGDIR / "FigS7_finnish_component_decomposition.png", dpi=600, bbox_inches='tight', pad_inches=0.2, facecolor='white')
fig.savefig(FIGDIR / "FigS7_finnish_component_decomposition.pdf", bbox_inches='tight', pad_inches=0.2, facecolor='white')
print("saved SI Finnish decomposition")
