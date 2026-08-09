#!/usr/bin/env python3
"""Render every figure in the paper.

    python src/figures/make_all_figures.py

Writes PNG and PDF into paper_figures/main and paper_figures/SI. Set
RESULTS_DIR / FIGURE_DIR to override the input and output locations.
Fig. S2 is skipped unless results/02_dutch_features.csv is present; it is a
large regenerable table and is not committed.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = [
    "fig1_study_design.py",
    "fig2_within_family.py",
    "fig3_era_windows.py",
    "fig4_crosscorpus.py",
    "fig_si01_melody_level.py",
    "fig_si02_feature_corr.py",
    "fig_si03_all_models.py",
    "fig_si04_compact_vs_full.py",
    "fig_si05_standardization.py",
    "fig_si06_leave_one_out.py",
    "fig_si_finnish.py",
    "fig_si_gaprate.py",
]

for name in SCRIPTS:
    print("--- " + name)
    runpy.run_path(str(HERE / name), run_name="__main__")
print("\nAll figures written.")
