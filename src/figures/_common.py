#!/usr/bin/env python3
"""Shared setup for the figure scripts: paths, plotting style, and input tables.

Every figure in the paper is produced by a script in this directory. The
notebook `notebooks/recreate_all_paper_figures.ipynb` only calls them, so there
is exactly one implementation of each figure.

Paths resolve without configuration from anywhere inside the repository:
  RESULTS_DIR  overrides where the committed tables are read from
  FIGURE_DIR   overrides where PNG and PDF are written
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_FILE = "04_pairwise_results.csv"


def _search_roots():
    here = Path.cwd().resolve()
    yield here
    for parent in here.parents:
        yield parent
    here = Path(__file__).resolve()
    for parent in here.parents:
        yield parent


def find_results_dir():
    env = os.environ.get("RESULTS_DIR")
    if env and (Path(env) / REQUIRED_FILE).exists():
        return Path(env).resolve()
    for root in _search_roots():
        for candidate in (root / "results", root, root / "03_outputs"):
            if (candidate / REQUIRED_FILE).exists():
                return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate {REQUIRED_FILE}. Run from inside the repository, or "
        "set RESULTS_DIR to the folder holding the committed result tables."
    )


RESULTS = find_results_dir()


def figure_dir(sub):
    """main/ or SI/ under paper_figures, honouring $FIGURE_DIR."""
    env = os.environ.get("FIGURE_DIR")
    out = Path(env) if env else Path.cwd() / "paper_figures" / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


# The style the main and SI figures were composed in. The three scripts that
# predate this module (fig4_crosscorpus, fig_si_finnish, fig_si_gaprate) reset
# to the matplotlib defaults instead and set their own font only; that
# difference is deliberate and is what their published layouts depend on.
NOTEBOOK_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(NOTEBOOK_STYLE)


def save_figure(fig, folder, stem):
    """Save as high-resolution PNG and vector PDF."""
    png_path = folder / f"{stem}.png"
    pdf_path = folder / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.2,
                facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def load_tables():
    """Every committed table the figures read, as a namespace-like dict.

    02_dutch_features.csv is large and not committed; only Fig. S2 needs it, so
    it is returned as None when absent.
    """
    features_path = RESULTS / "02_dutch_features.csv"
    return dict(
        dutch_features=pd.read_csv(features_path) if features_path.exists() else None,
        melody_models=pd.read_csv(RESULTS / "03_dutch_model_results.csv"),
        pairwise_results=pd.read_csv(RESULTS / "04_pairwise_results.csv"),
        pairwise_coefficients=pd.read_csv(RESULTS / "04_pairwise_coefficients.csv"),
        window_performance=pd.read_csv(RESULTS / "04_window_model_performance.csv"),
        window_coefficients=pd.read_csv(RESULTS / "04_window_coefficients.csv"),
        compact_vs_full=pd.read_csv(RESULTS / "04_compact_vs_full.csv"),
        final_model_comparison=pd.read_csv(RESULTS / "05_final_model_comparison.csv"),
        finnish_collections=pd.read_csv(RESULTS / "06_finnish_collection_scores.csv"),
        core_validation=pd.read_csv(RESULTS / "06_primary_core_validation.csv"),
        continuity_validation=pd.read_csv(RESULTS / "06_continuity_validation.csv"),
        augmented_validation=pd.read_csv(RESULTS / "06_augmented_validation.csv"),
        source_type_results=pd.read_csv(RESULTS / "07_source_type_results.csv"),
        symbtr_results=json.loads((RESULTS / "09_symbtr_results.json").read_text()),
        symbtr_pieces=pd.read_csv(RESULTS / "09_symbtr_piece_scores.csv"),
    )
