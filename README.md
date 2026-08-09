# Historical music collections share a melodic signature of vocal performance

Analysis code and derived results for a study of whether vocal production
constraints leave a detectable melodic signature in historical folk corpora.

The study compares vocal and instrumental melodies **within the same Dutch tune
families**, so that every between-family difference is removed by construction,
reduces the result to a three-feature unit-weighted score and applies that
score, with fixed signs and unit weights and without target-label fitting, to two separately constructed corpora: the
Digital Archive of Finnish Folk Tunes and the Turkish makam database SymbTr.

---

## Repository layout

```
.
├── src/          analysis scripts, run in the order below
├── src/figures/  one script per figure, plus make_all_figures.py
├── notebooks/    recreate_all_paper_figures.ipynb — calls the figure scripts
├── results/      derived tables (committed; every number in the paper comes from here)
└── data/         corpora are NOT included — see data/README.md
```

## Data

No corpus is redistributed here. All three are public.

| Corpus | Source |
|---|---|
| Meertens Tune Collections, MTC-FS-INST 2.0 | https://www.liederenbank.nl/mtc/ |
| Digital Archive of Finnish Folk Tunes | https://esavelmat.jyu.fi |
| SymbTr (Turkish makam symbolic database) | https://github.com/MTG/SymbTr |

`data/README.md` lists the exact files each script expects and where to put
them. Feature extraction reads only pitch, onset and duration; medium, year,
phrase annotations, lyrics and corpus-specific editorial fields are never used
as predictors.

## Pipeline

Run from the repository root. Scripts take `[DATA_DIR] [RESULTS_DIR]`
arguments and default to `data` and `results`.

| Order | Script | Produces | Used for |
|---|---|---|---|
| 1 | `src/01_inspect_data.py` | `01_file_inventory.csv`, `01_inspection_stats.json` | corpus QC |
| 2 | `src/extract_features.py` | `02_dutch_features.csv`, `02_finnish_features.csv`, `02_feature_qc.json` | Table S2 |
| 3 | `src/model_dutch.py` | `03_dutch_model_results.csv`, `03_dutch_coefficients.csv` | Table S3, Fig. S1 |
| 4 | `src/pairwise_era_adjusted_model.py` | `04_pairwise_results.csv`, `04_pairwise_coefficients.csv`, `04_incremental_performance.csv` | Tables S4–S5, Fig. 2A–B |
| 5 | `src/04_pairwise_window_stability.py` | `04_window_model_performance.csv`, `04_window_coefficients.csv`, `04_compact_vs_full.csv` | Figs. S3–S4 |
| 6 | `src/05_define_final_signatures.py` | `05_final_model_comparison.csv`, `05_primary_signature.json`, `05_continuity_augmented_signature.json` | Tables S6–S8, Fig. 3 |
| 7 | `src/06_validate_finnish_signatures.py` | `06_finnish_*.csv`, `06_*_validation.csv` | Tables S9–S11, Figs. 4A, 4C, S5–S7 |
| 8 | `src/06b_volume_sensitivity.py` | `06b_volume_sensitivity.csv` | merged-volume sensitivity |
| 9 | `src/07_source_type_sensitivity.py` | `07_source_type_results.csv`, `07_source_type_coefficients.csv` | Fig. 2C, SI Materials and Methods 6, Tables S12–S13 |
| 10 | `src/08_gap_threshold_sensitivity.py` | `gap_threshold_results.json` | Table S14, SI Materials and Methods 9 |
| 11 | `src/09_symbtr_transfer.py` | `09_symbtr_results.json`, `09_symbtr_piece_scores.csv` | third-corpus Results, Figs. 4B, S8, SI Materials and Methods 11 |
| 11b | `src/symbtr_dutchref_sensitivity.py` | `symbtr_dutchref_results.json` | Dutch-reference sensitivity, SI Materials and Methods 11 |
| 11c | `src/alignment_analysis.py` (protocol: `src/ALIGNMENT_PROTOCOL.md`) | `alignment_results*.json`, `alignment_pair_metrics.csv`, `alignment_report.md` | exploratory phrase-boundary alignment, SI Materials and Methods 12, Table S18 |
| 12 | `src/figures/make_all_figures.py` | all main and SI figures | — |

Steps 2 and 3 depend on step 2's feature table. Steps 4–6 depend on
`02_dutch_features.csv`; step 7 additionally needs `05_primary_signature.json`
and `02_finnish_features.csv`; step 9 additionally needs the MTC metadata
tables; step 10 (gap threshold) additionally needs `data/finfolktunes.mat`
and `results/06_finnish_melody_scores.csv`; step 11 (SymbTr) needs the unzipped `data/symbtr_txt/` note files. Step 11b reads only `results/09_symbtr_piece_scores.csv` (Dutch reference parameters are embedded, or recomputed with `--dutch-features`); step 11c additionally needs `data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz`. The figure notebook (step 12)
reads only files in `results/`.

`02_dutch_features.csv` and `02_finnish_features.csv` are **not committed**
(size); regenerate them with step 2 before running steps 3–7, 9, or 10.

## Design notes

**Mirrored within-family contrast.** For each tune family containing both
media, every feature is reduced to a vocal-minus-instrumental difference. Each
family contributes that difference twice, with opposite signs and opposite
labels, so the data are exactly balanced and antisymmetric. The logistic
regression is therefore fit without an intercept.

**Nested grouped cross-validation.** Outer fivefold, inner threefold, always
grouped by tune family, so both mirrors of a family and all its variants stay
on the same side of every split. Imputation, standardization and
hyperparameter selection happen inside training folds only.

**Cross-corpus transfer.** The scores applied to the Finnish and Turkish
corpora use unit weights only. No coefficient fitting, feature reselection,
reweighting, threshold optimization or outcome-informed standardization is
performed on either target corpus. Score-level tests in each target corpus are
Holm-corrected across the three transferred scores (pitch-core, gap-rate,
and augmented). No score is assigned sole-primary status.

**Seeds.** `SEED = 42` throughout. Bootstrap and exact-enumeration routines
are deterministic given the documented seeds.

## Environment

```
python >= 3.11
numpy
pandas
scipy
scikit-learn
matplotlib
```

> **Version pinning is incomplete.** Results in `results/` were produced in a
> specific interpreter, and the elastic-net solver's convergence path — and
> therefore the selected hyperparameters — can differ across scikit-learn
> releases. Differences observed across versions were at most 0.014 AUC and
> changed no reported conclusion, but exact reproduction of the committed
> tables requires the original environment. Record the working versions in
> `requirements.txt` before relying on byte-level reproduction.
>
> The Finnish and Turkish transfers are unaffected: they involve no model
> fitting, only unit weights, z-scores and exact enumeration.

## Reproducing the figures

Every figure has exactly one implementation, a script in `src/figures/`. Render
all twelve with:

```bash
MPLBACKEND=Agg python src/figures/make_all_figures.py
```

or render one at a time:

```bash
python src/figures/fig1_study_design.py
python src/figures/fig_si05_standardization.py
```

PNG and PDF go to `paper_figures/main` and `paper_figures/SI`. Run from anywhere
inside the repository; the scripts walk up from the working directory to find
`results/`. `RESULTS_DIR` and `FIGURE_DIR` override the input and output
locations.

The notebook renders the same twelve figures by calling the same scripts, so it
cannot drift away from the submitted versions:

```bash
MPLBACKEND=Agg jupyter nbconvert --execute --to notebook --inplace \
  notebooks/recreate_all_paper_figures.ipynb
```

**Reproduction status.** Both routes produce all twelve figures byte-identical
to the versions in the manuscript, given `results/02_dutch_features.csv`. That
table is large and not committed; without it Fig. S2 is skipped with a message
and the other eleven still match. Regenerate it with
`python src/extract_features.py`.

`MPLBACKEND=Agg` matters: Fig. S7 calls `tight_layout`, whose text metrics
differ between the Agg and macosx canvases, so a GUI backend produces a slightly
different layout from the committed figure.

Turkish form names are mapped to their human-readable spellings (`sazsemaisi`
-> `saz semaisi`), matching SI Table S16 and the main text.

**Figure vocabulary.** Axes that report an AUC are all labelled
"Vocal-instrumental separation (AUC)". *Continuity*, *pitch movement*,
*compact* and *combined* name fitted models; *pitch-core*, *gap-rate* and
*augmented* name only the three transferred scores.

## Citation

Manuscript in preparation. Citation details to be added on publication.

## License

Code and derived result tables are released under the MIT License (see
`LICENSE`). Note that the underlying corpora carry their own terms; consult
the Meertens Instituut and the University of Jyväskylä archive for reuse
conditions on the source data.
