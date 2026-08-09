#!/usr/bin/env python3
"""
model_dutch.py — Vocal vs instrumental classification within Dutch tune
families that contain both variants (MTC-FS-INST 2.0).

Design
------
- Unit of grouping: tune family. Only families with >= 1 vocal and >= 1
  instrumental melody are used. All cross-validation splits keep complete
  tune families together (StratifiedGroupKFold, groups = tunefamily).
- Model: elastic-net logistic regression (saga). The pipeline
  [median imputation -> standardization -> logistic regression] is fit
  entirely inside training folds; hyperparameters (C, l1_ratio) are chosen
  by an inner grouped, stratified 3-fold CV on the training fold only
  (nested CV). Outer CV: 5 folds.
- Feature sets:
    continuity : prop_gap_transitions, mean/max notes between gaps,
                 mean/max duration between gaps, note_density
    pitch      : pitch_range, max_abs_interval, mean_abs_interval_nonrep,
                 prop_repeated_pitch, prop_large_interval_ge5
    combined   : union of the two after removing redundant features by a
                 deterministic greedy rule (drop any feature with
                 |Pearson r| >= 0.8 to an already-kept feature, in fixed
                 list order, computed on the modeling subset)
  Baselines: chance (prior-probability dummy), n_notes alone, year alone,
  n_notes + year (same elastic-net pipeline for comparability).
- Metrics per outer test fold: ROC AUC and balanced accuracy (0.5 threshold).
- Coefficients: the same grouped grid search refit on the full modeling
  subset; coefficients are on the standardized (z-score) scale.

Inputs:  02_data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz (id, tunefamily,
         type, year only) and 03_outputs/02_dutch_features.csv.
Outputs: 03_outputs/03_dutch_model_results.csv
         03_outputs/03_dutch_coefficients.csv

Usage: python model_dutch.py [DATA_DIR] [OUT_DIR]
"""

import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.1, 0.5, 0.9]}
REDUNDANCY_R = 0.8

CONTINUITY = ["prop_gap_transitions", "mean_notes_between_gaps",
              "max_notes_between_gaps", "mean_dur_between_gaps",
              "max_dur_between_gaps", "note_density"]
PITCH = ["pitch_range", "max_abs_interval", "mean_abs_interval_nonrep",
         "prop_repeated_pitch", "prop_large_interval_ge5"]


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   max_iter=20000, random_state=SEED)),
    ])


def greedy_nonredundant(X, names, thr=REDUNDANCY_R):
    """Keep features in list order; drop any with |r| >= thr to a kept one."""
    kept, kept_ix = [], []
    for j, name in enumerate(names):
        col = X[:, j]
        ok = np.isfinite(col)
        redundant = False
        for ki in kept_ix:
            both = ok & np.isfinite(X[:, ki])
            if both.sum() > 2:
                r = np.corrcoef(col[both], X[both, ki])[0, 1]
                if abs(r) >= thr:
                    redundant = True
                    break
        if not redundant:
            kept.append(name)
            kept_ix.append(j)
    return kept


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"

    # ---- metadata: id -> (tunefamily, type, year) --------------------
    meta = {}
    fam_types = defaultdict(Counter)
    p = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["id"]] = (r["tunefamily"], r["type"], r["year"])
            if r["tunefamily"]:
                fam_types[r["tunefamily"]][r["type"]] += 1
    mixed = {tf for tf, c in fam_types.items()
             if c["vocal"] > 0 and c["instrumental"] > 0}

    # ---- features ----------------------------------------------------
    feats = {}
    with open(os.path.join(out_dir, "02_dutch_features.csv")) as fh:
        for row in csv.DictReader(fh):
            feats[row["melody_id"]] = row

    ids, groups, y, year, rows = [], [], [], [], []
    all_feature_names = CONTINUITY + PITCH + ["n_notes"]
    for mid, (tf, typ, yr) in meta.items():
        if tf not in mixed or mid not in feats:
            continue
        fr = feats[mid]
        ids.append(mid)
        groups.append(tf)
        y.append(1 if typ == "vocal" else 0)
        year.append(float(yr))
        rows.append([float(fr[k]) if fr[k] != "" else np.nan
                     for k in all_feature_names])
    X_all = np.array(rows)
    y = np.array(y)
    year = np.array(year)
    groups = np.array(groups)
    col = {k: j for j, k in enumerate(all_feature_names)}
    print(f"modeling subset: {len(ids)} melodies, "
          f"{len(set(groups))} tune families, "
          f"{y.sum()} vocal / {(1 - y).sum()} instrumental")

    combined = greedy_nonredundant(X_all[:, [col[k] for k in CONTINUITY + PITCH]],
                                   CONTINUITY + PITCH)
    print("combined (non-redundant) features:", combined)

    def matrix(names, add_year=False):
        M = X_all[:, [col[k] for k in names]] if names else \
            np.empty((len(y), 0))
        if add_year:
            M = np.column_stack([M, year])
        return M

    models = {
        "continuity": (matrix(CONTINUITY), CONTINUITY),
        "pitch_movement": (matrix(PITCH), PITCH),
        "combined": (matrix(combined), combined),
        "baseline_n_notes": (matrix(["n_notes"]), ["n_notes"]),
        "baseline_year": (matrix([], add_year=True), ["year"]),
        "baseline_n_notes_year": (matrix(["n_notes"], add_year=True),
                                  ["n_notes", "year"]),
    }

    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    results = []

    # chance model
    for f, (tr, te) in enumerate(outer.split(X_all, y, groups), 1):
        dummy = DummyClassifier(strategy="prior").fit(X_all[tr], y[tr])
        prob = dummy.predict_proba(X_all[te])[:, 1]
        results.append(["chance", f, len(te), len(set(groups[te])),
                        roc_auc_score(y[te], prob),
                        balanced_accuracy_score(y[te], dummy.predict(X_all[te])),
                        "", ""])

    for name, (X, fnames) in models.items():
        for f, (tr, te) in enumerate(outer.split(X, y, groups), 1):
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                         random_state=SEED)
            gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc",
                              cv=inner, n_jobs=-1)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            prob = gs.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            results.append([name, f, len(te), len(set(groups[te])),
                            roc_auc_score(y[te], prob),
                            balanced_accuracy_score(y[te], pred),
                            gs.best_params_["clf__C"],
                            gs.best_params_["clf__l1_ratio"]])
            print(f"{name} fold {f}: AUC={results[-1][4]:.3f} "
                  f"bal_acc={results[-1][5]:.3f} best={gs.best_params_}")

    # ---- results CSV (per fold + mean/sd) ------------------------------
    res_path = os.path.join(out_dir, "03_dutch_model_results.csv")
    with open(res_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "fold", "n_test", "n_test_families",
                    "roc_auc", "balanced_accuracy", "best_C", "best_l1_ratio"])
        for r in results:
            w.writerow([r[0], r[1], r[2], r[3],
                        f"{r[4]:.4f}", f"{r[5]:.4f}", r[6], r[7]])
        for name in ["chance"] + list(models):
            rs = [r for r in results if r[0] == name]
            auc = np.array([r[4] for r in rs])
            bac = np.array([r[5] for r in rs])
            w.writerow([name, "mean", sum(r[2] for r in rs),
                        len(set(groups)), f"{auc.mean():.4f}",
                        f"{bac.mean():.4f}", "", ""])
            w.writerow([name, "sd", "", "", f"{auc.std(ddof=1):.4f}",
                        f"{bac.std(ddof=1):.4f}", "", ""])
    print("wrote", res_path)

    # ---- coefficients from full-subset refit ---------------------------
    coef_path = os.path.join(out_dir, "03_dutch_coefficients.csv")
    with open(coef_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "term", "coefficient_std_scale",
                    "chosen_C", "chosen_l1_ratio"])
        for name, (X, fnames) in models.items():
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                         random_state=SEED)
            gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc",
                              cv=inner, n_jobs=-1)
            gs.fit(X, y, groups=groups)
            clf = gs.best_estimator_.named_steps["clf"]
            C = gs.best_params_["clf__C"]
            l1 = gs.best_params_["clf__l1_ratio"]
            w.writerow([name, "(intercept)", f"{clf.intercept_[0]:.4f}", C, l1])
            for fn, c in zip(fnames, clf.coef_[0]):
                w.writerow([name, fn, f"{c:.4f}", C, l1])
    print("wrote", coef_path)


if __name__ == "__main__":
    main()
