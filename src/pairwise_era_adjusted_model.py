#!/usr/bin/env python3
"""
pairwise_era_adjusted_model.py — Family-level pairwise (mirrored-difference)
reanalysis of the 433 mixed Dutch tune families.

Design
------
For each tune family containing both vocal and instrumental variants:

    Δfeature = mean(feature | vocal variants) − mean(feature | instrumental)

computed for every melodic feature and for documentation year. Each family
contributes two mirrored observations: (+Δ, label 1 = "vocal-first") and
(−Δ, label 0 = "instrumental-first"). Every family therefore has equal
weight, the dataset is exactly balanced, and classification tests
within-family ordering. Both mirrors of a family always share a CV fold
(groups = family).

The logistic regression is fit WITHOUT an intercept: mirrored data are
antisymmetric by construction, so a nonzero intercept could only encode the
arbitrary mirror labelling. Standardization (scale only; the mirrored mean
is exactly 0) and hyperparameter selection happen inside training folds
(nested grouped CV).

Compact a-priori feature sets (redundancy resolved by simplicity /
directness, NOT by outcome performance):
  continuity     : d_prop_gap_transitions, d_mean_notes_between_gaps,
                   d_note_density
                   (mean notes-between-gaps kept over its max- and
                   duration-based twins: r >= 0.8 block, and the mean note
                   count is the simplest member)
  pitch_movement : d_prop_repeated_pitch, d_mean_abs_interval_nonrep,
                   d_pitch_range, d_max_abs_interval
                   (max_abs_interval kept as the extreme-interval measure;
                   the alternative, prop_large_interval_ge5, correlates
                   r = 0.81 with mean_abs_interval_nonrep already in the set)
  combined       : union of the above (7 features)

Models: year_only; each melodic set alone; year + each melodic set.
Metrics on pooled out-of-fold predictions: pairwise accuracy (fraction of
families whose +Δ mirror gets p > 0.5), ROC AUC, balanced accuracy, log
loss. Uncertainty: percentile bootstrap over tune families (B = 2000),
paired resampling for incremental contrasts (year+X vs year_only).

Inputs : 02_data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz (id, tunefamily,
         type, year), 03_outputs/02_dutch_features.csv
Outputs: 03_outputs/04_pairwise_results.csv
         03_outputs/04_pairwise_coefficients.csv
         03_outputs/04_incremental_performance.csv

Usage: python pairwise_era_adjusted_model.py [DATA_DIR] [OUT_DIR]
"""

import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT = 2000
GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.1, 0.5, 0.9]}

CONTINUITY = ["prop_gap_transitions", "mean_notes_between_gaps", "note_density"]
PITCHMOV = ["prop_repeated_pitch", "mean_abs_interval_nonrep",
            "pitch_range", "max_abs_interval"]
COMBINED = CONTINUITY + PITCHMOV

MODELS = {
    "year_only": ["year"],
    "continuity": CONTINUITY,
    "pitch_movement": PITCHMOV,
    "combined": COMBINED,
    "year_plus_continuity": ["year"] + CONTINUITY,
    "year_plus_pitch_movement": ["year"] + PITCHMOV,
    "year_plus_combined": ["year"] + COMBINED,
}
INCREMENTS = [("year_plus_continuity", "year_only"),
              ("year_plus_pitch_movement", "year_only"),
              ("year_plus_combined", "year_only")]


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   fit_intercept=False, max_iter=50000,
                                   random_state=SEED)),
    ])


def build_deltas(data_dir, out_dir):
    """Return (families, delta matrix [n_fam x n_var], var names)."""
    meta = {}
    fam_types = defaultdict(lambda: defaultdict(list))
    p = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["id"]] = (r["tunefamily"], r["type"], float(r["year"]))

    feats = {}
    with open(os.path.join(out_dir, "02_dutch_features.csv")) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            feats[row["melody_id"]] = row

    for mid, (tf, typ, yr) in meta.items():
        if not tf or mid not in feats:
            continue
        fam_types[tf][typ].append(mid)

    variables = COMBINED + ["year"]
    fams, deltas = [], []
    for tf, sides in fam_types.items():
        if not sides["vocal"] or not sides["instrumental"]:
            continue
        row = []
        for v in variables:
            def side_mean(mids):
                vals = []
                for m in mids:
                    x = meta[m][2] if v == "year" else \
                        (float(feats[m][v]) if feats[m][v] != "" else np.nan)
                    vals.append(x)
                vals = np.array(vals, dtype=float)
                return np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            row.append(side_mean(sides["vocal"]) - side_mean(sides["instrumental"]))
        fams.append(tf)
        deltas.append(row)
    return np.array(fams), np.array(deltas), variables


def mirrored(delta_sub):
    """Stack +Δ (label 1) and −Δ (label 0)."""
    X = np.vstack([delta_sub, -delta_sub])
    y = np.concatenate([np.ones(len(delta_sub)), np.zeros(len(delta_sub))])
    return X, y


def oof_predictions(X, y, groups):
    """Pooled out-of-fold probabilities from nested grouped CV."""
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    prob = np.full(len(y), np.nan)
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=SEED)
        gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        prob[te] = gs.predict_proba(X[te])[:, 1]
    assert np.isfinite(prob).all()
    return prob


def metrics(y, prob, fam_index_plus):
    """fam_index_plus: row indices of the +Δ mirror of each family."""
    pred = (prob >= 0.5).astype(int)
    return {
        "pairwise_accuracy": float((prob[fam_index_plus] > 0.5).mean()
                                   + 0.5 * (prob[fam_index_plus] == 0.5).mean()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-12, 1 - 1e-12))),
    }


def boot_ci(stat_fn, n_fam, rng, B=N_BOOT):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n_fam, n_fam)
        vals.append(stat_fn(idx))
    v = np.array(vals)
    return np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"

    fams, D, variables = build_deltas(data_dir, out_dir)
    n_fam = len(fams)
    print(f"{n_fam} mixed families; Δ variables: {variables}")

    # Δ-space correlations among the a-priori combined set (report only)
    sub = D[:, [variables.index(v) for v in COMBINED]]
    ok = np.isfinite(sub).all(axis=1)
    C = np.corrcoef(sub[ok].T)
    high = [(COMBINED[i], COMBINED[j], round(float(C[i, j]), 3))
            for i in range(len(COMBINED)) for j in range(i + 1, len(COMBINED))
            if abs(C[i, j]) >= 0.8]
    print("Δ-space |r|>=0.8 pairs within combined set:", high or "none")

    vidx = {v: j for j, v in enumerate(variables)}
    groups2 = np.concatenate([fams, fams])          # mirrors share the group
    plus_ix = np.arange(n_fam)                      # +Δ rows come first

    rng = np.random.default_rng(SEED)
    oof = {}
    res_rows = []
    for name, vs in MODELS.items():
        Xd = D[:, [vidx[v] for v in vs]]
        X, y = mirrored(Xd)
        prob = oof_predictions(X, y, groups2)
        oof[name] = prob
        m = metrics(y, prob, plus_ix)

        def stat(idx, prob=prob):
            rows = np.concatenate([idx, idx + n_fam])
            yy = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
            pp = prob[rows]
            mm = metrics(yy, pp, np.arange(len(idx)))
            return [mm["pairwise_accuracy"], mm["roc_auc"],
                    mm["balanced_accuracy"], mm["log_loss"]]
        lo, hi = boot_ci(stat, n_fam, np.random.default_rng(SEED))
        keys = ["pairwise_accuracy", "roc_auc", "balanced_accuracy", "log_loss"]
        for k, key in enumerate(keys):
            res_rows.append([name, key, f"{m[key]:.4f}",
                             f"{lo[k]:.4f}", f"{hi[k]:.4f}"])
        print(name, {k: round(m[k], 3) for k in keys})

    with open(os.path.join(out_dir, "04_pairwise_results.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "estimate",
                    "ci95_lo_boot_families", "ci95_hi_boot_families"])
        w.writerows(res_rows)

    # ---- incremental performance (paired bootstrap) --------------------
    inc_rows = []
    for full, base in INCREMENTS:
        pf, pb = oof[full], oof[base]

        def diff_stat(idx):
            rows = np.concatenate([idx, idx + n_fam])
            yy = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
            mf = metrics(yy, pf[rows], np.arange(len(idx)))
            mb = metrics(yy, pb[rows], np.arange(len(idx)))
            return [mf["pairwise_accuracy"] - mb["pairwise_accuracy"],
                    mf["roc_auc"] - mb["roc_auc"],
                    mf["balanced_accuracy"] - mb["balanced_accuracy"],
                    mf["log_loss"] - mb["log_loss"]]
        y_full = np.concatenate([np.ones(n_fam), np.zeros(n_fam)])
        mf = metrics(y_full, pf, plus_ix)
        mb = metrics(y_full, pb, plus_ix)
        est = [mf["pairwise_accuracy"] - mb["pairwise_accuracy"],
               mf["roc_auc"] - mb["roc_auc"],
               mf["balanced_accuracy"] - mb["balanced_accuracy"],
               mf["log_loss"] - mb["log_loss"]]
        lo, hi = boot_ci(diff_stat, n_fam, np.random.default_rng(SEED))
        for k, key in enumerate(["d_pairwise_accuracy", "d_roc_auc",
                                 "d_balanced_accuracy", "d_log_loss"]):
            inc_rows.append([full, base, key, f"{est[k]:.4f}",
                             f"{lo[k]:.4f}", f"{hi[k]:.4f}"])

    with open(os.path.join(out_dir, "04_incremental_performance.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "reference", "metric", "estimate",
                    "ci95_lo_boot_families", "ci95_hi_boot_families"])
        w.writerows(inc_rows)

    # ---- coefficients: full-data refit + family-bootstrap CIs ----------
    coef_rows = []
    for name in ["year_only", "combined", "year_plus_combined"]:
        vs = MODELS[name]
        Xd = D[:, [vidx[v] for v in vs]]
        X, y = mirrored(Xd)
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=SEED)
        gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(X, y, groups=groups2)
        best = gs.best_params_
        coefs = gs.best_estimator_.named_steps["clf"].coef_[0]

        # bootstrap coefficients over families with fixed hyperparameters
        boot = np.zeros((500, len(vs)))
        rngc = np.random.default_rng(SEED)
        for b in range(500):
            idx = rngc.integers(0, n_fam, n_fam)
            Xb, yb = mirrored(Xd[idx])
            pipe = make_pipe()
            pipe.set_params(clf__C=best["clf__C"],
                            clf__l1_ratio=best["clf__l1_ratio"])
            pipe.fit(Xb, yb)
            boot[b] = pipe.named_steps["clf"].coef_[0]
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        for v, c, l, h in zip(vs, coefs, lo, hi):
            coef_rows.append([name, "d_" + v, f"{c:.4f}", f"{l:.4f}",
                              f"{h:.4f}", best["clf__C"],
                              best["clf__l1_ratio"]])

    with open(os.path.join(out_dir, "04_pairwise_coefficients.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "term", "coefficient_std_scale",
                    "ci95_lo_boot_families", "ci95_hi_boot_families",
                    "chosen_C", "chosen_l1_ratio"])
        w.writerows(coef_rows)

    print("wrote 04_pairwise_results.csv, 04_incremental_performance.csv, "
          "04_pairwise_coefficients.csv")
    return high


if __name__ == "__main__":
    main()
