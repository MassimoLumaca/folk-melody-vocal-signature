#!/usr/bin/env python3
"""
04_pairwise_window_stability.py — Era-window sensitivity analysis of the
mirrored within-family pairwise design (Dutch mixed tune families).

Continues the design of src/pairwise_era_adjusted_model.py: for each tune
family with both vocal and instrumental variants, Δ = mean(vocal) −
mean(instrumental) per variable; two mirrored observations per family
(+Δ = "vocal-first" = 1, −Δ = "instrumental-first" = 0); intercept-free
elastic-net logistic regression; both mirrors always share the CV fold
(groups = family); imputation, scaling and hyperparameter selection inside
training folds (outer 5-fold / inner 3-fold nested grouped CV).

New here: the analysis is repeated in nested era windows defined by the
family's absolute documentation-year difference |Δyear| ≤ 15, 25, 50, 75,
100 years, and unrestricted. Windows are fixed a priori as a sensitivity axis;
no window is selected on performance.

Models per window:
    year_only            : Δyear
    pitch_movement       : repeated-pitch prop., mean abs interval (non-rep),
                           pitch range, max abs interval
    combined             : full 7-feature melodic set (as in analysis 04)
    compact              : gap rate, repeated-pitch prop., pitch range,
                           max abs interval  (the four coefficients supported
                           in the unrestricted era-adjusted model)
    year_plus_<each melodic model>

Reported per window: n families; pooled out-of-fold AUC, pairwise accuracy
and log loss with family-bootstrap 95% CIs (B = 2000); paired-bootstrap
improvement of every model over year_only; coefficients of the
year-adjusted models with family-bootstrap CIs (B = 500 refits at fixed
hyperparameters), coefficient sign stability (share of refits matching the
point-estimate sign) and selection frequency (share of refits with a
nonzero coefficient). Compact vs full: paired bootstrap of melodic-only and
year-adjusted variants; the compact model is preferred if its AUC is within
0.02 of the full model and the paired bootstrap shows no clear loss (95% CI
of AUC difference not entirely below −0.02).

Inputs : 02_data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz,
         03_outputs/02_dutch_features.csv
Outputs: 03_outputs/04_window_model_performance.csv
         03_outputs/04_window_coefficients.csv
         03_outputs/04_compact_vs_full.csv

Usage: python 04_pairwise_window_stability.py [DATA_DIR] [OUT_DIR]
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
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_BOOT_METRIC = 2000
N_BOOT_COEF = 300
GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.1, 0.5, 0.9]}
ZERO_TOL = 1e-6
EQUIV_MARGIN = 0.02

FULL = ["prop_gap_transitions", "mean_notes_between_gaps", "note_density",
        "prop_repeated_pitch", "mean_abs_interval_nonrep", "pitch_range",
        "max_abs_interval"]
PITCHMOV = ["prop_repeated_pitch", "mean_abs_interval_nonrep",
            "pitch_range", "max_abs_interval"]
COMPACT = ["prop_gap_transitions", "prop_repeated_pitch",
           "pitch_range", "max_abs_interval"]

MELODIC_MODELS = {"pitch_movement": PITCHMOV, "combined": FULL,
                  "compact": COMPACT}
WINDOWS = [("le15", 15.0), ("le25", 25.0), ("le50", 50.0), ("le75", 75.0),
           ("le100", 100.0), ("unrestricted", np.inf)]
COEF_MODELS = ["year_only", "year_plus_combined", "year_plus_compact"]


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   fit_intercept=False, max_iter=50000,
                                   random_state=SEED)),
    ])


def build_deltas(data_dir, out_dir):
    meta = {}
    p = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["id"]] = (r["tunefamily"], r["type"], float(r["year"]))
    feats = {}
    with open(os.path.join(out_dir, "02_dutch_features.csv")) as fh:
        for row in csv.DictReader(fh):
            feats[row["melody_id"]] = row
    fam = defaultdict(lambda: defaultdict(list))
    for mid, (tf, typ, yr) in meta.items():
        if tf and mid in feats:
            fam[tf][typ].append(mid)
    variables = FULL + ["year"]
    fams, deltas = [], []
    for tf, sides in fam.items():
        if not sides["vocal"] or not sides["instrumental"]:
            continue
        row = []
        for v in variables:
            def smean(mids):
                vals = np.array([
                    meta[m][2] if v == "year" else
                    (float(feats[m][v]) if feats[m][v] != "" else np.nan)
                    for m in mids])
                return np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            row.append(smean(sides["vocal"]) - smean(sides["instrumental"]))
        fams.append(tf)
        deltas.append(row)
    return np.array(fams), np.array(deltas), variables


def mirrored(delta_sub):
    X = np.vstack([delta_sub, -delta_sub])
    y = np.concatenate([np.ones(len(delta_sub)), np.zeros(len(delta_sub))])
    return X, y


def oof_predictions(X, y, groups):
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


def metr(prob, n_fam, idx=None):
    """Metrics on families idx (default all). Rows: [+Δ block | −Δ block]."""
    if idx is None:
        idx = np.arange(n_fam)
    rows = np.concatenate([idx, idx + n_fam])
    yy = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
    pp = prob[rows]
    pw = float((pp[:len(idx)] > 0.5).mean() + 0.5 * (pp[:len(idx)] == 0.5).mean())
    return {"pairwise_accuracy": pw,
            "roc_auc": float(roc_auc_score(yy, pp)),
            "log_loss": float(log_loss(yy, np.clip(pp, 1e-12, 1 - 1e-12)))}


KEYS = ["pairwise_accuracy", "roc_auc", "log_loss"]


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"

    fams, D, variables = build_deltas(data_dir, out_dir)
    vidx = {v: j for j, v in enumerate(variables)}
    absdy = np.abs(D[:, vidx["year"]])

    perf_rows, coef_rows, cmp_rows = [], [], []

    for wname, wlim in WINDOWS:
        sel = absdy <= wlim
        Dw, fw = D[sel], fams[sel]
        n_fam = len(fw)
        groups2 = np.concatenate([fw, fw])
        print(f"== window {wname}: {n_fam} families")

        models = {"year_only": ["year"]}
        for mn, vs in MELODIC_MODELS.items():
            models[mn] = vs
            models["year_plus_" + mn] = ["year"] + vs

        # one shared set of bootstrap family draws per window -> all
        # contrasts are paired on identical resamples
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, n_fam, (N_BOOT_METRIC, n_fam))

        oof, boot_m, point_m = {}, {}, {}
        for name, vs in models.items():
            Xd = Dw[:, [vidx[v] for v in vs]]
            X, y = mirrored(Xd)
            oof[name] = oof_predictions(X, y, groups2)
            point_m[name] = metr(oof[name], n_fam)
            boot_m[name] = np.array(
                [[metr(oof[name], n_fam, idx)[k] for k in KEYS]
                 for idx in draws])
            lo, hi = np.percentile(boot_m[name], [2.5, 97.5], axis=0)
            for k, key in enumerate(KEYS):
                perf_rows.append([wname, n_fam, name, "", key,
                                  f"{point_m[name][key]:.4f}",
                                  f"{lo[k]:.4f}", f"{hi[k]:.4f}"])
            print(f"  {name}: AUC={point_m[name]['roc_auc']:.3f} "
                  f"pw={point_m[name]['pairwise_accuracy']:.3f}")

        # improvements over year_only (paired: same draws)
        for name in models:
            if name == "year_only":
                continue
            diffs = boot_m[name] - boot_m["year_only"]
            lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
            for k, key in enumerate(KEYS):
                perf_rows.append([
                    wname, n_fam, name, "year_only", "d_" + key,
                    f"{point_m[name][key] - point_m['year_only'][key]:.4f}",
                    f"{lo[k]:.4f}", f"{hi[k]:.4f}"])

        # compact vs full (paired: same draws), melodic-only and adjusted
        for a_name, b_name in [("compact", "combined"),
                               ("year_plus_compact", "year_plus_combined")]:
            est = {k: point_m[a_name][k] - point_m[b_name][k] for k in KEYS}
            diffs = boot_m[a_name] - boot_m[b_name]
            lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
            d_auc = est["roc_auc"]                      # compact − full
            auc_hi = hi[KEYS.index("roc_auc")]
            # compact preferred iff its AUC is within EQUIV_MARGIN of (or
            # better than) the full model AND the paired bootstrap does not
            # clearly support a loss (the 95% CI of d_auc is not entirely
            # below 0)
            equivalent = (d_auc >= -EQUIV_MARGIN) and (auc_hi >= 0.0)
            for k, key in enumerate(KEYS):
                cmp_rows.append([wname, n_fam, a_name, b_name, "d_" + key,
                                 f"{est[key]:.4f}", f"{lo[k]:.4f}",
                                 f"{hi[k]:.4f}",
                                 "compact_preferred" if equivalent else
                                 "full_preferred"])

        # coefficients, sign stability, selection frequency
        for name in COEF_MODELS:
            vs = models[name]
            Xd = Dw[:, [vidx[v] for v in vs]]
            X, y = mirrored(Xd)
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                         random_state=SEED)
            gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc",
                              cv=inner, n_jobs=-1)
            gs.fit(X, y, groups=groups2)
            best = gs.best_params_
            point = gs.best_estimator_.named_steps["clf"].coef_[0]

            rng = np.random.default_rng(SEED)
            boot = np.zeros((N_BOOT_COEF, len(vs)))
            for b in range(N_BOOT_COEF):
                idx = rng.integers(0, n_fam, n_fam)
                Xb, yb = mirrored(Xd[idx])
                pipe = make_pipe()
                pipe.set_params(clf__C=best["clf__C"],
                                clf__l1_ratio=best["clf__l1_ratio"],
                                clf__max_iter=5000, clf__tol=1e-3)
                pipe.fit(Xb, yb)
                boot[b] = pipe.named_steps["clf"].coef_[0]
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
            selfreq = (np.abs(boot) > ZERO_TOL).mean(axis=0)
            sign_stab = np.where(
                np.abs(point) > ZERO_TOL,
                (np.sign(boot) == np.sign(point)).mean(axis=0),
                (np.abs(boot) <= ZERO_TOL).mean(axis=0))
            for j, v in enumerate(vs):
                coef_rows.append([wname, n_fam, name, "d_" + v,
                                  f"{point[j]:.4f}", f"{lo[j]:.4f}",
                                  f"{hi[j]:.4f}", f"{sign_stab[j]:.3f}",
                                  f"{selfreq[j]:.3f}", best["clf__C"],
                                  best["clf__l1_ratio"]])

    with open(os.path.join(out_dir, "04_window_model_performance.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "n_families", "model", "reference", "metric",
                    "estimate", "ci95_lo", "ci95_hi"])
        w.writerows(perf_rows)
    with open(os.path.join(out_dir, "04_window_coefficients.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "n_families", "model", "term",
                    "coefficient_std_scale", "ci95_lo", "ci95_hi",
                    "sign_stability", "selection_frequency",
                    "chosen_C", "chosen_l1_ratio"])
        w.writerows(coef_rows)
    with open(os.path.join(out_dir, "04_compact_vs_full.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "n_families", "model", "reference", "metric",
                    "estimate", "ci95_lo", "ci95_hi", "verdict_auc_rule"])
        w.writerows(cmp_rows)
    print("wrote performance, coefficients, compact-vs-full CSVs")


if __name__ == "__main__":
    main()
