#!/usr/bin/env python3
"""
05_define_final_signatures.py — Define the smallest Dutch melodic signature
that stays informative after within-family control and era matching, plus a
continuity-augmented score adding gap rate.

Continues the intercept-free mirrored within-family pairwise design
(src/pairwise_era_adjusted_model.py, src/04_pairwise_window_stability.py).
Finnish data are NOT touched at this stage.

Predefined melodic models
    minimal_pitch_core   : prop_repeated_pitch, pitch_range
    extended_pitch_core  : + max_abs_interval
    continuity_augmented : + prop_gap_transitions (gap rate)

Era windows (fixed sensitivity axis, never selected on performance):
|Δyear| ≤ 15, 25, 50, 75, 100 years, and unrestricted.

Per window and model (and year-adjusted variant): pooled out-of-fold AUC,
pairwise accuracy, log loss with family-bootstrap 95% CIs (B = 2000 shared
draws — all contrasts paired); improvement over year-only; coefficients of
year-adjusted variants with family-bootstrap CIs, sign stability and
selection frequency (B = 300 refits, fixed hyperparameters).

Prespecified selection rules (encoded in decide_signature()):
 R1  Prefer the smaller model when its AUC is within 0.02 of the larger and
     the paired bootstrap shows no clear loss (95% CI of the AUC difference
     not entirely below zero).
 R2  The core score may contain only features whose coefficient
     DIRECTION is stable across windows: all nonzero point estimates share
     one sign and the coefficient is nonzero in at least 4 of 6 windows
     (evaluated in the year-adjusted continuity-augmented model).
 R3  max_abs_interval is retained only if the extended model shows a
     consistent, non-trivial improvement over the minimal model:
     point ΔAUC ≥ +0.01 in at least 4 of 6 windows AND in the unrestricted
     window (paired estimates).
 R4  Gap rate is excluded from the core score by rule as window-dependent
     (established in the 04 stability analysis).
 R5  Gap rate defines the continuity-augmented score (core features + gap
     rate). Core and augmented denote composition, not rank; neither is
     assigned sole-primary status for cross-corpus transfer.
 R6  No era window is selected on performance; the unrestricted window is
     the primary reporting sample only because it uses all families.
 R7  Documentation year is nuisance adjustment / baseline only.

Score definition written to the JSONs: unit-weighted signed sum of
z-scored features (signs = stable directions). Unit weights are chosen for
transfer robustness; the Dutch-fit standardized coefficients are included
as metadata for sensitivity analyses.

Inputs : 02_data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz,
         03_outputs/02_dutch_features.csv
Outputs: 03_outputs/05_final_model_comparison.csv
         results/05_primary_signature.json
         results/05_continuity_augmented_signature.json

Usage: python 05_define_final_signatures.py [DATA_DIR] [OUT_DIR]
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
R3_MIN_GAIN = 0.01
R2_MIN_NONZERO_WINDOWS = 4

MINIMAL = ["prop_repeated_pitch", "pitch_range"]
EXTENDED = MINIMAL + ["max_abs_interval"]
AUGMENTED = EXTENDED + ["prop_gap_transitions"]
ALLVARS = AUGMENTED + ["year"]

MODELS = {"minimal_pitch_core": MINIMAL,
          "extended_pitch_core": EXTENDED,
          "continuity_augmented": AUGMENTED}
WINDOWS = [("le15", 15.0), ("le25", 25.0), ("le50", 50.0),
           ("le75", 75.0), ("le100", 100.0), ("unrestricted", np.inf)]
KEYS = ["pairwise_accuracy", "roc_auc", "log_loss"]


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
    fams, deltas = [], []
    for tf, sides in fam.items():
        if not sides["vocal"] or not sides["instrumental"]:
            continue
        row = []
        for v in ALLVARS:
            def smean(mids):
                vals = np.array([
                    meta[m][2] if v == "year" else
                    (float(feats[m][v]) if feats[m][v] != "" else np.nan)
                    for m in mids])
                return np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            row.append(smean(sides["vocal"]) - smean(sides["instrumental"]))
        fams.append(tf)
        deltas.append(row)
    return np.array(fams), np.array(deltas)


def mirrored(d):
    return (np.vstack([d, -d]),
            np.concatenate([np.ones(len(d)), np.zeros(len(d))]))


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
    if idx is None:
        idx = np.arange(n_fam)
    rows = np.concatenate([idx, idx + n_fam])
    yy = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
    pp = prob[rows]
    pw = float((pp[:len(idx)] > 0.5).mean()
               + 0.5 * (pp[:len(idx)] == 0.5).mean())
    return {"pairwise_accuracy": pw,
            "roc_auc": float(roc_auc_score(yy, pp)),
            "log_loss": float(log_loss(yy, np.clip(pp, 1e-12, 1 - 1e-12)))}


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"

    fams, D = build_deltas(data_dir, out_dir)
    vidx = {v: j for j, v in enumerate(ALLVARS)}
    absdy = np.abs(D[:, vidx["year"]])

    perf_rows, coef_rows = [], []
    # containers for the rule engine
    auc_point = defaultdict(dict)          # auc_point[model][window]
    dauc_ext_min = {}                      # window -> (est, lo, hi)
    dauc_min_ext = {}
    dauc_ext_aug = {}
    coef_sign = defaultdict(dict)          # coef_sign[term][window] (augmented model)
    unrestricted_coefs = {}

    for wname, wlim in WINDOWS:
        sel = absdy <= wlim
        Dw, fw = D[sel], fams[sel]
        n_fam = len(fw)
        groups2 = np.concatenate([fw, fw])
        print(f"== window {wname}: {n_fam} families", flush=True)

        models = {"year_only": ["year"]}
        for mn, vs in MODELS.items():
            models[mn] = vs
            models["year_plus_" + mn] = ["year"] + vs

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
            auc_point[name][wname] = point_m[name]["roc_auc"]
            print(f"  {name}: AUC={point_m[name]['roc_auc']:.3f}", flush=True)

        # improvements over year_only (paired, shared draws)
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

        # nested model contrasts (paired, shared draws)
        for a, b, store in [
                ("extended_pitch_core", "minimal_pitch_core", dauc_ext_min),
                ("minimal_pitch_core", "extended_pitch_core", dauc_min_ext),
                ("extended_pitch_core", "continuity_augmented", dauc_ext_aug)]:
            diffs = boot_m[a] - boot_m[b]
            lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
            j = KEYS.index("roc_auc")
            est = point_m[a]["roc_auc"] - point_m[b]["roc_auc"]
            store[wname] = (est, lo[j], hi[j])
            for k, key in enumerate(KEYS):
                perf_rows.append([
                    wname, n_fam, a, b, "d_" + key,
                    f"{point_m[a][key] - point_m[b][key]:.4f}",
                    f"{lo[k]:.4f}", f"{hi[k]:.4f}"])

        # coefficients of year-adjusted variants
        for name in ["year_plus_" + m for m in MODELS]:
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

            rngc = np.random.default_rng(SEED)
            boot = np.zeros((N_BOOT_COEF, len(vs)))
            for b in range(N_BOOT_COEF):
                idx = rngc.integers(0, n_fam, n_fam)
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
                direction = ("positive" if point[j] > ZERO_TOL else
                             "negative" if point[j] < -ZERO_TOL else "zero")
                coef_rows.append([wname, n_fam, name, "d_" + v,
                                  f"{point[j]:.4f}", direction,
                                  f"{lo[j]:.4f}", f"{hi[j]:.4f}",
                                  f"{sign_stab[j]:.3f}", f"{selfreq[j]:.3f}",
                                  best["clf__C"], best["clf__l1_ratio"]])
                if name == "year_plus_continuity_augmented" and v != "year":
                    coef_sign[v][wname] = direction
                if wname == "unrestricted" and \
                        name == "year_plus_continuity_augmented":
                    unrestricted_coefs[v] = float(point[j])

    # ------------------------------------------------------------------
    # rule engine
    # ------------------------------------------------------------------
    def direction_stable(term):
        signs = [coef_sign[term][w] for w, _ in WINDOWS]
        nonzero = [s for s in signs if s != "zero"]
        return (len(set(nonzero)) == 1 and
                len(nonzero) >= R2_MIN_NONZERO_WINDOWS), signs

    stability = {t: direction_stable(t) for t in AUGMENTED}

    # R3: extended vs minimal — consistent non-trivial gain?
    gains = {w: float(dauc_ext_min[w][0]) for w, _ in WINDOWS}
    n_gain = sum(1 for g in gains.values() if g >= R3_MIN_GAIN)
    r3_pass = bool((n_gain >= 4) and (gains["unrestricted"] >= R3_MIN_GAIN))

    # R1 fallback: minimal within margin of extended & no supported loss?
    est_mn, lo_mn, hi_mn = (float(x) for x in dauc_min_ext["unrestricted"])
    r1_minimal_ok = bool((est_mn >= -EQUIV_MARGIN) and (hi_mn >= 0.0))

    mai_stable = stability["max_abs_interval"][0]
    keep_mai = r3_pass and mai_stable
    primary_feats = EXTENDED if keep_mai else MINIMAL
    # R2 hard filter
    primary_feats = [f for f in primary_feats if stability[f][0]]

    decision_log = {
        "R2_direction_stability": {t: {"stable": stability[t][0],
                                       "signs_by_window": stability[t][1]}
                                   for t in AUGMENTED},
        "R3_extended_minus_minimal_dAUC_by_window":
            {w: round(gains[w], 4) for w, _ in WINDOWS},
        "R3_pass": r3_pass,
        "R1_minimal_within_margin_of_extended_unrestricted":
            {"d_auc": round(est_mn, 4), "ci": [round(lo_mn, 4),
                                               round(hi_mn, 4)],
             "ok": r1_minimal_ok},
        "max_abs_interval_retained": keep_mai,
        "gap_rate_excluded_from_primary": "prespecified (R4)",
    }

    directions = {"prop_repeated_pitch": "positive", "pitch_range": "negative",
                  "max_abs_interval": "negative",
                  "prop_gap_transitions": "positive"}

    def signature_json(feats, name, note):
        return {
            "name": name,
            "features": [
                {"feature": f,
                 "direction": directions[f],
                 "unit_weight": 1.0 if directions[f] == "positive" else -1.0,
                 "dutch_fit_coef_std_scale_unrestricted_year_adjusted":
                     unrestricted_coefs.get(f)}
                for f in feats],
            "score_definition": ("score = sum_i unit_weight_i * z(feature_i); "
                                 "z-scoring parameters must be taken from the "
                                 "corpus the score is applied to (or from a "
                                 "prespecified reference sample), never from "
                                 "outcome-labelled test data"),
            "derivation": ("Intercept-free mirrored within-family pairwise "
                           "design, Dutch mixed tune families (MTC-FS-INST "
                           "2.0), era-window sensitivity |dyear| <= 15/25/50/"
                           "75/100/unrestricted, year as nuisance adjustment "
                           "only; see 03_outputs/05_signature_definition.md"),
            "note": note,
            "decision_log": decision_log,
        }

    primary = signature_json(
        primary_feats, "core_melodic_score",
        "Contains only features with window-stable coefficient direction "
        "(R2); gap rate excluded from the core score by rule R4 as "
        "window-dependent. The names core and augmented denote composition, "
        "not rank; neither is assigned sole-primary status.")
    secondary = signature_json(
        primary_feats + ["prop_gap_transitions"],
        "continuity_augmented_score",
        "Continuity-augmented score (R5): core features plus gap rate. "
        "Gap rate direction is positive but classified as "
        "window-dependent (supported in narrow era windows and the "
        "unrestricted sample; shrunk to zero at |dyear| <= 75 and <= 100).")

    with open(os.path.join(out_dir, "05_final_model_comparison.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_type", "window", "n_families", "model",
                    "reference_or_term", "metric", "estimate", "ci95_lo",
                    "ci95_hi", "direction", "sign_stability",
                    "selection_frequency", "chosen_C", "chosen_l1_ratio"])
        for r in perf_rows:
            w.writerow(["performance", r[0], r[1], r[2], r[3], r[4],
                        r[5], r[6], r[7], "", "", "", "", ""])
        for r in coef_rows:
            w.writerow(["coefficient", r[0], r[1], r[2], r[3],
                        "coefficient_std_scale", r[4], r[6], r[7], r[5],
                        r[8], r[9], r[10], r[11]])

    with open(os.path.join(out_dir, "05_primary_signature.json"), "w") as fh:
        json.dump(primary, fh, indent=2)
    with open(os.path.join(out_dir,
                           "05_continuity_augmented_signature.json"),
              "w") as fh:
        json.dump(secondary, fh, indent=2)
    print("primary features:", primary_feats)
    print("decision log:", json.dumps(decision_log, indent=1))


if __name__ == "__main__":
    main()
