#!/usr/bin/env python3
"""
08_gap_threshold_sensitivity.py — sensitivity of every gap-rate-dependent
result to the 0.5-beat gap threshold, repeated at 0.25 and 1.0 beats.

Motivation
----------
A gap is a silence of at least 0.5 beats between consecutive notes. That
threshold sets what counts as an interruption of melodic flow, and the
continuity component of the transferred score depends on it. This script brackets
the choice by recomputing all gap-dependent results at 0.25 and 1.0 beats.
The 0.5-beat values are recomputed as an internal consistency check and must
reproduce the published features.

What is recomputed at each threshold t in {0.25, 0.5, 1.0}
---------------------------------------------------------
Dutch (mirrored within-family design, unrestricted sample, same machinery as
pairwise_era_adjusted_model.py, seed 42):
  - pooled out-of-fold AUC of the combined seven-feature model with gap
    rate(t) substituted;
  - standardized gap-rate coefficient in the year-adjusted combined model
    (full-data grouped grid search).

Finnish (unit-weighted score transfer, label-blind standardization):
  - exact 10-collection test for the continuity component z(gap(t));
  - exact 10-collection test for the augmented score
    z(rep) - z(range) + z(gap(t)), with z(rep) and z(range) unchanged;
  - complete-separation flag, boundary margin, one-sided exact P (floor 0.0083).

Gap definition at threshold t: transition i is a gap iff
onset(i+1) - [onset(i) + duration(i)] >= t beats. Feature construction is
otherwise identical to extract_features.py.

Inputs : data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz
         data/finfolktunes.mat            (Finnish MIDI-Toolbox note matrices)
         results/02_dutch_features.csv     (regenerate with extract_features.py)
         results/06_finnish_melody_scores.csv
Outputs: results/gap_threshold_results.json

This script reports Table S14 and SI Materials and Methods 9.

Usage: python 08_gap_threshold_sensitivity.py [DATA_DIR] [RESULTS_DIR]
"""
import csv
import gzip
import importlib.util
import itertools
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = [0.25, 0.5, 1.0]
VOCAL_FIN = {"LS1", "LS2", "LS3", "LS4", "RS1", "RS2", "HS1"}
SEED = 42


def load_pairwise_module():
    """Import pairwise_era_adjusted_model.py so the pipeline is shared, not copied."""
    path = os.path.join(HERE, "pairwise_era_adjusted_model.py")
    spec = importlib.util.spec_from_file_location("pairwise_era_adjusted_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load_pairwise_module()


def frac(x):
    return float(Fraction(x)) if x is not None else 0.0


def gap_rate(onset, dur, t):
    if len(onset) < 2:
        return np.nan
    rest = onset[1:] - (onset[:-1] + dur[:-1])
    return float((rest >= t - 1e-9).sum() / (len(onset) - 1))


def auc(v, i):
    v = np.asarray(v)[:, None]
    i = np.asarray(i)[None, :]
    return float(((v > i).sum() + 0.5 * (v == i).sum()) / (v.size * i.size))


def exact_test(voc, ins):
    vals = np.array(list(voc) + list(ins))
    obs = auc(voc, ins)
    null = []
    for c in itertools.combinations(range(len(vals)), len(ins)):
        m = np.ones(len(vals), bool)
        m[list(c)] = False
        null.append(auc(vals[m], vals[~m]))
    null = np.array(null)
    return obs, float((null >= obs - 1e-12).mean())


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    seq_path = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    mat_path = os.path.join(data_dir, "finfolktunes.mat")
    dutch_feat_path = os.path.join(results_dir, "02_dutch_features.csv")
    finnish_score_path = os.path.join(results_dir, "06_finnish_melody_scores.csv")

    res = {"thresholds_beats": THRESHOLDS}

    # ---------------- Dutch: per-melody gap(t) --------------------------
    dutch_gap = {t: {} for t in THRESHOLDS}
    meta = {}
    with gzip.open(seq_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            f = r["features"]
            dur = np.array(f["duration"], dtype=float)
            rest = np.array([frac(x) for x in f["restduration_frac"]])
            n = len(dur)
            onset = np.zeros(n)
            if n > 1:
                onset[1:] = np.cumsum(dur[:-1] + rest[:-1])
            for t in THRESHOLDS:
                dutch_gap[t][r["id"]] = gap_rate(onset, dur, t)
            meta[r["id"]] = (r["tunefamily"], r["type"], float(r["year"]))

    feats = {r["melody_id"]: r for r in csv.DictReader(open(dutch_feat_path))}
    diffs = [abs(dutch_gap[0.5][m] - float(feats[m]["prop_gap_transitions"]))
             for m in feats]
    res["dutch_gap05_matches_published_max_abs_diff"] = float(max(diffs))

    fam = defaultdict(lambda: defaultdict(list))
    for mid, (tf, typ, yr) in meta.items():
        if tf and mid in feats:
            fam[tf][typ].append(mid)

    OTHER = ["mean_notes_between_gaps", "note_density",
             "prop_repeated_pitch", "mean_abs_interval_nonrep",
             "pitch_range", "max_abs_interval"]

    res["dutch"] = {}
    for t in THRESHOLDS:
        fams, D = [], []
        for tf, sides in fam.items():
            if not sides["vocal"] or not sides["instrumental"]:
                continue

            def smean(mids, getter):
                v = np.array([getter(m) for m in mids], dtype=float)
                return np.nanmean(v) if np.isfinite(v).any() else np.nan

            row = [smean(sides["vocal"], lambda m: dutch_gap[t][m])
                   - smean(sides["instrumental"], lambda m: dutch_gap[t][m])]
            for col in OTHER:
                g = lambda m, c=col: (float(feats[m][c])
                                      if feats[m][c] != "" else np.nan)
                row.append(smean(sides["vocal"], g)
                           - smean(sides["instrumental"], g))
            row.append(smean(sides["vocal"], lambda m: meta[m][2])
                       - smean(sides["instrumental"], lambda m: meta[m][2]))
            fams.append(tf)
            D.append(row)
        fams, D = np.array(fams), np.array(D)
        names = ["gap_t"] + OTHER + ["year"]

        # combined melodic model (7 features), out-of-fold AUC
        cols = [0] + [names.index(c) for c in OTHER]
        X, y = B.mirrored(D[:, cols])
        groups = np.concatenate([fams, fams])
        prob = B.oof_predictions(X, y, groups)
        auc_comb = float(B.roc_auc_score(y, prob))

        # year-adjusted combined coefficients (full-data grid search)
        Xy, yy = B.mirrored(D)
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        gs = GridSearchCV(B.make_pipe(), B.GRID, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(Xy, yy, groups=groups)
        coefs = dict(zip(names, gs.best_estimator_.named_steps["clf"].coef_[0]))
        res["dutch"][str(t)] = {
            "n_families": int(len(fams)),
            "combined_model_oof_auc": auc_comb,
            "year_adjusted_gap_coefficient": float(coefs["gap_t"]),
            "year_adjusted_year_coefficient": float(coefs["year"])}

    # ---------------- Finnish: per-melody gap(t) -------------------------
    mfile = loadmat(mat_path, squeeze_me=True)
    nm = mfile["nm"]
    scores = list(csv.DictReader(open(finnish_score_path)))
    assert len(scores) == nm.shape[0] == 8613
    fin_gap = {t: np.zeros(8613) for t in THRESHOLDS}
    for i, a in enumerate(nm):
        a = np.atleast_2d(np.asarray(a, dtype=float))
        onset, dur, pitch = a[:, 0], a[:, 1], a[:, 3]
        keep = pitch > 0
        onset, dur, pitch = onset[keep], dur[keep], pitch[keep]
        if len(onset) > 1 and (np.diff(onset) == 0).any():
            order = np.lexsort((pitch, onset))
            onset, dur, pitch = onset[order], dur[order], pitch[order]
            last = np.concatenate((np.diff(onset) != 0, [True]))
            onset, dur, pitch = onset[last], dur[last], pitch[last]
        for t in THRESHOLDS:
            fin_gap[t][i] = gap_rate(onset, dur, t)
    pub = np.array([float(r["prop_gap_transitions"]) for r in scores])
    res["finnish_gap05_matches_published_max_abs_diff"] = \
        float(np.max(np.abs(fin_gap[0.5] - pub)))

    colls = np.array([r["collection"] for r in scores])
    zrep_minus_zrange = np.array(
        [float(r["core_finnish_blind"]) for r in scores])

    res["finnish"] = {}
    for t in THRESHOLDS:
        g = fin_gap[t]
        zg = (g - g.mean()) / g.std(ddof=1)
        aug = zrep_minus_zrange + zg
        out = {}
        for label, vec in [("continuity_alone", zg), ("augmented", aug)]:
            means = {c: float(vec[colls == c].mean()) for c in np.unique(colls)}
            voc = [means[c] for c in means if c in VOCAL_FIN]
            ins = [means[c] for c in means if c not in VOCAL_FIN]
            a_, p = exact_test(voc, ins)
            out[label] = {"collection_auc": a_,
                          "complete_separation": bool(min(voc) > max(ins)),
                          "min_margin_sd": float(min(voc) - max(ins)),
                          "p_one_sided_exact": p}
        out["pct_melodies_with_zero_gaps"] = float((g == 0).mean())
        res["finnish"][str(t)] = out

    with open(os.path.join(results_dir, "gap_threshold_results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
