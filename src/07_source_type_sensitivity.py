#!/usr/bin/env python3
"""
07_source_type_sensitivity.py — source-type-matched re-run of the mirrored
within-family pairwise analysis.

Motivation
----------
Medium and source type are confounded in MTC-FS-INST 2.0. Almost every
instrumental melody derives from a notated source (2,364 print and 6,549
manuscript against one audio record), whereas 4,029 of 9,195 vocal melodies
(43.8%) are transcriptions of field recordings. A vocal/instrumental contrast
in the full corpus therefore also contrasts kinds of document. This script
asks whether the within-family melodic signal survives when both sides of a
family are restricted to notated sources.

Design
------
Identical to pairwise_era_adjusted_model.py — same features, same mirrored
antisymmetric construction, same intercept-free elastic net, same nested
grouped cross-validation, same seed. The only thing that changes is which
melodies contribute to the family means:

  full     all melodies (reproduction check: must match 04_pairwise_results)
  notated  audio-sourced melodies dropped from the family means; a family is
           kept if at least one vocal and one instrumental melody remain
  strict   any family containing an audio-sourced melody is dropped entirely

Source type comes from the official MTC-FS-INST-2.0 metadata tables, where
each source carries a type field with values {manuscript, print, audio}.

Inputs : data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz
         data/mtc_metadata/MTC-FS-INST-2.0.csv
         data/mtc_metadata/MTC-FS-INST-2.0-fieldnames.csv
         data/mtc_metadata/MTC-FS-INST-2.0-sources.csv
         data/mtc_metadata/MTC-FS-INST-2.0-sources-fieldnames.csv
         results/02_dutch_features.csv
Outputs: results/07_source_type_results.csv        (AUC + bootstrap CI per sample)
         results/07_source_type_coefficients.csv   (standardized coefficients)

Usage: python 07_source_type_sensitivity.py [DATA_DIR] [RESULTS_DIR]
"""

import csv
import gzip
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pairwise_module():
    """Import pairwise_era_adjusted_model.py so the pipeline is shared, not copied."""
    path = os.path.join(HERE, "pairwise_era_adjusted_model.py")
    spec = importlib.util.spec_from_file_location("pairwise_era_adjusted_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load_pairwise_module()

SAMPLES = ["full", "notated", "strict"]
REPORT_MODELS = ["year_only", "continuity", "pitch_movement",
                 "combined", "year_plus_combined"]


def source_type_map(data_dir):
    """melody filename -> {'print', 'manuscript', 'audio'}"""
    meta_dir = os.path.join(data_dir, "mtc_metadata")
    sf = next(csv.reader(open(os.path.join(meta_dir, "MTC-FS-INST-2.0-sources-fieldnames.csv"))))
    sources = {r[0]: dict(zip(sf, r))
               for r in csv.reader(open(os.path.join(meta_dir, "MTC-FS-INST-2.0-sources.csv")))}
    mf = next(csv.reader(open(os.path.join(meta_dir, "MTC-FS-INST-2.0-fieldnames.csv"))))
    out = {}
    for row in csv.reader(open(os.path.join(meta_dir, "MTC-FS-INST-2.0.csv"))):
        rec = dict(zip(mf, row))
        out[rec["filename"]] = sources[rec["source_id"]]["type"]
    return out


def build_deltas(sample, stype, data_dir, results_dir):
    """As B.build_deltas, but with the source-type filter applied."""
    meta = {}
    families = defaultdict(lambda: defaultdict(list))
    seq = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    with gzip.open(seq, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["id"]] = (r["tunefamily"], r["type"], float(r["year"]))

    feats = {row["melody_id"]: row for row in
             csv.DictReader(open(os.path.join(results_dir, "02_dutch_features.csv")))}

    for mid, (tf, typ, _) in meta.items():
        if tf and mid in feats:
            families[tf][typ].append(mid)

    variables = B.COMBINED + ["year"]
    fams, deltas, n_mel, n_dropped = [], [], 0, 0
    for tf, sides in families.items():
        vocal, instrumental = list(sides["vocal"]), list(sides["instrumental"])
        if not vocal or not instrumental:
            continue
        if sample == "strict":
            if any(stype.get(m) == "audio" for m in vocal + instrumental):
                continue
        elif sample == "notated":
            before = len(vocal) + len(instrumental)
            vocal = [m for m in vocal if stype.get(m) != "audio"]
            instrumental = [m for m in instrumental if stype.get(m) != "audio"]
            n_dropped += before - len(vocal) - len(instrumental)
            if not vocal or not instrumental:
                continue
        n_mel += len(vocal) + len(instrumental)
        row = []
        for v in variables:
            def side_mean(mids):
                vals = []
                for m in mids:
                    x = (meta[m][2] if v == "year"
                         else (float(feats[m][v]) if feats[m][v] != "" else np.nan))
                    vals.append(x)
                arr = np.array(vals, dtype=float)
                return np.nanmean(arr) if np.isfinite(arr).any() else np.nan
            row.append(side_mean(vocal) - side_mean(instrumental))
        fams.append(tf)
        deltas.append(row)
    info = dict(n_families=len(fams), n_melodies=n_mel, audio_dropped=n_dropped)
    return np.array(fams), np.array(deltas), variables, info


def evaluate(D, variables, fams, models):
    rng = np.random.default_rng(B.SEED)
    out = {}
    for name in models:
        cols = [variables.index(v) for v in B.MODELS[name]]
        sub = D[:, cols]
        ok = np.isfinite(sub).all(axis=1)
        sub, groups_1 = sub[ok], fams[ok]
        X, y = B.mirrored(sub)
        groups = np.concatenate([groups_1, groups_1])
        prob = B.oof_predictions(X, y, groups)
        n = len(sub)
        auc = float(B.roc_auc_score(y, prob))
        acc = float((prob[:n] > 0.5).mean() + 0.5 * (prob[:n] == 0.5).mean())
        boots = []
        for _ in range(B.N_BOOT):
            idx = rng.integers(0, n, n)
            yy = np.concatenate([np.ones(n), np.zeros(n)])
            pp = np.concatenate([prob[:n][idx], prob[n:][idx]])
            boots.append(B.roc_auc_score(yy, pp))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = dict(auc=auc, pairwise_accuracy=acc, lo=float(lo), hi=float(hi))
    return out


def coefficients(D, variables, fams):
    """Standardized coefficients of the year-adjusted combined model."""
    cols = [variables.index(v) for v in B.COMBINED + ["year"]]
    sub = D[:, cols]
    ok = np.isfinite(sub).all(axis=1)
    sub, groups_1 = sub[ok], fams[ok]
    X, y = B.mirrored(sub)
    groups = np.concatenate([groups_1, groups_1])
    inner = StratifiedGroupKFold(n_splits=B.INNER_FOLDS, shuffle=True, random_state=B.SEED)
    gs = GridSearchCV(B.make_pipe(), B.GRID, scoring="roc_auc", cv=inner, n_jobs=-1)
    gs.fit(X, y, groups=groups)
    coefs = gs.best_estimator_.named_steps["clf"].coef_[0]
    return dict(zip(B.COMBINED + ["year"], (float(c) for c in coefs)))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    stype = source_type_map(data_dir)

    perf_rows, coef_rows = [], []
    for sample in SAMPLES:
        fams, D, variables, info = build_deltas(sample, stype, data_dir, results_dir)
        print(f"{sample:<8} {info['n_families']} families, {info['n_melodies']} melodies, "
              f"{info['audio_dropped']} audio melodies dropped")
        res = evaluate(D, variables, fams, REPORT_MODELS)
        for model, v in res.items():
            perf_rows.append([sample, info["n_families"], info["n_melodies"], model,
                              "roc_auc", f"{v['auc']:.4f}", f"{v['lo']:.4f}", f"{v['hi']:.4f}"])
            perf_rows.append([sample, info["n_families"], info["n_melodies"], model,
                              "pairwise_accuracy", f"{v['pairwise_accuracy']:.4f}", "", ""])
            print(f"    {model:<22} AUC={v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        for term, c in coefficients(D, variables, fams).items():
            coef_rows.append([sample, info["n_families"], "year_plus_combined",
                              "d_" + term, f"{c:.4f}"])

    with open(os.path.join(results_dir, "07_source_type_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "n_families", "n_melodies", "model", "metric",
                    "estimate", "ci95_lo_boot_families", "ci95_hi_boot_families"])
        w.writerows(perf_rows)
    with open(os.path.join(results_dir, "07_source_type_coefficients.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "n_families", "model", "term", "coefficient_std_scale"])
        w.writerows(coef_rows)
    print("\nwrote 07_source_type_results.csv, 07_source_type_coefficients.csv")


if __name__ == "__main__":
    main()
