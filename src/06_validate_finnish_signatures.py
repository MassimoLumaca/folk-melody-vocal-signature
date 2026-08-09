#!/usr/bin/env python3
"""
06_validate_finnish_signatures.py — Cross-corpus transfer of the unit-weighted
Dutch melodic scores to the Finnish archive (finfolktunes).

Score definitions (results/05_primary_signature.json,
results/05_continuity_augmented_signature.json) — evaluated EXACTLY as
unit-weighted; no coefficient refitting, no feature selection, no threshold
optimization, no modification after observing Finnish results:

    core_score       = z(prop_repeated_pitch) − z(pitch_range)
    continuity_score = z(prop_gap_transitions)
    augmented_score  = core_score + continuity_score

Excluded by prespecification: documentation year, max absolute interval,
phrase annotations, melody length, any Finnish-fitted coefficient.

Standardization
    PRIMARY  "finnish_blind": z-scores use mean/SD of the complete Finnish
             corpus (all 8,613 melodies), computed without medium labels —
             label-blind domain standardization, not model fitting.
    SENSITIVITY "dutch_ref": z-scores use fixed means/SDs of the full
             Dutch reference corpus (MTC-FS-INST 2.0, all 18,109 melodies
             in 02_dutch_features.csv). Prespecified; no performance-based
             choice between schemes.

Primary validation unit: the 10 Finnish source collections (medium is
assigned at collection level): vocal = LS1–LS4, RS1, RS2, HS1 (7);
instrumental = KT1, Kantelesävelmiä, Jouhikkosävelmiä (3).

Primary hypothesis (direction fixed in advance): vocal collections have
higher collection-mean core scores. Exact one-sided Mann–Whitney /
label-permutation test over all C(10,3) = 120 label assignments; also
two-sided exact p, collection-level AUC (U / 21), rank-biserial (2·AUC−1),
and all ten collection values.

Secondary (prespecified): continuity score higher in vocal; augmented
separates better than core (exact permutation of AUC difference);
repeated-pitch higher / pitch range lower / gap rate higher in vocal, with
Holm correction across these three feature tests.

Robustness: melody-level AUC (descriptive only); both standardization
schemes; leave-one-collection-out; perfect-separation check.

Inputs : 03_outputs/02_finnish_features.csv, 03_outputs/02_dutch_features.csv,
         results/05_primary_signature.json (read for the score spec)
Outputs: 03_outputs/06_finnish_melody_scores.csv
         03_outputs/06_finnish_collection_scores.csv
         results/06_primary_core_validation.csv
         03_outputs/06_continuity_validation.csv
         03_outputs/06_augmented_validation.csv

Usage: python 06_validate_finnish_signatures.py [OUT_DIR]
(all inputs/outputs under OUT_DIR = 03_outputs by default)
"""

import csv
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

FEATS = ["prop_repeated_pitch", "pitch_range", "prop_gap_transitions"]
SIGN = {"prop_repeated_pitch": +1.0, "pitch_range": -1.0,
        "prop_gap_transitions": +1.0}

VOCAL = {"LS1", "LS2", "LS3", "LS4", "RS1", "RS2", "HS1"}
INSTR = {"KT1", "Kantelesävelmiä", "Jouhikkosävelmiä"}

# filename prefix -> collection (metadata join not needed: prefixes encode
# the collection; kjs_0001–0221 = Kantelesävelmiä, kjs_0222+ = Jouhikkosävelmiä
# is NOT assumed — instead we re-derive collection from the metadata file if
# available; fall back to the features CSV's voc_inst plus prefix)


def load_features(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def collection_of(finnish_meta_path):
    """filename -> collection from the Finnish metadata TSV."""
    m = {}
    with open(finnish_meta_path, encoding="latin-1") as fh:
        next(fh)
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            m[f[2]] = f[1]
    return m


def auc_from_groups(voc_vals, ins_vals):
    """P(vocal > instrumental) + 0.5 P(=) over all pairs."""
    v = np.asarray(voc_vals)[:, None]
    i = np.asarray(ins_vals)[None, :]
    return float(((v > i).sum() + 0.5 * (v == i).sum()) / (v.size * i.size))


def exact_tests(values, is_vocal, direction):
    """Exact permutation over all C(10,3) assignments of 3 'instrumental'
    labels. Statistic: collection-level AUC (equivalent to Mann–Whitney U).
    direction: +1 -> H1 vocal higher; -1 -> H1 vocal lower."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    ins_idx = np.flatnonzero(~np.asarray(is_vocal))
    obs_auc = auc_from_groups(values[np.asarray(is_vocal)],
                              values[ins_idx])
    obs_stat = direction * (obs_auc - 0.5)
    null_aucs = []
    for combo in itertools.combinations(range(n), len(ins_idx)):
        mask = np.ones(n, bool)
        mask[list(combo)] = False
        null_aucs.append(auc_from_groups(values[mask], values[~mask]))
    null_aucs = np.array(null_aucs)
    null_stat = direction * (null_aucs - 0.5)
    p_one = float((null_stat >= obs_stat - 1e-12).mean())
    p_two = float((np.abs(null_aucs - 0.5) >= abs(obs_auc - 0.5) - 1e-12).mean())
    return {"auc": obs_auc, "rank_biserial": 2 * obs_auc - 1,
            "p_one_sided_exact": p_one, "p_two_sided_exact": p_two,
            "n_permutations": len(null_aucs)}


def exact_auc_difference(vals_a, vals_b, is_vocal):
    """Exact permutation comparison of collection-level AUCs of two scores
    computed on the same collections. One-sided: H1 AUC_a > AUC_b."""
    vals_a, vals_b = np.asarray(vals_a), np.asarray(vals_b)
    n = len(vals_a)
    k = int((~np.asarray(is_vocal)).sum())
    voc = np.asarray(is_vocal)
    obs = (auc_from_groups(vals_a[voc], vals_a[~voc])
           - auc_from_groups(vals_b[voc], vals_b[~voc]))
    null = []
    for combo in itertools.combinations(range(n), k):
        mask = np.ones(n, bool)
        mask[list(combo)] = False
        null.append(auc_from_groups(vals_a[mask], vals_a[~mask])
                    - auc_from_groups(vals_b[mask], vals_b[~mask]))
    null = np.array(null)
    return {"d_auc": float(obs),
            "p_one_sided_exact": float((null >= obs - 1e-12).mean()),
            "n_permutations": len(null)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (preserving order)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (m - rank) * pvals[ix])
        adj[ix] = min(1.0, running)
    return adj.tolist()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "03_outputs"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "02_data"

    fin = load_features(os.path.join(out_dir, "02_finnish_features.csv"))
    dut = load_features(os.path.join(out_dir, "02_dutch_features.csv"))
    coll_map = collection_of(os.path.join(
        data_dir, "finfolktunes_data_corrected.txt"))

    X = np.array([[float(r[f]) for f in FEATS] for r in fin])
    colls = np.array([coll_map[r["melody_id"]] for r in fin])
    labels = np.array([r["voc_inst"] for r in fin])

    # standardization parameters
    mu_fin, sd_fin = X.mean(axis=0), X.std(axis=0, ddof=1)
    Xd = np.array([[float(r[f]) for f in FEATS] for r in dut])
    mu_dut, sd_dut = Xd.mean(axis=0), Xd.std(axis=0, ddof=1)

    schemes = {"finnish_blind": (mu_fin, sd_fin),
               "dutch_ref": (mu_dut, sd_dut)}
    w = np.array([SIGN[f] for f in FEATS])

    scores = {}
    for sc, (mu, sd) in schemes.items():
        Z = (X - mu) / sd
        core = w[0] * Z[:, 0] + w[1] * Z[:, 1]
        cont = w[2] * Z[:, 2]
        scores[sc] = {"core": core, "continuity": cont,
                      "augmented": core + cont}

    # ---- melody-level CSV ----------------------------------------------
    with open(os.path.join(out_dir, "06_finnish_melody_scores.csv"), "w",
              newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["melody_id", "collection", "voc_inst"] + FEATS +
                      [f"{s}_{sc}" for sc in schemes
                       for s in ["core", "continuity", "augmented"]])
        for i, r in enumerate(fin):
            wcsv.writerow([r["melody_id"], colls[i], labels[i]] +
                          [f"{X[i, j]:.6g}" for j in range(3)] +
                          [f"{scores[sc][s][i]:.6f}" for sc in schemes
                           for s in ["core", "continuity", "augmented"]])

    # ---- collection-level table ----------------------------------------
    coll_names = sorted(set(colls), key=lambda c: (c not in VOCAL, c))
    coll_rows = []
    coll_stats = {sc: {} for sc in schemes}
    for sc in schemes:
        for c in coll_names:
            m = colls == c
            entry = {"n": int(m.sum()),
                     "medium": "vocal" if c in VOCAL else "instrumental"}
            for s in ["core", "continuity", "augmented"]:
                entry[f"mean_{s}"] = float(scores[sc][s][m].mean())
                entry[f"median_{s}"] = float(np.median(scores[sc][s][m]))
            for j, f in enumerate(FEATS):
                entry[f"mean_{f}"] = float(X[m, j].mean())
            coll_stats[sc][c] = entry
        # ranks (1 = highest mean score)
        for s in ["core", "continuity", "augmented"]:
            vals = {c: coll_stats[sc][c][f"mean_{s}"] for c in coll_names}
            for rank, c in enumerate(sorted(vals, key=vals.get,
                                            reverse=True), 1):
                coll_stats[sc][c][f"rank_{s}"] = rank

    with open(os.path.join(out_dir, "06_finnish_collection_scores.csv"),
              "w", newline="") as fh:
        wcsv = csv.writer(fh)
        hdr = ["scheme", "collection", "medium", "n_melodies"]
        for s in ["core", "continuity", "augmented"]:
            hdr += [f"mean_{s}", f"median_{s}", f"rank_{s}"]
        hdr += [f"mean_{f}" for f in FEATS]
        wcsv.writerow(hdr)
        for sc in schemes:
            for c in coll_names:
                e = coll_stats[sc][c]
                row = [sc, c, e["medium"], e["n"]]
                for s in ["core", "continuity", "augmented"]:
                    row += [f"{e['mean_' + s]:.4f}",
                            f"{e['median_' + s]:.4f}", e[f"rank_{s}"]]
                row += [f"{e['mean_' + f]:.4f}" for f in FEATS]
                wcsv.writerow(row)

    is_vocal = np.array([c in VOCAL for c in coll_names])

    def coll_means(sc, s):
        return [coll_stats[sc][c][f"mean_{s}"] for c in coll_names]

    def coll_feat_means(sc, f):
        return [coll_stats[sc][c][f"mean_{f}"] for c in coll_names]

    def melody_auc(sc, s):
        v = scores[sc][s][labels == "vocal"]
        i = scores[sc][s][labels == "instrumental"]
        return auc_from_groups(v, i)

    def loco(sc, s):
        """leave-one-collection-out AUC of collection means."""
        out = {}
        vals = np.array(coll_means(sc, s))
        for j, c in enumerate(coll_names):
            mask = np.ones(len(coll_names), bool)
            mask[j] = False
            vv = vals[mask][is_vocal[mask]]
            ii = vals[mask][~is_vocal[mask]]
            out[c] = auc_from_groups(vv, ii) if len(vv) and len(ii) else np.nan
        return out

    # ---- primary: core --------------------------------------------------
    prim_rows = []
    for sc in schemes:
        t = exact_tests(coll_means(sc, "core"), is_vocal, +1)
        vals = coll_means(sc, "core")
        sep = min(np.array(vals)[is_vocal]) > max(np.array(vals)[~is_vocal])
        l = loco(sc, "core")
        prim_rows.append([sc, "core", "vocal_higher",
                          f"{t['auc']:.4f}", f"{t['rank_biserial']:.4f}",
                          f"{t['p_one_sided_exact']:.5f}",
                          f"{t['p_two_sided_exact']:.5f}",
                          t["n_permutations"],
                          f"{melody_auc(sc, 'core'):.4f}",
                          "yes" if sep else "no",
                          f"{min(l.values()):.4f}",
                          f"{max(l.values()):.4f}",
                          ";".join(f"{c}={v:.3f}" for c, v in l.items())])
    with open(os.path.join(out_dir, "06_primary_core_validation.csv"), "w",
              newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["scheme", "score", "prespecified_direction",
                       "collection_auc", "rank_biserial",
                       "p_one_sided_exact", "p_two_sided_exact",
                       "n_permutations", "melody_auc_descriptive",
                       "all_vocal_above_all_instrumental",
                       "loco_auc_min", "loco_auc_max", "loco_auc_by_left_out"])
        wcsv.writerows(prim_rows)

    # ---- secondary: continuity score + three features -------------------
    cont_rows = []
    for sc in schemes:
        t = exact_tests(coll_means(sc, "continuity"), is_vocal, +1)
        l = loco(sc, "continuity")
        cont_rows.append([sc, "continuity_score", "vocal_higher",
                          f"{t['auc']:.4f}", f"{t['rank_biserial']:.4f}",
                          f"{t['p_one_sided_exact']:.5f}",
                          f"{t['p_two_sided_exact']:.5f}", "",
                          f"{melody_auc(sc, 'continuity'):.4f}",
                          f"{min(l.values()):.4f}", f"{max(l.values()):.4f}"])
    # feature tests are standardization-free (raw feature means)
    ft_specs = [("prop_repeated_pitch", +1, "vocal_higher"),
                ("pitch_range", -1, "vocal_lower"),
                ("prop_gap_transitions", +1, "vocal_higher")]
    ft_p = []
    ft_tmp = []
    for f, direction, dname in ft_specs:
        t = exact_tests(coll_feat_means("finnish_blind", f), is_vocal,
                        direction)
        ft_tmp.append((f, dname, t))
        ft_p.append(t["p_one_sided_exact"])
    ft_adj = holm(ft_p)
    for (f, dname, t), padj in zip(ft_tmp, ft_adj):
        cont_rows.append(["raw_features", f, dname,
                          f"{t['auc']:.4f}", f"{t['rank_biserial']:.4f}",
                          f"{t['p_one_sided_exact']:.5f}",
                          f"{t['p_two_sided_exact']:.5f}",
                          f"{padj:.5f}", "", "", ""])
    with open(os.path.join(out_dir, "06_continuity_validation.csv"), "w",
              newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["scheme_or_block", "score_or_feature",
                       "prespecified_direction", "collection_auc",
                       "rank_biserial", "p_one_sided_exact",
                       "p_two_sided_exact", "p_holm_adjusted_3_features",
                       "melody_auc_descriptive", "loco_auc_min",
                       "loco_auc_max"])
        wcsv.writerows(cont_rows)

    # ---- augmented: tests + comparison with core ------------------------
    aug_rows = []
    for sc in schemes:
        t = exact_tests(coll_means(sc, "augmented"), is_vocal, +1)
        l = loco(sc, "augmented")
        vals = coll_means(sc, "augmented")
        sep = min(np.array(vals)[is_vocal]) > max(np.array(vals)[~is_vocal])
        cmp_ = exact_auc_difference(coll_means(sc, "augmented"),
                                    coll_means(sc, "core"), is_vocal)
        aug_rows.append([sc, "augmented", "vocal_higher",
                         f"{t['auc']:.4f}", f"{t['rank_biserial']:.4f}",
                         f"{t['p_one_sided_exact']:.5f}",
                         f"{t['p_two_sided_exact']:.5f}",
                         f"{melody_auc(sc, 'augmented'):.4f}",
                         "yes" if sep else "no",
                         f"{min(l.values()):.4f}", f"{max(l.values()):.4f}",
                         f"{cmp_['d_auc']:.4f}",
                         f"{2 * cmp_['d_auc']:.4f}",
                         f"{cmp_['p_one_sided_exact']:.5f}"])
    with open(os.path.join(out_dir, "06_augmented_validation.csv"), "w",
              newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["scheme", "score", "prespecified_direction",
                       "collection_auc", "rank_biserial",
                       "p_one_sided_exact", "p_two_sided_exact",
                       "melody_auc_descriptive",
                       "all_vocal_above_all_instrumental",
                       "loco_auc_min", "loco_auc_max",
                       "d_auc_augmented_minus_core",
                       "d_rank_biserial_augmented_minus_core",
                       "p_one_sided_exact_auc_comparison"])
        wcsv.writerows(aug_rows)

    # console summary
    for sc in schemes:
        print(f"[{sc}] core coll-AUC={exact_tests(coll_means(sc,'core'), is_vocal, +1)['auc']:.3f}")
    print("done")


if __name__ == "__main__":
    main()
