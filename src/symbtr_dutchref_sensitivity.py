#!/usr/bin/env python3
"""Turkish (SymbTr) Dutch-reference standardization sensitivity.

Repeats the SymbTr form-level transfer (09_symbtr_transfer.py) with z scores
computed from the fixed means/SDs of the full Dutch reference corpus
(18,109 melodies, ddof=1) instead of label-blind SymbTr parameters,
mirroring the Finnish dutch_ref scheme in 06_validate_finnish_signatures.py.
Reported in SI Materials and Methods 11. Sensitivity analysis: raw exact
P values plus Holm across the three scores within this scheme.

Reads : results/09_symbtr_piece_scores.csv  (raw rep / rng / gap per piece)
        02_dutch_features.csv via --dutch-features (optional; otherwise the
        embedded reference parameters below, recorded from that file, are used)
Writes: results/symbtr_dutchref_results.json

Validation gate: label-blind z columns of the piece CSV are reproduced from
the raw features (max abs err 0.0) before the Dutch reference is applied.
"""
import argparse, csv, itertools, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FEATS = ["prop_repeated_pitch", "pitch_range", "prop_gap_transitions"]
# Recorded from 02_dutch_features.csv (18,109 melodies, ddof=1):
DUTCH_MU = [0.19732191648251367, 14.327737588368175, 0.015804483714680666]
DUTCH_SD = [0.14248952372745445, 4.2489612542793405, 0.029866782576320666]


def auc(a, b):
    a = np.asarray(a)[:, None]; b = np.asarray(b)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def exact(vv, ii):
    vals = np.array(vv + ii); k = len(ii)
    obs = auc(vv, ii); null = []
    for c in itertools.combinations(range(len(vals)), k):
        m = np.ones(len(vals), bool); m[list(c)] = False
        null.append(auc(vals[m].tolist(), vals[~m].tolist()))
    return obs, float((np.array(null) >= obs - 1e-12).mean()), len(null)


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for rank, ix in enumerate(order):
        run = max(run, (m - rank) * ps[ix]); adj[ix] = min(1.0, run)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dutch-features", default=None,
                    help="path to 02_dutch_features.csv (recomputes reference parameters)")
    args = ap.parse_args()

    mu_d, sd_d = np.array(DUTCH_MU), np.array(DUTCH_SD)
    if args.dutch_features:
        dut = list(csv.DictReader(open(args.dutch_features)))
        Xd = np.array([[float(r[f]) for f in FEATS] for r in dut])
        mu_d, sd_d = Xd.mean(axis=0), Xd.std(axis=0, ddof=1)

    rows = list(csv.DictReader(open(os.path.join(RESULTS, "09_symbtr_piece_scores.csv"))))
    X = np.array([[float(r["rep"]), float(r["rng"]), float(r["gap"])] for r in rows])

    # validation gate: reproduce label-blind z columns
    mu_s, sd_s = X.mean(axis=0), X.std(axis=0, ddof=1)
    Zb = (X - mu_s) / sd_s
    err = max(abs(Zb[:, 0] - [float(r["zrep"]) for r in rows]).max(),
              abs(Zb[:, 1] - [float(r["zrng"]) for r in rows]).max(),
              abs(Zb[:, 2] - [float(r["zgap"]) for r in rows]).max())
    assert err < 1e-9, f"label-blind z reproduction failed: {err}"

    Zd = (X - mu_d) / sd_d
    for i, r in enumerate(rows):
        r["core_d"] = Zd[i, 0] - Zd[i, 1]
        r["zgap_d"] = Zd[i, 2]
        r["aug_d"] = r["core_d"] + r["zgap_d"]
        r["core_b"], r["zgap_b"], r["aug_b"] = (float(r["core"]), float(r["zgap"]), float(r["aug"]))

    forms = defaultdict(list)
    for r in rows:
        if r["group"] in ("vocal", "instrumental"):
            forms[(r["group"], r["form"])].append(r)
    fm = {k: v for k, v in forms.items() if len(v) >= 4}

    out = {"n_pieces": len(rows), "n_forms": len(fm),
           "reference_parameters": {"features": FEATS,
                                    "dutch_mu": mu_d.tolist(), "dutch_sd": sd_d.tolist(),
                                    "symbtr_mu": mu_s.tolist(), "symbtr_sd": sd_s.tolist()},
           "validation": {"blind_z_reproduction_max_abs_err": float(err)}}
    for scheme, keys in [("symbtr_blind", ("aug_b", "core_b", "zgap_b")),
                         ("dutch_ref", ("aug_d", "core_d", "zgap_d"))]:
        means = {k: {kk: float(np.mean([p[kk] for p in v])) for kk in keys} | {"n": len(v)}
                 for k, v in fm.items()}
        res, ps = {}, []
        for name, kk in zip(["augmented", "core", "continuity"], keys):
            vv = [means[k][kk] for k in means if k[0] == "vocal"]
            ii = [means[k][kk] for k in means if k[0] == "instrumental"]
            a, p, n = exact(vv, ii)
            res[name] = {"auc": a, "p_one_sided_exact": p, "n_permutations": n,
                         "complete_separation": bool(min(vv) > max(ii))}
            ps.append(p)
        for i, name in enumerate(["augmented", "core", "continuity"]):
            res[name]["p_holm"] = holm(ps)[i]
        pv = {kk: [float(r[kk]) for r in rows if r["group"] == "vocal"] for kk in keys}
        pi = {kk: [float(r[kk]) for r in rows if r["group"] == "instrumental"] for kk in keys}
        res["piece_level_descriptive"] = {name: auc(pv[kk], pi[kk])
                                          for name, kk in zip(["augmented", "core", "continuity"], keys)}
        res["form_ranking_by_augmented"] = [
            {"form": k[1], "group": k[0], "aug": round(means[k][keys[0]], 4)}
            for k in sorted(means, key=lambda k: -means[k][keys[0]])]
        out[scheme] = res
    ar = [r["aug_d"] for r in rows if r["form"] == "aranagme"]
    out["aranagme_dutch_ref"] = {
        "n": len(ar), "mean_aug": float(np.mean(ar)),
        "mean_aug_vocal": float(np.mean([r["aug_d"] for r in rows if r["group"] == "vocal"])),
        "mean_aug_instr": float(np.mean([r["aug_d"] for r in rows if r["group"] == "instrumental"]))}
    bl = [x["form"] for x in out["symbtr_blind"]["form_ranking_by_augmented"]]
    dr = {x["form"]: i for i, x in enumerate(out["dutch_ref"]["form_ranking_by_augmented"])}
    from scipy.stats import spearmanr
    out["rank_agreement_augmented_spearman"] = float(
        spearmanr(range(len(bl)), [dr[f] for f in bl]).statistic)

    dst = os.path.join(RESULTS, "symbtr_dutchref_results.json")
    json.dump(out, open(dst, "w"), indent=1, ensure_ascii=False)
    print("wrote", dst)


if __name__ == "__main__":
    main()
