#!/usr/bin/env python3
"""
06b_volume_sensitivity.py — POST HOC structural sensitivity analysis of the
Finnish cross-corpus transfer (constructed after the transfer test in
06_validate_finnish_signatures.py; reported as a robustness check, not a
prespecified test).

Motivation: the ten Finnish collections derive from one publication series,
Suomen Kansan Sävelmiä (1893–1933), issued in five series under three
editors (esavelmat.jyu.fi/collection.html): I Hengellisiä sävelmiä (HS1,
1898, Krohn); II Laulusävelmiä (LS1–LS4, 1904–1933, Krohn); III
Kansantansseja (KT1, 1893, Krohn); IV Runosävelmiä (RS1 Ingria 1910, RS2
Karelia 1930, Launis); V Kantele- ja jouhikkosävelmiä (1928, Väisänen).
Kantelesävelmiä and Jouhikkosävelmiä are the two instrument sections of the
single 1928 volume and share one filename prefix (kjs_) in the archive, so
they are not independent units.

Two aggregations of the unit-weighted augmented score (scores read from
03_outputs/06_finnish_melody_scores.csv; nothing is refit):
  1. "merged_volume_V": Kantelesävelmiä + Jouhikkosävelmiä as one
     melody-weighted unit -> 9 units (7 vocal, 2 instrumental),
     C(9,2) = 36 allocations, attainable minimum one-sided P = 1/36.
  2. "five_series": all collections aggregated to series I–V
     (vocal: I, II, IV; instrumental: III, V), C(5,2) = 10 allocations,
     attainable minimum one-sided P = 1/10 = 0.10 (design ceiling).

Reported per scheme (finnish_blind, dutch_ref): complete-separation flag,
minimum boundary margin (lowest vocal unit minus highest instrumental
unit, SD units of the scheme), exact one-sided P.

Output: 03_outputs/06b_volume_sensitivity.csv
Usage: python 06b_volume_sensitivity.py [OUT_DIR]
"""

import csv
import itertools
import os
import sys

import numpy as np

VOCAL = ["LS1", "LS2", "LS3", "LS4", "RS1", "RS2", "HS1"]
SERIES = {  # series -> (medium, member collections)
    "I_Hengellisia": ("vocal", ["HS1"]),
    "II_Laulusavelmia": ("vocal", ["LS1", "LS2", "LS3", "LS4"]),
    "III_Kansantansseja": ("instrumental", ["KT1"]),
    "IV_Runosavelmia": ("vocal", ["RS1", "RS2"]),
    "V_Kantele_ja_jouhikko": ("instrumental",
                              ["Kantelesävelmiä", "Jouhikkosävelmiä"]),
}


def auc(v, i):
    v = np.asarray(v)[:, None]
    i = np.asarray(i)[None, :]
    return float(((v > i).sum() + 0.5 * (v == i).sum()) / (v.size * i.size))


def exact_p(voc_vals, ins_vals):
    vals = np.array(list(voc_vals) + list(ins_vals))
    k = len(ins_vals)
    obs = auc(voc_vals, ins_vals)
    null = []
    for combo in itertools.combinations(range(len(vals)), k):
        mask = np.ones(len(vals), bool)
        mask[list(combo)] = False
        null.append(auc(vals[mask], vals[~mask]))
    null = np.array(null)
    return obs, float((null >= obs - 1e-12).mean()), len(null)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "03_outputs"
    rows = list(csv.DictReader(
        open(os.path.join(out_dir, "06_finnish_melody_scores.csv"))))

    out = [["scheme", "aggregation", "n_units", "n_vocal", "n_instrumental",
            "complete_separation", "min_boundary_margin_sd",
            "collection_auc", "p_one_sided_exact", "n_permutations",
            "unit_values"]]
    for scheme in ["finnish_blind", "dutch_ref"]:
        col = f"augmented_{scheme}"
        melodies = {}
        for r in rows:
            melodies.setdefault(r["collection"], []).append(float(r[col]))
        coll_mean = {c: float(np.mean(v)) for c, v in melodies.items()}

        # 1) merged volume V
        merged = float(np.mean(melodies["Kantelesävelmiä"]
                               + melodies["Jouhikkosävelmiä"]))
        voc = [coll_mean[c] for c in VOCAL]
        ins = [coll_mean["KT1"], merged]
        a, p, n = exact_p(voc, ins)
        margin = min(voc) - max(ins)
        units = {**{c: coll_mean[c] for c in VOCAL}, "KT1": coll_mean["KT1"],
                 "V_merged": merged}
        out.append([scheme, "merged_volume_V", 9, 7, 2,
                    "yes" if margin > 0 else "no", f"{margin:.4f}",
                    f"{a:.4f}", f"{p:.5f}", n,
                    ";".join(f"{k}={v:.3f}" for k, v in units.items())])

        # 2) five source series
        voc_s, ins_s, units = [], [], {}
        for s, (medium, members) in SERIES.items():
            pool = sum((melodies[m] for m in members), [])
            val = float(np.mean(pool))
            units[s] = val
            (voc_s if medium == "vocal" else ins_s).append(val)
        a, p, n = exact_p(voc_s, ins_s)
        margin = min(voc_s) - max(ins_s)
        out.append([scheme, "five_series", 5, 3, 2,
                    "yes" if margin > 0 else "no", f"{margin:.4f}",
                    f"{a:.4f}", f"{p:.5f}", n,
                    ";".join(f"{k}={v:.3f}" for k, v in units.items())])

    path = os.path.join(out_dir, "06b_volume_sensitivity.csv")
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(out)
    for r in out[1:]:
        print(r[:10])
    print("wrote", path)


if __name__ == "__main__":
    main()
