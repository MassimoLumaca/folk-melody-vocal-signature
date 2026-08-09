#!/usr/bin/env python3
"""
alignment_analysis.py — Positional differences between aligned vocal and
instrumental variants within the 433 Dutch mixed tune families.
Run per ALIGNMENT_PROTOCOL.md (committed before this script executed).
EXPLORATORY; no directional claims.

Archived from TEMP_alignment_test on 2026-08-08; only the I/O paths below
were adapted to the repository layout (data/ input, results/ outputs).
The analysis logic is byte-identical to the pre-committed run.

Reads (read-only): ../data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz
    (override with the MTC_SEQ environment variable; see data/README.md)
Writes (../results/): alignment_pair_metrics.csv,
    alignment_results.json (primary + quality>=0.5 + sensitivity scoring)

Usage: python alignment_analysis.py [--sensitivity]
"""
import gzip
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.environ.get("MTC_SEQ", os.path.join(
    HERE, "..", "data", "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz"))
OUTDIR = os.path.join(HERE, "..", "results")
SEED = 42
NBOOT = 2000
GAP_BEATS = 0.5

# primary scoring
PRIM = dict(match=2.0, near=0.0, mis=-1.0, gap=-1.5, near_tol=2)
# prespecified sensitivity scoring
SENS = dict(match=1.0, near=-1.0, mis=-1.0, gap=-1.0, near_tol=0)


def frac(x):
    return float(Fraction(x)) if x is not None else 0.0


def load_families():
    fam = defaultdict(lambda: defaultdict(list))
    with gzip.open(SEQ, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["tunefamily"]:
                continue
            f = r["features"]
            dur = np.array(f["duration"], float)
            rest = np.array([frac(x) for x in f["restduration_frac"]])
            n = len(dur)
            onset = np.zeros(n)
            if n > 1:
                onset[1:] = np.cumsum(dur[:-1] + rest[:-1])
            fam[r["tunefamily"]][r["type"]].append(dict(
                id=r["id"],
                pitch=np.array(f["midipitch"], float),
                onset=onset, dur=dur,
                beatstrength=np.array(
                    [x if x is not None else np.nan
                     for x in f["beatstrength"]], float),
                phrase_end=np.array(f["phrase_end"], bool)))
    return {tf: s for tf, s in fam.items()
            if s["vocal"] and s["instrumental"]}


def nw_score(a, b, sc):
    """Score-only NW, linear gaps, row-vectorized."""
    n, m = len(a), len(b)
    g = sc["gap"]
    prev = np.arange(m + 1, dtype=float) * g
    for i in range(1, n + 1):
        d = np.abs(b - a[i - 1])
        s = np.where(d == 0, sc["match"],
                     np.where(d <= sc["near_tol"], sc["near"], sc["mis"]))
        cand = np.maximum(prev[:-1] + s, prev[1:] + g)
        row = np.empty(m + 1)
        row[0] = i * g
        t = np.maximum(np.concatenate(([row[0]], cand)),
                       -1e18)
        # resolve left-gap dependency: row[j]=max(cand[j-1], row[j-1]+g)
        idx = np.arange(m + 1)
        u = t - idx * g
        row = np.maximum.accumulate(u) + idx * g
        prev = row
    return prev[-1]


def nw_align(a, b, sc):
    """Full NW with traceback. Returns list of (i, j) ops with None for gaps."""
    n, m = len(a), len(b)
    g = sc["gap"]
    H = np.zeros((n + 1, m + 1))
    H[0, :] = np.arange(m + 1) * g
    H[:, 0] = np.arange(n + 1) * g
    for i in range(1, n + 1):
        d = np.abs(b - a[i - 1])
        s = np.where(d == 0, sc["match"],
                     np.where(d <= sc["near_tol"], sc["near"], sc["mis"]))
        cand = np.maximum(H[i - 1, :-1] + s, H[i - 1, 1:] + g)
        idx = np.arange(m + 1)
        t = np.concatenate(([H[i, 0]], cand)) - idx * g
        H[i] = np.maximum.accumulate(t) + idx * g
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            d = abs(a[i - 1] - b[j - 1])
            s = sc["match"] if d == 0 else (
                sc["near"] if d <= sc["near_tol"] else sc["mis"])
            if abs(H[i, j] - (H[i - 1, j - 1] + s)) < 1e-9:
                ops.append((i - 1, j - 1)); i -= 1; j -= 1; continue
        if i > 0 and abs(H[i, j] - (H[i - 1, j] + g)) < 1e-9:
            ops.append((i - 1, None)); i -= 1; continue
        ops.append((None, j - 1)); j -= 1
    return ops[::-1]


def pair_metrics(v, ins, sc):
    """v, ins: melody dicts (v vocal, ins instrumental)."""
    pv, pi = v["pitch"], ins["pitch"]
    base = int(round(np.median(pv) - np.median(pi)))
    best, bestc = -1e18, 0
    for c in (base - 1, base, base + 1):
        s = nw_score(pv, pi + c, sc)
        if s > best:
            best, bestc = s, c
    pit = pi + bestc
    ops = nw_align(pv, pit, sc)

    anchors = [(x, y) for x, y in ops if x is not None and y is not None]
    n_anch = len(anchors)
    quality = sum(1 for x, y in anchors if pv[x] == pit[y]) / \
        max(1, min(len(pv), len(pit)))
    out = dict(quality=quality, n_anchors=n_anch, transposition=bestc)
    if n_anch < 4:
        return None

    # Q1a leap<->step at consecutive anchors
    ls = sl = 0
    for (x1, y1), (x2, y2) in zip(anchors[:-1], anchors[1:]):
        iv = pv[x2] - pv[x1]
        ii = pit[y2] - pit[y1]
        if abs(ii) >= 5 and abs(iv) <= 2:
            ls += 1
        if abs(iv) >= 5 and abs(ii) <= 2:
            sl += 1
    out["q1a_leapstep_asym"] = (ls - sl) / (n_anch - 1)

    # insertions per side
    vins = [x for x, y in ops if y is None]
    iins = [y for x, y in ops if x is None]
    out["q2_ins_rate_vocal"] = len(vins) / n_anch
    out["q2_ins_rate_instr"] = len(iins) / n_anch
    out["q2_len_diff"] = len(pv) - len(pit)

    def rep_share(idxs, p):
        reps = sum(1 for k in idxs if k > 0 and p[k] == p[k - 1])
        return reps / len(idxs) if idxs else np.nan
    rv = rep_share(vins, pv)
    ri = rep_share(iins, pit)
    out["q1b_repins_vocal"] = rv
    out["q1b_repins_instr"] = ri
    out["q1b_repins_asym"] = (rv - ri) if (not np.isnan(rv)
                                           and not np.isnan(ri)) else np.nan

    # Q6' inserted-material structure
    def ins_absint(idxs, p):
        vals = [abs(p[k] - p[k - 1]) for k in idxs if k > 0]
        return float(np.mean(vals)) if vals else np.nan
    out["q6_insint_vocal"] = ins_absint(vins, pv)
    out["q6_insint_instr"] = ins_absint(iins, pit)

    # gaps at anchor transitions
    rest_v = v["onset"][1:] - (v["onset"][:-1] + v["dur"][:-1])
    rest_i = ins["onset"][1:] - (ins["onset"][:-1] + ins["dur"][:-1])

    def has_gap(rest, k1, k2):
        seg = rest[k1:k2]
        return bool((seg >= GAP_BEATS - 1e-9).any()) if len(seg) else False
    vg_only = ig_only = both = 0
    vgap_pos = []            # vocal note index before a vocal-specific gap
    for (x1, y1), (x2, y2) in zip(anchors[:-1], anchors[1:]):
        gv = has_gap(rest_v, x1, x2)
        gi = has_gap(rest_i, y1, y2)
        if gv and not gi:
            vg_only += 1
            vgap_pos.append(x1)
        elif gi and not gv:
            ig_only += 1
        elif gv and gi:
            both += 1
    out["q2_gap_asym"] = (vg_only - ig_only) / (n_anch - 1)

    # Q3: run length before vocal-specific gaps vs before non-gap transitions
    runlen = np.zeros(len(pv))       # uninterrupted beats before note k+1
    acc = 0.0
    for k in range(len(pv) - 1):
        acc += v["dur"][k]
        if rest_v[k] >= GAP_BEATS - 1e-9:
            acc = 0.0
        runlen[k + 1] = acc
    if vgap_pos:
        pre_gap = float(np.mean([runlen[k] for k in vgap_pos]))
        nong = [runlen[k] for k in range(1, len(pv) - 1)
                if rest_v[k - 1] < GAP_BEATS and k - 1 not in vgap_pos]
        out["q3_prerun_gap_minus_nongap"] = pre_gap - float(np.mean(nong)) \
            if nong else np.nan
        # Q5 at vocal-specific gaps
        bs = v["beatstrength"]
        mbs = np.nanmean(bs)
        out["q5_beatstrength_rel"] = float(
            np.nanmean([bs[k] for k in vgap_pos]) - mbs)
        pe = v["phrase_end"]
        out["q5_phrase_end_coincidence"] = float(
            np.mean([pe[k] for k in vgap_pos]) - pe.mean())
    else:
        out["q3_prerun_gap_minus_nongap"] = np.nan
        out["q5_beatstrength_rel"] = np.nan
        out["q5_phrase_end_coincidence"] = np.nan

    # Q4 register slope
    xs, ys = [], []
    ranks = pv.argsort().argsort() / max(1, len(pv) - 1)
    for x, y in anchors:
        xs.append(ranks[x])
        ys.append(pit[y] - pv[x])
    xs, ys = np.array(xs), np.array(ys)
    if xs.std() > 0:
        out["q4_register_slope"] = float(np.polyfit(xs, ys, 1)[0])
    else:
        out["q4_register_slope"] = np.nan
    return out


METRICS = ["q1a_leapstep_asym", "q1b_repins_vocal", "q1b_repins_instr",
           "q1b_repins_asym", "q2_ins_rate_vocal", "q2_ins_rate_instr",
           "q2_len_diff", "q2_gap_asym", "q3_prerun_gap_minus_nongap",
           "q4_register_slope", "q5_beatstrength_rel",
           "q5_phrase_end_coincidence", "q6_insint_vocal",
           "q6_insint_instr", "quality"]


def aggregate(rows, tag):
    """family means -> across-family mean with bootstrap CI."""
    byfam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in METRICS:
            if k in r and r[k] is not None and np.isfinite(r[k]):
                byfam[r["family"]][k].append(r[k])
    out = {}
    rng = np.random.default_rng(SEED)
    for k in METRICS:
        fm = np.array([np.mean(v[k]) for v in byfam.values() if v[k]])
        if len(fm) < 10:
            continue
        bs = [fm[rng.integers(0, len(fm), len(fm))].mean()
              for _ in range(NBOOT)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = dict(n_families=int(len(fm)), mean=float(fm.mean()),
                      ci95=[float(lo), float(hi)])
    return {tag: out}


def main():
    sc = SENS if "--sensitivity" in sys.argv else PRIM
    tagbase = "sensitivity" if "--sensitivity" in sys.argv else "primary"
    fams = load_families()
    rows = []
    skipped = 0
    for tf, sides in fams.items():
        for v in sides["vocal"]:
            for ins in sides["instrumental"]:
                m = pair_metrics(v, ins, sc)
                if m is None:
                    skipped += 1
                    continue
                m["family"] = tf
                m["vocal_id"] = v["id"]
                m["instr_id"] = ins["id"]
                rows.append(m)
    res = {"scoring": tagbase, "n_pairs": len(rows),
           "n_pairs_skipped_lt4_anchors": skipped,
           "quality_median": float(np.median([r["quality"] for r in rows])),
           "n_pairs_quality_ge_0.5":
               int(sum(1 for r in rows if r["quality"] >= 0.5))}
    res.update(aggregate(rows, "all_pairs"))
    res.update(aggregate([r for r in rows if r["quality"] >= 0.5],
                         "quality_ge_0.5"))

    suffix = "" if tagbase == "primary" else "_sensitivity"
    if tagbase == "primary":
        import csv as _csv
        with open(os.path.join(OUTDIR, "alignment_pair_metrics.csv"), "w",
                  newline="") as fh:
            w = _csv.writer(fh)
            cols = ["family", "vocal_id", "instr_id", "transposition",
                    "n_anchors"] + METRICS
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) for c in cols])
    with open(os.path.join(OUTDIR, f"alignment_results{suffix}.json"),
              "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("all_pairs", "quality_ge_0.5")},
                     indent=1))
    for tag in ["all_pairs", "quality_ge_0.5"]:
        print(f"== {tag}")
        for k, v in res[tag].items():
            print(f"  {k:<28} n={v['n_families']:>3} mean={v['mean']:+.4f} "
                  f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
