#!/usr/bin/env python3
"""
extract_features.py — Corpus-neutral melodic feature extraction for the Dutch
(MTC-FS-INST 2.0) and Finnish (finfolktunes) corpora.

One feature function is applied to both corpora after each melody is reduced to
the same representation: arrays of (midi_pitch, onset_beats, duration_beats),
with onset/duration in quarter-note beat units and onset of the first note = 0.

Exclusions honoured (by construction — these inputs are never read):
  - Dutch phrase annotations (phrase_end, phrase_ix, phrasepos, GPR/LBDM fields)
  - lyrics / text-derived variables
  - corpus-specific derived annotations (contours, ima*, scaledegree, etc.)
  - documentation year

Usage:
    python extract_features.py [DATA_DIR] [OUT_DIR]
Defaults: DATA_DIR = 02_data, OUT_DIR = 03_outputs

Outputs:
    OUT_DIR/02_dutch_features.csv
    OUT_DIR/02_finnish_features.csv
    OUT_DIR/02_feature_qc.json    (missingness + correlation matrices, raw)

The script is read-only with respect to DATA_DIR.
"""

import csv
import gzip
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.io import loadmat

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------

GAP_BEATS = 0.5   # a transition is a "gap" if the inter-note rest is >= 0.5
                  # quarter-note beats (matches GAP_PRIMARY used elsewhere in
                  # this project, 08_finnish_crossval/finnish_crossval.py)

# Finnish medium mapping, documented at the source-collection level
# (esavelmat.jyu.fi/collection.html; same sets as finnish_crossval.py)
FIN_VOCAL = {"LS1", "LS2", "LS3", "LS4", "RS1", "RS2", "HS1"}
FIN_INSTR = {"KT1", "Kantelesävelmiä", "Jouhikkosävelmiä"}

FEATURES = [
    "prop_gap_transitions",
    "mean_notes_between_gaps",
    "max_notes_between_gaps",
    "mean_dur_between_gaps",
    "max_dur_between_gaps",
    "note_density",
    "pitch_range",
    "max_abs_interval",
    "mean_abs_interval_nonrep",
    "prop_repeated_pitch",
    "prop_large_interval_ge5",
    "n_notes",
]


# ----------------------------------------------------------------------
# shared feature function (identical math for both corpora)
# ----------------------------------------------------------------------

def melody_features(pitch, onset, dur, gap_beats=GAP_BEATS):
    """Compute the 12 features from parallel arrays (pitch, onset, dur).

    pitch : int MIDI numbers; onset/dur : floats in quarter-note beats,
    onset non-decreasing. Returns dict; NaN where undefined.
    """
    pitch = np.asarray(pitch, dtype=float)
    onset = np.asarray(onset, dtype=float)
    dur = np.asarray(dur, dtype=float)
    n = len(pitch)
    out = {k: np.nan for k in FEATURES}
    out["n_notes"] = n
    if n == 0:
        return out

    offset = onset + dur
    span = offset.max() - onset[0]
    out["pitch_range"] = pitch.max() - pitch.min()
    out["note_density"] = n / span if span > 0 else np.nan

    if n < 2:
        return out

    # transitions
    ivl = np.diff(pitch)
    a = np.abs(ivl)
    t = n - 1
    out["max_abs_interval"] = a.max()
    out["prop_repeated_pitch"] = (ivl == 0).sum() / t
    out["prop_large_interval_ge5"] = (a >= 5).sum() / t
    nonrep = a[ivl != 0]
    out["mean_abs_interval_nonrep"] = nonrep.mean() if len(nonrep) else np.nan

    # gaps: rest between consecutive notes >= gap_beats
    rest = onset[1:] - offset[:-1]          # may be < 0 only if overlapping
    is_gap = rest >= gap_beats - 1e-9
    out["prop_gap_transitions"] = is_gap.sum() / t

    # segments delimited by gap transitions (first and last segments included)
    bounds = np.flatnonzero(is_gap)         # gap after note index b
    starts = np.concatenate(([0], bounds + 1))
    ends = np.concatenate((bounds, [n - 1]))
    seg_notes = ends - starts + 1
    seg_dur = offset[ends] - onset[starts]
    out["mean_notes_between_gaps"] = seg_notes.mean()
    out["max_notes_between_gaps"] = int(seg_notes.max())
    out["mean_dur_between_gaps"] = seg_dur.mean()
    out["max_dur_between_gaps"] = seg_dur.max()
    return out


# ----------------------------------------------------------------------
# corpus loaders -> (melody_id, voc_inst, pitch, onset, dur)
# ----------------------------------------------------------------------

def _frac(x):
    return float(Fraction(x)) if x is not None else 0.0


def iter_dutch(data_dir):
    """MTC-FS-INST: onsets rebuilt as cumulative (duration + rest) in quarter
    units. This equals the corpus's own IOI exactly (verified: max deviation
    ~4e-16 over all 18,109 records); the integer onsettick grid is NOT used
    because it is lossy for 56 tuplet-containing records."""
    p = os.path.join(data_dir, "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            f = rec["features"]
            dur = np.array(f["duration"], dtype=float)
            rest = np.array([_frac(r) for r in f["restduration_frac"]])
            n = len(dur)
            onset = np.zeros(n)
            if n > 1:
                onset[1:] = np.cumsum(dur[:-1] + rest[:-1])
            yield rec["id"], rec["type"], np.array(f["midipitch"], dtype=float), onset, dur


def iter_finnish(data_dir, qc):
    """finfolktunes: metadata row i (1-based key) <-> nm{i}. Latin-1 TSV; the
    96 short KT1 rows are right-padded. Notes with pitch <= 0 are dropped
    (implausible MIDI 0 encodings; counted in qc). Melodies with simultaneous
    onsets are reduced to a single line by the skyline rule: at each distinct
    onset keep the highest-pitched note (counted in qc)."""
    txt = os.path.join(data_dir, "finfolktunes_data_corrected.txt")
    lines = open(txt, encoding="latin-1").read().splitlines()
    rows = []
    for l in lines[1:]:
        if not l.strip():
            continue
        fields = l.split("\t")
        fields = (fields + [""] * 11)[:11]
        rows.append(fields)

    m = loadmat(os.path.join(data_dir, "finfolktunes.mat"), squeeze_me=True)
    nm = m["nm"]
    assert len(rows) == nm.shape[0], "metadata/matrix count mismatch"

    for i, fields in enumerate(rows):
        coll, fname = fields[1], fields[2]
        voc_inst = ("vocal" if coll in FIN_VOCAL
                    else "instrumental" if coll in FIN_INSTR else "unknown")
        a = np.atleast_2d(np.asarray(nm[i], dtype=float))
        onset, dur, pitch = a[:, 0], a[:, 1], a[:, 3]

        keep = pitch > 0
        if (~keep).any():
            qc["fin_notes_dropped_pitch0"] += int((~keep).sum())
        onset, dur, pitch = onset[keep], dur[keep], pitch[keep]

        # skyline reduction for simultaneous onsets
        if len(onset) > 1 and (np.diff(onset) == 0).any():
            order = np.lexsort((pitch, onset))        # onset asc, pitch asc
            onset, dur, pitch = onset[order], dur[order], pitch[order]
            last = np.concatenate((np.diff(onset) != 0, [True]))  # keep highest
            qc["fin_notes_removed_skyline"] += int((~last).sum())
            qc["fin_melodies_skyline"] += 1
            onset, dur, pitch = onset[last], dur[last], pitch[last]

        onset = onset - onset[0] if len(onset) else onset
        yield fname, voc_inst, pitch, onset, dur


# ----------------------------------------------------------------------
# QC helpers
# ----------------------------------------------------------------------

def qc_table(rows):
    """missingness per feature + Pearson correlations on complete pairs."""
    X = np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
    miss = {k: int(np.isnan(X[:, j]).sum()) for j, k in enumerate(FEATURES)}
    corr = {}
    for j in range(len(FEATURES)):
        for k in range(j + 1, len(FEATURES)):
            ok = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
            if ok.sum() > 2 and X[ok, j].std() > 0 and X[ok, k].std() > 0:
                r = float(np.corrcoef(X[ok, j], X[ok, k])[0, 1])
                corr[f"{FEATURES[j]}~{FEATURES[k]}"] = round(r, 4)
    return miss, corr


def write_csv(path, ids, labels, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["melody_id", "voc_inst"] + FEATURES)
        for i, lab, r in zip(ids, labels, rows):
            w.writerow([i, lab] + [("" if isinstance(r[k], float) and np.isnan(r[k])
                                    else r[k]) for k in FEATURES])


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"
    os.makedirs(out_dir, exist_ok=True)

    qc = {"fin_notes_dropped_pitch0": 0, "fin_notes_removed_skyline": 0,
          "fin_melodies_skyline": 0}
    report = {"gap_beats": GAP_BEATS, "qc_counts": qc}

    for corpus, it, fname in [
        ("dutch", iter_dutch(data_dir), "02_dutch_features.csv"),
        ("finnish", iter_finnish(data_dir, qc), "02_finnish_features.csv"),
    ]:
        ids, labels, rows = [], [], []
        for mid, lab, pitch, onset, dur in it:
            ids.append(mid)
            labels.append(lab)
            rows.append(melody_features(pitch, onset, dur))
        write_csv(os.path.join(out_dir, fname), ids, labels, rows)
        miss, corr = qc_table(rows)
        report[corpus] = {
            "n_melodies": len(ids),
            "missing_per_feature": miss,
            "correlations": corr,
            "high_correlations_abs_ge_0.8": {
                k: v for k, v in corr.items() if abs(v) >= 0.8},
        }
        print(f"{corpus}: {len(ids)} melodies -> {fname}")

    with open(os.path.join(out_dir, "02_feature_qc.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("QC ->", os.path.join(out_dir, "02_feature_qc.json"))


if __name__ == "__main__":
    main()
