#!/usr/bin/env python3
"""
01_inspect_data.py — Inventory and quality inspection of the Dutch (MTC) and
Finnish (finfolktunes) corpus files in 02_data.

Usage:
    python 01_inspect_data.py [DATA_DIR] [OUT_DIR]

Defaults: DATA_DIR = 02_data, OUT_DIR = 03_outputs

Outputs:
    OUT_DIR/01_file_inventory.csv
    OUT_DIR/01_inspection_stats.json   (raw numbers used by the report)

Only standard-library + numpy/scipy (for the .mat file) are used.
The script is read-only with respect to DATA_DIR.
"""

import csv
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.io import loadmat


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def human(nbytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024.0


def inspect_mtc_jsonl(path, note_keys=("midipitch", "onsettick", "duration")):
    """Stream-parse an MTC/Essen *_sequences.jsonl.gz file and collect stats."""
    stats = {
        "n_records": 0,
        "n_json_errors": 0,
        "json_error_lines": [],
        "n_notes_total": 0,
        "note_count_min": None,
        "note_count_max": None,
        "type_counts": Counter(),
        "n_missing_id": 0,
        "n_duplicate_id": 0,
        "n_missing_tunefamily": 0,
        "n_missing_year": 0,          # year is None / 0 / absent
        "year_min": None,
        "year_max": None,
        "n_freemeter_true": 0,
        "n_missing_note_key": Counter(),      # records lacking a pitch/onset/duration key
        "n_records_unequal_feature_lengths": 0,
        "n_null_midipitch": 0,        # None values inside midipitch lists
        "n_null_duration": 0,
        "n_null_onsettick": 0,
        "n_zero_length_sequences": 0,
        "feature_keys": None,         # keys of first record (reference set)
        "n_records_missing_feature_keys": 0,  # records whose feature set != reference
        "has_ann_bgcorpus": False,
        "origin_counts": Counter(),   # essen only
    }
    seen_ids = set()
    years = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["n_json_errors"] += 1
                if len(stats["json_error_lines"]) < 10:
                    stats["json_error_lines"].append(lineno)
                continue
            stats["n_records"] += 1

            rid = rec.get("id")
            if rid in (None, ""):
                stats["n_missing_id"] += 1
            elif rid in seen_ids:
                stats["n_duplicate_id"] += 1
            else:
                seen_ids.add(rid)

            tf = rec.get("tunefamily")
            if tf in (None, ""):
                stats["n_missing_tunefamily"] += 1

            yr = rec.get("year")
            if yr in (None, "", 0):
                stats["n_missing_year"] += 1
            else:
                try:
                    years.append(int(yr))
                except (TypeError, ValueError):
                    stats["n_missing_year"] += 1

            stats["type_counts"][rec.get("type", "<absent>")] += 1
            if rec.get("freemeter"):
                stats["n_freemeter_true"] += 1
            if "ann_bgcorpus" in rec:
                stats["has_ann_bgcorpus"] = True
            if "origin" in rec:
                stats["origin_counts"][rec["origin"]] += 1

            feats = rec.get("features", {})
            if stats["feature_keys"] is None:
                stats["feature_keys"] = sorted(feats.keys())
            elif sorted(feats.keys()) != stats["feature_keys"]:
                stats["n_records_missing_feature_keys"] += 1

            # note-level checks
            lengths = {len(v) for v in feats.values() if isinstance(v, list)}
            if len(lengths) > 1:
                stats["n_records_unequal_feature_lengths"] += 1
            n_notes = len(feats.get("midipitch", [])) if isinstance(
                feats.get("midipitch"), list) else 0
            if n_notes == 0:
                stats["n_zero_length_sequences"] += 1
            stats["n_notes_total"] += n_notes
            stats["note_count_min"] = n_notes if stats["note_count_min"] is None \
                else min(stats["note_count_min"], n_notes)
            stats["note_count_max"] = n_notes if stats["note_count_max"] is None \
                else max(stats["note_count_max"], n_notes)

            for k in note_keys:
                v = feats.get(k)
                if not isinstance(v, list):
                    stats["n_missing_note_key"][k] += 1
                else:
                    nnull = sum(1 for x in v if x is None)
                    if k == "midipitch":
                        stats["n_null_midipitch"] += nnull
                    elif k == "duration":
                        stats["n_null_duration"] += nnull
                    elif k == "onsettick":
                        stats["n_null_onsettick"] += nnull

    if years:
        stats["year_min"], stats["year_max"] = min(years), max(years)
    stats["type_counts"] = dict(stats["type_counts"])
    stats["n_missing_note_key"] = dict(stats["n_missing_note_key"])
    stats["origin_counts"] = dict(stats["origin_counts"])
    return stats


def inspect_finnish_txt(path, expected_ncols=11):
    """Inspect the tab-separated Finnish metadata file (Latin-1 encoded)."""
    stats = {
        "n_rows": 0,
        "header": None,
        "n_malformed_rows": 0,       # wrong number of tab-separated fields
        "malformed_examples": [],
        "missing_per_column": None,
        "collection_counts": Counter(),
        "year_nonempty": 0,
        "year_values_sample": Counter(),
        "n_duplicate_row_keys": 0,
    }
    seen_keys = set()
    with open(path, "r", encoding="latin-1") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r").split("\t")
        stats["header"] = header
        missing = Counter()
        for lineno, line in enumerate(fh, 2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").rstrip("\r").split("\t")
            stats["n_rows"] += 1
            if len(fields) != expected_ncols:
                stats["n_malformed_rows"] += 1
                if len(stats["malformed_examples"]) < 10:
                    stats["malformed_examples"].append(
                        {"line": lineno, "n_fields": len(fields)})
                fields = (fields + [""] * expected_ncols)[:expected_ncols]
            for i, f in enumerate(fields):
                if f.strip() == "":
                    missing[header[i] + f"[{i}]"] += 1
            rk = fields[0]
            if rk in seen_keys:
                stats["n_duplicate_row_keys"] += 1
            seen_keys.add(rk)
            stats["collection_counts"][fields[1]] += 1
            if fields[7].strip():
                stats["year_nonempty"] += 1
                if len(stats["year_values_sample"]) < 30:
                    stats["year_values_sample"][fields[7].strip()] += 1
        stats["missing_per_column"] = dict(missing)
    stats["collection_counts"] = dict(stats["collection_counts"])
    stats["year_values_sample"] = dict(stats["year_values_sample"])
    return stats


def inspect_finnish_mat(path):
    """Inspect the finfolktunes.mat MIDI-Toolbox note-matrix archive."""
    m = loadmat(path, squeeze_me=True)
    nm = m["nm"]
    stats = {
        "matlab_header": str(m.get("__header__")),
        "n_entries": int(nm.shape[0]),
        "n_empty": 0,
        "n_not_7_cols": 0,
        "n_notes_total": 0,
        "note_count_min": None,
        "note_count_max": None,
        "n_entries_with_nan": 0,
        "n_entries_neg_duration": 0,
        "n_entries_nonmono_onset": 0,   # onsets not non-decreasing
        "pitch_min": None,
        "pitch_max": None,
    }
    for e in nm:
        a = np.atleast_2d(np.asarray(e, dtype=float)) if np.asarray(e).size else np.zeros((0, 7))
        if a.size == 0:
            stats["n_empty"] += 1
            continue
        if a.shape[1] != 7:
            stats["n_not_7_cols"] += 1
            continue
        n = a.shape[0]
        stats["n_notes_total"] += n
        stats["note_count_min"] = n if stats["note_count_min"] is None \
            else min(stats["note_count_min"], n)
        stats["note_count_max"] = n if stats["note_count_max"] is None \
            else max(stats["note_count_max"], n)
        if np.isnan(a).any():
            stats["n_entries_with_nan"] += 1
        if (a[:, 1] < 0).any():
            stats["n_entries_neg_duration"] += 1
        if (np.diff(a[:, 0]) < 0).any():
            stats["n_entries_nonmono_onset"] += 1
        pmin, pmax = a[:, 3].min(), a[:, 3].max()
        stats["pitch_min"] = pmin if stats["pitch_min"] is None else min(stats["pitch_min"], pmin)
        stats["pitch_max"] = pmax if stats["pitch_max"] is None else max(stats["pitch_max"], pmax)
    return stats


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "02_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "03_outputs"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    mtc_files = {
        "MTC-FS-INST-2.0": "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz",
        "MTC-ANN-2.0.1": "MTC-ANN-2.0.1_sequences-1.1.jsonl.gz",
        "Essen": "essen_sequences-1.1.jsonl.gz",
    }
    for name, fn in mtc_files.items():
        p = os.path.join(data_dir, fn)
        if os.path.exists(p):
            print(f"Inspecting {name} ...", flush=True)
            results[name] = inspect_mtc_jsonl(p)
        else:
            results[name] = {"error": "file not found"}

    p_txt = os.path.join(data_dir, "finfolktunes_data_corrected.txt")
    if os.path.exists(p_txt):
        print("Inspecting Finnish metadata txt ...", flush=True)
        results["finfolktunes_txt"] = inspect_finnish_txt(p_txt)
    else:
        results["finfolktunes_txt"] = {"error": "file not found"}

    p_mat = os.path.join(data_dir, "finfolktunes.mat")
    if os.path.exists(p_mat):
        print("Inspecting Finnish .mat note matrices ...", flush=True)
        results["finfolktunes_mat"] = inspect_finnish_mat(p_mat)
    else:
        results["finfolktunes_mat"] = {"error": "file not found"}

    with open(os.path.join(out_dir, "01_inspection_stats.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("Wrote", os.path.join(out_dir, "01_inspection_stats.json"))


if __name__ == "__main__":
    main()
