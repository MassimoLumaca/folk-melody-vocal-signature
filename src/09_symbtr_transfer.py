#!/usr/bin/env python3
"""
09_symbtr_transfer.py — unit-weighted score transfer to a third corpus (SymbTr,
Turkish makam art music), applied without refitting.

The two unit-weighted scores defined on the Dutch corpus are applied to SymbTr with
no coefficient fitting, threshold optimization, or feature reselection:
    core      = z(repeated-pitch proportion) - z(pitch range)
    augmented = core + z(gap rate)
z-scores are label-blind over all analyzed pieces. Performance medium is
documented at the level of makam form, so forms with at least four pieces are
the validation units and medium is tested by exact enumeration of the
instrumental-label assignments (mirroring the Finnish collection-level design).
The analysis is deterministic: exact enumeration and deterministic z-scores,
no random component.

Label rule (from the form field of the filename): the vocal and instrumental
form sets are fixed a priori. Mixed-practice and pedagogical forms are flagged
out of the primary contrast; non-repertoire example material is excluded.
aranagme (instrumental renditions of song material) is reported separately as
a provenance check.

Feature derivation from SymbTr-txt: structure rows (Kod 51) are skipped; rest
rows (Nota53 == 'Es' or Koma53 == -1) accumulate time only; pitch is the
53-comma value converted to semitones (Koma53 * 12/53); note duration is
4*Pay/Payda quarter-note beats (grace notes keep duration 0); a gap is an
inter-note silence of at least 0.5 beats. Pieces with fewer than three notes
are excluded and counted. Definitions match src/extract_features.py.

Inputs : data/symbtr_txt/            (SymbTr-txt note files; see data/README.md)
Outputs: results/09_symbtr_results.json
         results/09_symbtr_piece_scores.csv  (per-piece transferred scores; SI Fig. S8)

This script reports the third-corpus Results section and SI Materials and
Methods 11.

Usage: python 09_symbtr_transfer.py [DATA_DIR] [RESULTS_DIR]
"""
import os, sys, csv, json, itertools
import numpy as np
from collections import defaultdict

VOCAL = {"sarki", "turku", "ilahi", "beste", "yuruksemai", "agirsemai", "fantezi",
         "rumeliturkusu", "nakis", "murabba", "kar", "kanto", "nefes", "divan",
         "popsarkisi", "ninni", "mersiye", "durak", "destan", "cocuksarkisi",
         "bozlak", "kosma", "kalenderi", "kar_i_natik", "kar_i_nev", "karce",
         "sugul", "tevsihilahi", "tesbih", "tekbir", "salatuselam", "salatiummiye",
         "miraciye", "selam", "guvende"}
INSTR = {"pesrev", "sazsemaisi", "longa", "sirto", "sazeseri", "oyunhavasi",
         "medhal", "mandra", "kasaphavasi", "kurthavasi"}
FLAGGED = {"aranagme", "zeybek", "kocekce", "mars", "mehter", "karsilama",
           "tavsanca", "etud"}
EXCL = {"seyir", "kupe", "ornek_oz"}


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    d = os.path.join(data_dir, "symbtr_txt")
    out = os.path.join(results_dir, "09_symbtr_results.json")

    pieces = []; n_short = 0; n_parsefail = 0
    for fn in sorted(os.listdir(d)):
        form = fn.split('--')[1]
        if form in EXCL: continue
        grp = 'vocal' if form in VOCAL else 'instrumental' if form in INSTR else \
            'flagged' if form in FLAGGED else None
        if grp is None: continue
        pitch, onset, dur = [], [], []
        t = 0.0
        try:
            for r in csv.DictReader(open(os.path.join(d, fn), encoding='utf-8'),
                                    delimiter='\t'):
                if r['Kod'] == '51': continue
                try: koma = int(r['Koma53'])
                except: continue
                try:
                    b = 4 * float(r['Pay']) / float(r['Payda']) \
                        if r['Payda'] not in ('', '0') else 0.0
                except: b = 0.0
                if koma == -1 or r['Nota53'] == 'Es':
                    t += b; continue
                pitch.append(koma); onset.append(t); dur.append(b); t += b
        except Exception:
            n_parsefail += 1; continue
        if len(pitch) < 3:
            n_short += 1; continue
        p = np.array(pitch, float); o = np.array(onset); du = np.array(dur)
        ivl = np.diff(p); ttrans = len(p) - 1
        rest = o[1:] - (o[:-1] + du[:-1])
        pieces.append(dict(file=fn, form=form, group=grp, n=len(p),
            rep=float((ivl == 0).sum() / ttrans),
            rng=float((p.max() - p.min()) * 12 / 53),
            gap=float((rest >= 0.5 - 1e-9).sum() / ttrans)))

    for k in ['rep', 'rng', 'gap']:
        v = np.array([x[k] for x in pieces]); mu, sd = v.mean(), v.std(ddof=1)
        for x in pieces: x['z' + k] = (x[k] - mu) / sd
    for x in pieces:
        x['core'] = x['zrep'] - x['zrng']; x['aug'] = x['core'] + x['zgap']

    piece_csv = os.path.join(results_dir, "09_symbtr_piece_scores.csv")
    cols = ['file', 'form', 'group', 'n', 'rep', 'rng', 'gap',
            'zrep', 'zrng', 'zgap', 'core', 'aug']
    with open(piece_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for x in pieces:
            w.writerow([x[c] for c in cols])
    print('wrote', piece_csv)

    res = {'n_pieces': len(pieces), 'excluded_short': n_short,
           'parse_failures': n_parsefail}
    forms = defaultdict(list)
    for x in pieces:
        if x['group'] in ('vocal', 'instrumental'):
            forms[(x['group'], x['form'])].append(x)
    fm = [(g, f, len(v), np.mean([p['aug'] for p in v]),
           np.mean([p['core'] for p in v]), np.mean([p['zgap'] for p in v]),
           np.mean([p['rep'] for p in v]), np.mean([p['rng'] for p in v]),
           np.mean([p['gap'] for p in v]))
          for (g, f), v in forms.items() if len(v) >= 4]
    voc = [x for x in fm if x[0] == 'vocal']; ins = [x for x in fm if x[0] == 'instrumental']
    res['form_table'] = [dict(zip(['group', 'form', 'n', 'aug', 'core', 'zgap',
        'rep_raw', 'range_st', 'gap_raw'], x)) for x in sorted(fm, key=lambda x: -x[3])]

    def auc(a, b):
        a = np.asarray(a)[:, None]; b = np.asarray(b)[None, :]
        return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))

    def exact(vv, ii):
        vals = np.array(vv + ii); k = len(ii)
        obs = auc(vv, ii); null = []
        for c in itertools.combinations(range(len(vals)), k):
            m = np.ones(len(vals), bool); m[list(c)] = False
            null.append(auc(vals[m], vals[~m]))
        null = np.array(null)
        return obs, float((null >= obs - 1e-12).mean()), len(null)

    for ix, name in [(3, 'augmented'), (4, 'core'), (5, 'continuity')]:
        vv = [x[ix] for x in voc]; ii = [x[ix] for x in ins]
        a, p, n = exact(vv, ii)
        res[f'form_test_{name}'] = dict(n_vocal_forms=len(vv),
            n_instr_forms=len(ii), auc=a,
            complete_separation=bool(min(vv) > max(ii)),
            p_one_sided_exact=p, n_permutations=n)

    pv = [x['aug'] for x in pieces if x['group'] == 'vocal']
    pi = [x['aug'] for x in pieces if x['group'] == 'instrumental']
    res['piece_level_descriptive'] = dict(n_vocal=len(pv), n_instr=len(pi),
        auc_augmented=auc(pv, pi),
        auc_core=auc([x['core'] for x in pieces if x['group'] == 'vocal'],
                     [x['core'] for x in pieces if x['group'] == 'instrumental']),
        auc_continuity=auc([x['zgap'] for x in pieces if x['group'] == 'vocal'],
                           [x['zgap'] for x in pieces if x['group'] == 'instrumental']))
    ar = [x['aug'] for x in pieces if x['form'] == 'aranagme']
    res['aranagme_sensitivity'] = dict(n=len(ar), mean_aug=float(np.mean(ar)),
        mean_aug_vocal=float(np.mean(pv)), mean_aug_instr=float(np.mean(pi)))
    json.dump(res, open(out, 'w'), indent=2, default=float)
    print('wrote', out)


if __name__ == '__main__':
    main()
