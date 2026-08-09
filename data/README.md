# Data

Neither corpus is redistributed in this repository. Both are publicly
available from their custodians.

## Dutch — Meertens Tune Collections, MTC-FS-INST 2.0

Obtain from https://www.liederenbank.nl/mtc/ (download requires a short
registration form). Place here:

```
data/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz
data/mtc_metadata/MTC-FS-INST-2.0.csv
data/mtc_metadata/MTC-FS-INST-2.0-fieldnames.csv
data/mtc_metadata/MTC-FS-INST-2.0-sources.csv
data/mtc_metadata/MTC-FS-INST-2.0-sources-fieldnames.csv
```

The sequences file supplies melody id, tune-family id, medium and
documentation year, and is also read by `alignment_analysis.py`
(exploratory phrase-boundary alignment, `src/ALIGNMENT_PROTOCOL.md`). The metadata tables supply the source `type` field
(`manuscript`, `print`, `audio`) used by `07_source_type_sensitivity.py`.

Reference: van Kranenburg P, de Bruin MJ (2019) *The Meertens Tune
Collections: MTC-FS-INST 2.0*. Meertens Online Reports 2019-1.

## Finnish — Digital Archive of Finnish Folk Tunes

Obtain from https://esavelmat.jyu.fi. Place the MATLAB note matrices and the
metadata table here as expected by `extract_features.py`:

```
data/finfolktunes.mat            (MIDI-Toolbox note matrices; variable `nm`)
data/finfolktunes_data_corrected.txt   (Latin-1 metadata: collection, type, year, ...)
```

`finfolktunes.mat` is read directly by `06_validate_finnish_signatures.py`
and by `08_gap_threshold_sensitivity.py`.

Reference: Eerola T, Toiviainen P (2004) *Suomen Kansan eSävelmät: Digital
Archive of Finnish Folk Tunes*. University of Jyväskylä.

## Turkish — SymbTr (Turkish makam music symbolic database)

Obtain the SymbTr-txt collection from https://github.com/MTG/SymbTr and unzip
the per-piece note files here:

```
data/symbtr_txt/*.txt
```

Each filename encodes `makam--form--usul--title--composer`;
`09_symbtr_transfer.py` reads the form field for performance medium and the
note rows (`Koma53`, `Pay`, `Payda`, `Nota53`, `Kod`) for pitch and timing.

Reference: Karaosmanoğlu MK (2012) *A Turkish makam music symbolic database
for music information retrieval: SymbTr*. Proc ISMIR 13:223-228.

## Note on medium labels

Dutch medium is recorded per melody in the record type field. Finnish medium
is documented at source-collection level, not per melody, which is why the
Finnish and Turkish analyses treat source collections and makam forms, respectively, rather than melodies as inferential units.
