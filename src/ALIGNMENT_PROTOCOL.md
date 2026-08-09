# Within-family alignment analysis — protocol (fixed before any alignment is computed)

STATUS: EXPLORATORY. This analysis measures POSITIONAL DIFFERENCES between
aligned vocal and instrumental variants of the same tune family. It makes
no directional claims: tune-family variants are relatives, not observed
adaptation events, and nothing in the data orders them in time. All
language of "change/insertion/replacement" below is shorthand for
side-specific differences under alignment, and will be reported as such.
The original question's directional sub-question (whether
instrumental-to-vocal adaptation is more constrained than the reverse) is
NOT answerable with these data and is replaced by a distributional
asymmetry comparison (Q6').

## Data and pairs

All 433 Dutch mixed tune families; all vocal x instrumental melody pairs
within a family (5,135 pairs; no selection by similarity). Analyses
aggregate pair-level metrics to FAMILY means (each family one unit,
regardless of its pair count), then report across-family means with
2,000-resample family-bootstrap 95% CIs. No melody excluded a priori.

## Alignment (parameters fixed now)

Global Needleman-Wunsch on MIDI pitch sequences after transposition of
the instrumental melody by c semitones, c in {round(median_v - median_i)
- 1, +0, +1}; the transposition with the best alignment score is kept.
Primary scoring: +2 exact pitch match, 0 if |difference| <= 2 semitones,
-1 otherwise; linear gap penalty -1.5. Prespecified sensitivity scoring:
+1 match / -1 mismatch / -1 gap. Alignment quality = fraction of exact
matches over min(length); reported for all pairs; all metrics computed on
(a) all pairs and (b) pairs with quality >= 0.5, with the excluded pairs
counted — low-quality pairs are the most-transformed ones, so (b) is a
conservative view, not the primary one.

## Metrics (all reported as family-mean, across-family mean [95% CI];
asymmetries are sign-symmetric under side relabeling, so a CI excluding
zero indicates a medium-linked asymmetry)

Q1a Leap-step correspondence: over consecutive aligned anchors, count
transitions where the instrumental interval is a leap (>=5 st) while the
vocal interval is a step/repeat (<=2 st), minus the reverse, per anchor
transition.
Q1b Repetition insertions: among vocal-side unaligned notes (insertions),
the share whose pitch equals the preceding vocal note; same for
instrumental-side insertions; difference. (Text-setting prediction:
vocal insertions are disproportionately repetitions.)
Q2  Insertion rates per side (unaligned notes / aligned length) and net
length difference.
Q3  Vocal-specific gaps: aligned anchor transitions with a vocal rest
>= 0.5 beats but instrumental rest < 0.5. For each melody, mean length
(beats) of the uninterrupted vocal run preceding vocal-specific gaps vs
preceding matched non-gap transitions; paired difference.
Q4  Register: slope of (instrumental - vocal) aligned pitch difference
(post-transposition) on the vocal note's within-melody pitch percentile.
Positive slope = the instrumental side is relatively higher where the
melody is high, i.e. expansion concentrated at the top.
Q5  Location of vocal-specific gaps: beatstrength of the preceding vocal
note relative to that melody's mean beatstrength; and coincidence with
annotated vocal phrase_end vs the melody's phrase_end base rate.
CAVEAT: phrase annotations were excluded from the paper's signature
analyses; they enter here descriptively and inherit encoding conventions.
Q6' Inserted-material structure: for instrumental-only vs vocal-only
insertions, mean absolute interval to the preceding same-side note and
repetition share. A difference is a distributional asymmetry, not a
statement about adaptation direction.

One run under the primary scoring; the sensitivity scoring is run once,
after, and reported regardless of outcome.
