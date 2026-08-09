# Positional differences between aligned vocal and instrumental variants

EXPLORATORY, per `ALIGNMENT_PROTOCOL.md` (committed before computation).
No directional claims: these are side-specific differences under
alignment within tune families, not observed adaptations. Script:
`alignment_analysis.py`; per-pair table: `alignment_pair_metrics.csv`;
raw aggregates: `alignment_results.json` (primary scoring) and
`alignment_results_sensitivity.json`. All estimates are across-family
means of family means (each of the 433 families one unit), 95%
family-bootstrap CIs, 2,000 resamples. "hq" = pairs with alignment
identity >= 0.5 (3,682 of 5,135 pairs; the excluded third are the
most-transformed pairs, so hq is the conservative view).

## Feasibility

All 5,135 cross-medium pairs aligned (0 skipped); median identity 0.70,
family-mean 0.77. The within-family alignment premise holds: variant
pairs share a recoverable common skeleton.

## Findings by question

**Q1a — leaps are NOT locally replaced by steps.** The leap-step
substitution asymmetry at aligned anchors is null: +0.0006 per anchor
transition [-0.0018, +0.0032] (hq: +0.0017 [-0.0000, +0.0034]). The
corpus-level interval differences between media do not arise by
substituting a step where the relative has a leap at the same position.

**Q1b — vocal-side insertions are disproportionately repeated pitches.**
9.1% of vocal-side unaligned notes repeat the preceding pitch vs 5.3% of
instrumental-side ones; paired family asymmetry +0.026 [+0.012, +0.040].
This is the note-splitting signature predicted by the text-setting
account of repetition, observed mechanically at alignment level.

**Q2 — vocal variants carry more unaligned material and more
side-specific gaps.** Insertion rates: vocal 0.243 vs instrumental 0.150
per anchor; net length difference +3.0 notes [+0.1, +6.7].
Vocal-specific gaps (vocal rest >= 0.5 beats where the aligned
instrumental transition has none) exceed instrumental-specific gaps by
+0.0064 per anchor transition [+0.0032, +0.0098] — the alignment-level
counterpart of the corpus Δgap.

**Q3 — vocal-specific gaps follow long uninterrupted runs, but the
estimate is inflated by construction.** The run preceding a
vocal-specific gap is +3.9 beats longer [+2.7, +5.2] than before non-gap
transitions. CAVEAT (do not quote the magnitude): part of this is an
inspection artifact — the run before a gap is by definition a complete
inter-gap segment, while the run before a random non-gap transition is a
partial one, so even random gap placement produces a positive value. A
confirmatory version needs a within-melody permutation null for gap
positions. Direction is suggestive; magnitude is not interpretable as
reported.

**Q4 — register: treat the anchor-level slope as unreliable; the
extreme-based answer stands.** The aligned-position slope is negative
(-1.45 [-1.63, -1.28]; hq -0.90), nominally meaning the instrumental
side is relatively higher where the vocal melody is low and lower where
it is high. This statistic regresses a difference on one of its
components and is therefore contaminated by regression-to-the-mean; the
sign cannot be trusted. The transposition-robust answer to Q4 remains
the extreme-based one from the range analysis: the instrumental side
sits higher at both ends (top +1.8 st, bottom +1.3 st) with net widening
of ~0.5 st, and — combined with Q2 — the extra instrumental compass
lives substantially in unaligned (inserted) material rather than in
shifted versions of shared notes.

**Q5 — vocal-specific gaps sit at strong metric positions and at
annotated phrase ends.** Beatstrength of the note preceding a
vocal-specific gap exceeds the melody mean by +0.20 [+0.17, +0.23], and
coincidence with annotated vocal phrase ends exceeds the base rate by
+0.60 [+0.55, +0.63]. Vocal-specific gaps are overwhelmingly
phrase-boundary events. CAVEAT: phrase annotations inherit encoding
conventions and were excluded from the paper's signature analyses; this
is descriptive.

**Q6' — inserted material has the same interval structure on both
sides.** Mean absolute interval of inserted notes: vocal 2.48 vs
instrumental 2.53 st (CIs overlap almost completely). The sides differ
in *how much* they insert (Q2) and in the repetition share of what they
insert (Q1b), not in the interval size of inserted material. The
directional "which adaptation is more constrained" question remains
unanswerable with these data.

## Sensitivity

The prespecified alternative scoring (+1/-1/-1) is run once and reported
in `alignment_results_sensitivity.json` regardless of outcome; see the
results files for side-by-side values.

## One-paragraph synthesis (descriptive)

Within families, the vocal and instrumental variants share a
well-recoverable melodic skeleton (median identity 0.70). The media do
not differ by substituting intervals at shared positions; they differ in
what is added around the skeleton and where flow is broken: the vocal
side adds more material, its additions are more often repeated pitches
(the text-setting signature), and it breaks continuity at phrase
boundaries on strong beats — after long uninterrupted runs, though that
last magnitude is partly mechanical. The instrumental side's wider
compass is carried more by its inserted figuration than by re-pitching
of shared notes. All of this is exploratory and non-directional.
