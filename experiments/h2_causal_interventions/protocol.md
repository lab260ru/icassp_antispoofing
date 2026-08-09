# H2 protocol — quality-gated causal interventions

**Status:** pre-score execution protocol locked; confirmatory interpretation is
conditional on the H1 candidate freeze, pair-quality gate, and scorer panel.

## Question and prediction

If a selected cue reflects detector reliance, a paired waveform transformation
that changes the cue while preserving speech content will shift detector scores
consistently across datasets and architectures.

## Inputs and panel

Use H1-frozen candidate cues, 500 examples per class per core dataset, and the
four-model intervention panel in `configs/study.yaml`. Do not choose examples
after viewing detector responses. The explicit score-independent input-freeze
and quality contracts are in `H2_PRE_SCORE_QUALITY.md`.

## Transformations

The only arms admitted for the current frozen
Spectra-AASIST/`crest_factor_db` candidate are:

- Loudness-matched dynamic-range compression targeting 3 and 6 dB crest-factor
  reduction (`drc_cf3`, `drc_cf6`).
- A fixed +0.1 dB gain and polarity reversal as amplitude/phase negative
  controls (`small_gain_plus_0p1db`, `polarity`).

The implementation registry also contains silence, tilt, and all-pass methods,
but those arms are not in this candidate's pre-score ledger and must not be
added after detector responses are inspected.

## Quality gates and metric

Retain a transformed example only if STOI is at least 0.95, transcript WER
against the original is at most 5%, loudness drift is at most 0.2 LU, and no
clipping occurs, and the registered target-feature direction holds. Preserve a
quality row for every failure. Drop an intervention arm if fewer than 90% pass.
No detector may read a pair before the quality table is frozen. Report paired
score deltas with clustered bootstrap intervals. Declare causal sensitivity only
under the pre-registered criterion in `configs/study.yaml`.

## Outputs

`results/intervention_manifest.parquet`, `results/paired_deltas.parquet`,
quality reports, plots, and `analysis.md`.
