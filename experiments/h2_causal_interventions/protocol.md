# H2 protocol — quality-gated causal interventions

**Status:** CONFIRMATORY conditional on the H1 candidate freeze.

## Question and prediction

If a selected cue reflects detector reliance, a paired waveform transformation
that changes the cue while preserving speech content will shift detector scores
consistently across datasets and architectures.

## Inputs and panel

Use H1-frozen candidate cues, 500 examples per class per core dataset, and the
four-model intervention panel in `configs/study.yaml`. Do not choose examples
after viewing detector responses.

## Transformations

- Loudness-matched dynamic-range compression targeting 3 and 6 dB crest-factor reduction.
- VAD-based leading/trailing silence standardization and silence equalization.
- Spectral tilt at plus/minus 3 dB per octave.
- Stable all-pass phase perturbation.
- Gain and polarity transformations as negative controls.

## Quality gates and metric

Retain a transformed example only if STOI is at least 0.95, transcript WER
against the original is at most 5%, loudness drift is at most 0.2 LU, and no
clipping occurs. Drop an intervention arm if fewer than 90% pass. Report paired
score deltas with clustered bootstrap intervals. Declare causal sensitivity only
under the pre-registered criterion in `configs/study.yaml`.

## Outputs

`results/intervention_manifest.parquet`, `results/paired_deltas.parquet`,
quality reports, plots, and `analysis.md`.
