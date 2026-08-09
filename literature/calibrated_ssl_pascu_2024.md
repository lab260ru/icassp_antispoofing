# Generalisation and calibration with frozen SSL representations

## Verified source

Pascu et al., *Towards generalisable and calibrated audio deepfake detection
with self-supervised representations*, Interspeech 2024, pp. 4828--4832,
DOI: `10.21437/Interspeech.2024-1302`.

- Official record: <https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html>
- Verified 2026-08-09 from the official ISCA Archive metadata and abstract.

## What the source establishes

- The paper frames generalisation and calibrated predictions as practical
  requirements for audio deepfake detection.
- It reports that large frozen SSL representations with a simple logistic
  regression classifier substantially improve its eight-dataset benchmark
  relative to the named RawNet2 comparison.

## Paper implication

This is relevant to the conditional H3 direction, especially a low-parameter
late fusion or feature-conditioned head. It is not evidence that waveform
feature conditioning is beneficial; H3 must remain conditional on H2 and use
equivalent recipes, held-out EER, and calibration-aware reporting.

## Citation status

Verified for later bibliography use; not yet inserted into `paper/references.bib`.
