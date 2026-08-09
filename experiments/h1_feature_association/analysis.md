# H1 analysis

## run_001 — score-artifact and ID-join validation

**Type:** input-integrity validation; not a test of H1.

- Dataset: `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` at
  `9492c4a85ad91508b6da03c92c98c58aeaa02424`.
- Model artifact: `lab260/Spectra-AASIST` at
  `eb65c2662d9e646d72557b3f4bdd08b000068c7f`.
- Result: all 71,237 published scores joined one-to-one with authoritative
  labels; 7,355 bonafide and 63,882 spoof trials.
- Score semantics: the raw score is increasing bonafide evidence, so the
  canonical analysis variable is its negation (`score_spoof`).

This validates exact utterance-ID joins and score orientation without rerunning
Spectra-AASIST. It does not yet support a feature-reliance or causal claim.

Next: complete the 200-utterance audio-feature pipeline pilot, then ingest the
remaining published score panel and run the locked full H1 extraction.

## run_002 — feature-pipeline pilot

**Type:** implementation validation; not a test of H1.

- Input: deterministic 100-per-class ASVspoof 2019 LA sample selected solely
  from the pinned labels table.
- Output: 600 feature rows, covering all 200 samples across the full-waveform,
  deterministic-crop, and pre-emphasized-crop views.
- Integrity: no duplicate `(sample_id, view)` keys; all non-voice features had
  zero missingness; voice-quality features were missing on 5% of records, which
  is retained as missing rather than imputed.

The data schema, FLAC decoding, resampling, view generation, and frozen
feature registry are operational. Full confirmatory ASVspoof 2019 LA extraction
can begin without changing H1 definitions.
