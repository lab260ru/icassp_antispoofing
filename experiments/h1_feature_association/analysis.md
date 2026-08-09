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

## run_003 — complete eight-model score panel

**Type:** input-integrity validation; not a test of H1.

- The pinned ASVspoof 2019 LA score panel contains one normalized
  spoof-evidence score per published trial for each of the eight prespecified
  models: 569,896 model--trial rows, 71,237 trials per model, and no duplicate
  `(model, sample_id)` keys.
- This makes the eight-model panel available without recomputing Arena
  baselines. The source score files and result metadata remain on the HDD
  volume; the compact catalog is versioned in the repository.

## run_004 — locked full discovery feature extraction

**Type:** input-integrity validation; prerequisite for H1.

- Input: the deterministic ASVspoof 2019 LA discovery subset, seed 2609,
  5,000 trials per class.
- Output: 30,000 rows: 10,000 unique trials by full-waveform,
  deterministic-crop, and pre-emphasized-crop views.
- Integrity: class balance and all `(sample_id, view)` keys are exact. The
  dataset exposes 67 speaker IDs but no attack ID. The latter is represented
  as one explicit missing metadata category; it is never inferred from labels.

## run_005 — first discovery association screen

**Type:** exploratory discovery screen under the locked H1 estimator; not a
portable or causal conclusion.

- Inputs: the run_004 features joined by exact IDs to every model's run_003
  score panel. Each model has 30,000 joined feature-view rows (10,000 trials
  by three views).
- Output: 1,344 tests = 8 models $\times$ 2 within-label strata $\times$ 3
  views $\times$ 28 features. BH adjustment is applied across this complete
  screen; 1,004 partial-association rows have $q<0.05$, so effect size and
  cross-corpus replication, rather than p-value count, govern follow-up.
- Crest-factor check: for spoof trials, Spectra-AASIST has partial Spearman
  $\rho=-0.454$ (full waveform), $-0.445$ (deterministic crop), and $-0.327$
  (pre-emphasized crop); each has BH $q<5\times10^{-122}$. The same feature is
  strong and negative in particular other architectures (for example,
  RawBMamba and XLSR-SLS) but weak, null, or positive in others depending on
  class and view. It is therefore an architecture- and representation-specific
  discovery pattern, not evidence of a universal crest-factor shortcut.
- Guardrail: the first invocation detected that integrated loudness was both a
  candidate feature and a requested control. It stopped before writing output.
  The rerun excludes the tested variable from its own control design and is
  covered by focused regression tests (`6 passed`).

Next: complete the two remaining discovery datasets, aggregate only under the
pre-registered portability rule, then freeze a provenance-bearing candidate
manifest before touching either held-out confirmation corpus or bootstrap CIs.
