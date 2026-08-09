# H2 analysis record

## 2026-08-09 — pre-execution checkpoint

The discovery-to-H2 candidate manifest freezes only the
Spectra-AASIST/`crest_factor_db` family before confirmation data are read. The
two ONNX runners have baseline-order parity, and the waveform transforms plus
ASR/WER runtime are implemented. No paired intervention manifest, waveform
quality table, transformed-audio detector score, bootstrap interval, or causal
result exists yet.

The next permissible result-producing step is to freeze a score-independent
pair manifest, apply only the candidate-relevant crest-factor arms plus the
registered controls, evaluate all waveform quality gates, and retain the
failures. Detector scoring may begin only after that immutable quality manifest
exists.
