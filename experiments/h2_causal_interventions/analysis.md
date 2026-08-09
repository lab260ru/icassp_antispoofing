# H2 analysis record

## 2026-08-09 — pre-execution checkpoint

The discovery-to-H2 candidate manifest freezes only the
Spectra-AASIST/`crest_factor_db` family before confirmation data are read. The
Spectra and AASIST ONNX runners, plus the exact source PyTorch Res2TCNGuard
evaluator, have baseline-order parity. The bundled Res2 ONNX graph is blocked
by its preserved mismatch diagnostic. The waveform transforms plus ASR/WER and
pre-score quality runtimes are implemented. No paired intervention manifest,
waveform quality table, transformed-audio detector score, bootstrap interval,
or causal result exists yet.

The next permissible result-producing step is to freeze a score-independent
pair manifest, apply only the candidate-relevant crest-factor arms plus the
registered controls, evaluate all waveform quality gates, and retain the
failures. Detector scoring may begin only after that immutable quality manifest
exists.
