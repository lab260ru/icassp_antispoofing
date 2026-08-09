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

## 2026-08-09 — interrupted quality screen and corrected successor

The detector-free `h2_quality_full_001` run was intentionally stopped after
133 of 4,000 pair checkpoints. It exposed a gate implementation error: requiring
both source and transformed clipping fractions to be zero rejected source clips
that were already peak-clipped, even when the registered polarity or +0.1 dB
control added no clipping. No detector was loaded or scored. The preserved HDD
checkpoints and transcript cache for this run are diagnostic artifacts only and
must not enter pass-rate, paired-score, or causal analysis.

Commit `eb6298c` corrects the condition to allow only non-positive
transform-induced clipping while retaining both raw fractions. The separate
`h2_quality_full_002` run on GPU 3 now performs the same detector-free,
pre-score quality screening under that corrected protocol. It is not a quality
result until the complete manifest is frozen; detector scoring remains barred.
