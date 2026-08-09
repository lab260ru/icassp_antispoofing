# Implementation Notes

## 2026-08-09 - ICASSP anti-spoofing autoresearch bootstrap

- Decision: Treat published SpeechAntiSpoofingBenchmarks Arena scores as authoritative inputs; analyze them but do not spend compute reproducing them.
- Decision: Use public core datasets first and reserve In-the-Wild and ASVspoof 5 as held-out confirmation data for feature selection made on the discovery datasets.
- Decision: Use one independent job per RTX 6000 Ada GPU; newly trained models use BF16 and per-backbone throughput measurement.
- Tradeoff: The current ICASSP 2027-specific kit is not yet available, so the initial manuscript will use the generic IEEE conference template and be migrated once the official kit is published.
- Constraint: Telegram, GitHub, and Hugging Face runtime credentials are not currently configured. Sensitive values are intentionally not retained here.
- Changed: The live Arena manifest pins dataset revisions that differ from the current heads of several dataset repositories. The catalogue therefore uses Arena-pinned revisions, not repository heads, for every audio/score join.
- Validation: Installed the public audio/analysis runtime; CUDA ONNX Runtime exposes TensorRT, CUDA, and CPU providers. The 28-feature synthetic-signal test passes after fixing the registry emission order.
- Validation: The first authoritative score artifact joined 100% of ASVspoof2019 LA trials by utterance ID; model-score polarity is normalized from labels and retained in the result metadata.
- Validation: The 200-utterance feature pilot produced three views per sample with no duplicate keys. Voice-quality measures are genuinely unavailable on some unvoiced examples, so downstream analysis preserves a missingness indicator.
- Validation: The score downloader now uses byte-validated `curl` range-resume transfers after the Hub client's compressed-transfer path failed. It completed all 72 score/result artifacts for the locked five-dataset, eight-model core panel. The ASVspoof2019_LA parquet panel has 569,896 rows and no duplicate `(model, sample_id)` keys.
- Validation: Full ASVspoof2019_LA extraction completed from the frozen 5,000-per-class manifest: 30,000 rows across three views, no duplicate `(sample_id, view)` keys. Dataset metadata includes 67 speakers but no attack identifier; its control column is deliberately represented as missing, not inferred.
- Guardrail: The first association invocation stopped before output when integrated loudness was both an outcome feature and a partial-correlation control, creating a duplicated column. The corrected estimator will exclude a tested variable from its own controls and receive a regression test before the same locked inputs are rerun.
- Validation: The corrected ASVspoof2019_LA H1 screen produced 1,344 exact-join tests (8 models × 2 classes × 3 views × 28 features). Spectra-AASIST spoof scores show large negative crest-factor partial correlations across the three registered views, while signs and sizes vary across architectures. The result is explicitly discovery-only.
- Guardrail: `partial_spearman` now removes the test/response variables from requested controls and deduplicates controls. A separate confirmation mode accepts only an explicit frozen candidate manifest, records its SHA-256, and uses deterministic label-stratified speaker/source-cluster bootstrap CIs (2,000 by default). Focused tests pass (`6 passed`).
- Follow-up: Finish the two remaining discovery datasets, aggregate only under the locked portability criterion, freeze candidates, and only then run held-out confirmation and bootstrap CIs.
