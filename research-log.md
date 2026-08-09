# Research Log

Chronological, append-only record of decisions and results.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-09 | bootstrap | Created the reproducible workspace, durable research objective, protocol-first experiment structure, HDD storage registry, and credential-safe communication queue. Arena scores are designated authoritative inputs rather than targets for recomputation. |
| 2 | 2026-08-09 | inner-loop | H1 run_001 validated the first pinned score artifact: all 71,237 ASVspoof2019 LA labels joined exactly to Spectra-AASIST scores. Raw scores increase bonafide evidence and are negated for a consistent spoof-evidence analysis variable. |
| 3 | 2026-08-09 | inner-loop | H1 run_002 feature-pipeline pilot extracted all 28 frozen descriptors over three waveform views for a deterministic 200-utterance balanced sample. No duplicate keys; only expected unvoiced-record voice-quality missingness (5%). |
| 4 | 2026-08-09 | infrastructure | Downloaded and validated 72 immutable Arena score/result artifacts (8 models × 5 core datasets). The complete ASVspoof2019 LA score panel contains 569,896 unique model–utterance rows, with one normalized spoof-evidence score per published trial and no duplicate model–sample keys. |
| 5 | 2026-08-09 | inner-loop | H1 run_004 completed locked full feature extraction for ASVspoof2019 LA: 5,000 bona fide and 5,000 spoof trials, each represented by full-waveform, deterministic-crop, and pre-emphasized-crop views (30,000 rows). All `(sample_id, view)` keys are unique; `attack_id` is absent in this dataset and is retained as an explicit missing control. |
| 6 | 2026-08-09 | analysis-guard | The initial association screen halted before writing results because testing integrated loudness duplicated a partial-correlation control column. The design now removes a tested variable from its own control set; no estimates from the failed attempt are retained. |
