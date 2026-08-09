# Pending notifications

## 2026-08-09 — bootstrap

Research bootstrap is complete: the workspace, reproducible protocols, HDD
registry, and four-GPU plan are ready. Public Hugging Face access is being
tested next. Telegram delivery is queued because `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` are not currently present in the runtime environment.

## 2026-08-09 — first validated artifact

The Arena catalogue is pinned and public downloads work. ASVspoof 2019 LA is
downloaded at its Arena revision; all 71,237 published Spectra-AASIST scores
joined exactly to labels. The ongoing work is the audio-feature pilot and the
remaining model/dataset downloads. GitHub pushing remains queued until a valid
runtime token is available.

## 2026-08-09 — complete first score panel

All 72 pinned Arena score/result artifacts for the five core datasets are now
downloaded with byte-size validation. The first full analysis panel covers
eight detectors and 71,237 ASVspoof2019 LA trials per detector (569,896 score
rows); the CPU feature extraction is the remaining prerequisite for the first
association result. Delivery remains queued because the Telegram credentials
are not present in the runtime environment.

## 2026-08-09 — H1 discovery data ready

The locked ASVspoof2019 LA discovery subset is fully extracted: 10,000 balanced
utterances and three reproducible views (30,000 rows). Before creating any
association table, the analysis caught and stopped on a self-control edge case
for the loudness feature. The correction is under test; no provisional result
was kept from that failed invocation.

## 2026-08-09 — first H1 discovery result

The corrected ASVspoof2019 LA screen completed 1,344 within-class association
tests across eight published detectors, 28 signal descriptors, and three
waveform views. Crest factor is strongly negatively associated with spoof
evidence in Spectra-AASIST (and some other architectures), but it is not
directionally uniform across the panel. This is a promising heterogeneous-
sensitivity result, not yet a cross-dataset or causal shortcut finding. The
two remaining discovery corpora are now the priority before any candidate is
frozen.
