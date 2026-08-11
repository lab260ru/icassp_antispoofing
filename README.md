# Same-item pair ranking in speech deepfake detection

This repository is the complete research record for an ICASSP working draft on
controlled cross-corpus speech deepfake detection. It began with a signal-cue
audit around `lab260/Spectra-AASIST` and published
SpeechAntiSpoofingBenchmarks models, then pivoted through a sealed independent
protocol to its current H9 paired-ranking study.

The current working title is **Same-Item Pair Ranking for Cross-Corpus
Speech Deepfake Detection**. The paper reports a narrow controlled result:
under one fresh Res2TCNGuard/ODSS protocol, documented same-item ranking
outperforms same-pool BCE and an equal-budget random-pair ranking control on
two fixed external targets. It makes no state-of-the-art,
architecture-general, causal-representation, or blind-evaluation claim. The
failed crest study remains preserved as prior research history, not paper
evidence. Read `AGENTS.md` before extending any result.

Start with `AGENTS.md`, then read `research-state.yaml`, `findings.md`, and the
relevant experiment protocol. Human-readable artifacts are versioned here;
datasets, models, checkpoints, and full logs live under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/` and are referenced by pinned
manifests and checksums.

For a compact map from every research loop to its protocol, result note, HDD
ledger, paper artifact, and external continuation condition, read
`ARTIFACT_INDEX.md`.

## Current objective

Through 2026-08-11 13:00 UTC, maintain the H9 initial ICASSP paper draft,
reproducible experiment artifacts, and a Codex-only internal review bundle.
Existing Arena scores are treated as authoritative and are not reproduced with
new inference.
