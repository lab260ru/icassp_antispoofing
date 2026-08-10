# Signal-feature causality in speech anti-spoofing

This repository is the complete research record for an ICASSP 2027 draft on
cross-dataset signal cues in speech anti-spoofing detectors, beginning with
`lab260/Spectra-AASIST` and extending to published SpeechAntiSpoofingBenchmarks
models and datasets.

The current evidence-conservative title is **Beyond Crest Factor: A
Five-Corpus Audit of Signal Associations in Speech Anti-Spoofing**. The paper
reports a failed portability test for one discovery-frozen crest slice and a
quality-gate stop before detector scoring; it makes no causal or mitigation
claim. The title and main claim may change only through the documented
outer-loop pivot policy in `research-state.yaml`.

Start with `AGENTS.md`, then read `research-state.yaml`, `findings.md`, and the
relevant experiment protocol. Human-readable artifacts are versioned here;
datasets, models, checkpoints, and full logs live under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/` and are referenced by pinned
manifests and checksums.

For a compact map from every research loop to its protocol, result note, HDD
ledger, paper artifact, and external continuation condition, read
`ARTIFACT_INDEX.md`.

## Current objective

Through 2026-08-10 20:00 UTC, maintain an evidence-backed ICASSP-format
initial paper draft, reproducible experiment artifacts, a Codex-only internal
review bundle, and a prioritized continuation roadmap. Existing Arena scores
are treated as authoritative and are not reproduced with new inference.
