# Main-paper Figure 1 — crest evidence boundary

The current manuscript uses the reviewed **v2** asset
`fig_crest_evidence_boundary_main_v2.pdf`. It is a display-only rendering of
the already sealed H1 frozen crest slice and H2 detector-free quality summary;
it computes no test, interval, pooled result, ranking, or causal estimate.

## Input and visual contract

The v1 contract locks only five H1 association summaries, two discovery-frozen
held-out confirmation rows, and one H2 quality summary. H4/H5/H6/S1, raw
scores, models, audio, ASR, H2B, and feature tables are excluded. The layout
only v2 contract additionally pins the exact v1 parser and eight-input manifest
hashes before it opens those sources.

| Item | Location / SHA-256 |
| --- | --- |
| v1 protocol | `experiments/paper_extension/main_crest_evidence_boundary_protocol.md` |
| v2 layout protocol | `experiments/paper_extension/main_crest_evidence_boundary_v2_protocol.md` |
| v1 source manifest | `experiments/paper_extension/main_crest_evidence_boundary_input_manifest.json`, `10d6c6cdc0a694f8d2c93a67a3c69714f2ff98c7463da20570c80f43426b2501` |
| preserved v1 vector / PNG / metadata | `f545dbed4a7d03e635c8d35500d6590d207a18d73577d8a0c7a4e5b0c677f50b` / `12699310824d5cf257819bd9824fc6d6c40e58bf21afe61493034efee79eef67` / `08fa66e478c35cc7ba891c3e0f893e34eae4372f56010650f2943c4919e2560d` |
| authoritative v2 vector / PNG / metadata | `a308cd826f913bc3192e25690dc9a0c8cbd75a585c5afd6074af78edc23bc65c` / `9863288acd890c5ad0ad9f06a167be79c949869fdb8731e994a2d885ab824155` / `0a559b5c73f9aad1e6fc5ecebaf1ee4c43f22116b08e319028ace61581a1d9c9` |

V2 preserves the fixed H1 corpus order, H2 arm order, values, held-out
intervals, and scales from v1. It only: (1) makes the three discovery and two
held-out H1 rows visibly distinct; (2) names H2 as the one-corpus,
1,000-clip ASVspoof2019 LA detector-free screen; (3) calls out the absence of
a score-eligible retained-pair manifest; and (4) gives the individually
passing polarity control a neutral rather than affirmative colour.

The visual inspection and final v2 verdict are recorded in
`paper/reviews/codex_main_figure_20260810/v2_verification.md`. The v1 files
remain preserved as an inspected iteration; do not overwrite either asset.
