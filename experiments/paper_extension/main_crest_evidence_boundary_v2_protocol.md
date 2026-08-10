# Main-paper Fig. 1 v2 protocol — layout-only clarification

**Status:** drafted after three independent Codex-only reviews of the rendered
v1 figure and before any v2 renderer or output is created. This is a
non-overwriting, layout-and-wording-only successor to
`main_crest_evidence_boundary_protocol.md`; it cannot alter the eight sealed
inputs, values, confidence intervals, arm order, scales, or research claims.

## Reason for a separate version

Reviewers identified two reader risks in v1: its shared ``across five corpora``
title could appear to apply to the H2 panel, which is a single 1,000-clip
ASVspoof2019 LA detector-free quality screen; and Panel A did not visually
separate the three discovery from two held-out confirmation rows. A green
polarity bar could also be misconstrued as authorizing scoring when the full
four-arm panel failed.

The v1 PDF/PNG/metadata remain preserved. Version 2 is a new output with the
following and only the following presentation changes:

1. the figure-level title no longer says ``across five corpora``;
2. Panel A identifies its five-corpus scope and separates the locked discovery
   rows from held-out confirmation rows with a fixed horizontal divider and
   fixed group labels;
3. Panel B names the ASVspoof2019 LA, 1,000-clip detector-free screen, states
   that no score-eligible retained-pair manifest was frozen, and uses neutral
   gray for the individually passing polarity control; and
4. the paper caption will say that polarity alone clears the per-arm rate but
   cannot authorize scoring while the full four-arm panel remains unfreezable.

## Immutable dependency contract

Version 2 must call the already committed v1 loader only after verifying both
the v1 source-module SHA-256 and its eight-source input-manifest SHA-256
against literals in the v2 renderer. Thus it inherits exactly these sources and
nothing else: five H1 association summaries, two H1 confirmation rows, and one
H2 detector-free quality summary. It must not open H4/H5/H6/S1/H2B, score
artifacts, models, audio, ASR, or raw feature tables.

The corpus order remains `ASVspoof2019_LA`, `ASVspoof2021_LA`,
`ASVspoof2021_DF`, `InTheWild`, `ASVspoof5`. The H2 arm order remains
DRC-3, DRC-6, +0.1 dB gain, polarity; Panel A's horizontal scale remains
`[-0.55, 0.08]`; Panel B's scale remains `[0, 103]` percent; and the gate
remains 90%. The exact values are copied from the sealed source only and no
test, bootstrap, pooled statistic, rank, threshold, or selection is computed.

## Outputs and stop rule

After the v2 code and this protocol are committed, an explicitly authorized
render may write only the following new, non-overwritable assets:

- `paper/figures/fig_crest_evidence_boundary_main_v2.pdf`;
- `paper/figures/fig_crest_evidence_boundary_main_v2.png`; and
- `paper/figures/fig_crest_evidence_boundary_main_v2.metadata.json`.

If the v1 module/manifest hash, any source hash/schema, or visual inspection
fails, stop. Do not replace v1, alter input scope, or turn the control bar into
a causal or model-performance claim. A `main.tex` edit is permitted only after
v2 passes its sealed input, file, and visual checks; it must immediately be
compiled with the refreshed `paper/build/main.pdf` committed in the same paper
change.
