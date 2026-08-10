# H6 supplementary figure 001 — within-class agreement atlas

Status: rendered and visually inspected display-only supplementary artifact.
It consumes only the three literal, hash-validated compact outputs sealed by
`H6_ANALYSIS_001.md`, following
`H6_SUPPLEMENTARY_FIGURE_PROTOCOL.md`. It does not parse a raw score value,
open an audio/model/feature artifact, compute a new statistic, or change an H6
decision.

## What it displays

The fixed 28-row by 10-column heatmap displays every registered H6
`model-pair × (corpus, class)` Spearman agreement cell. Rows follow the
registered unordered model-pair order; columns follow the registered five
corpora with bona fide then spoof class; neither is sorted, filtered, ranked,
or highlighted by a result. The blue--neutral--vermillion scale is fixed to
`[-1, 1]` rather than fitted to the observed range.

This is descriptive context for variation among the fixed published-score
artifacts. It is not a detector-quality ranking, architectural-independence
measurement, cue-reliance result, causal sensitivity result, model-selection
criterion, or basis for changing H1/H2/H2B/H3/H4/H5/H6.

## HDD ledger

Outputs are in
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/figures_within_class_agreement_v1/`.

| Artifact | SHA-256 |
| --- | --- |
| vector PDF | `c6b43ef6eaf0455ce33b42fd2bd78cc96e42cb86d48646d8d83516ad744ce939` |
| 300-DPI PNG | `ed9185df9b85486a091a701c78522a01b1cdbce7e80dee1e7b404ba7b30fb0bb` |
| metadata JSON | `c531ad89db9f5dddba55b0c8be98bca6e6c7090e3fe98a91e7f946a0daf3f600` |
| sealed 280-cell matrix input | `105e99fca829af8bf1600fa73374ddcde0f4121b6716a8beff0b1fd503eec2e5` |
| sealed 28-summary input | `294faa27e8e9e7cce4292c0b5f9e05bbfb2667e88e1bb719113f81914a0326e8` |
| sealed analysis provenance | `3e9d4f628fdffea5430859db589c32dce3ce5e9ce7c9b1b2448928ae4bd433d8` |

The visual inspection record is
`H6_SUPPLEMENTARY_FIGURE_001_VISUAL_INSPECTION.md`. The source matrix and
summary remain immutable; do not rerender into this output directory or alter
the display order, scale, or boundary wording after inspection.
