# H5 supplementary concordance figure 001

Status: rendered and visually inspected display-only supplementary artifact.
It uses only the two hash-validated compact H5 output tables prescribed in
`H5_SUPPLEMENTARY_CONCORDANCE_FIGURE_PROTOCOL.md`; it computes no new research
statistic and changes no H5 result or decision.

The authoritative rendering is the layout-corrected `v2` directory below. An
earlier `figures_median_concordance` rendering is retained on the HDD as an
implementation artifact after visual inspection found colliding panel headings;
it has the same sealed inputs and encodings but is not the recommended display.
The `v2` rerender changes only panel-title line wrapping and available canvas
width, preserving every input hash, feature/view order, fixed `[0, 1]` viridis
scale, unavailable marker, and terminal-descriptive marker.

## What it displays

Three fixed panels show all 84 predeclared H5 view-pair/feature aggregations in
registry order. Each cell is the already-sealed five-corpus median Spearman
concordance. Light-gray `x` cells mark the six unavailable aggregations; green
squares mark the 17 already-sealed terminal descriptive units. Neither marker
is a ranking, selection, training, detector, or causal signal.

## HDD ledger

Authoritative outputs are under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/figures_median_concordance_v2/`.

| Artifact | SHA-256 |
|---|---|
| vector PDF | `f011d2ab951423d46be015057dc9524cac573b19bab6303e83d27def861e9f24` |
| 300-DPI PNG | `3733f55e7b3972a511859071d1d8c31a500ded7136e2319a4bf19124b36d8eaa` |
| metadata JSON | `57699bdebb7c3f2adf807dc802890444f1420768acddf719269d0a417cbdb45c` |
| H5 matrix input | `a6ea23155943aa38327b5c65da0422c6677741c8d678581ae1b99e93d1fd8438` |
| H5 aggregation input | `f1cb7c85ad291ab6fcaac0fc5d25c91787b6accc9ca8e33ed6f6fd613216bdb6` |

The visual inspection record is
`H5_SUPPLEMENTARY_CONCORDANCE_FIGURE_001_VISUAL_INSPECTION.md`. This figure is
supplementary only and must not be used to revise a score, label, causal, or
feature-selection conclusion.
