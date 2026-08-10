# H5 supplementary figure protocol — sealed five-corpus median concordance

**Status:** locked on 2026-08-10 after H5 analysis was sealed and before any
H5 supplementary-figure input is read, rendered, or inspected. This is a
display-only protocol for a completed H5 atlas. It does not create a new
statistic, hypothesis, selection, or result.

## Immutable input boundary

The renderer may read only these two compact outputs from the completed,
hash-revalidated `h5_analysis_001` directory:

| Artifact | Required absolute path |
|---|---|
| H5 matrix | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/h5_view_invariance_matrix.csv` |
| H5 aggregation | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/h5_view_invariance_aggregation.csv` |

It derives the sibling `h5_analysis_provenance.json` only to revalidate both
literal input paths, byte sizes, and SHA-256 values. The renderer rejects an
alias, a different run directory, a changed input, a response-like path, or a
CSV header containing label/source/score/logit/detector/model/EER/Arena/audio/
ASR/H1/H2/H2B/H4/result/summary-like text. It has no option to receive a
feature table, waveform, label, score artifact, model, ASR path, or prior
result.

## Locked display

The figure contains exactly three panels, in the H5 protocol's fixed
lexicographic view-pair order:

1. `deterministic_crop` / `full_waveform`;
2. `deterministic_crop` / `preemphasized_crop`;
3. `full_waveform` / `preemphasized_crop`.

Each panel contains the 28 registered features in registry order, for exactly
84 predeclared aggregation units. One cell per row encodes the sealed
five-corpus **median Spearman concordance**, not a newly calculated or ranked
quantity. The `viridis` colorblind-safe scale is fixed to `[0, 1]` across every
panel. No sorting, filtering, threshold sweep, top-k display, rescaling, or
cross-experiment comparison is permitted.

If an aggregation unit has any unavailable/failed dataset cell, the renderer
uses a light-gray cell overlaid with a black `×`; it never imputes a value. A
small green square marks only the already-sealed
`view_stable_terminal_descriptive` flag. This marker is descriptive and
terminal: it is not a ranking, feature-selection, training, detector, or causal
signal.

## Validation and outputs

Before drawing, the renderer requires the exact H5 matrix and aggregation CSV
schemas, 420 unique dataset/view-pair/feature rows, 84 unique aggregation
units, the locked five datasets, three view pairs, 28 features, exact matrix ↔
aggregation unit agreement, and hash-valid provenance. It recomputes only
integrity identities already carried in the aggregation (availability, median,
and terminal flag) to reject mismatched/tampered compact files; it does not
estimate a new research quantity.

The output directory must be new and contains only:

- `h5_view_invariance_median_concordance_heatmap.pdf` — vector PDF;
- `h5_view_invariance_median_concordance_heatmap.png` — 300-DPI PNG;
- `h5_view_invariance_median_concordance_heatmap.metadata.json` — input/output
  identities, fixed display configuration, and figure hashes.

The figure is supplementary only. It may state agreement/disagreement of H5
feature measurements across registered waveform views, but cannot infer
preprocessing effects, label leakage, detector sensitivity, causality, corpus
quality, or support any H1--H4/H2B/H3 decision.
