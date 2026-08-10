# H6 analysis 001 — within-class published-score agreement atlas

Status: complete, exhaustive descriptive matrix. H6 reads only the sealed
label freeze and 40 sealed original raw-score text artifacts; it emits no raw
score values and reads no waveform, feature, model, ASR, detector, or
prior-result artifact.

The frozen-manifest-only analyzer revalidated the protocol, Arena index, all
label and score byte identities, and the independent seed-2611 selection before
parsing score text. It then materialized exactly all
`5 corpora x 2 classes x 28 unordered model pairs = 280` registered cells,
each with 5,000 exact selected-ID joins and 200 fixed sample-cluster bootstrap
attempts.

## Registered result

- The 280-cell matrix is complete: every cell has status `ok`; there are no
  unavailable joins, duplicate IDs, orientation failures, substitutions, or
  intersection reselections.
- The 28 ten-cell pair summaries are complete. The distribution of the 280
  within-class Spearman agreements has median 0.332477 and range
  −0.629261 to 0.868770. Median agreement across the 28 fixed model pairs
  ranges from 0.029535 (Spectra-AASIST/Res2TCNGuard) to 0.652363
  (Res2TCNGuard/RawBMamba).
- Cross-corpus/class heterogeneity is substantial in the fixed registry: the
  largest per-pair ten-cell range is 1.486509 for AASIST/Res2TCNGuard. The
  lowest cell is InTheWild, label 0, Res2TCNGuard/RawBMamba
  (rho −0.629261; 95% bootstrap CI [−0.646543, −0.611638]); the highest is
  ASVspoof2021_DF, label 0, AASIST/Res2TCNGuard
  (rho 0.868770; 95% CI [0.861255, 0.876059]).

These are descriptive within-class rank-agreement measurements for the sealed
published artifacts. They do not establish model independence, a quality
ordering, feature reliance, causal sensitivity, or any conclusion about the
failed H1/H2/H2B/H3/H4/H5 gates.

## HDD ledger

All large outputs remain under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001/`.

| Artifact | SHA-256 |
|---|---|
| `h6_within_class_score_agreement_matrix.csv` | `105e99fca829af8bf1600fa73374ddcde0f4121b6716a8beff0b1fd503eec2e5` |
| `h6_model_pair_agreement_summary.csv` | `294faa27e8e9e7cce4292c0b5f9e05bbfb2667e88e1bb719113f81914a0326e8` |
| `h6_analysis_provenance.json` | `3e9d4f628fdffea5430859db589c32dce3ce5e9ce7c9b1b2448928ae4bd433d8` |
| H6 label-selection manifest | `159351198de98f4a3821495a3e40012a2cd152e6a2d7f796c36f910a6b907c39` |

The provenance records `complete_280_cell_matrix: true`, no failed cell,
`raw_score_only: true`, and `score_values_emitted: false`. H6 is now sealed:
do not tune the sample cap, seed, orientation threshold, model/dataset registry,
pair order, bootstrap count, or interpretation after seeing this result.
