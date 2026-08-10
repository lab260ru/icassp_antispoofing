# H7 supplementary feature-transfer figure — protocol

## Purpose and scope

This display-only rendering visualizes the already sealed H7 five-cell
feature-only leave-one-corpus-out AUROC matrix. It computes no metric,
bootstrap, ranking, feature attribution, threshold, detector response, or
causal estimate. It cannot alter H1/H2/H2B/H3 or select a cue/training input.

## Exact inputs

The renderer must consume only these exact compact H7 artifacts, relative to
the repository root, after verifying their SHA-256 bytes:

| Artifact | SHA-256 |
| --- | --- |
| `results/h7_input_freeze_001/h7_selected_sources.csv` | `42592ee49a3867b86f41f83620d328272de4362b32649021028a4e1253fd194f` |
| `results/h7_input_freeze_001/h7_input_freeze_provenance.json` | `236a95a309d201b8b24276e1436d1c3043c3d5d20c6e0685e7b0aaa1503248ec` |
| `results/h7_analysis_001/h7_leave_one_corpus_out.csv` | `d8ab31532c7a3867477d3978cc7b3ebc1a9dbf6ef71e4b47086f9fc1dd541e79` |
| `results/h7_analysis_001/h7_analysis_provenance.json` | `9a044edf921ba732e50f0300c06a3ff6484edf1286f7092fb7e96fa652e32cc2` |

The input CSV must have exactly five rows in the fixed corpus order
ASVspoof2019_LA, ASVspoof2021_LA, ASVspoof2021_DF, InTheWild, ASVspoof5;
AUROC and its 95% interval must be finite and ordered; each cell must have
10,000 test rows, a 40,000-row train pool, 500/500 valid source-ID bootstrap
replicates, and converged fit. The implementation rejects response-like input
paths/headers, substitutions, hash mismatch, missing rows, invalid CIs, and
nonempty output directories.

## Locked rendering

Render one fixed-order horizontal AUROC forest plot, not a ranking:

- corpus rows retain the exact protocol order;
- an Okabe--Ito blue point and 95% CI line represent each held-out AUROC;
- a neutral dashed AUROC=0.5 reference identifies chance label separation;
- axis range is fixed to [0.45, 1.00];
- title and annotation explicitly say `H7 score-free, feature-only baseline`,
  `all 28 full-waveform features`, `LOO train: 40k`, `held-out: 10k`, and
  `no detector score or causal claim`; and
- export vector PDF plus 300-DPI PNG and hash-bound metadata JSON.

Write non-overwriting outputs below
`results/h7_analysis_001/figures_feature_transfer_001/`. This is a future
authorized supplement artifact, not a main-paper figure or proof that the
conference accepts supplementary material.
