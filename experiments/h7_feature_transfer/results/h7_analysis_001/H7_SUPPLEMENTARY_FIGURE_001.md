# H7 supplementary figure 001 — sealed feature-transfer display

## Status

**Authoritative display-only figure.** This rendering visualizes the already
sealed H7 matrix; it did not run a fit, bootstrap, thresholding, ranking, or
feature attribution. It is a repository supplementary artifact, not a
main-paper figure and not evidence that a conference supplement is authorized.

## Fixed result shown

The all-28-feature, score-free leave-one-corpus-out baseline has held-out
label AUROC (95% source-ID percentile CI) of:

| Held-out corpus | AUROC | 95% CI |
| --- | ---: | --- |
| ASVspoof2019_LA | 0.907785 | [0.902070, 0.913709] |
| ASVspoof2021_LA | 0.845107 | [0.837789, 0.854125] |
| ASVspoof2021_DF | 0.780957 | [0.771824, 0.789614] |
| InTheWild | 0.534590 | [0.522942, 0.546265] |
| ASVspoof5 | 0.588320 | [0.577296, 0.599532] |

All five cells use the frozen 40,000-row leave-one-corpus-out training pool,
10,000 held-out rows, and 500/500 valid source-ID bootstrap replicates. This
is feature-only label-transfer context; it does not establish detector
reliance, causality, cue selection, model ranking, or mitigation performance.
The sealed cohorts have singleton source IDs, so these are utterance-level
cluster bootstrap intervals rather than repeated-source dependence estimates.

## Hash-validated inputs

| Input | SHA-256 |
| --- | --- |
| `../h7_input_freeze_001/h7_selected_sources.csv` | `42592ee49a3867b86f41f83620d328272de4362b32649021028a4e1253fd194f` |
| `../h7_input_freeze_001/h7_input_freeze_provenance.json` | `236a95a309d201b8b24276e1436d1c3043c3d5d20c6e0685e7b0aaa1503248ec` |
| `h7_leave_one_corpus_out.csv` | `d8ab31532c7a3867477d3978cc7b3ebc1a9dbf6ef71e4b47086f9fc1dd541e79` |
| `h7_analysis_provenance.json` | `9a044edf921ba732e50f0300c06a3ff6484edf1286f7092fb7e96fa652e32cc2` |

The renderer validates all four byte hashes, the fixed five-row corpus order,
40,000/10,000 sizes, 500/500 bootstrap completeness, fit convergence,
finite ordered AUROC CIs, freeze linkage, and the no-claim provenance before
writing an output. Its protocol is
[`../../H7_SUPPLEMENTARY_FIGURE_PROTOCOL.md`](../../H7_SUPPLEMENTARY_FIGURE_PROTOCOL.md).

## Authoritative files and visual inspection

| File | SHA-256 |
| --- | --- |
| `figures_feature_transfer_001/h7_feature_transfer_auroc.pdf` | `c9492875cdd519cd9c3359acb768a026a33ca761690ca22402863a4a5c3cddd7` |
| `figures_feature_transfer_001/h7_feature_transfer_auroc.png` | `be8baef92454c7698f98e80087c3d58224de91ffc0e6b81ee5b322a5ee8fa63d` |
| `figures_feature_transfer_001/h7_feature_transfer_auroc.metadata.json` | `cf0a58033e6578717a7977d4c9b03ff010674ab5e705e8cef7a5a1bae9909b84` |

The PDF is a one-page vector document; the PNG is 1895 × 863 at the requested
300-DPI export setting. Manual visual inspection passed: all five protocol
corpus rows, 95% intervals, chance reference, fixed x-axis, scope footer, and
numeric annotations are legible. Two pre-publication layout trials are kept
under `figures_feature_transfer_001_superseded_layout_001/` and
`figures_feature_transfer_001_superseded_layout_002/`; each has identical
sealed inputs but is non-authoritative because its chance annotation collided
with a tick or clipped at the plot boundary. No data or scientific encoding
changed between these display iterations.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/plot_h7_feature_transfer.py
```

The command refuses a nonempty output directory. It was run after committing
the renderer at `1d0249b`.
