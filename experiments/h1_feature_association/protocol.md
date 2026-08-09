# H1 protocol — cross-dataset score-feature association

**Status:** CONFIRMATORY, locked before result generation.

## Question and prediction

At least one pre-registered waveform feature has a directionally stable
within-class association with published detector scores after controlling for
label, duration, loudness, corpus, codec, speaker, and attack effects where
metadata permits. Crest factor is one candidate, not the presupposed answer.

## Inputs

- Pinned per-sample Arena score artifacts from `data/arena-index.yaml`.
- Core datasets in `configs/study.yaml` with stable sample-ID joins only.
- The v1_28 feature registry, extracted on the three registered waveform views.

## Design

Run discovery on ASVspoof 2019 LA, ASVspoof 2021 LA, and ASVspoof 2021 DF.
Freeze candidates and analysis choices, then evaluate In-the-Wild and ASVspoof
5. Use the eight-model score panel only where published score artifacts exist.
For datasets above 50,000 examples, use the deterministic class-balanced seed
2609 subset defined in the study configuration.

## Primary analyses

Compute within-class Spearman and partial score-feature correlations, feature
label AUROC, feature-only cross-dataset EER, clustered bootstrap intervals, and
Benjamini-Hochberg-adjusted q-values. A portable association requires the exact
criterion in `configs/study.yaml`; models/datasets absent from Arena artifacts
are reported as missing, never imputed.

## Exclusions and failure conditions

Exclude only samples lacking a stable ID, a readable waveform, required labels,
or a published score; report every count. A score-direction or label-convention
mismatch halts analysis for that artifact until catalogue metadata resolves it.

## Outputs

`results/association_long.parquet`, `results/association_summary.csv`,
`results/join_report.json`, plots, and `analysis.md`.
