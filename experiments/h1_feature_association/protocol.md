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

### Execution guardrails

The partial rank design controls duration, integrated loudness, and available
speaker/attack metadata. A tested feature or response variable is removed from
its own control set before construction of the design matrix: conditioning a
variable on itself is undefined. This clarification was validated after a
pre-output implementation stop and before the first successful H1 screen; it
does not change the registered estimand or acceptance criterion.

Bootstrap intervals are a distinct confirmation stage. The screen always emits
the complete matrix; it never silently promotes its strongest rows. Any
bootstrap request must supply an explicit, frozen CSV or Parquet candidate
manifest recording candidate key, selection split, selection basis, and freeze
time. The manifest is hashed into its output. It may be constructed only after
the three discovery datasets and the selection rule are frozen, and before the
held-out confirmation scores are interpreted. See `BOOTSTRAP_CLI.md` for the
required schema and deterministic 2,000-replicate procedure.

## Exclusions and failure conditions

Exclude only samples lacking a stable ID, a readable waveform, required labels,
or a published score; report every count. A score-direction or label-convention
mismatch halts analysis for that artifact until catalogue metadata resolves it.

## Outputs

`results/<dataset>/association_summary.csv`,
`results/<dataset>/feature_label_metrics.csv`,
`results/<dataset>/association_join_report.json`, optional
`results/<dataset>/association_confirmation_bootstrap.csv`, explicit
multi-dataset `results/combined/` tables, plots, and `analysis.md`. Flat result
paths are intentionally not used because a later single-dataset run must never
overwrite an earlier corpus.
