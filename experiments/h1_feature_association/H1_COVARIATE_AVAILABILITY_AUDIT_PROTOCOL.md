# H1 covariate-availability reporting audit — protocol

## Purpose and boundary

This is a post-result reporting audit of the already completed H1
rank-residual adjustment and its two held-out bootstrap confirmations. It
creates no feature--score association, p-value, confidence interval, ranking,
candidate, or intervention decision. It cannot alter H1/H2/H2B/H3 or promote
any H4--H6 result.

The sole question is: which H1 adjustment and bootstrap metadata fields were
available in the exact five pinned feature cohorts? The resulting table is a
supplementary disclosure for the manuscript, not a new empirical result.

## Fixed inputs

The audit reads only the following five pinned `v1_28` feature Parquets on the
HDD, one row per `sample_id` selected by the existing `full_waveform` view:

| Dataset | Path |
| --- | --- |
| ASVspoof2019_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet` |
| ASVspoof2021_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet` |
| ASVspoof2021_DF | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet` |
| InTheWild | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet` |
| ASVspoof5 | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet` |

The audit projects exactly `sample_id`, `source_id`, `label`, `view`,
`duration_seconds`, `integrated_lufs`, `speaker_id`, and `attack_id`; it must
never materialize registry feature values. The pinned Parquets are known to
contain feature columns because they are the original H1 products, so those
unprojected columns are not themselves an error. The tool must reject score,
logit, detector, model, EER, AUROC, audio, or ASR paths and columns. It must
reject other views, duplicate `sample_id`s within a corpus, missing labels, or
a mismatch between the five locked corpus names and paths.

The two existing confirmation notes are fixed contextual inputs only:
`results/InTheWild/CONFIRMATION_RUN_20260809.md` and
`results/ASVspoof5/CONFIRMATION_RUN_20260809.md`. Their already recorded
bootstrap cluster counts may be transcribed, but no bootstrap is rerun.

## Locked reporting calculations

For every corpus and label, report only:

1. total selected samples;
2. finite counts for duration and integrated loudness;
3. nonempty counts and distinct-value counts for speaker, attack, and source
   IDs; and
4. the exact algorithmic treatment already implemented in
   `src.statistics.partial_spearman`: intercept; rank-transformed duration and
   loudness with median fill when their column is present; categorical
   speaker/attack dummies with an explicit missing category; and removal of a
   tested variable from the controls.

The report must also state the fixed confirmation bootstrap rule
(`speaker_id` when nonempty, otherwise `source_id`), label stratification,
2,000 replicates, seed 2609, and the already sealed selected-spoof cluster
counts. It may not recalculate any score-dependent quantity.

## Output and hard stops

Outputs are a compact CSV plus Markdown report beneath
`experiments/h1_feature_association/results/covariate_availability_audit_001/`.
They must include each source file byte hash and schema. Outputs are
non-overwriting.

Stop with an explicit error for an unexpected input/schema, non-unique
full-waveform sample IDs, a response-like field, or an attempt to write over
an existing report. Do not substitute a corpus, infer a missing field, or read
model-score/audio data.
