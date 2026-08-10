# H5 protocol — score-free paired waveform-view invariance atlas

Status: locked on 2026-08-10 before any H5 input freeze, feature analysis,
score access, waveform decoding, model import, or H5 result inspection.

## Question and prediction

The three registered waveform views may alter feature measurement enough to
contextualize view-dependent descriptive results, independently of labels and
detectors. H5 predicts that some registered features will have low paired rank
concordance or material paired shifts between views. A null result is retained
equally. H5 cannot explain a detector, select a feature, or reopen H1, H2, H2B,
H3, or H4.

## Immutable score-free inputs and firewall

H5 may read only the following five v1_28 feature tables after their SHA-256
values are sealed by a dedicated H5 input freeze:

| Dataset | Feature product |
|---|---|
| ASVspoof2019_LA | /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet |
| ASVspoof2021_LA | /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet |
| ASVspoof2021_DF | /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet |
| InTheWild | /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet |
| ASVspoof5 | /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet |

The input products may carry label and source metadata because they are shared
feature containers, but the H5 loader must project and materialize only
sample_id, view, and the 28 committed feature columns. It must reject any
label/source field passed through its API or emitted in a manifest, matrix, or
summary. The implementation rejects all score/logit/detector/model/eer columns
and paths containing score, logit, detector, model, eer, arena, audio, ASR, H1,
H2, H2B, H4, result, or summary. It must not decode waveforms, initialize ASR
or a model, train, or read any detector response or prior-result table.

## Independent input freeze

For each dataset, deterministically select at most 10,000 distinct sample IDs
using SHA-256 seed 2610, retaining exactly the three registered views for each
selected sample. The H5 selection is independent of all H1/H2/H2B/H4 manifests.
The freeze records absolute input paths, byte sizes, SHA-256 values, projected
schema, per-view row counts, selected sample IDs, and hashes. Missing feature
values remain missing. No imputation, view substitution, corpus substitution,
cap/seed change, or use of labels is allowed.

## Registered exhaustive matrix

For five datasets, three unordered view pairs, and 28 features, materialize
exactly 420 cells. Each cell uses finite paired values for selected sample IDs
only and records:

- paired finite sample count;
- Spearman rank concordance between the two views;
- IQR-normalized median paired shift, defined as median(view_b - view_a)
  divided by the IQR of their pooled finite values; and
- paired-sample percentile bootstrap 95% intervals for both quantities using
  200 resamples. Each cell receives a SHA-256 derived generator seed from the
  base seed and its full identity.

The view-pair order is fixed lexicographically. A missing selected sample,
duplicate sample/view row, absent view, non-finite pair shortage, or zero pooled
IQR is an explicit failed cell; it is never repaired by changing the estimator.

## Aggregation and interpretation

The exhaustive 84 view-pair/feature aggregation units report every dataset
cell, the median concordance, and the median absolute normalized shift across
the five corpora. A view-stable designation is terminal/descriptive only and
requires all five cells to have concordance at least 0.90, concordance intervals
whose lower bound is at least 0.80, and absolute normalized-shift intervals
contained within 0.10. The all-cell matrix is primary; stable status cannot
rank, select, train, freeze, or otherwise promote a feature.

H5 may state only feature-measurement agreement or disagreement across the
registered waveform-view definitions. It must not infer preprocessing behavior,
label leakage, detector sensitivity, causality, or corpus quality.

## Stop rule

Stop and record a failed H5 run on any input-hash/schema/firewall failure,
incomplete fresh freeze, or incomplete 420-cell matrix. Do not relax a threshold
or alter views/features/bootstrap count after inspection. Any downstream study
requires its own new protocol and independent discovery process.
