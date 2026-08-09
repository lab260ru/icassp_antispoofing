# H4 protocol — cross-corpus label--cue transportability atlas

**Status:** locked before H4 input freezing or analysis. H4 is a score-free,
classifier-free descriptive study. It does not select a detector cue, modify H1,
repair H2/H2B, authorize H3, or support a causal claim.

## Question

For the already registered `v1_28` waveform features, does a feature's raw
binary-label relationship have a consistent direction across five benchmark
corpora? This asks about corpus transportability of label--cue separation, not
detector reliance or waveform causality.

## Immutable allowed inputs

H4 may read only these five local feature products, after SHA-256 validation:

| Dataset | Input |
|---|---|
| ASVspoof2019_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet` |
| ASVspoof2021_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet` |
| ASVspoof2021_DF | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet` |
| InTheWild | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet` |
| ASVspoof5 | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet` |

Allowed columns are `sample_id`, `source_id`, `label`, `view`, and the 28
feature names in the committed `v1_28` registry. The runner must reject any
response-like column or path containing `score`, `logit`, `detector`, `model`,
or `eer`; it must reject association summaries, score catalogs, Arena artifact
directories, model directories, and audio paths. It must not decode waveforms,
initialize ASR, import a detector, inspect scores, or train a model.

## Pre-analysis input freeze

Before computing an AUROC, a dedicated H4 input-freeze command must record
each input's absolute path, byte size, SHA-256, allowed-column schema, per-view
row counts, and binary-label counts. It must deterministically choose at most
5,000 distinct `source_id` values per `(dataset, label)` using SHA-256 seed
2609, then retain all three views for each selected source. The manifest and
provenance report are non-overwritable and are committed before H4 analysis.

Missing feature values are retained as missing. For a feature/view/dataset,
the estimator uses finite feature values only and reports its finite count by
label. No imputation, feature re-extraction, corpus substitution, or sample
ceiling change is permitted after the freeze.

## Registered output matrix

The primary matrix is all 420 `dataset × view × feature` cells:

- 5 fixed datasets;
- 3 fixed waveform views;
- 28 fixed `v1_28` descriptors.

For each cell, compute raw binary label AUROC and signed separation
`delta_auc = 2 * AUROC - 1`; retain the natural feature orientation rather than
flipping it. Generate a 95% source-cluster percentile bootstrap interval using
500 label-stratified resamples with deterministic seed 2609. A source cluster
is resampled within label; singleton-source datasets use the same documented
source-ID resampling rule. An unavailable finite class, duplicate key, missing
view, or malformed source ID is an explicit failed cell, never a replacement
analysis.

The aggregation contains the exact 84 `view × feature` units. An
**atlas-only stable label association** is reported only if all of the following
hold:

1. `delta_auc` has the same nonzero sign in at least four of five corpora;
2. both held-out corpora have intervals excluding zero in that same sign;
3. both held-out absolute `delta_auc` values are at least 0.10; and
4. finite feature availability is at least 80% in each label of every corpus.

This designation is descriptive and terminal: it may not rank, freeze, choose,
or otherwise supply a feature to H1, H2, H2B, H3, a model, or a new training
experiment. Report every one of the 420 cells and all 84 aggregation units,
including failures and missingness.

## Interpretation and stopping rule

H4 quantifies corpus dependence in feature--label separation. It does not
explain any detector-score association, establish label leakage, validate an
anti-spoofing method, or identify a causal mechanism. After the complete matrix
and aggregation are materialized, preserve the result (including a zero-stable
unit outcome) and do not tune feature direction, thresholds, samples, views,
or bootstrap count. A new hypothesis would require a new protocol and fresh
score-independent freeze.
