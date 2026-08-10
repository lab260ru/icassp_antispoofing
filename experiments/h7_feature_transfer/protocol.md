# H7 — fixed-vector cross-corpus feature-transfer audit

## Question and status

H7 asks a deliberately different, score-free question: **does the complete
fixed 28-feature full-waveform vector support label transfer to a held-out
corpus under one locked linear baseline?** It is a descriptive feature-only
baseline, not a detector analysis, causal cue test, intervention, mitigation,
or H3 training result.

This post-H1/H4 protocol is new and independent. It must not use H1 scores,
H1/H4/H5/H6 result values, feature rankings, model coefficients, waveform
audio, ASR, or detector code to choose a feature, model, hyperparameter, or
corpus. Its complete five-fold matrix is terminal and cannot select an H2,
H2B, or H3 follow-up.

## Locked inputs and score firewall

Only these five existing HDD Parquets are permitted:

| Dataset | Path |
| --- | --- |
| ASVspoof2019_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet` |
| ASVspoof2021_LA | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet` |
| ASVspoof2021_DF | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet` |
| InTheWild | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet` |
| ASVspoof5 | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet` |

Read exactly `sample_id`, `source_id`, `label`, `view`, and the frozen
`src.audio_features.FEATURE_NAMES` list. Project the `full_waveform` view
only. Source Parquets are expected to retain feature columns but must not have
response-like (`score`, `logit`, `detector`, `model`, `eer`) columns or paths.
The implementation must reject raw-score/Arena/model/audio/ASR paths and
imports.

## Independent freeze

Before any H7 fit, hash every source Parquet and freeze at most 5,000 unique
source IDs per `(dataset, label)` using the ascending SHA-256 key
`2612|dataset|label|source_id`. Select all matching full-waveform rows, require
one row per selected source ID, preserve both binary labels, and write the
complete selected-ID manifest and provenance. No score, H1/H4/H5/H6 result,
or feature statistic may be read in this freeze.

## Locked model and complete evaluation matrix

For each of the five corpora in the fixed order above, fit exactly one
leave-one-corpus-out model on the other four frozen cohorts and evaluate it
only on the held-out frozen cohort:

- feature vector: all 28 registered features, with no feature ranking or
  subset selection;
- preprocessing fit on training rows only: replace infinities with missing,
  median-impute every feature, append missingness indicators, standardize; and
- classifier: `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
  max_iter=1000, random_state=2612)` with no class weighting, calibration,
  threshold tuning, hyperparameter sweep, or coefficient interpretation.

Report all five held-out cells: train/test sample counts, AUROC, equal-error
rate (EER), feature count after missing indicators, fit convergence, and a
95% source-cluster percentile bootstrap interval for AUROC. Bootstrap exactly
500 held-out source-cluster resamples per cell using cell-specific SHA-256
derived seeds. If every source ID is singleton, this is explicitly reported as
an utterance-level cluster bootstrap; no alternate grouping is inferred.
EER is the linear interpolation of the FPR/FNR crossing on the empirical ROC
curve; no test-set threshold is fitted or reused.

The primary descriptive summary is the unweighted median held-out AUROC across
the five cells. It has no acceptance threshold and cannot rank/select a
feature, detector, waveform view, or mitigation method.

## Outputs and hard stops

The implementation must be committed before execution. It writes a
non-overwriting input freeze, exact five-row result CSV/Parquet, provenance
JSON, and `H7_ANALYSIS_001.md` under
`experiments/h7_feature_transfer/results/`. It must stop on input hash/schema
mismatch, nonbinary labels, missing/duplicate selected IDs, unexpected view,
non-finite metric, incomplete five-cell matrix, output overwrite, score/model
path access, or unapproved model configuration.

Do not add H7 to the main paper until its complete matrix is validated and a
separate evidence/presentation review determines that it fits the four-page
story without diluting H1/H2's central negative result.
