# H5 paired waveform-view invariance atlas CLI

This document implements the locked [H5 protocol](protocol.md). H5 is a
terminal descriptive measurement-agreement audit. It never reads labels,
detector outputs, score artifacts, model/ASR code, waveform paths, audio, or
any prior-result table.

The existing v1_28 feature containers also support other frozen studies and
may physically carry non-H5 metadata. That metadata is not an H5 input: the
freeze uses the exact `sample_id, view` projection, and the analyzer uses the
exact `sample_id, view` plus 28-feature projection. A label or source field is
not accepted in an H5 API, manifest, output table, or estimator. Any
score/logit/detector/model/EER/Arena/audio/ASR/H1/H2/H2B/H4/result/summary-like
path or source header is rejected before values are projected.

## 1. Freeze identities and views before reading a feature value

Choose a new output directory on the HDD. All five arguments are mandatory
and must be the literal absolute protocol paths; an alias, symlink to another
target, response-like path, or non-protocol corpus is rejected.

```bash
PYTHONPATH=. python3 scripts/freeze_h5_inputs.py \
  --asvspoof2019-la /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet \
  --asvspoof2021-la /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet \
  --asvspoof2021-df /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet \
  --inthe-wild /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet \
  --asvspoof5 /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_input_freeze_001
```

The freeze computes no feature statistic. It reads only the identity/view
projection, records source byte identities and the projected schema, validates
unique `(sample_id, view)` keys and all three registered views, and ranks
distinct sample IDs with SHA-256 seed **2610**. It retains at most 10,000 IDs
per corpus. It never uses labels or another experiment's manifest. Its output
directory is non-overwritable.

Artifacts:

- `h5_input_selection_manifest.parquet`: only `dataset`, `sample_id`,
  deterministic rank, and selection-key hash;
- `h5_input_freeze_provenance.json`: input paths/sizes/hashes, identity/view
  validation, exact H5 projections, independent seed/cap, and manifest hash.

If the provenance reports `failed_input_integrity`, do not run the analyzer or
repair the selection. Record the failed run and create a new protocol only if
future scientific scope warrants it.

## 2. Analyze only a validated, committed freeze

```bash
PYTHONPATH=. python3 scripts/run_h5_view_invariance.py \
  --input-freeze-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_input_freeze_001/h5_input_freeze_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001
```

The analyzer has no feature-path option. Before it reads a feature value, it
rechecks the protocol hash, exact source paths, source byte hashes/sizes,
identity/view validation, and the independently recomputed selection manifest.
Input or manifest drift is a hard error. It then projects exactly the H5
identity/view/feature schema and materializes every registered cell.

The new output directory contains:

- `h5_view_invariance_matrix.csv`: all 420 `5 corpora × 3 lexicographic view
  pairs × 28 features` cells. Each records paired finite count, tie-aware
  Spearman concordance, IQR-normalized median shift, two paired-sample
  percentile-bootstrap 95% intervals, and an identity-derived SHA-256 seed.
  Missing/duplicate views or samples, finite-pair shortage, rank degeneracy,
  or zero pooled IQR become explicit failed cells; none is imputed or
  substituted. A degenerate bootstrap draw is retained as an invalid attempt,
  not silently redrawn.
- `h5_view_invariance_aggregation.csv`: all 84 view-pair/feature units, their
  five-corpus medians, and every component of the terminal view-stability rule.
- `h5_analysis_provenance.json`: input-freeze/output hashes and locked
  configuration.

The pair order is fixed lexicographically:
`deterministic_crop/full_waveform`,
`deterministic_crop/preemphasized_crop`, then
`full_waveform/preemphasized_crop`.

The stability flag is descriptive only. It requires all five successful cells,
all concordances at least 0.90, all concordance-interval lower bounds at least
0.80, and every normalized-shift interval within `[-0.10, 0.10]`. It cannot
rank/select a feature, support a detector claim, or reopen H1--H4/H2B/H3.

## Runtime and stop rule

The complete workload is CPU-bound. It uses 200 exact paired-sample bootstrap
resamples for each valid cell, batched only to avoid constructing pandas frames.
It does not use a GPU, read waveform files, initialize ASR/a model, or train.

Stop and record a failed H5 run on a firewall/hash/schema failure, an incomplete
freeze, or any incomplete 420-cell matrix. Do not change the seed, sample cap,
view list, feature list, pair order, bootstrap count, or thresholds after any
inspection.
