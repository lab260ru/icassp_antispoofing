# H4 score-free input freeze and label--cue atlas CLI

This document implements the locked [H4 protocol](protocol.md). H4 is a
descriptive, score-free and classifier-free feature--label audit. It must not
read an Arena artifact, score/logit catalog, detector/model directory, audio
path, or any H1/H2/H2B/H3 result.

## 1. Freeze inputs before computing a statistic

Choose a new output directory on the HDD. The five input arguments are all
mandatory and must be the exact absolute paths below; aliases, symlinks to a
different target, response-like paths, and non-protocol datasets are rejected.

```bash
PYTHONPATH=. python3 scripts/freeze_h4_inputs.py \
  --asvspoof2019-la /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2019_LA/features_wide.parquet \
  --asvspoof2021-la /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_LA/features_wide.parquet \
  --asvspoof2021-df /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof2021_DF/features_wide.parquet \
  --inthe-wild /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet \
  --asvspoof5 /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/ASVspoof5/features_wide.parquet \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h4_label_transportability/h4_input_freeze_001
```

The command projects only `sample_id`, `source_id`, `label`, `view`, and the
28 registered `v1_28` features. It records the whole Parquet schema but
rejects any response-like field (`score`, `logit`, `detector`, `model`, or
`eer`) before feature values are read. Harmless extractor provenance fields
are not consumed. It validates exact paths, byte sizes and SHA-256 values;
records binary-label and per-view counts; and deterministically ranks source
IDs with SHA-256 seed 2609, retaining at most 5,000 source IDs per
dataset/label. The freeze output directory is non-overwritable.

Its artifacts are:

- `h4_input_selection_manifest.parquet`: selected `(dataset, label,
  source_id)` source clusters and deterministic ranks;
- `h4_input_freeze_provenance.json`: exact source paths, schema, size, hashes,
  validation records, cap, seed, and selection-manifest hash.

Commit these compact artifacts or their committed provenance/hash record before
starting the analysis. Do not alter an existing freeze to repair an input.

## 2. Analyze only a validated freeze

```bash
PYTHONPATH=. python3 scripts/run_h4_label_transportability.py \
  --input-freeze-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h4_label_transportability/h4_input_freeze_001/h4_input_freeze_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h4_label_transportability/h4_analysis_001
```

The analyzer has no input-path option. It first rechecks the frozen paths,
current sizes/hashes, allowed projection, protocol hash and selection-manifest
hash. Changed inputs or a hand-edited/moved manifest are hard errors; do not
silently re-freeze or substitute data.

The run writes, in a new non-overwritable output directory:

- `h4_label_cue_matrix.csv`: all 420 `5 datasets × 3 views × 28 features`
  cells, including explicit failed cells, selected and finite counts by label,
  availability, raw-orientation AUROC, signed `delta_auc`, and 500
  label-stratified source-cluster bootstrap CI fields (seed 2609);
- `h4_label_cue_aggregation.csv`: all 84 `view × feature` units and every
  component of the exact atlas-only stable-label-association rule;
- `h4_analysis_provenance.json`: input-freeze hashes, output hashes, locked
  seed/replicates and complete-cell counts.

No cell is imputed or rerouted. Missing finite classes, duplicate sample/view
keys, malformed source IDs, and absent views become explicit failed cells.
The aggregation designation is terminal and descriptive; it cannot select a
feature, cue, model, training experiment, or causal follow-up.

## Runtime expectation

The analysis is CPU/memory-bandwidth intensive: the locked workload contains
420 cells and 500 source-cluster bootstrap replicates per valid cell. The
implementation computes the exact tie-aware weighted Mann--Whitney resamples
in batches of 32 rather than creating 210,000 pandas resample frames. With the
maximum 5,000 source clusters per label, expect the complete run to take
minutes to hours depending on finite rows and number of recordings per source.
It does not use GPU, ASR, audio decoding, model inference, or training.
