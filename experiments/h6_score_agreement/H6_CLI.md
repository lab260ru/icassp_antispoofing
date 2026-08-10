# H6 within-class published-score agreement CLI

This document implements the locked [H6 protocol](protocol.md). H6 is a
descriptive check of agreement among the eight original, published Arena score
artifacts. It does not open audio/waveforms, feature products, model code or
weights, ASR, detector runners, H1/H2/H2B/H3/H4 tables, or any score summary.
It cannot identify a cue, prove model independence, establish a causal effect,
or select a model or follow-up experiment.

## 1. Freeze the label panel before parsing a score

Use a new directory on the HDD. The command has no input override: it derives
exactly the five registered label files and the 40 registered original
score-file locations from the pinned `data/arena-index.yaml`.

```bash
PYTHONPATH=. python3 scripts/freeze_h6_inputs.py \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_input_freeze_001
```

The freeze reads the label schema and exact label-ID/label projection only.
It chooses at most 5,000 unique normalized sample IDs per `(dataset, label)`
with SHA-256 seed **2611**, then writes the complete label-only manifest before
any score text is parsed, ranked, joined, or emitted. A source/speaker column
is used for bootstrap clustering only when that column exists in the pinned
label table; otherwise the stable sample ID is the cluster.

For provenance, the freeze also hashes the bytes of every original score file
and records its path, model revision, size, and SHA-256. Hashing does not parse
or materialize a score value. It records a failed freeze rather than repairing
a missing artifact or index mismatch. The new directory contains:

- `h6_label_selection_manifest.parquet`: selected labels, sample IDs, allowed
  bootstrap-cluster IDs, deterministic ranks, and selection-key hashes;
- `h6_input_freeze_provenance.json`: label schema/revisions/byte identities,
  all 40 raw-score byte identities, registry/configuration, and manifest hash.

Commit the compact provenance/hash record before running the analyzer. Do not
edit a freeze to repair an input or use a manifest from H1/H4/H5.

## 2. Analyze only a validated freeze

```bash
PYTHONPATH=. python3 scripts/run_h6_score_agreement.py \
  --input-freeze-provenance /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_input_freeze_001/h6_input_freeze_provenance.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001
```

The analyzer has no score, label, waveform, feature, model, or summary path
option. Before opening score text it rechecks the protocol hash, Arena-index
hash, source paths/revisions/sizes/SHA-256 values, label validation, and the
independently recomputed selection manifest. It parses only the original
two-column raw score artifacts and converts each one to the existing
increasing-spoof orientation using the frozen two-class panel. It never uses a
processed score panel.

The new output directory contains:

- `h6_within_class_score_agreement_matrix.csv`: exactly 280 cells: five
  datasets × two classes × all 28 unordered model pairs. Every complete cell
  gives the exact joined count, within-class Spearman agreement, 200
  sample-cluster percentile bootstrap attempts and a 95% interval. Its seed is
  SHA-256-derived from the full cell identity and base seed 2611.
- `h6_model_pair_agreement_summary.csv`: exactly 28 pair summaries, each over
  its ten registered dataset/class cells, with median agreement and cross-cell
  range.
- `h6_analysis_provenance.json`: freeze/output hashes, orientation validation
  statuses, fixed registry/configuration, and a stop status.

No score is imputed, model substituted, or intersection-reselected. Missing
selected scores, duplicate model/sample IDs, malformed raw text, ambiguous
orientation, and non-finite within-class ranks are written as explicit
unavailable cells. The matrix remains 280 rows, but the provenance reports a
stopped run; it is not a partial scientific result. A tampered freeze or byte
identity mismatch is a hard pre-analysis failure.

## Runtime and interpretation boundary

This CPU-only analysis may rank-resample up to 280 cells × 200 bootstrap
replicates. It uses no GPU and no inference/training. Stop and record failure
instead of changing the sample cap, seed, model/dataset registry, pair order,
orientation rule, bootstrap count, or interpretation after inspection.

Agreement is descriptive architecture-diversity evidence only. Neither high
nor low agreement establishes architecture independence, feature reliance,
model quality, a causal sensitivity, or any result from another hypothesis.
