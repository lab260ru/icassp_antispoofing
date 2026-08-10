# H9-PCR terminal evaluator contract

`scripts/evaluate_h9_pcr_targets.py` is the only H9-PCR component allowed to
open SONAR or ArAD target labels. It is a terminal analysis, not a source
selection, early-stopping, model-selection, or tuning interface. It must run
only after all 12 source-selected checkpoints are frozen.

No target data were acquired, decoded, or evaluated while this implementation
note and its synthetic tests were produced.

## Inputs that must exist before the terminal call

The future target materializer must produce one JSON manifest per fixed target,
with this exact metadata-only shape. `records_path` points to a CSV or Parquet
file with the exact ordered columns listed below; it must contain **no label
column**.

```json
{
  "artifact_kind": "h9_pcr_canonical_target_manifest",
  "version": "h9-pcr-terminal-evaluation-v1",
  "dataset": "SONAR",
  "dataset_revision": "eca7c72ebdf0f7936a644605a56735ac8564dbd9",
  "records_path": "/absolute/HDD/path/sonar_trials.csv",
  "records_sha256": "<sha256>",
  "labels": {"path": "/absolute/HDD/path/sonar_labels.csv", "sha256": "<sha256>"},
  "target_labels_read": false,
  "target_metrics_read": false
}
```

The target record file must have exactly:

```text
sample_id,audio_path,audio_sha256,audio_bytes,canonical_fingerprint
```

All hashes/fingerprints are 64-character SHA-256 values. The evaluator
byte-hashes every declared waveform before inference. The label file must be a
separate CSV/Parquet table with exactly `sample_id,label`, one unique binary
label per canonical target sample. Its values are not decoded while its hash is
checked.

The source-training handoff must create an immutable JSON
`h9_pcr_frozen_checkpoint_ledger` (version
`h9-pcr-terminal-evaluation-v1`) with:

- hash-pinned `PLAN.md` and `DATA.md`;
- a materialized source manifest containing `sample_id` and a nonempty
  canonical audio fingerprint, plus the exact P and B2 source-pair paths and
  hashes;
- an architecture bundle path and `_net.py` hash;
- a source-only lambda-selection artifact path/hash and selected value;
- a locked training envelope with batch size 24, four workers, six epochs,
  CUDA enabled, the fixed margin, and shared optimizer values; and
- exactly twelve unique checkpoint records: `B1`, `B2`, and `P` for seeds
  9101--9104. Each record binds the checkpoint plus its source-only training
  sidecar by SHA-256.

Every sidecar must say `target_labels_read: false`, `target_audio_read: false`,
must prove `fresh_seeded` initialization with the matching seed and architecture
hash, and must contain this exact source binding:

```json
"source_artifact_hashes": {
  "source_manifest_sha256": "<sha256>",
  "p_pairs_sha256": "<sha256>",
  "b2_pairs_sha256": "<sha256>"
}
```

The checkpoint itself must carry the same source-manifest hash, the exact
training config, the source-only lower-EER/lower-epoch checkpoint rule, and
fresh-initialization provenance. A missing/duplicate matrix member, stale hash,
pretrained provenance, source binding mismatch, or configuration drift is a
hard stop before target inference or label loading.

## Terminal invocation

Only after the source fit ledger has been frozen, execute one exhaustive call
on a CUDA GPU, with a fresh new output directory on the project HDD:

```bash
PYTHONPATH=. python3 scripts/evaluate_h9_pcr_targets.py \
  --sonar-manifest /absolute/HDD/path/sonar_manifest.json \
  --arad-manifest /absolute/HDD/path/arad_manifest.json \
  --checkpoint-ledger /absolute/HDD/path/h9_frozen_checkpoint_ledger.json \
  --plan experiments/h9_paired_counterfactual/PLAN.md \
  --data-contract experiments/h9_paired_counterfactual/DATA.md \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_terminal_evaluation_001 \
  --terminal-evaluation
```

Without `--terminal-evaluation`, the evaluator returns before it reads a
target manifest, waveform, or label artifact. The command exposes no flags for
checkpoint choice, seed subset, source split, model, loss, lambda, epoch,
augmentation, target subset, metric, bootstrap count, or threshold.

Before inference it rejects any exact source--target canonical-audio
fingerprint collision. It then evaluates **both** targets, all three methods,
and all four frozen seeds. It writes immutable raw, label-free long-form
predictions first; only then it opens the two target label files once to write
per-target EER/AUROC, seed-level EER/AUROC, the labeled reconstruction, and the
fixed 2,000-replicate (`seed=2909`) label-stratified shared-ID bootstrap of
macro EER `P-B1` and `P-B2`.

The decision JSON passes only when P has at least 10% relative lower two-target
macro EER than both B1/B2, strictly lower EER on both SONAR and ArAD for both
comparators, both bootstrap upper confidence limits are below zero, no exact
fingerprint collision exists, and the manifest/prediction reconstruction audit
passes. Any failure stops H9-PCR; the evaluator has no rescue path.
