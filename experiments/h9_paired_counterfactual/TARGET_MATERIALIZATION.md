# H9-PCR terminal-target materialization contract

`scripts/materialize_h9_pcr_target.py` builds exactly one canonical,
score-free SONAR or ArAD manifest after the source-only checkpoint ledger is
sealed. It has no model, prediction, metric, hyperparameter, or target-subset
argument. Do not run it before `h9_pcr_frozen_checkpoint_ledger.json` exists:
the CLI validates that complete source handoff before it opens a target path.

## Fixed roles and raw input

The `--dataset` argument accepts only the locked role/revision pair:

| Dataset | Required raw directory suffix |
| --- | --- |
| `SONAR` | `eca7c72ebdf0f7936a644605a56735ac8564dbd9/raw/data` |
| `ArAD` | `350184966eeb5b46ff2acdabd8f4d12e41e582da/raw/data` |

The raw directory must contain Parquet shards whose Arrow schema has string
`path`, a struct-valued `audio.bytes` binary payload, and an integer `label`.
The adapter accepts only the two containers present in the pinned inputs:
RIFF/WAVE and FLAC. It preserves the original container extension while
copying every byte, and fingerprints their shared decoded-PCM representation.
It rejects an unknown revision, missing/changed field, another container,
duplicate raw path, nonbinary label, empty class, changed copy hash, or
existing output directory. It does not guess a schema, remap a label, or
select trials.

## Label firewall and output order

The materializer uses two non-overlapping raw-column passes:

1. It reads only `path,audio`, copies every byte-identical RIFF/WAVE or FLAC
   payload (with its real extension), and
   writes the canonical record CSV and audio audit. The record table has
   exactly `sample_id,audio_path,audio_sha256,audio_bytes,canonical_fingerprint`;
   it has no label column.
2. Only after that record CSV is written and byte-hashed does it read
   `path,label` to write the separate `sample_id,label` CSV. It never opens a
   model, prediction, score, metric, target-selection interface, or target
   label in the record phase.

The final manifest retains the terminal evaluator's
`target_labels_read: false` / `target_metrics_read: false` sentinel: no label
has been supplied to training, prediction, selection, or metrics. Its adjacent
materialization provenance separately and truthfully records that the raw
label column was opened solely in the second phase to create the sealed label
artifact. The terminal evaluator is still the first component permitted to
join that label artifact with predictions and calculate metrics.

The canonical fingerprint follows the already frozen H9 mono-16-kHz PCM
policy. It is used later only for the exact source--target collision hard stop;
it does not modify the copied training/inference waveform.

## Future invocation — do not run until the source ledger exists

```bash
PYTHONPATH=. python3 scripts/materialize_h9_pcr_target.py \
  --dataset SONAR \
  --raw-shard-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/sonar/eca7c72ebdf0f7936a644605a56735ac8564dbd9/raw/data \
  --checkpoint-ledger /absolute/HDD/h9_pcr_frozen_checkpoint_ledger.json \
  --plan experiments/h9_paired_counterfactual/PLAN.md \
  --data-contract experiments/h9_paired_counterfactual/DATA.md \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_sonar_target_materialization_001
```

Run the same fixed command with `--dataset ArAD`, its revision-qualified raw
directory, and a distinct fresh output directory. Pass the two emitted
`h9_<dataset>_target_manifest.json` paths unchanged to the one terminal
evaluation call.
