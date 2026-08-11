# H10 CD-ADD adapter and terminal evaluator

**Status:** synthetic-contract implementation only. Do not execute either
production command until the prospective H10 protocol, its metadata audit, and
this code are committed. The commands have no synthetic-mode CLI option.

## Contracts

The adapter accepts only `SpeechAntiSpoofingBenchmarks/CD-ADD` at revision
`b03c6cf3463d67c1525ba20738c495d120675876`, and only an explicit
`<revision>/data` directory. It verifies the unchanged H9 checkpoint ledger
against SHA-256
`3e0530e2b850cbb08329d14a8982a7448a319a4a30e75ee2d51dc4ecb7f2325d` before it
opens the audit or CD-ADD directory. Its metadata-audit JSON must be a
pre-target, score-free object with this required shape:

```json
{
  "artifact_kind": "h10_cdadd_target_metadata_audit",
  "repository": "SpeechAntiSpoofingBenchmarks/CD-ADD",
  "revision": "b03c6cf3463d67c1525ba20738c495d120675876",
  "license": "CC-BY-4.0",
  "card_url": "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD/blob/b03c6cf3463d67c1525ba20738c495d120675876/README.md",
  "card_sha256": "<lowercase sha256 of pinned public card>",
  "target_rows_read": false,
  "target_audio_read": false,
  "target_labels_read": false,
  "target_scores_read": false,
  "target_predictions_read": false,
  "odss_overlap_audit": {
    "known_recording_source_overlap": false,
    "known_speaker_list_overlap": false,
    "known_released_item_overlap": false
  }
}
```

Each manifest, materialization provenance, and terminal provenance also records
the SHA-256 of this locked H10 protocol plus the frozen H9 `PLAN.md` and
`DATA.md`. It declares a literal all-audio-bearing-Parquet-trials contract:
no target subset, generator stratum, or path-based exclusion is representable
in the command or in a valid manifest.

The values are statements from the dedicated card/lineage audit; the adapter
does not infer them from a zero waveform collision. Any known or unresolved
ODSS recording-source, speaker-list, or released-item overlap blocks target
materialization.

## Future materialization command

This copies every raw audio-bearing Parquet trial byte-for-byte. It writes and
hashes a label-free record table after reading only `path,audio`; only then it
uses a separate `path,label` pass to create the sealed label artifact.

```bash
PYTHONPATH=. python3 scripts/materialize_h10_cdadd_target.py \
  --raw-shard-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/datasets/CD-ADD/b03c6cf3463d67c1525ba20738c495d120675876/data \
  --checkpoint-ledger /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_pcr_frozen_checkpoint_ledger.json \
  --metadata-audit /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/h10_cdadd_metadata_audit_001.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/h10_cdadd_target_materialization_001
```

The output is a manifest with exactly five record columns
`sample_id,audio_path,audio_sha256,audio_bytes,canonical_fingerprint` and a
separate `sample_id,label` artifact. The materialization provenance carries
the card/license audit, every raw shard hash/size/Arrow schema/row count, the
canonical-audio policy, and the pre-target H9-ledger validation.

## Future terminal command

Only run this once on the emitted manifest. The evaluator byte-validates each
copied waveform and every raw shard audit, replays the exact H9 source ledger,
hard-stops on source/target canonical PCM collisions, then writes all 12
`B1/B2/P × 9101–9104` label-free predictions before it opens the label file.
It reports all-trial EER/AUROC, all per-seed metrics, the fixed 2,000-draw
shared sample-ID label-stratified bootstrap (`seed=2909`), and the fixed H10
gate. There are no target-subset, model, seed, bootstrap, or rescue flags.

```bash
PYTHONPATH=. python3 scripts/evaluate_h10_cdadd_terminal.py \
  --target-manifest /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/h10_cdadd_target_materialization_001/h10_cdadd_target_manifest.json \
  --checkpoint-ledger /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h9_paired_counterfactual/h9_pcr_frozen_checkpoint_ledger.json \
  --output-dir /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h10_cdadd/h10_cdadd_terminal_evaluation_001 \
  --terminal-evaluation
```

The terminal result is conditional on the four frozen H9 checkpoints. A failed
gate is terminal for this target and must not select another target, subset,
loss, seed, threshold, or source rerun.
