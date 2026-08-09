# H2B Q0 run 001 — DeepVoice score-blind input freeze

**Status:** completed metadata-only input freeze; not a waveform, feature, ASR,
transform, detector, score, EER, or causal experiment.

## Declared source and scope

Q0 uses `SpeechAntiSpoofingBenchmarks/DeepVoice` at pinned revision
`cc3bdf544cfc09bd9cc788f7f022ba1af9daf701`, as declared in commit
`26f9eac` before acquisition.  The downloaded labels table matches the
Arena-index SHA-256
`74efc4b38a3a678ac8246a871a1e1b6e330fa92e2b78529f3bda040c1a00cf5c`.
It contains 5,053 trials: 628 label-0 and 4,425 label-1.  The HDD source tree
contains labels, audio Parquet shards, and provenance files but no score-like
file.

The two commands below invoked only `src.arena_io.load_labels` through the
score-free input builder and deterministic SHA-256 metadata selection.  They
did not enumerate/decode audio shards, extract a feature, initialize ASR,
apply a transform, import a detector, or access any published score artifact.

```bash
PYTHONPATH=. python3 scripts/build_h2_input_csv.py \
  --dataset DeepVoice \
  --output-csv experiments/future_directions/results/h2b_q0_deepvoice_input.csv \
  --output-provenance experiments/future_directions/results/h2b_q0_deepvoice_input.provenance.json

PYTHONPATH=. python3 scripts/freeze_h2b_q0_manifest.py \
  --input-csv experiments/future_directions/results/h2b_q0_deepvoice_input.csv \
  --input-provenance experiments/future_directions/results/h2b_q0_deepvoice_input.provenance.json \
  --dataset DeepVoice \
  --revision cc3bdf544cfc09bd9cc788f7f022ba1af9daf701 \
  --per-label 128 --seed 2609 \
  --output-manifest experiments/future_directions/results/h2b_q0_deepvoice_manifest.csv \
  --output-provenance experiments/future_directions/results/h2b_q0_deepvoice_manifest.provenance.json
```

## Frozen result

The manifest contains exactly 256 rows: 128 per binary label, all from
DeepVoice.  Every row is marked `q0_frozen_score_blind`; the JSON says
`detector_scoring_allowed: false` and preserves the guard that no audio,
feature, ASR, transform, score, or detector was read.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `h2b_q0_deepvoice_input.csv` | 640,558 | `f11f24aa9fee5f4a737c4aad553d9a2f2ddc92a83a4e9f8d054c7ab8d4d5065d` |
| `h2b_q0_deepvoice_input.provenance.json` | 1,722 | `6b136e8f549557499d7cb201ceaa77531cfaf80eb6d6179b159851ca38b9e905` |
| `h2b_q0_deepvoice_manifest.csv` | 102,577 | `640853b5e663bb15c9f5501d57e2dc749684ddefba5c101b5586bccf412eb622` |
| `h2b_q0_deepvoice_manifest.provenance.json` | 2,154 | `c133ca6b390c52af84bb51cfa505afe59cc4f79f4e8df3f71db03c85ece9674c` |

## Consequence

Q0 authorizes only the next **committed** Q1 quality-calibration
implementation to decode the listed source audio.  It does not authorize
detector scoring.  Q1 still requires a finite parameter manifest, explicit
family-specific minimum target changes, quality-only runner, and tests before
execution.
