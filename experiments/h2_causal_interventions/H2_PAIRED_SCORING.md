# H2 quality-frozen paired detector scoring

`scripts/run_h2_paired_scoring.py` is the only implemented detector stage for
the current H2 panel. It is intentionally downstream of the committed
post-quality freeze, and it is not a causal analysis or a model-training run.
It can score only the three parity-eligible runners:

| Model | Runner and input contract | Batch | Physical GPU |
|---|---|---:|---:|
| Spectra-AASIST | CUDA ONNX; external `preemphasis_0.97` | 8 | 0 |
| AASIST | CUDA ONNX; `raw` | 2 | 1 |
| Res2TCNGuard | pinned source PyTorch `evaluate.py`; `raw` | 2 | 2 |

The bundled Res2TCNGuard ONNX graph remains blocked and is never selected.
There is no fourth scorer in this implementation.

## Required quality-freeze contract

Pass explicit paths to the committed output of
`scripts/freeze_h2_quality_manifest.py`:

- `score_eligible_pairs.csv`, not the mutable HDD `quality_pairs.parquet`;
- `quality_freeze.json` adjacent to that CSV;
- the committed pre-score `input_manifest.csv` and `arm_ledger.json` that
  produced the quality run; and
- the committed `data/arena-index.yaml`.

The CSV must be retained-only and have one unique `pair_id` per row. Each row
is required to carry these fields: quality-row version; pair/sample/dataset,
label, source, and selection identities; arm plus arm-definition SHA-256;
`retained=True`, `quality_status=passed`, and
`detector_stage=eligible_after_quality_freeze`; source/transformed waveform
SHA-256s; and positive source/transformed sample counts and sample rate. The
quality-freeze JSON must declare `h2_quality_freeze_v1`, `score_eligible`, a
passed 90%-retained gate for every registered arm, hashes of the input
manifest/ledger/index/quality-run artifacts, no previous detector scoring, and
the exact byte SHA-256 of the CSV.

Before constructing a detector, the runner verifies the committed artifacts,
the CSV byte hash, all quality identities and arm hashes, and regenerates each
source waveform and its registered transform. It rejects any source or
transformed waveform hash/length/sample-rate mismatch. Arena score artifacts
are not read by this command: compact parity reports only validate the fixed
runner choice and orientation.

## Launches

Use one process per physical GPU; each process sees its assigned device as
logical `cuda:0`. Substitute a newly committed freeze directory and a single
explicit run ID. The ONNX launches must use `run_h2_cuda.sh` for the CUDA-13
wheel libraries.

```bash
FREEZE=experiments/h2_causal_interventions/results/quality_freezes/RUN_ID
INPUT=experiments/h2_causal_interventions/results/pre_score/h2_crest_pre_score_001_20260809T205114Z

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. scripts/run_h2_cuda.sh scripts/run_h2_paired_scoring.py \
  --run-id h2_paired_score_001 --model Spectra-AASIST \
  --quality-manifest "$FREEZE/score_eligible_pairs.csv" --quality-freeze "$FREEZE/quality_freeze.json" \
  --input-manifest "$INPUT/input_manifest.csv" --arm-ledger "$INPUT/arm_ledger.json"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. scripts/run_h2_cuda.sh scripts/run_h2_paired_scoring.py \
  --run-id h2_paired_score_001 --model AASIST \
  --quality-manifest "$FREEZE/score_eligible_pairs.csv" --quality-freeze "$FREEZE/quality_freeze.json" \
  --input-manifest "$INPUT/input_manifest.csv" --arm-ledger "$INPUT/arm_ledger.json"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_h2_paired_scoring.py \
  --run-id h2_paired_score_001 --model Res2TCNGuard \
  --quality-manifest "$FREEZE/score_eligible_pairs.csv" --quality-freeze "$FREEZE/quality_freeze.json" \
  --input-manifest "$INPUT/input_manifest.csv" --arm-ledger "$INPUT/arm_ledger.json"
```

The command refuses an unset, remapped, or multi-GPU `CUDA_VISIBLE_DEVICES`
value. It therefore cannot silently place a model on a different GPU than the
recorded throughput result.

## Resumption and outputs

For run ID `R` and model `M`, immutable per-pair response checkpoints and a
fixed contract live under:

`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/paired_scoring/R/M/`

- `score_provenance.json` — immutable freeze/parity/model/GPU contract;
- `pair_rows/<pair_id>.json` — immutable, content-hashed paired response; and
- `paired_scores.parquet` — deterministic materialization of those segments.

Each response preserves original/transformed logits, raw scores, canonical
spoof-evidence scores, their paired delta, detector-input hashes, waveform
hashes, orientation, batch/GPU setting, and runner provenance. A resume must
match the full contract and re-validates every input waveform; an existing
valid checkpoint is reused rather than recomputed. A conflicting contract or
checkpoint is a hard error, never an overwrite.

## Focused verification

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_h2_paired_scoring.py \
  tests/test_h2_quality_freeze.py \
  tests/test_h2_quality_runner.py \
  tests/test_h2_pre_score_pairs.py
```

The tests use synthetic audio and a fake scorer. They do not load corpus audio,
ASR, ONNX Runtime, model weights, or a production detector.
