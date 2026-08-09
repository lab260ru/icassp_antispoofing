# H2 detector-free waveform-quality runner

`scripts/run_h2_quality.py` is the execution layer between the committed
pre-score input freeze and any future H2 scorer.  It is intentionally unable to
import a detector runner, load detector weights, call a scorer, or write a
detector-response column.  It evaluates only the four committed crest-factor
arms, waveform feature changes, STOI, loudness/clipping diagnostics, and the
pinned Whisper transcript-WER gate.

## Preconditions and validation

The runner defaults to the committed artifacts in
`results/pre_score/h2_crest_pre_score_001_20260809T205114Z/` and refuses to
run unless both `input_manifest.csv` and `arm_ledger.json` are tracked, clean
against `HEAD`, and valid.  It verifies all of the following before audio or
ASR is touched:

- exact `h2_pre_score_pairs_v1` manifest marker and non-duplicate identities;
- exact SHA-256 definition hashes for only `drc_cf3`, `drc_cf6`,
  `small_gain_plus_0p1db`, and `polarity`;
- every selected dataset's Arena repo/revision against `data/arena-index.yaml`;
- score-independence column guard and source/selection hash formats.

The source ASR runtime is `openai-whisper==20250625`, `small.en`, FP16, and
the model/checkpoint identity documented in `H2_ASR_WER_RUNTIME.md`.  The
runner records its configured CUDA visibility and logical Whisper device in
provenance.  With the requested physical GPU assignment, launch it as below:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/run_h2_quality.py \
  --run-id h2_quality_full_001
```

Inside that process, physical GPU 3 is exposed as logical `cuda:0`, which is
the pinned Whisper device.  The command receives the standard `pystoi` score,
registered full-waveform feature extractor, registered waveform transforms,
and the protocol's STOI/WER/loudness/added-clipping/target-direction gates through
`evaluate_quality_pair`.

## Resumption and outputs

For a run ID `R`, durable artifacts live under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/R/`:

- `pair_rows/<pair_id>.json` — immutable, hash-checked checkpoint for every
  input--arm pair, including failures;
- `original_transcripts/<hash>.json` — a durable original-waveform transcript
  or original-ASR failure cache shared by all four arms of one sample;
- `quality_pairs.parquet` — deterministic materialization of all immutable
  pair rows written so far;
- `quality_provenance.json` and `quality_summary.json` — fixed run contract
  and compact state summary.

The corresponding compact summary is refreshed at
`experiments/h2_causal_interventions/results/quality_runs/R.summary.json`.
That summary is a research artifact to commit after a completed run; raw
waveforms, Whisper weights, and full pair tables remain on HDD.

A rerun with the same ID requires the same manifest/ledger/index/mode/limit
contract.  Existing pair IDs are hash-checked and skipped.  A different row
for an existing ID is a hard error rather than an overwrite.

Original transcripts are evaluated once per sample and reused for every arm;
transformed waveforms are transcribed independently.  An original-transcript
failure is cached as well, so every arm receives a visible retained failure
instead of silently retrying or disappearing.

## Pilot mode

A bounded run is explicitly non-confirmatory and cannot serve as a panel-gate
artifact.  Both flags are required:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/run_h2_quality.py \
  --run-id h2_quality_pilot_001 --limit 8 --pilot
```

The output mode and summary say `pilot_not_panel_gate`; it remains blocked
from detector use even if all sampled rows pass their individual quality
metrics.  An unbounded full run also remains `not_frozen` for detector use
until the separate arm-level 90% pass-rate and quality-freeze protocol is
performed.

## Focused verification

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_h2_quality_runner.py \
  tests/test_h2_pre_score_pairs.py \
  tests/test_h2_asr_wer.py \
  tests/test_h2_waveform_transforms.py
```

These tests do not access corpus audio, load Whisper, or run a detector.
