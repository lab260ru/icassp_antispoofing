# H2 ONNX baseline-parity gate

**Status:** implementation complete; calibration outcome must be read from the
compact artifact bundle, not inferred from model-card class names.

## Purpose and boundary

`scripts/calibrate_h2_onnx_parity.py` checks whether a local executable ONNX
model reproduces the *ordering* of its pinned Arena baseline score artifact on
a score-independent set of original clips. It is an H2 readiness gate, not an
intervention result: it neither transforms audio nor makes a causal claim.

The initial supported models are the locally present, audited fixed-input
artifacts:

- `Spectra-AASIST/spectra-aasist.onnx`
- `AASIST/aasist.onnx`

The revision-pinned Res2TCNGuard bundle is now present under the HDD model
store. Its `res2tcnguard.onnx` signature is independently verified as
`float32[batch,64600] -> float32[batch,2]`, matching its model card's
first-window/tile-repeat contract. It is an *admitted parity candidate*, not a
validated H2 scorer, until an explicit `--model Res2TCNGuard` calibration run
passes the locked orientation and ordering gates. It deliberately remains out
of the default two-model command below.

Their model paths, model revisions, and source score files are resolved solely
from `data/arena-index.yaml`.

## Frozen calibration contract

For each dataset, the script makes exactly 64 deterministic draws per label
(128 clips for binary ASVspoof2019_LA) from the authoritative labels table.
The selection uses `pandas.sample(random_state=2609 + label)`, is sorted by
`label, sample_id`, and is frozen in
`calibration_manifest.parquet` plus an inspection-friendly CSV. On subsequent
runs the script recomputes the expected selection and refuses to proceed if an
existing manifest differs. Selection happens before model loading and never
uses detector scores.

The detector input contract is deliberately separate from H1 feature crops:

1. decode the pinned 16 kHz waveform;
2. apply an explicitly named preprocessing candidate;
3. retain its first 64,600 samples, or tile-repeat a shorter clip;
4. feed `float32[B,64600]` to ONNX Runtime and save both logits.

No random crop, padding with zeros, sample-rate conversion, or TensorRT engine
is permitted by this path.

## Audited preprocessing decision

The script evaluates the only two admitted candidates: `raw` and external
`preemphasis_0.97`. It does not assume that Spectra's PyTorch model-card
pre-emphasis belongs outside its ONNX graph. For each candidate it saves the
two ONNX logits, the predeclared selected raw scalar (`logit_1` by default),
the Arena raw/canonical baselines, input hashes, and provider information.

The canonical `score_spoof` mapping is copied from the *pinned Arena baseline*
using `src.arena_io.load_model_scores`; it is never selected from an H2 arm.
A candidate is eligible only when its raw-score direction matches that fixed
orientation and its raw scalar has Spearman correlation at least `0.999` with
the matching Arena raw score over all 128 clips. The predeclared choice is:

1. use `raw` if it passes every gate;
2. otherwise use the sole passing non-raw candidate;
3. otherwise block the scorer.

This least-assumption tie break is set before treatment data are visible.

## Execution and artifacts

Run from the repository root after the focused tests:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_onnx_fixed_window.py tests/test_h2_parity_calibration.py
PYTHONPATH=. scripts/run_h2_cuda.sh scripts/calibrate_h2_onnx_parity.py
```

`run_h2_cuda.sh` makes the CUDA 13 libraries bundled in the active Python
environment discoverable to ONNX Runtime; the host CUDA installation is 12.6
and cannot supply `libcublasLt.so.13` by itself. ONNX Runtime tries
`CUDAExecutionProvider` first, then safely retries on CPU if CUDA session
creation or its first inference fails. The actual provider and whether the
fallback occurred are persisted; CPU results can diagnose parity but do not
establish the GPU performance configuration.

The compact, commit-eligible output bundle is
`experiments/h2_causal_interventions/results/parity_calibration/<dataset>/`:

- `calibration_manifest.{parquet,csv}`
- one `*_scores.parquet` per model/preprocessing candidate
- `parity_summary.csv`
- `parity_report.json`

Large weights and all raw audio remain under the configured HDD root. A failed
or blocked parity gate is itself an H2 artifact and must not be deleted.

The launcher is resumable by model: a re-run for one model replaces only that
model's two preprocessing rows and retains the completed rows for other
models. This permits a conservative batch size for a large model without
discarding a completed small-model calibration.
