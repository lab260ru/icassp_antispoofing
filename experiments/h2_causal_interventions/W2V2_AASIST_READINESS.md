# W2V2-AASIST future fourth-scorer readiness

**Audit time:** 2026-08-09T22:07:22Z  
**Scope:** read-only metadata/source audit. No model, checkpoint, code, engine,
or data was downloaded; no waveform was scored. This is an engineering
readiness note, not an H2 result or a permission to reopen the failed H2 run.

## Decision

The **ONNX route is conditionally feasible** and is the only reasonable route
for a future W2V2-AASIST fourth scorer. It can reuse the project's generic
fixed-window ONNX machinery *only after* the downloaded graph proves the
required one-input/two-logit contract and clears the pinned-Arena parity gate.
It is presently **blocked**: the artifact is absent locally and its graph has
not been inspected.

Do not repair or use the advertised PyTorch/TensorRT wrapper. At the pinned
revision its README names `w2v2_aasist.py` and `_net.py`, but neither file is
present in the revision's root tree. The available `trt_w2v2_aasist.py` imports
the missing `w2v2_aasist` module. Even with those files, the PyTorch path also
requires the separate `xlsr2_300m.pt` fairseq checkpoint. This makes the ONNX
artifact the bounded, reproducible acquisition target.

The current crest-factor H2 panel remains ineligible for a separate reason:
`H2_QUALITY_RUN_002.md` records that three of four arms miss the 90% retention
gate, so no score-eligible pair manifest exists. A W2V2 readiness pass must not
score those pairs or turn the existing exploratory crest candidate into a
causal claim.

## Pinned source and artifact identity

| Item | Exact identity |
|---|---|
| Hub repository | `SpeechAntiSpoofingBenchmarks/W2V2-AASIST` |
| Revision (required) | `196128e5a5101d5cb6ac7701597891bc7de7e7b5` |
| Intended executable artifact | `w2v2-aasist.onnx` |
| ONNX size | 1,264,789,106 bytes |
| ONNX LFS content SHA-256 | `837169def567cd68d94f7b5a6bd7ef55a7b64ea9cec61364944bcb66521e92d3` |
| ONNX Git blob ID | `f333fa6bfc22191d6c5dc27ee0b79e4deee5296f` |
| Non-selected PyTorch checkpoint | `LA_model.pth`, 1,271,633,441 bytes; LFS SHA-256 `bd6f36097259fe54e7004eb983651e5304d807be81156dbd04faccb70d91e10c` |
| Intended local directory | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/models/W2V2-AASIST/196128e5a5101d5cb6ac7701597891bc7de7e7b5` |

The source is the [pinned Hub revision](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/tree/196128e5a5101d5cb6ac7701597891bc7de7e7b5).
The repository's `data/arena-index.yaml` pins the same revision and local
directory. A direct, read-only Hub tree query at the audit time reported the
two large files above, the three listed runtime files, `.gitattributes`, and
the `.eval_results` directory; it did not list either advertised wrapper. The
declared local revision directory has no top-level model runtime file: an
explicit check for `w2v2-aasist.onnx` returned “No such file or directory.”

## What is known versus what must be verified

The pinned README says the Arena scorer consumes 16-kHz mono raw audio,
applies a deterministic first-64,600-sample window (tile-repeat when shorter),
and uses logit/index 1 as increasing bona-fide evidence. The project’s pinned
Arena score catalog independently records W2V2-AASIST orientation as
`negated_raw_is_spoof`. These are useful *predeclared calibration hypotheses*,
not a substitute for inspecting or executing the ONNX graph.

`src/onnx_fixed_window.py` already has a minimal generic runner. It accepts
exactly one ONNX input and one output, requires shapes `[batch, 64600]` and
`[batch, 2]`, feeds `float32` audio, retains both logits, and has explicit raw
and pre-emphasis candidates. `scripts/calibrate_h2_onnx_parity.py` can freeze
the score-independent 128-clip ASVspoof2019-LA calibration panel, join the
pinned Arena `scores.txt`, enforce matching raw-score rank order
(Spearman >= 0.999), and preserve both preprocessing outcomes. Therefore a
minimal runner is feasible **if and only if** graph inspection confirms that
contract. W2V2 is intentionally absent from its `ONNX_FILENAMES` allow-list,
from the default model panel, and from `src.h2_paired_scoring.SCORER_SPECS`.
Those are safeguards, not omissions to bypass.

## Safe future acquisition and validation plan

This sequence applies only after a newly approved outer-loop protocol supplies
quality-passed pairs and a pre-score-frozen W2V2 panel. It must run before any
future transformed-audio score is viewed.

1. Download only `w2v2-aasist.onnx`, `README.md`, and `meta.yaml` with the
   exact revision into the stated HDD directory. Do not fetch `LA_model.pth`,
   `xlsr2_300m.pt`, or an engine. Hash the completed ONNX file and require an
   exact match to `837169def567cd68d94f7b5a6bd7ef55a7b64ea9cec61364944bcb66521e92d3`.
   Record the downloader/library version, revision, path, byte count, and hash
   in a committed acquisition manifest. A partial or mismatched artifact is a
   hard stop.
2. Before adding any allow-list entry, inspect the graph offline with ONNX
   Runtime and record every input/output name, dtype, static/dynamic dimension,
   and external-data dependency. Continue only if there is exactly one
   float32 waveform input compatible with `[B,64600]` and exactly one
   float32 `[B,2]` logits output. Otherwise write a new protocol and an
   explicitly tested adapter; never coerce it through `FixedWindowOnnxRunner`.
3. Add W2V2 to `ONNX_FILENAMES` in a dedicated, pre-results implementation
   commit, with a signature regression test. Run the existing calibration
   script explicitly for W2V2 on the frozen 128-clip ASVspoof2019-LA panel,
   initially with `raw` preprocessing and scalar index 1, through
   `scripts/run_h2_cuda.sh` on physical GPU 3. Preserve both logits, exact
   detector-input hashes, provider, model hash, and all parity rows. Acceptance
   requires CUDA execution without fallback, correct predeclared orientation,
   and raw-score vs. pinned-Arena Spearman >= 0.999. A failure remains a
   committed parity artifact; neither orientation nor windowing may be changed
   after treatment scores are examined.
4. Only after acceptance, run the recorded batch sweep (1, 2, 4, 8, 16, 32)
   on the unmodified frozen calibration clips, retaining provider, exact hash,
   throughput, and peak VRAM. Freeze the selected batch in a dedicated scorer
   spec and add paired-scoring tests before adding W2V2 to `SCORER_SPECS`.
5. A later causal scoring run still needs a newly valid quality freeze that
   clears every arm and validation of all four model scores. The previous
   `h2_quality_full_002` output may not be reused or selectively filtered.

## Audit evidence and checks

- Local source/protocol evidence: `data/arena-index.yaml`,
  `experiments/h2_causal_interventions/model_capability_audit.md`,
  `H2_PAIRED_SCORING.md`, `H2_QUALITY_RUN_002.md`,
  `src/onnx_fixed_window.py`, `scripts/calibrate_h2_onnx_parity.py`, and
  `src/h2_paired_scoring.py`.
- Read-only pinned-Hub tree and raw README/helper queries resolved the revision,
  LFS hashes, file sizes, missing wrappers, documented windowing, and logit
  convention. They did not retrieve any model binary.
- Framework regression check:
  `PYTHONPATH=. python3 -m pytest -q tests/test_onnx_fixed_window.py tests/test_h2_parity_calibration.py tests/test_h2_paired_scoring.py`
  passed **11 tests** in 0.54 s. These tests establish the current generic
  runner and post-quality scorer guards; they do not validate W2V2 itself.
