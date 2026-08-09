# H2 parity run 001 — baseline-only outcome

**Run time:** 2026-08-09T19:39Z (CUDA-validated rerun of the same frozen
manifest)  
**Scope:** original, unmodified ASVspoof2019_LA calibration clips only. This is
not an intervention result and does not test any causal hypothesis.

## Frozen inputs

- Dataset: `ASVspoof2019_LA` revision
  `9492c4a85ad91508b6da03c92c98c58aeaa02424`.
- Calibration manifest: 128 clips, 64 per authoritative label, seed 2609;
  SHA-256 `0a1918b15fc972f9ffb2b40c2150099d097cc2daff38372673a6e15dbd0cbbd1`.
- Detector contract: 16 kHz float32; named preprocessing followed by first
  64,600 samples or tile-repeat for short inputs; selected raw scalar = logit
  index 1. Both logits and the waveform hashes are retained in the score
  tables.
- Baseline orientation: the independently pinned Arena artifacts establish
  `negated_raw_is_spoof` for both models. The runner's canonical score is
  therefore `score_spoof = -logit_1`; it was not inferred from intervention
  responses.

## Results

| Model | ONNX SHA-256 | Batch | Raw preprocessing rho | Pre-emphasis rho | Frozen choice |
|---|---|---:|---:|---:|---|
| AASIST | `130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1` | 4 | 0.999994 | 0.882542 | raw |
| Spectra-AASIST | `85b217c72cf2ff1fc820827fa65e3dc49891fba49746931b06f1a5ad8a336a9b` | 4 | 0.880888 | 0.999971 | pre-emphasis 0.97 |

Each value is the Spearman correlation of the runner's selected raw scalar
with the matching pinned Arena raw score over all 128 clips. The threshold was
predeclared as 0.999; the failed preprocessing candidate is retained in the
artifact bundle. The selection followed the predeclared rule: use raw when it
passes, otherwise use the sole passing non-raw candidate.

## Runtime validation

The first CPU-fallback attempt exposed missing dynamic-library discovery for
`libcublasLt.so.13`. The runtime library is present in the active Python
environment's CUDA 13 wheel; launching through `scripts/run_h2_cuda.sh` made
it visible before Python started. The rerun above records
`CUDAExecutionProvider` and `used_cpu_fallback=false` for all four candidate
scorings (Spectra on CUDA device 0 and AASIST on device 1). These are
functional parity gates, not throughput benchmarks; select H2 production batch
sizes only after a separate, recorded sweep on the frozen calibration input.

## Compact artifacts

The complete compact bundle is in
`results/parity_calibration/ASVspoof2019_LA/`, including the frozen manifest,
four per-clip raw-logit tables, `parity_summary.csv`, and `parity_report.json`.

## Consequence for H2

The two executable scorers pass their *baseline parity* gate with the choices
above. H2 remains blocked for causal claims until the H1 candidate/arm freeze,
the fixed ASR WER quality gate, and a validated fourth scorer or pre-frozen
fallback are complete, as specified in `model_capability_audit.md`.
