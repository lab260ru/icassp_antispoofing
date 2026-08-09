# H2 throughput run 001 — parity-validated ONNX scorer sweep

**Run time:** 2026-08-09T19:43Z  
**Purpose:** engineering measurement on frozen, unmodified calibration inputs;
not an intervention, causal, or training result.

## Contract

- Inputs: the 128-clip ASVspoof2019_LA manifest and model-specific parity
  choices from `PARITY_RUN_001.md`.
- Provider: CUDAExecutionProvider, with no CPU fallback. CUDA libraries are
  exposed using `scripts/run_h2_cuda.sh`.
- Measurement: three full passes over the same 128 prepared clips for each
  batch size; report median wall time. Model loading and waveform decoding are
  excluded after the warm-up. This is a scorer throughput measurement, not an
  end-to-end H2 pipeline throughput figure.
- Candidate batches: 1, 2, 4, 8, 16, and 32. No out-of-memory event occurred.
  Peak VRAM was not captured by this initial utility, so it must not be inferred
  from the 48 GB device capacity.

## Results

| Model | GPU | Frozen preprocessing | Fastest measured batch | Median throughput |
|---|---:|---|---:|---:|
| Spectra-AASIST | 0 | pre-emphasis 0.97 | 8 | 236.32 clips/s |
| AASIST | 1 | raw | 2 | 333.44 clips/s |

The complete per-batch three-repeat timings are committed in
`results/throughput_benchmark/ASVspoof2019_LA/`. The selected batches are
valid only for the named ONNX hash, CUDA provider, frozen waveform contract,
and 128-clip workload. Re-benchmark after model, provider, input-length, or
hardware changes.

## Consequence

Use Spectra batch 8 and AASIST batch 2 as initial safe settings for the
future quality-passed H2 pair scoring pipeline. They do not select a feature,
intervention arm, or H3 training batch size.
