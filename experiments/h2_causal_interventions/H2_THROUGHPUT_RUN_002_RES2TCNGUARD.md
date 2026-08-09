# H2 throughput run 002 — Res2TCNGuard source-PyTorch scorer sweep

**Purpose:** a bounded engineering measurement on the frozen, unmodified
ASVspoof2019_LA 128-clip calibration workload. It is not a waveform
intervention, training run, or causal result.

## Contract

- Scorer: the parity-validated revision-pinned `evaluate.py` + `_net.py` +
  `best_1.495.pth` source-PyTorch bundle. The blocked Res2TCNGuard ONNX export
  is not used.
- Inputs: the established 64-per-label calibration manifest, SHA-256
  `0a1918b15fc972f9ffb2b40c2150099d097cc2daff38372673a6e15dbd0cbbd1`.
- Device: physical GPU 2, exposed directly as `cuda:2` with an unset
  `CUDA_VISIBLE_DEVICES` mapping. The result report records the observed
  device/provider and PyTorch/CUDA versions.
- Candidate batches: 1, 2, 4, 8, 16, and 32. A candidate that throws CUDA OOM
  is retained as an `oom` row and cannot be selected.
- Measurement: one unmeasured candidate-batch warm-up, followed by three
  synchronized full 128-clip passes. Waveform decoding, preparation, and
  model loading are excluded. The result records all wall times, median
  throughput and per-clip latency, plus the maximum PyTorch allocated and
  reserved CUDA bytes over those passes. Memory peaks include persistent model
  allocations because the peak statistics are reset after warm-up, not after
  unloading the scorer.
- Selection: highest successful median clips/s; exact ties resolve to the
  smaller batch. This is deterministic and does not use detector outcomes.

## Execution

From the repository root after the focused non-GPU unit test:

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_h2_res2_throughput.py
PYTHONPATH=. python3 scripts/benchmark_h2_res2_pytorch.py --device cuda:2
```

The compact result artifacts are
`results/throughput_benchmark/ASVspoof2019_LA/Res2TCNGuard_pytorch_batch_sweep.csv`
and `Res2TCNGuard_pytorch_throughput_report.json`. They must be read for the
actual selected batch and observed values; this protocol intentionally does
not prestate a result.

## Observed result — 2026-08-09T20:50Z

The source-PyTorch runner used `PyTorch CUDA` on `cuda:2` (RTX 6000 Ada;
PyTorch `2.11.0+cu130`, CUDA `13.0`) with no `CUDA_VISIBLE_DEVICES` remapping.
All six candidates completed; no OOM row was produced.

| Batch | Median clips/s | Median latency (ms/clip) | Peak allocated / reserved (MiB) |
|---:|---:|---:|---:|
| 1 | 193.23 | 5.175 | 268.75 / 336 |
| 2 | **198.09** | **5.048** | 527.33 / 788 |
| 4 | 193.91 | 5.157 | 1046.38 / 1676 |
| 8 | 173.83 | 5.753 | 2078.86 / 3450 |
| 16 | 167.64 | 5.965 | 4150.30 / 6992 |
| 32 | 159.62 | 6.265 | 8285.32 / 14076 |

**Selected setting:** batch **2**, the highest-throughput completed candidate
under the locked tie break. This choice is limited to this model revision,
source runner, GPU, fixed windowing contract, and 128 original calibration
clips; it is not a training batch choice or a detector-reliance finding.
