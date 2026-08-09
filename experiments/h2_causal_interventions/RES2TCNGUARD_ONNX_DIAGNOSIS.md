# Res2TCNGuard ONNX parity diagnosis

**Status:** PyTorch alternative eligible; the current ONNX artifact is blocked.

This is a baseline-only implementation diagnostic on the pre-existing frozen
128-clip `ASVspoof2019_LA` calibration manifest. It includes no transformed
waveforms, pair-quality gate, causal estimate, or H2 claim.

## Fixed inputs

- Dataset revision: `9492c4a85ad91508b6da03c92c98c58aeaa02424`.
- Manifest: `results/parity_calibration/ASVspoof2019_LA/calibration_manifest.parquet`,
  64 clips per class, SHA-256
  `0a1918b15fc972f9ffb2b40c2150099d097cc2daff38372673a6e15dbd0cbbd1`.
- Model bundle: `SpeechAntiSpoofingBenchmarks/Res2TCNGuard` revision
  `4624265fa5e88c0abe425e37c278f3a9288aa914`.
- Both paths received the exact same float32 first-64,600/tile-repeat windows:
  all 128 detector-input SHA-256 values matched.

## Diagnostic result

The ONNX `logit_1` has the expected bona-fide direction, but fails the locked
ordering threshold (`rho >= 0.999`) against the pinned Arena baseline:

| Executable path | Raw-score vs. Arena Spearman | Raw-score MAE | Gate |
|---|---:|---:|---|
| `res2tcnguard.onnx`, raw | 0.911686 | not used for eligibility | blocked |
| `res2tcnguard.onnx`, pre-emphasis 0.97 | 0.750177 | not used for eligibility | blocked |
| pinned `evaluate.py` + `best_1.495.pth`, raw | 0.999994 | 0.002439 | passes |

The direct PyTorch-to-ONNX comparison on those same raw windows has Spearman
`0.911675` and `logit_1` MAE `3.210553`; the two PyTorch/Arena logits agree to
within a maximum absolute difference of `0.016964`. Thus the mismatch is not
caused by score orientation, clip selection, audio decoding, or windowing. It
is an ONNX export/runtime divergence.

## Consequence

`res2tcnguard.onnx` must not be used for H2 scoring. The small, exact PyTorch
bundle is the executable Res2TCNGuard candidate instead, subject to the same
separate pair-quality and panel-completion gates as every other H2 scorer.
The reproducible raw-only gate is
`scripts/calibrate_h2_res2_pytorch_parity.py`; it writes its own score table
and JSON report without overwriting the blocked ONNX evidence.

## Reproduction artifacts

```bash
PYTHONPATH=. python3 -m pytest -q tests/test_res2tcn_pytorch.py \
  tests/test_onnx_fixed_window.py tests/test_h2_parity_calibration.py
PYTHONPATH=. python3 scripts/calibrate_h2_res2_pytorch_parity.py \
  --device cuda:2 --batch-size 4
```

The gate writes
`results/parity_calibration/ASVspoof2019_LA/Res2TCNGuard__pytorch_raw__scores.parquet`
and `Res2TCNGuard__pytorch_parity_report.json`. The latter pins the three
defining bundle-file hashes as well as manifest, device, batch, and gate
provenance.
