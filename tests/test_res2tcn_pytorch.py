from __future__ import annotations

import numpy as np
import torch

from src.onnx_fixed_window import WINDOW_SAMPLES
from src.res2tcn_pytorch import Res2TCNGuardPyTorchRunner


class _FakeRes2Model:
    def __call__(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = waveforms[:, 0]
        logits = torch.stack((raw + 1.0, raw - 1.0), dim=1)
        return waveforms, logits


def test_injected_runner_uses_exact_first_window_tiling_and_logit_one() -> None:
    runner = Res2TCNGuardPyTorchRunner("unused", device="cpu", model=_FakeRes2Model())
    result = runner.score([np.array([2.0, -3.0], dtype=np.float32), np.array([4.0], dtype=np.float32)])
    assert result.detector_waveforms.shape == (2, WINDOW_SAMPLES)
    assert np.array_equal(result.detector_waveforms[0, :6], np.array([2.0, -3.0, 2.0, -3.0, 2.0, -3.0]))
    assert np.array_equal(result.logits, np.array([[3.0, 1.0], [5.0, 3.0]], dtype=np.float32))
    assert np.array_equal(result.raw_score, np.array([1.0, 3.0], dtype=np.float32))


def test_injected_runner_rejects_wrong_prewindowed_length() -> None:
    runner = Res2TCNGuardPyTorchRunner("unused", device="cpu", model=_FakeRes2Model())
    try:
        runner.score(np.zeros((2, WINDOW_SAMPLES - 1), dtype=np.float32))
    except ValueError as error:
        assert "Pre-windowed" in str(error)
    else:
        raise AssertionError("Expected pre-window length validation")
