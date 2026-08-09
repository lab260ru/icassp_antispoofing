from __future__ import annotations

import numpy as np
import pytest

from src.onnx_fixed_window import (
    WINDOW_SAMPLES,
    FixedWindowOnnxRunner,
    apply_preprocessing,
    canonicalize_raw_score,
    fixed_first_window_tiled,
)


class _Node:
    def __init__(self, name: str, shape: list[object]) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self) -> None:
        self.last_feed: np.ndarray | None = None

    def get_inputs(self) -> list[_Node]:
        return [_Node("wav", ["batch", WINDOW_SAMPLES])]

    def get_outputs(self) -> list[_Node]:
        return [_Node("logits", ["batch", 2])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["logits"]
        self.last_feed = feeds["wav"].copy()
        values = np.arange(feeds["wav"].shape[0], dtype=np.float32)
        return [np.stack([values, values + 10.0], axis=1)]


def test_fixed_window_uses_first_samples_and_tiles_short_audio() -> None:
    long = np.arange(WINDOW_SAMPLES + 12, dtype=np.float32)
    assert np.array_equal(fixed_first_window_tiled(long), long[:WINDOW_SAMPLES])
    short = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    tiled = fixed_first_window_tiled(short)
    assert tiled.shape == (WINDOW_SAMPLES,)
    assert np.array_equal(tiled[:9], np.tile(short, 3))
    with pytest.raises(ValueError, match="empty"):
        fixed_first_window_tiled(np.array([], dtype=np.float32))


def test_preemphasis_is_explicit_and_not_applied_to_raw() -> None:
    audio = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    assert np.array_equal(apply_preprocessing(audio, "raw"), audio)
    assert np.allclose(apply_preprocessing(audio, "preemphasis_0.97"), [1.0, 1.03, 2.06])
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        apply_preprocessing(audio, "untracked")


def test_runner_retains_two_logits_and_selected_raw_scalar() -> None:
    session = _FakeSession()
    runner = FixedWindowOnnxRunner("not-needed.onnx", session=session, preprocessing="raw", raw_scalar_index=1)
    result = runner.score([np.array([0.1, 0.2], dtype=np.float32), np.array([0.3], dtype=np.float32)])
    assert result.logits.shape == (2, 2)
    assert np.array_equal(result.raw_score, np.array([10.0, 11.0], dtype=np.float32))
    assert session.last_feed is not None
    assert session.last_feed.shape == (2, WINDOW_SAMPLES)
    assert np.array_equal(session.last_feed[0, :6], np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2], dtype=np.float32))


def test_canonicalization_uses_preexisting_arena_orientation() -> None:
    raw = np.array([2.0, -3.0])
    assert np.array_equal(canonicalize_raw_score(raw, "raw_is_spoof"), raw)
    assert np.array_equal(canonicalize_raw_score(raw, "negated_raw_is_spoof"), -raw)
    with pytest.raises(ValueError, match="Unsupported"):
        canonicalize_raw_score(raw, "guessed")
