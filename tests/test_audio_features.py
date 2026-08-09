import numpy as np

from src.audio_features import FEATURE_NAMES, compute_features, deterministic_crop, preemphasize


def test_registry_and_signal_sanity() -> None:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    sine = 0.3 * np.sin(2 * np.pi * 160 * time)
    features = compute_features(sine)
    assert tuple(features) == FEATURE_NAMES
    assert len(features) == 28
    assert 140 < features["f0_median_hz"] < 180
    assert features["crest_factor_db"] > 2


def test_crop_and_preemphasis_are_deterministic() -> None:
    waveform = np.arange(100, dtype=np.float32)
    first = deterministic_crop(waveform, target_samples=257)
    second = deterministic_crop(waveform, target_samples=257)
    assert np.array_equal(first, second)
    emphasized = preemphasize(first)
    assert emphasized.shape == first.shape
