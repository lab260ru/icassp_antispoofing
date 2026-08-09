import numpy as np
import pytest

from src.h2_waveform_transforms import (
    apply_first_order_allpass_cascade,
    apply_gain,
    apply_polarity,
    apply_spectral_tilt,
    compress_crest_factor,
    standardize_endpoint_silence,
)


SAMPLE_RATE = 16_000


def _tone(seconds: float = 0.4, amplitude: float = 0.1) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)


def test_gain_and_polarity_preserve_float32_and_report_clipping() -> None:
    waveform = _tone()
    gained = apply_gain(waveform, SAMPLE_RATE, 6.0)
    inverted = apply_polarity(waveform, SAMPLE_RATE)
    assert gained.waveform.dtype == np.float32
    assert gained.waveform.shape == waveform.shape
    assert np.allclose(gained.waveform, waveform * (10.0 ** (6.0 / 20.0)))
    assert gained.diagnostics["gain_db"] == 6.0
    assert gained.diagnostics["output_clipping_fraction"] == 0.0
    assert inverted.waveform.dtype == np.float32
    assert np.array_equal(inverted.waveform, -waveform)
    assert inverted.diagnostics["crest_factor_delta_db"] == pytest.approx(0.0, abs=1e-6)


def test_endpoint_silence_replaces_only_endpoints_around_energy_vad_core() -> None:
    core = _tone(seconds=0.15)
    waveform = np.concatenate((np.zeros(800, dtype=np.float32), core, np.zeros(400, dtype=np.float32)))
    result = standardize_endpoint_silence(
        waveform,
        SAMPLE_RATE,
        leading_silence_s=0.05,
        trailing_silence_s=0.05,
        frame_ms=10.0,
        hop_ms=10.0,
        guard_ms=0.0,
    )
    start = int(result.diagnostics["speech_start_sample"])
    end = int(result.diagnostics["speech_end_sample"])
    target = 800
    assert result.waveform.dtype == np.float32
    assert np.array_equal(result.waveform[target : target + (end - start)], waveform[start:end])
    assert np.array_equal(result.waveform[:target], np.zeros(target, dtype=np.float32))
    assert np.array_equal(result.waveform[-target:], np.zeros(target, dtype=np.float32))
    assert result.diagnostics["target_leading_samples"] == target
    assert result.diagnostics["target_trailing_samples"] == target
    with pytest.raises(ValueError, match="no speech-like"):
        standardize_endpoint_silence(np.zeros(2_000, dtype=np.float32), SAMPLE_RATE, 0.05, 0.05)


def test_drc_reduces_crest_factor_and_preserves_loudness_diagnostics() -> None:
    waveform = _tone(seconds=1.0)
    waveform[2_000] = 0.8
    waveform[7_000] = -0.8
    result = compress_crest_factor(waveform, SAMPLE_RATE, target_reduction_db=3.0)
    assert result.waveform.dtype == np.float32
    assert result.waveform.shape == waveform.shape
    assert result.diagnostics["achieved_crest_reduction_db"] > 0.5
    assert result.diagnostics["output_crest_factor_db"] < result.diagnostics["input_crest_factor_db"]
    assert result.diagnostics["loudness_matching_requested"] is True
    assert "loudness_delta_lu" in result.diagnostics
    identity = compress_crest_factor(waveform, SAMPLE_RATE, target_reduction_db=0.0)
    assert np.array_equal(identity.waveform, waveform)


def test_spectral_tilt_is_deterministic_float32_and_moves_slope_in_requested_direction() -> None:
    waveform = np.random.default_rng(2609).normal(0.0, 0.03, size=2 * SAMPLE_RATE).astype(np.float32)
    plus = apply_spectral_tilt(waveform, SAMPLE_RATE, tilt_db_per_octave=3.0)
    minus = apply_spectral_tilt(waveform, SAMPLE_RATE, tilt_db_per_octave=-3.0)
    assert plus.waveform.dtype == np.float32
    assert plus.waveform.shape == waveform.shape
    assert np.isfinite(plus.waveform).all()
    assert plus.diagnostics["tilt_n_fft"] == 1024
    assert plus.diagnostics["tilt_hop_length"] == 256
    assert plus.diagnostics["achieved_global_spectral_tilt_db_per_octave"] > 1.5
    assert minus.diagnostics["achieved_global_spectral_tilt_db_per_octave"] < -1.5
    identity = apply_spectral_tilt(waveform, SAMPLE_RATE, tilt_db_per_octave=0.0)
    assert np.array_equal(identity.waveform, waveform)
    with pytest.raises(ValueError, match="within"):
        apply_spectral_tilt(waveform, SAMPLE_RATE, tilt_db_per_octave=13.0)
    with pytest.raises(ValueError, match="Nyquist"):
        apply_spectral_tilt(waveform, SAMPLE_RATE, tilt_db_per_octave=3.0, reference_hz=8_000.0)


def test_allpass_is_stable_and_rejects_unstable_coefficients() -> None:
    waveform = _tone(seconds=1.0)
    result = apply_first_order_allpass_cascade(waveform, SAMPLE_RATE, coefficients=(0.3, -0.4))
    assert result.waveform.dtype == np.float32
    assert result.waveform.shape == waveform.shape
    assert np.isfinite(result.waveform).all()
    assert result.diagnostics["allpass_order"] == 2
    trimmed_input = waveform[1_000:-1_000]
    trimmed_output = result.waveform[1_000:-1_000]
    assert np.sqrt(np.mean(trimmed_output**2)) == pytest.approx(np.sqrt(np.mean(trimmed_input**2)), rel=0.02)
    with pytest.raises(ValueError, match="strictly inside"):
        apply_first_order_allpass_cascade(waveform, SAMPLE_RATE, coefficients=(1.0,))
