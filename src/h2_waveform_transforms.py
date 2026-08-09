"""Deterministic waveform transforms for H2 quality-gated interventions.

This module deliberately performs no model scoring, data loading, or quality
gate decisions. Each transform preserves float32 mono waveform representation
and returns the diagnostics needed by the later H2 manifest builder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyloudnorm as pyln
from scipy import signal


EPS = 1e-10
CLIP_THRESHOLD = 0.999


@dataclass(frozen=True)
class TransformResult:
    """A waveform transform plus manifest-friendly before/after diagnostics."""

    waveform: np.ndarray
    diagnostics: Mapping[str, Any]


def _validate_waveform(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 1:
        raise ValueError(f"Expected a mono one-dimensional waveform, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("Waveform must contain at least one sample")
    waveform = array.astype(np.float32, copy=True)
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform contains non-finite samples")
    return waveform


def _validate_sample_rate(sample_rate: int) -> int:
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return rate


def _integrated_lufs(waveform: np.ndarray, sample_rate: int) -> float:
    """Return integrated loudness or NaN when a clip is too short for EBU R128."""
    try:
        return float(pyln.Meter(sample_rate).integrated_loudness(waveform.astype(np.float64)))
    except Exception:
        return float("nan")


def waveform_diagnostics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    """Compute compact diagnostics without deciding whether a pair passes H2."""
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(waveform)))
    crest = float(20.0 * np.log10(max(peak / max(rms, EPS), EPS)))
    return {
        "samples": int(waveform.size),
        "duration_s": float(waveform.size / rate),
        "peak_abs": peak,
        "rms": rms,
        "crest_factor_db": crest,
        "integrated_lufs": _integrated_lufs(waveform, rate),
        "clipping_fraction": float(np.mean(np.abs(waveform) >= CLIP_THRESHOLD)),
    }


def _result(
    arm: str,
    original: np.ndarray,
    transformed: np.ndarray,
    sample_rate: int,
    extra: Mapping[str, Any] | None = None,
) -> TransformResult:
    before = waveform_diagnostics(original, sample_rate)
    after = waveform_diagnostics(transformed, sample_rate)
    diagnostics: dict[str, Any] = {"arm": arm}
    diagnostics.update({f"input_{key}": value for key, value in before.items()})
    diagnostics.update({f"output_{key}": value for key, value in after.items()})
    diagnostics["loudness_delta_lu"] = float(after["integrated_lufs"] - before["integrated_lufs"])
    diagnostics["crest_factor_delta_db"] = float(after["crest_factor_db"] - before["crest_factor_db"])
    if extra:
        diagnostics.update(extra)
    return TransformResult(waveform=np.asarray(transformed, dtype=np.float32), diagnostics=diagnostics)


def apply_gain(audio: np.ndarray, sample_rate: int, gain_db: float) -> TransformResult:
    """Apply a signed global gain without clipping prevention or normalization."""
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    if not np.isfinite(gain_db):
        raise ValueError(f"gain_db must be finite, got {gain_db}")
    linear_gain = float(10.0 ** (float(gain_db) / 20.0))
    transformed = (waveform * linear_gain).astype(np.float32)
    return _result("gain", waveform, transformed, rate, {"gain_db": float(gain_db), "linear_gain": linear_gain})


def apply_polarity(audio: np.ndarray, sample_rate: int) -> TransformResult:
    """Invert waveform polarity; all amplitude-only diagnostics should remain equal."""
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    return _result("polarity", waveform, (-waveform).astype(np.float32), rate)


def _energy_vad_bounds(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    relative_threshold_db: float,
    absolute_floor: float,
    guard_ms: float,
) -> tuple[int, int, dict[str, float | int]]:
    if frame_ms <= 0 or hop_ms <= 0 or guard_ms < 0:
        raise ValueError("frame_ms and hop_ms must be positive; guard_ms must be non-negative")
    if relative_threshold_db >= 0:
        raise ValueError("relative_threshold_db must be negative")
    if absolute_floor <= 0:
        raise ValueError("absolute_floor must be positive")
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    starts = np.arange(0, waveform.size, hop, dtype=np.int64)
    ends = np.minimum(starts + frame, waveform.size)
    energy = np.concatenate(([0.0], np.cumsum(waveform.astype(np.float64) ** 2)))
    rms = np.sqrt((energy[ends] - energy[starts]) / np.maximum(ends - starts, 1))
    maximum = float(np.max(rms))
    threshold = max(float(absolute_floor), maximum * (10.0 ** (relative_threshold_db / 20.0)))
    active = rms >= threshold
    if not active.any() or maximum < absolute_floor:
        raise ValueError("Energy VAD found no speech-like active region")
    first, last = np.flatnonzero(active)[[0, -1]]
    guard = int(round(sample_rate * guard_ms / 1000.0))
    speech_start = max(0, int(starts[first]) - guard)
    speech_end = min(waveform.size, int(ends[last]) + guard)
    return speech_start, speech_end, {
        "vad_frame_ms": float(frame_ms),
        "vad_hop_ms": float(hop_ms),
        "vad_relative_threshold_db": float(relative_threshold_db),
        "vad_absolute_floor": float(absolute_floor),
        "vad_guard_ms": float(guard_ms),
        "vad_threshold_rms": float(threshold),
        "vad_max_frame_rms": maximum,
    }


def standardize_endpoint_silence(
    audio: np.ndarray,
    sample_rate: int,
    leading_silence_s: float,
    trailing_silence_s: float,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    relative_threshold_db: float = -40.0,
    absolute_floor: float = 1e-4,
    guard_ms: float = 20.0,
) -> TransformResult:
    """Replace endpoint regions with deterministic zero padding around an energy-VAD core.

    The selected core is copied exactly. Output duration may change, so later H2
    analysis must retain its detector-window position diagnostics.
    """
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    if leading_silence_s < 0 or trailing_silence_s < 0:
        raise ValueError("Requested endpoint silence durations must be non-negative")
    speech_start, speech_end, vad = _energy_vad_bounds(
        waveform,
        rate,
        frame_ms,
        hop_ms,
        relative_threshold_db,
        absolute_floor,
        guard_ms,
    )
    leading = int(round(float(leading_silence_s) * rate))
    trailing = int(round(float(trailing_silence_s) * rate))
    core = waveform[speech_start:speech_end]
    transformed = np.concatenate((np.zeros(leading, dtype=np.float32), core, np.zeros(trailing, dtype=np.float32)))
    extra: dict[str, Any] = {
        **vad,
        "speech_start_sample": speech_start,
        "speech_end_sample": speech_end,
        "speech_core_samples": int(core.size),
        "original_leading_samples": speech_start,
        "original_trailing_samples": int(waveform.size - speech_end),
        "target_leading_samples": leading,
        "target_trailing_samples": trailing,
        "position_changed": bool(leading != speech_start),
    }
    return _result("endpoint_silence", waveform, transformed, rate, extra)


def _block_envelope(waveform: np.ndarray, sample_rate: int, frame_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    starts = np.arange(0, waveform.size, frame, dtype=np.int64)
    ends = np.minimum(starts + frame, waveform.size)
    energy = np.concatenate(([0.0], np.cumsum(waveform.astype(np.float64) ** 2)))
    rms = np.sqrt((energy[ends] - energy[starts]) / np.maximum(ends - starts, 1))
    centers = (starts + ends - 1) / 2.0
    return centers, rms


def _compress_with_threshold(
    waveform: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    frame_ms: float,
) -> np.ndarray:
    centers, block_rms = _block_envelope(waveform, sample_rate, frame_ms)
    step_s = frame_ms / 1000.0
    attack = float(np.exp(-step_s / max(attack_ms / 1000.0, EPS)))
    release = float(np.exp(-step_s / max(release_ms / 1000.0, EPS)))
    smoothed = np.empty_like(block_rms)
    current = float(block_rms[0])
    for index, value in enumerate(block_rms):
        coefficient = attack if value > current else release
        current = coefficient * current + (1.0 - coefficient) * float(value)
        smoothed[index] = current
    level_db = 20.0 * np.log10(np.maximum(smoothed, EPS))
    gain_reduction_db = np.maximum(level_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    sample_gain_db = np.interp(
        np.arange(waveform.size, dtype=np.float64),
        centers,
        -gain_reduction_db,
        left=-float(gain_reduction_db[0]),
        right=-float(gain_reduction_db[-1]),
    )
    return (waveform.astype(np.float64) * (10.0 ** (sample_gain_db / 20.0))).astype(np.float32)


def compress_crest_factor(
    audio: np.ndarray,
    sample_rate: int,
    target_reduction_db: float,
    *,
    ratio: float = 8.0,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
    frame_ms: float = 10.0,
    match_integrated_loudness: bool = True,
    search_steps: int = 28,
) -> TransformResult:
    """Apply deterministic block-envelope DRC, selecting a threshold for a crest target.

    Loudness restoration is a scalar gain, intentionally without a limiter. A
    later H2 quality gate must reject clips whose restored waveform clips or
    misses the prescribed loudness tolerance.
    """
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    if target_reduction_db < 0 or not np.isfinite(target_reduction_db):
        raise ValueError("target_reduction_db must be finite and non-negative")
    if ratio <= 1 or attack_ms <= 0 or release_ms <= 0 or frame_ms <= 0 or search_steps <= 0:
        raise ValueError("ratio must exceed 1; timing parameters and search_steps must be positive")
    before = waveform_diagnostics(waveform, rate)
    original_crest = float(before["crest_factor_db"])
    target = float(target_reduction_db)
    if target == 0.0:
        transformed = waveform.copy()
        threshold_db = float("nan")
    else:
        _, envelope = _block_envelope(waveform, rate, frame_ms)
        lower = -120.0
        upper = float(20.0 * np.log10(max(float(np.max(envelope)), EPS)) + 6.0)
        transformed = waveform.copy()
        threshold_db = upper
        best_error = float("inf")
        for _ in range(int(search_steps)):
            trial_threshold = (lower + upper) / 2.0
            trial = _compress_with_threshold(waveform, rate, trial_threshold, ratio, attack_ms, release_ms, frame_ms)
            trial_crest = float(waveform_diagnostics(trial, rate)["crest_factor_db"])
            reduction = original_crest - trial_crest
            error = abs(reduction - target)
            if error < best_error:
                transformed, threshold_db, best_error = trial, trial_threshold, error
            if reduction < target:
                upper = trial_threshold
            else:
                lower = trial_threshold
    compressed_lufs = _integrated_lufs(transformed, rate)
    reference_lufs = float(before["integrated_lufs"])
    loudness_gain_db = 0.0
    loudness_matched = False
    if match_integrated_loudness and np.isfinite(reference_lufs) and np.isfinite(compressed_lufs):
        loudness_gain_db = float(reference_lufs - compressed_lufs)
        transformed = (transformed * (10.0 ** (loudness_gain_db / 20.0))).astype(np.float32)
        loudness_matched = True
    after = waveform_diagnostics(transformed, rate)
    achieved = original_crest - float(after["crest_factor_db"])
    return _result(
        "drc_crest_factor",
        waveform,
        transformed,
        rate,
        {
            "target_crest_reduction_db": target,
            "achieved_crest_reduction_db": float(achieved),
            "crest_target_error_db": float(achieved - target),
            "compressor_threshold_db": float(threshold_db),
            "compressor_ratio": float(ratio),
            "compressor_attack_ms": float(attack_ms),
            "compressor_release_ms": float(release_ms),
            "compressor_frame_ms": float(frame_ms),
            "loudness_matching_requested": bool(match_integrated_loudness),
            "loudness_matched": loudness_matched,
            "loudness_restore_gain_db": loudness_gain_db,
        },
    )


def apply_first_order_allpass_cascade(
    audio: np.ndarray,
    sample_rate: int,
    coefficients: Sequence[float],
) -> TransformResult:
    """Apply a stable cascade H(z)=(a + z^-1)/(1 + a z^-1) for each |a| < 1."""
    waveform = _validate_waveform(audio)
    rate = _validate_sample_rate(sample_rate)
    values = tuple(float(value) for value in coefficients)
    if not values:
        raise ValueError("At least one all-pass coefficient is required")
    if not all(np.isfinite(value) and abs(value) < 1.0 for value in values):
        raise ValueError("All-pass coefficients must be finite and strictly inside (-1, 1)")
    transformed = waveform.astype(np.float64)
    for coefficient in values:
        transformed = signal.lfilter([coefficient, 1.0], [1.0, coefficient], transformed)
    return _result(
        "allpass_phase",
        waveform,
        transformed.astype(np.float32),
        rate,
        {"allpass_coefficients": values, "allpass_order": len(values)},
    )
