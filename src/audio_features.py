"""Versioned, interpretable waveform feature registry for H1."""

from __future__ import annotations

import math

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy import signal


FEATURE_VERSION = "v1_28"
TARGET_SAMPLE_RATE = 16_000
CROP_SAMPLES = 64_600
N_FFT = 1_024
HOP = 256
EPS = 1e-10

FEATURE_NAMES = (
    "crest_factor_db",
    "peak_dbfs",
    "rms_dbfs",
    "integrated_lufs",
    "silence_fraction",
    "clipping_fraction",
    "frame_rms_cv",
    "frame_rms_range_db",
    "spectral_centroid_hz",
    "spectral_centroid_std_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "spectral_slope_db_per_khz",
    "low_high_energy_ratio_db",
    "spectral_flux",
    "f0_median_hz",
    "f0_iqr_hz",
    "voiced_fraction",
    "hnr_db",
    "cpp_proxy_db",
    "jitter_proxy",
    "shimmer_proxy_db",
    "f0_delta_hz",
    "group_delay_var",
    "inst_freq_dispersion",
    "modulation_centroid_hz",
    "modulation_entropy",
)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), EPS)))


def _safe_mean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else float("nan")


def _safe_median(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(np.median(valid)) if valid.size else float("nan")


def _safe_iqr(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(np.quantile(valid, 0.75) - np.quantile(valid, 0.25)) if valid.size else float("nan")


def resample_mono(audio: np.ndarray, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != target_sample_rate:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=target_sample_rate, res_type="soxr_hq")
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def deterministic_crop(audio: np.ndarray, target_samples: int = CROP_SAMPLES) -> np.ndarray:
    """Center crop or repeat-pad deterministically; never use random evaluation crops."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if y.size >= target_samples:
        start = (y.size - target_samples) // 2
        return y[start : start + target_samples]
    repeats = math.ceil(target_samples / y.size)
    return np.tile(y, repeats)[:target_samples]


def preemphasize(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return y
    return signal.lfilter([1.0, -coefficient], [1.0], y).astype(np.float32)


def waveform_views(audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    full = resample_mono(audio, sample_rate)
    crop = deterministic_crop(full)
    return {
        "full_waveform": full,
        "deterministic_crop": crop,
        "preemphasized_crop": preemphasize(crop),
    }


def _frames(y: np.ndarray) -> np.ndarray:
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    return librosa.util.frame(y, frame_length=N_FFT, hop_length=HOP).T.copy()


def _spectral_features(y: np.ndarray, frame_rms: np.ndarray) -> dict[str, float]:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=False)
    magnitude = np.abs(stft) + EPS
    power = magnitude**2
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SAMPLE_RATE)
    normalized = power / np.maximum(power.sum(axis=0, keepdims=True), EPS)
    centroids = (frequencies[:, None] * normalized).sum(axis=0)
    bandwidths = np.sqrt((((frequencies[:, None] - centroids[None, :]) ** 2) * normalized).sum(axis=0))
    cumulative = np.cumsum(power, axis=0)
    rolloff_index = (cumulative >= 0.85 * cumulative[-1:, :]).argmax(axis=0)
    rolloffs = frequencies[rolloff_index]
    flatness = np.exp(np.mean(np.log(magnitude), axis=0)) / np.maximum(np.mean(magnitude, axis=0), EPS)
    x = frequencies / 1000.0
    x_centered = x - x.mean()
    log_mag = 20.0 * np.log10(magnitude)
    slopes = (x_centered[:, None] * (log_mag - log_mag.mean(axis=0, keepdims=True))).sum(axis=0) / np.maximum((x_centered**2).sum(), EPS)
    low = power[(frequencies >= 0) & (frequencies < 1000)].sum(axis=0)
    high = power[(frequencies >= 4000) & (frequencies <= 8000)].sum(axis=0)
    normalized_mag = magnitude / np.maximum(magnitude.sum(axis=0, keepdims=True), EPS)
    flux = np.sqrt(np.square(np.diff(normalized_mag, axis=1)).sum(axis=0)) if magnitude.shape[1] > 1 else np.array([0.0])
    phase = np.angle(stft)
    phase_freq = np.unwrap(phase, axis=0)
    group_delay = -np.diff(phase_freq, axis=0) / (2.0 * np.pi * np.diff(frequencies)[:, None])
    phase_time = np.unwrap(phase, axis=1)
    instantaneous = np.diff(phase_time, axis=1) / (2.0 * np.pi * (HOP / TARGET_SAMPLE_RATE))
    return {
        "spectral_centroid_hz": _safe_mean(centroids),
        "spectral_centroid_std_hz": float(np.nanstd(centroids)),
        "spectral_bandwidth_hz": _safe_mean(bandwidths),
        "spectral_rolloff85_hz": _safe_mean(rolloffs),
        "spectral_flatness": _safe_mean(flatness),
        "spectral_slope_db_per_khz": _safe_mean(slopes),
        "low_high_energy_ratio_db": _safe_mean(10.0 * np.log10(np.maximum(low, EPS) / np.maximum(high, EPS))),
        "spectral_flux": _safe_mean(flux),
        "group_delay_var": float(np.nanmedian(np.nanvar(group_delay, axis=0))) if group_delay.size else float("nan"),
        "inst_freq_dispersion": float(np.nanmedian(np.nanvar(instantaneous, axis=0))) if instantaneous.size else float("nan"),
    }


def _voice_quality_features(y: np.ndarray, frames: np.ndarray, frame_rms: np.ndarray) -> dict[str, float]:
    padded = y if y.size >= 2 * N_FFT else np.pad(y, (0, 2 * N_FFT - y.size))
    try:
        f0, voiced, _ = librosa.pyin(
            padded,
            fmin=50.0,
            fmax=500.0,
            sr=TARGET_SAMPLE_RATE,
            frame_length=2 * N_FFT,
            hop_length=HOP,
            center=False,
        )
    except Exception:
        f0 = np.full(max(1, frames.shape[0]), np.nan)
        voiced = np.zeros_like(f0, dtype=bool)
    voiced = np.asarray(voiced, dtype=bool)
    f0 = np.asarray(f0, dtype=float)
    usable = min(len(f0), len(frames), len(frame_rms))
    f0, voiced, used_frames, used_rms = f0[:usable], voiced[:usable], frames[:usable], frame_rms[:usable]
    valid = voiced & np.isfinite(f0) & (f0 > 0)
    f0_valid = f0[valid]
    if not f0_valid.size:
        return {
            "f0_median_hz": float("nan"), "f0_iqr_hz": float("nan"), "voiced_fraction": 0.0,
            "hnr_db": float("nan"), "cpp_proxy_db": float("nan"), "jitter_proxy": float("nan"),
            "shimmer_proxy_db": float("nan"), "f0_delta_hz": float("nan"),
        }
    voice_frames = used_frames[valid] * np.hanning(N_FFT)[None, :]
    autocorrelation = np.fft.irfft(np.abs(np.fft.rfft(voice_frames, axis=1)) ** 2, n=N_FFT, axis=1)
    lag = np.clip(np.rint(TARGET_SAMPLE_RATE / f0_valid).astype(int), 1, N_FFT - 1)
    r0 = autocorrelation[:, 0]
    rp = autocorrelation[np.arange(len(lag)), lag]
    hnr = 10.0 * np.log10(np.maximum(rp, EPS) / np.maximum(r0 - rp, EPS))
    log_magnitude = np.log(np.abs(np.fft.rfft(voice_frames, axis=1)) + EPS)
    cepstrum = np.fft.irfft(log_magnitude, n=N_FFT, axis=1)
    quefrency_low, quefrency_high = int(TARGET_SAMPLE_RATE / 500.0), int(TARGET_SAMPLE_RATE / 50.0)
    cepstral_band = cepstrum[:, quefrency_low:quefrency_high]
    cpp = cepstral_band.max(axis=1) - np.median(cepstral_band, axis=1)
    periods = 1.0 / f0_valid
    jitter = np.abs(np.diff(periods)).mean() / np.maximum(periods.mean(), EPS) if len(periods) > 1 else float("nan")
    voiced_rms = used_rms[valid]
    shimmer = 20.0 * np.log10(np.maximum(np.abs(np.diff(voiced_rms)).mean(), EPS) / np.maximum(voiced_rms.mean(), EPS)) if len(voiced_rms) > 1 else float("nan")
    f0_delta = np.abs(np.diff(f0_valid)).mean() if len(f0_valid) > 1 else float("nan")
    return {
        "f0_median_hz": _safe_median(f0_valid),
        "f0_iqr_hz": _safe_iqr(f0_valid),
        "voiced_fraction": float(valid.mean()),
        "hnr_db": _safe_mean(hnr),
        "cpp_proxy_db": _safe_mean(cpp),
        "jitter_proxy": float(jitter),
        "shimmer_proxy_db": float(shimmer),
        "f0_delta_hz": float(f0_delta),
    }


def _modulation_features(frame_rms: np.ndarray) -> dict[str, float]:
    envelope = frame_rms - frame_rms.mean()
    if envelope.size < 4 or not np.any(np.abs(envelope) > EPS):
        return {"modulation_centroid_hz": 0.0, "modulation_entropy": 0.0}
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(len(envelope)))) ** 2
    frequencies = np.fft.rfftfreq(len(envelope), d=HOP / TARGET_SAMPLE_RATE)
    spectrum[0] = 0.0
    probability = spectrum / np.maximum(spectrum.sum(), EPS)
    entropy = -np.sum(probability[probability > 0] * np.log(probability[probability > 0]))
    return {
        "modulation_centroid_hz": float((frequencies * probability).sum()),
        "modulation_entropy": float(entropy / np.log(max(len(probability), 2))),
    }


def compute_features(audio: np.ndarray) -> dict[str, float]:
    """Compute the frozen 28-feature registry for a mono, 16 kHz waveform."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        y = np.zeros(N_FFT, dtype=np.float32)
    frames = _frames(y)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    try:
        integrated_lufs = float(pyln.Meter(TARGET_SAMPLE_RATE).integrated_loudness(y.astype(np.float64)))
    except Exception:
        integrated_lufs = float("nan")
    silence_threshold = max(frame_rms.max() * (10.0 ** (-50.0 / 20.0)), EPS)
    output = {
        "crest_factor_db": _db(peak / max(rms, EPS)),
        "peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "integrated_lufs": integrated_lufs,
        "silence_fraction": float((frame_rms <= silence_threshold).mean()),
        "clipping_fraction": float((np.abs(y) >= 0.999).mean()),
        "frame_rms_cv": float(frame_rms.std() / max(frame_rms.mean(), EPS)),
        "frame_rms_range_db": _db(np.quantile(frame_rms, 0.95) / max(np.quantile(frame_rms, 0.05), EPS)),
    }
    spectral = _spectral_features(y, frame_rms)
    phase = {
        "group_delay_var": spectral.pop("group_delay_var"),
        "inst_freq_dispersion": spectral.pop("inst_freq_dispersion"),
    }
    output.update(spectral)
    output.update(_voice_quality_features(y, frames, frame_rms))
    output.update(phase)
    output.update(_modulation_features(frame_rms))
    if tuple(output) != FEATURE_NAMES:
        missing = sorted(set(FEATURE_NAMES).difference(output))
        extra = sorted(set(output).difference(FEATURE_NAMES))
        raise RuntimeError(f"Feature registry mismatch; missing={missing}, extra={extra}")
    return output
