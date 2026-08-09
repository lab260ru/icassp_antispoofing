"""Auditable fixed-window ONNX inference for H2 parity checks.

This module deliberately implements only the waveform contract shared by the
locally audited Spectra-AASIST and AASIST ONNX artifacts.  It does not infer a
model's class orientation: that mapping is established by a separately pinned
Arena score artifact in the parity calibration script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WINDOW_SAMPLES = 64_600
PREPROCESSING_CHOICES = ("raw", "preemphasis_0.97")


def fixed_first_window_tiled(audio: np.ndarray, target_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """Return the first fixed detector window, repeat-tiling short waveforms.

    This is intentionally different from the H1 deterministic centre crop.
    The two H2 ONNX scorer cards specify first-window inference; preserving
    that positional convention is necessary for score-artifact parity.
    """
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if waveform.size == 0:
        raise ValueError("Cannot tile an empty waveform for detector inference")
    if not np.isfinite(waveform).all():
        raise ValueError("Detector waveform contains non-finite values")
    if waveform.size >= target_samples:
        return np.ascontiguousarray(waveform[:target_samples], dtype=np.float32)
    repeats = math.ceil(target_samples / waveform.size)
    return np.ascontiguousarray(np.tile(waveform, repeats)[:target_samples], dtype=np.float32)


def apply_preprocessing(audio: np.ndarray, preprocessing: str) -> np.ndarray:
    """Apply an explicitly named external preprocessing candidate.

    The raw and pre-emphasis paths are both calibration candidates.  This
    function does not select one based on detector outcomes.
    """
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if preprocessing == "raw":
        return waveform.copy()
    if preprocessing == "preemphasis_0.97":
        if waveform.size == 0:
            return waveform.copy()
        result = np.empty_like(waveform)
        result[0] = waveform[0]
        result[1:] = waveform[1:] - np.float32(0.97) * waveform[:-1]
        return result
    choices = ", ".join(PREPROCESSING_CHOICES)
    raise ValueError(f"Unknown preprocessing {preprocessing!r}; choose one of {choices}")


def canonicalize_raw_score(raw_score: np.ndarray, arena_orientation: str) -> np.ndarray:
    """Map a matching raw scalar to increasing spoof evidence.

    ``arena_orientation`` comes only from ``src.arena_io.load_model_scores``
    on the pinned baseline score artifact.  It is never inferred from an H2
    intervention response.
    """
    values = np.asarray(raw_score, dtype=np.float64)
    if arena_orientation == "raw_is_spoof":
        return values
    if arena_orientation == "negated_raw_is_spoof":
        return -values
    raise ValueError(f"Unsupported Arena orientation: {arena_orientation!r}")


@dataclass(frozen=True)
class OnnxScoreBatch:
    """Raw ONNX output and its explicitly selected scalar column."""

    logits: np.ndarray
    raw_score: np.ndarray
    detector_waveforms: np.ndarray


class FixedWindowOnnxRunner:
    """Run a fixed-length waveform ONNX classifier with CUDA-first execution.

    A caller can inject a session for focused unit tests.  Production sessions
    try CUDA first (without TensorRT) and fall back to CPU if session creation
    or first inference fails.  The actual provider is exposed for provenance.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        preprocessing: str = "raw",
        raw_scalar_index: int = 1,
        cuda_device_id: int = 0,
        force_cpu: bool = False,
        session: Any | None = None,
    ) -> None:
        if preprocessing not in PREPROCESSING_CHOICES:
            raise ValueError(f"Unsupported preprocessing: {preprocessing}")
        if raw_scalar_index not in (0, 1):
            raise ValueError("raw_scalar_index must be 0 or 1 for a two-logit model")
        self.model_path = Path(model_path)
        self.preprocessing = preprocessing
        self.raw_scalar_index = raw_scalar_index
        self.cuda_device_id = int(cuda_device_id)
        self.force_cpu = bool(force_cpu)
        self._session = session
        self._using_cpu_fallback = force_cpu
        if self._session is None:
            self._session = self._create_session(prefer_cuda=not force_cpu)
        self._validate_signature()

    def _create_session(self, *, prefer_cuda: bool) -> Any:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX artifact: {self.model_path}")
        import onnxruntime as ort

        if prefer_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                return ort.InferenceSession(
                    str(self.model_path),
                    providers=[
                        ("CUDAExecutionProvider", {"device_id": str(self.cuda_device_id)}),
                        "CPUExecutionProvider",
                    ],
                )
            except Exception:
                # The report records this fallback; do not silently route to
                # TensorRT or an unrecorded mixed-precision engine.
                self._using_cpu_fallback = True
        return ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def _validate_signature(self) -> None:
        inputs = list(self._session.get_inputs())
        outputs = list(self._session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one ONNX input and one output")
        input_shape = tuple(inputs[0].shape)
        output_shape = tuple(outputs[0].shape)
        if len(input_shape) != 2 or input_shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"Expected waveform input [batch,{WINDOW_SAMPLES}], got {input_shape}")
        if len(output_shape) != 2 or output_shape[1] != 2:
            raise ValueError(f"Expected two-logit output [batch,2], got {output_shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    @property
    def provider(self) -> str:
        providers = list(self._session.get_providers())
        return str(providers[0]) if providers else "unknown"

    @property
    def used_cpu_fallback(self) -> bool:
        return self._using_cpu_fallback or self.provider == "CPUExecutionProvider"

    def prepare(self, audio: np.ndarray) -> np.ndarray:
        """Apply the named preprocessor then detector's first-window policy."""
        return fixed_first_window_tiled(apply_preprocessing(audio, self.preprocessing))

    def score(self, waveforms: Sequence[np.ndarray] | np.ndarray) -> OnnxScoreBatch:
        """Score waveforms and retain exact detector inputs for hashing/logging."""
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 2:
            prepared = np.ascontiguousarray(waveforms, dtype=np.float32)
            if prepared.shape[1] != WINDOW_SAMPLES:
                raise ValueError(f"Pre-windowed batch must have {WINDOW_SAMPLES} samples, got {prepared.shape}")
        else:
            prepared_rows = [self.prepare(np.asarray(waveform)) for waveform in waveforms]
            if not prepared_rows:
                raise ValueError("Cannot score an empty waveform batch")
            prepared = np.stack(prepared_rows).astype(np.float32, copy=False)
        logits = self._run(prepared)
        return OnnxScoreBatch(
            logits=logits,
            raw_score=logits[:, self.raw_scalar_index].copy(),
            detector_waveforms=prepared,
        )

    def _run(self, prepared: np.ndarray) -> np.ndarray:
        try:
            values = self._session.run([self.output_name], {self.input_name: prepared})[0]
        except Exception:
            # CUDA can initialise successfully but fail only on first use.  A
            # one-time CPU retry makes the calibration runnable while retaining
            # an explicit provider/fallback record in its compact report.
            if self.force_cpu or self.provider != "CUDAExecutionProvider":
                raise
            self._using_cpu_fallback = True
            self._session = self._create_session(prefer_cuda=False)
            self._validate_signature()
            values = self._session.run([self.output_name], {self.input_name: prepared})[0]
        logits = np.asarray(values, dtype=np.float32)
        if logits.ndim != 2 or logits.shape != (prepared.shape[0], 2):
            raise ValueError(f"Unexpected ONNX logits shape {logits.shape}; expected ({prepared.shape[0]}, 2)")
        return logits
