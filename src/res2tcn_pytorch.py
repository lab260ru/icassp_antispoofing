"""Exact, provenance-preserving PyTorch inference for Res2TCNGuard.

This runner deliberately uses the revision-pinned ``evaluate.py`` and
``best_1.495.pth`` bundle rather than the separately exported ONNX graph.  It
is intended for a baseline-parity gate before any H2 waveform interventions;
it does not define score orientation or perform transformations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np
import torch

from src.onnx_fixed_window import WINDOW_SAMPLES, fixed_first_window_tiled


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Return the digest used to identify a pinned bundle file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_evaluator(bundle_dir: Path) -> ModuleType:
    """Load the bundle-local evaluator without assuming package installation."""
    evaluator_path = bundle_dir / "evaluate.py"
    network_path = bundle_dir / "_net.py"
    if not evaluator_path.is_file() or not network_path.is_file():
        raise FileNotFoundError("Res2TCNGuard bundle must contain evaluate.py and _net.py")
    # ``evaluate.py`` imports its sibling as ``_net``.  Put the pinned bundle
    # first, just as its documented standalone invocation does.
    bundle_text = str(bundle_dir)
    if bundle_text not in sys.path:
        sys.path.insert(0, bundle_text)
    module_name = f"_pinned_res2tcn_evaluate_{sha256_file(evaluator_path)[:12]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load evaluator from {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # The standalone evaluator uses ``from _net import TestModel``.  Load
        # its sibling explicitly for the duration of that import so an
        # unrelated module named ``_net`` in a long-lived research process
        # cannot silently substitute an architecture.
        prior_network = sys.modules.get("_net")
        network_spec = importlib.util.spec_from_file_location("_net", network_path)
        if network_spec is None or network_spec.loader is None:
            raise ImportError(f"Unable to load network from {network_path}")
        network_module = importlib.util.module_from_spec(network_spec)
        sys.modules["_net"] = network_module
        try:
            network_spec.loader.exec_module(network_module)
            spec.loader.exec_module(module)
        finally:
            if prior_network is None:
                sys.modules.pop("_net", None)
            else:
                sys.modules["_net"] = prior_network
    return module


@dataclass(frozen=True)
class Res2TCNScoreBatch:
    """Two PyTorch logits, selected raw scalar, and exact detector windows."""

    logits: np.ndarray
    raw_score: np.ndarray
    detector_waveforms: np.ndarray


class Res2TCNGuardPyTorchRunner:
    """Run the exact bundle evaluator on its first-window/tile-repeat inputs."""

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        device: str = "cuda:2",
        model: Any | None = None,
        pad_fixed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.device = str(device)
        self._evaluator: ModuleType | None = None
        if model is None:
            checkpoint = self.bundle_dir / "best_1.495.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing Res2TCNGuard checkpoint: {checkpoint}")
            self._evaluator = _load_evaluator(self.bundle_dir)
            self._model = self._evaluator.load_model(checkpoint, device=self.device)
            self._pad_fixed = self._evaluator.pad_fixed
        else:
            # This narrow injection seam keeps shape/window tests independent of
            # the private model bundle. Production callers never supply it.
            self._model = model
            self._pad_fixed = pad_fixed_fn or fixed_first_window_tiled

    @property
    def provenance(self) -> dict[str, str]:
        """Expose hashes for the files defining this executable scorer."""
        required = ("evaluate.py", "_net.py", "best_1.495.pth")
        values: dict[str, str] = {}
        for name in required:
            path = self.bundle_dir / name
            if path.is_file():
                values[f"{name}_sha256"] = sha256_file(path)
        return values

    def prepare(self, audio: np.ndarray) -> np.ndarray:
        """Apply exactly the bundle's fixed first-window/tile-repeat contract."""
        prepared = self._pad_fixed(np.asarray(audio, dtype=np.float32))
        prepared = np.ascontiguousarray(prepared, dtype=np.float32).reshape(-1)
        if prepared.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"Res2TCNGuard prepared window must be ({WINDOW_SAMPLES},), got {prepared.shape}")
        return prepared

    @torch.no_grad()
    def score(self, waveforms: Sequence[np.ndarray] | np.ndarray) -> Res2TCNScoreBatch:
        """Return logits and raw ``logit_1`` without applying an orientation map."""
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 2:
            prepared = np.ascontiguousarray(waveforms, dtype=np.float32)
            if prepared.shape[1] != WINDOW_SAMPLES:
                raise ValueError(f"Pre-windowed batch must have {WINDOW_SAMPLES} samples, got {prepared.shape}")
        else:
            prepared_rows = [self.prepare(np.asarray(waveform)) for waveform in waveforms]
            if not prepared_rows:
                raise ValueError("Cannot score an empty waveform batch")
            prepared = np.stack(prepared_rows).astype(np.float32, copy=False)
        output = self._model(torch.from_numpy(prepared).to(self.device))
        logits = output[1] if isinstance(output, tuple) else output
        logits = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits.shape != (prepared.shape[0], 2):
            raise ValueError(f"Expected two logits per Res2TCNGuard input, got {logits.shape}")
        return Res2TCNScoreBatch(
            logits=logits,
            raw_score=logits[:, 1].copy(),
            detector_waveforms=prepared,
        )
