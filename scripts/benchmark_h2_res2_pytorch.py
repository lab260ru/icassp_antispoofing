#!/usr/bin/env python3
"""Benchmark the parity-validated Res2TCNGuard PyTorch scorer on frozen clips.

This is an engineering throughput measurement on the unmodified 128-clip
ASVspoof2019_LA calibration workload.  It neither transforms audio nor
produces an H2 causal result.  In particular, it deliberately uses the
revision-pinned source PyTorch evaluator, not the separately blocked ONNX
export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_h2_onnx_parity import build_calibration_manifest, load_calibration_audio
from src.res2tcn_pytorch import Res2TCNGuardPyTorchRunner, sha256_file


DATASET = "ASVspoof2019_LA"
MODEL = "Res2TCNGuard"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument(
        "--manifest",
        default="experiments/h2_causal_interventions/results/parity_calibration/ASVspoof2019_LA/calibration_manifest.parquet",
        help="Required pre-existing frozen calibration manifest; never regenerated here.",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-root",
        default="experiments/h2_causal_interventions/results/throughput_benchmark",
    )
    return parser.parse_args()


def _is_cuda_oom(error: BaseException) -> bool:
    """Recognize PyTorch OOM errors without turning other failures into rows."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(error, oom_type) or "out of memory" in str(error).lower()


def select_fastest_safe(rows: pd.DataFrame) -> dict[str, object]:
    """Select max throughput among successful rows, breaking exact ties by batch."""
    safe = rows.loc[rows["status"] == "ok"].copy()
    if safe.empty:
        raise RuntimeError("No safe Res2TCNGuard batch size completed")
    safe = safe.sort_values(["clips_per_second", "batch_size"], ascending=[False, True], kind="stable")
    return safe.iloc[0].to_dict()


def validate_frozen_manifest(manifest_path: Path, dataset: dict[str, object]) -> pd.DataFrame:
    """Refuse a measurement if its input selection differs from the parity set."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required frozen calibration manifest is missing: {manifest_path}")
    manifest = pd.read_parquet(manifest_path).sort_values(["label", "sample_id"]).reset_index(drop=True)
    expected = build_calibration_manifest(dataset, seed=2609, per_label=64)
    if list(manifest.columns) != list(expected.columns) or not manifest.equals(expected):
        raise ValueError("Calibration manifest is not the established frozen ASVspoof2019_LA contract")
    return manifest


def benchmark(
    runner: Res2TCNGuardPyTorchRunner,
    inputs: np.ndarray,
    *,
    model_sha256: str,
    batch_sizes: list[int],
    repeats: int,
) -> pd.DataFrame:
    """Measure synchronized full-workload passes and per-batch peak CUDA memory."""
    if inputs.ndim != 2 or not np.isfinite(inputs).all():
        raise ValueError("Expected a finite two-dimensional prepared input array")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    device = torch.device(runner.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Res2TCNGuard throughput requires an available CUDA device")
    device_index = torch.cuda._get_device_index(device, optional=False)
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        try:
            # Warm-up invokes the same exact scorer once, but its latency and
            # allocations are deliberately excluded from all measurements.
            runner.score(inputs[: min(batch_size, len(inputs))])
            torch.cuda.synchronize(device_index)
            elapsed_seconds: list[float] = []
            peak_allocated_bytes: list[int] = []
            peak_reserved_bytes: list[int] = []
            for _ in range(repeats):
                # Reset after warm-up. Existing model allocations remain in the
                # baseline, so peaks describe the active scorer footprint.
                torch.cuda.reset_peak_memory_stats(device_index)
                torch.cuda.synchronize(device_index)
                started = time.perf_counter()
                for offset in range(0, len(inputs), batch_size):
                    runner.score(inputs[offset : offset + batch_size])
                torch.cuda.synchronize(device_index)
                elapsed_seconds.append(time.perf_counter() - started)
                peak_allocated_bytes.append(int(torch.cuda.max_memory_allocated(device_index)))
                peak_reserved_bytes.append(int(torch.cuda.max_memory_reserved(device_index)))
            median_seconds = float(np.median(elapsed_seconds))
            rows.append(
                {
                    "model": MODEL,
                    "model_sha256": model_sha256,
                    "runner": "pinned_pytorch_evaluate_py",
                    "provider": "PyTorch CUDA",
                    "device": runner.device,
                    "n_clips": len(inputs),
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "status": "ok",
                    "median_seconds": median_seconds,
                    "clips_per_second": float(len(inputs) / median_seconds),
                    "median_latency_ms_per_clip": float(1000 * median_seconds / len(inputs)),
                    "peak_cuda_allocated_bytes": max(peak_allocated_bytes),
                    "peak_cuda_reserved_bytes": max(peak_reserved_bytes),
                    "peak_cuda_allocated_mib": float(max(peak_allocated_bytes) / (1024**2)),
                    "peak_cuda_reserved_mib": float(max(peak_reserved_bytes) / (1024**2)),
                    "all_seconds": json.dumps([round(value, 9) for value in elapsed_seconds]),
                    "all_peak_cuda_allocated_bytes": json.dumps(peak_allocated_bytes),
                    "all_peak_cuda_reserved_bytes": json.dumps(peak_reserved_bytes),
                    "error": "",
                }
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            # Preserve the failed candidate, then release cached workspace
            # before attempting a smaller or later independently safe batch.
            torch.cuda.empty_cache()
            rows.append(
                {
                    "model": MODEL,
                    "model_sha256": model_sha256,
                    "runner": "pinned_pytorch_evaluate_py",
                    "provider": "PyTorch CUDA",
                    "device": runner.device,
                    "n_clips": len(inputs),
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "status": "oom",
                    "median_seconds": float("nan"),
                    "clips_per_second": float("nan"),
                    "median_latency_ms_per_clip": float("nan"),
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
                    "peak_cuda_allocated_mib": float(torch.cuda.max_memory_allocated(device_index) / (1024**2)),
                    "peak_cuda_reserved_mib": float(torch.cuda.max_memory_reserved(device_index) / (1024**2)),
                    "all_seconds": "[]",
                    "all_peak_cuda_allocated_bytes": "[]",
                    "all_peak_cuda_reserved_bytes": "[]",
                    "error": f"{type(error).__name__}: {str(error).splitlines()[0]}",
                }
            )
    return pd.DataFrame(rows).sort_values("batch_size").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    batch_sizes = sorted(set(args.batch_sizes or DEFAULT_BATCH_SIZES))
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = yaml.safe_load(Path(args.index).read_text())
    dataset = index["datasets"][args.dataset]
    model_card = index["models"][MODEL]
    manifest_path = Path(args.manifest)
    manifest = validate_frozen_manifest(manifest_path, dataset)
    records = load_calibration_audio(dataset, manifest)
    runner = Res2TCNGuardPyTorchRunner(model_card["local_dir"], device=args.device)
    inputs = np.stack([runner.prepare(record["waveform"]) for record in records]).astype(np.float32, copy=False)
    table = benchmark(
        runner,
        inputs,
        model_sha256=runner.provenance["best_1.495.pth_sha256"],
        batch_sizes=batch_sizes,
        repeats=args.repeats,
    )
    fastest = select_fastest_safe(table)
    output_dir = Path(args.output_root) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "Res2TCNGuard_pytorch_batch_sweep.csv", index=False)
    device_index = torch.cuda._get_device_index(torch.device(args.device), optional=False)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "H2 engineering throughput benchmark on frozen parity inputs; not an intervention result",
        "dataset": args.dataset,
        "dataset_revision": str(dataset["revision"]),
        "model": MODEL,
        "model_revision": str(model_card["revision"]),
        "runner": "pinned_pytorch_evaluate_py",
        "provider": "PyTorch CUDA",
        "device": args.device,
        "cuda_device_index": device_index,
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "parity_report_path": "experiments/h2_causal_interventions/results/parity_calibration/ASVspoof2019_LA/Res2TCNGuard__pytorch_parity_report.json",
        "parity_report_sha256": hashlib.sha256(
            Path("experiments/h2_causal_interventions/results/parity_calibration/ASVspoof2019_LA/Res2TCNGuard__pytorch_parity_report.json").read_bytes()
        ).hexdigest(),
        "n_clips": len(inputs),
        "batch_sizes": batch_sizes,
        "repeats": args.repeats,
        "warmup_policy": "one unmeasured scorer call at each candidate batch before synchronized full-workload passes",
        "measurement_policy": "three synchronized full 128-clip passes per candidate; preparation, decoding, and model load excluded; per-repeat peak allocator/reserved memory after warm-up",
        "runner_provenance": runner.provenance,
        "fastest_safe_row": fastest,
    }
    (output_dir / "Res2TCNGuard_pytorch_throughput_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{MODEL}: fastest safe batch={int(fastest['batch_size'])} "
        f"throughput={float(fastest['clips_per_second']):.2f} clips/s; wrote {output_dir}"
    )


if __name__ == "__main__":
    main()
