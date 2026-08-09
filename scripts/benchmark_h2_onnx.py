#!/usr/bin/env python3
"""Benchmark parity-validated H2 ONNX scorer throughput on frozen inputs.

This is an engineering measurement, not an H2 intervention experiment. Run it
through ``scripts/run_h2_cuda.sh`` so ONNX Runtime can discover CUDA 13 wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_h2_onnx_parity import (
    DEFAULT_DATASET,
    DEFAULT_MODELS,
    ONNX_FILENAMES,
    load_calibration_audio,
    load_or_create_manifest,
    sha256_file,
)
from src.onnx_fixed_window import FixedWindowOnnxRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--parity-root",
        default="experiments/h2_causal_interventions/results/parity_calibration",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/h2_causal_interventions/results/throughput_benchmark",
    )
    return parser.parse_args()


def benchmark_model(
    model_name: str,
    model_path: Path,
    preprocessing: str,
    waveforms: list[np.ndarray],
    *,
    batch_sizes: list[int],
    repeats: int,
    cuda_device: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    runner = FixedWindowOnnxRunner(
        model_path,
        preprocessing=preprocessing,
        cuda_device_id=cuda_device,
    )
    if runner.provider != "CUDAExecutionProvider" or runner.used_cpu_fallback:
        raise RuntimeError(
            f"CUDA throughput is required but {model_name} used {runner.provider} "
            f"(fallback={runner.used_cpu_fallback})"
        )
    inputs = np.stack([runner.prepare(waveform) for waveform in waveforms]).astype(np.float32, copy=False)
    if not np.isfinite(inputs).all():
        raise ValueError("Prepared calibration inputs contain non-finite values")
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        # Warm-up separates graph/provider initialization from measured passes.
        runner.score(inputs[: min(batch_size, len(inputs))])
        elapsed: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            for offset in range(0, len(inputs), batch_size):
                runner.score(inputs[offset : offset + batch_size])
            elapsed.append(time.perf_counter() - start)
        median_seconds = float(np.median(elapsed))
        rows.append(
            {
                "model": model_name,
                "model_sha256": sha256_file(model_path),
                "preprocessing": preprocessing,
                "provider": runner.provider,
                "cuda_device": cuda_device,
                "n_clips": len(inputs),
                "batch_size": batch_size,
                "repeats": repeats,
                "median_seconds": median_seconds,
                "clips_per_second": float(len(inputs) / median_seconds),
                "all_seconds": json.dumps([round(value, 9) for value in elapsed]),
            }
        )
    result = pd.DataFrame(rows).sort_values("batch_size").reset_index(drop=True)
    fastest = result.loc[result["clips_per_second"].idxmax()].to_dict()
    return result, fastest


def main() -> None:
    args = parse_args()
    batch_sizes = list(dict.fromkeys(args.batch_sizes or [1, 2, 4, 8, 16, 32]))
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    index = yaml.safe_load(Path(args.index).read_text())
    if args.dataset not in index["datasets"]:
        raise KeyError(f"Unknown dataset: {args.dataset}")
    models = tuple(args.models or DEFAULT_MODELS)
    for model in models:
        if model not in ONNX_FILENAMES:
            raise ValueError(f"No supported fixed-window ONNX contract for {model}")
    parity_dir = Path(args.parity_root) / args.dataset
    report = json.loads((parity_dir / "parity_report.json").read_text())
    manifest = load_or_create_manifest(
        parity_dir / "calibration_manifest.parquet",
        index["datasets"][args.dataset],
        seed=int(report["selection"]["seed"]),
        per_label=int(report["selection"]["per_label"]),
    )
    records = load_calibration_audio(index["datasets"][args.dataset], manifest)
    waveforms = [record["waveform"] for record in records]
    output_dir = Path(args.output_root) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, object] = {}
    for offset, model_name in enumerate(models):
        decision = report["decisions"].get(model_name)
        if not decision or decision.get("status") != "eligible":
            raise ValueError(f"No eligible frozen parity decision for {model_name}")
        device = args.cuda_device if args.cuda_device is not None else offset
        model_path = Path(index["models"][model_name]["local_dir"]) / ONNX_FILENAMES[model_name]
        table, fastest = benchmark_model(
            model_name,
            model_path,
            str(decision["chosen_preprocessing"]),
            waveforms,
            batch_sizes=batch_sizes,
            repeats=args.repeats,
            cuda_device=device,
        )
        table.to_csv(output_dir / f"{model_name}_batch_sweep.csv", index=False)
        outcomes[model_name] = fastest
        print(
            f"{model_name}: fastest batch={int(fastest['batch_size'])} "
            f"throughput={float(fastest['clips_per_second']):.2f} clips/s"
        )
    output = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "H2 engineering throughput benchmark on frozen parity inputs; not an intervention result",
        "dataset": args.dataset,
        "dataset_revision": index["datasets"][args.dataset]["revision"],
        "parity_report_sha256": hashlib.sha256((parity_dir / "parity_report.json").read_bytes()).hexdigest(),
        "batch_sizes": batch_sizes,
        "repeats": args.repeats,
        "fastest_rows": outcomes,
    }
    (output_dir / "throughput_report.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
