#!/usr/bin/env python3
"""Validate the exact Res2TCNGuard PyTorch bundle against Arena baselines.

This is a readiness gate on original frozen clips only. It intentionally does
not modify the ONNX calibration artifacts or score transformed audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_h2_onnx_parity import build_calibration_manifest, load_calibration_audio
from src.arena_io import load_model_scores
from src.onnx_fixed_window import canonicalize_raw_score
from src.res2tcn_pytorch import Res2TCNGuardPyTorchRunner, sha256_file


DATASET = "ASVspoof2019_LA"
MODEL = "Res2TCNGuard"


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def rank_correlation(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", default=DATASET, choices=(DATASET,))
    parser.add_argument(
        "--manifest",
        default="experiments/h2_causal_interventions/results/parity_calibration/ASVspoof2019_LA/calibration_manifest.parquet",
    )
    parser.add_argument("--output-dir", default="experiments/h2_causal_interventions/results/parity_calibration")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-spearman", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not 0 < args.min_spearman <= 1:
        raise ValueError("min-spearman must be in (0, 1]")
    index = yaml.safe_load(Path(args.index).read_text())
    dataset = index["datasets"][args.dataset]
    model_card = index["models"][MODEL]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required pre-existing frozen calibration manifest is missing: {manifest_path}")
    manifest = pd.read_parquet(manifest_path).sort_values(["label", "sample_id"]).reset_index(drop=True)
    expected = build_calibration_manifest(dataset, seed=2609, per_label=64)
    if list(manifest.columns) != list(expected.columns) or not manifest.equals(expected):
        raise ValueError("Calibration manifest is not the established frozen ASVspoof2019_LA contract")
    records = load_calibration_audio(dataset, manifest)
    runner = Res2TCNGuardPyTorchRunner(model_card["local_dir"], device=args.device)
    rows: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        result = runner.score([item["waveform"] for item in chunk])
        for item, detector_waveform, logits, raw_score in zip(
            chunk, result.detector_waveforms, result.logits, result.raw_score, strict=True
        ):
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "source_id": item["source_id"],
                    "label": item["label"],
                    "dataset": args.dataset,
                    "dataset_revision": str(dataset["revision"]),
                    "model": MODEL,
                    "runner": "pinned_pytorch_evaluate_py",
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "sample_rate": item["sample_rate"],
                    "source_samples": item["source_samples"],
                    "detector_samples": int(detector_waveform.size),
                    "window_rule": "first_64600_or_tile_repeat_exact_pinned_evaluate_py",
                    "original_wave_sha256": item["original_wave_sha256"],
                    "detector_input_sha256": sha256_array(detector_waveform),
                    "logit_0": float(logits[0]),
                    "logit_1": float(logits[1]),
                    "selected_raw_score": float(raw_score),
                }
            )
    output = pd.DataFrame(rows)
    arena = load_model_scores(index, args.dataset, MODEL)
    output = output.merge(
        arena[["sample_id", "raw_score", "score_spoof", "orientation"]].rename(
            columns={"raw_score": "arena_raw_score", "score_spoof": "arena_score_spoof", "orientation": "arena_orientation"}
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(records):
        raise ValueError(f"Only {len(output)}/{len(records)} frozen clips joined to Arena baselines")
    orientations = output["arena_orientation"].unique().tolist()
    if orientations != ["negated_raw_is_spoof"]:
        raise ValueError(f"Unexpected or ambiguous frozen Arena orientation: {orientations}")
    output["score_spoof"] = canonicalize_raw_score(output["selected_raw_score"].to_numpy(), orientations[0])
    raw_vs_arena = rank_correlation(output["selected_raw_score"], output["arena_raw_score"])
    raw_label = rank_correlation(output["selected_raw_score"], output["label"])
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "H2 pre-intervention exact-PyTorch baseline parity gate; not an intervention result",
        "dataset": args.dataset,
        "dataset_revision": str(dataset["revision"]),
        "model": MODEL,
        "model_revision": str(model_card["revision"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_clips": len(output),
        "n_bonafide": int((output["label"] == 0).sum()),
        "n_spoof": int((output["label"] == 1).sum()),
        "runner": "pinned_pytorch_evaluate_py",
        "device": args.device,
        "batch_size": args.batch_size,
        "raw_scalar": "logit_1",
        "arena_orientation": orientations[0],
        "observed_raw_label_spearman": raw_label,
        "raw_vs_arena_spearman": raw_vs_arena,
        "raw_vs_arena_mae": float(np.mean(np.abs(output["selected_raw_score"] - output["arena_raw_score"]))),
        "raw_vs_arena_max_absdiff": float(np.max(np.abs(output["selected_raw_score"] - output["arena_raw_score"]))),
        "min_spearman": args.min_spearman,
        "ordering_gate_pass": bool(np.isfinite(raw_vs_arena) and raw_vs_arena >= args.min_spearman),
        "runner_provenance": runner.provenance,
    }
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_dir / "Res2TCNGuard__pytorch_raw__scores.parquet", index=False)
    (output_dir / "Res2TCNGuard__pytorch_parity_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
