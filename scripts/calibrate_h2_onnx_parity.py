#!/usr/bin/env python3
"""Freeze and score a 128-clip H2 ONNX parity calibration set.

The script is intentionally a gate, not an intervention experiment.  It only
scores original, pinned ASVspoof2019_LA clips and compares the selected ONNX
raw scalar with the Arena's pinned baseline ``scores.txt`` artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arena_io import decode_audio, iter_selected_audio, load_labels, load_model_scores
from src.onnx_fixed_window import (
    PREPROCESSING_CHOICES,
    WINDOW_SAMPLES,
    FixedWindowOnnxRunner,
    canonicalize_raw_score,
)


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DATASET = "ASVspoof2019_LA"
DEFAULT_MODELS = ("Spectra-AASIST", "AASIST")
ONNX_FILENAMES = {"Spectra-AASIST": "spectra-aasist.onnx", "AASIST": "aasist.onnx"}


def sha256_bytes(value: bytes | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value, dtype=np.float32).tobytes()
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_calibration_manifest(dataset: dict[str, Any], *, seed: int, per_label: int) -> pd.DataFrame:
    """Create a deterministic, score-independent stratified clip manifest."""
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    labels = load_labels(dataset)
    selections: list[pd.DataFrame] = []
    for label, group in labels.groupby("label", sort=True):
        if len(group) < per_label:
            raise ValueError(f"Label {label} has {len(group)} clips; need {per_label} for calibration")
        selected = group.sample(n=per_label, random_state=seed + int(label)).copy()
        selections.append(selected)
    result = pd.concat(selections, ignore_index=True).sort_values(["label", "sample_id"]).reset_index(drop=True)
    result["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    result["selection_seed"] = int(seed)
    result["selection_rule"] = f"stratified_fixed_{per_label}_per_label_pandas_random_state_seed_plus_label"
    result["dataset_revision"] = str(dataset["revision"])
    return result[
        [
            "manifest_schema_version",
            "dataset_revision",
            "selection_seed",
            "selection_rule",
            "sample_id",
            "source_id",
            "label",
        ]
    ]


def load_or_create_manifest(path: Path, dataset: dict[str, Any], *, seed: int, per_label: int) -> pd.DataFrame:
    """Freeze a manifest once; later invocations must reproduce it exactly."""
    expected = build_calibration_manifest(dataset, seed=seed, per_label=per_label)
    if path.exists():
        existing = pd.read_parquet(path).sort_values(["label", "sample_id"]).reset_index(drop=True)
        if list(existing.columns) != list(expected.columns) or not existing.equals(expected):
            raise ValueError(f"Existing calibration manifest does not match frozen deterministic contract: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_parquet(path, index=False)
    expected.to_csv(path.with_suffix(".csv"), index=False)
    return expected


def load_calibration_audio(dataset: dict[str, Any], manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """Decode exact selected clips, retaining source bytes and documented IDs."""
    selected = manifest[["sample_id", "source_id", "label"]]
    records = {record.sample_id: record for record in iter_selected_audio(dataset, selected)}
    ordered: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        record = records.get(row.sample_id)
        if record is None:
            raise ValueError(f"Frozen calibration clip was not found locally: {row.sample_id}")
        waveform, sample_rate = decode_audio(record.audio_bytes)
        if sample_rate != 16_000:
            raise ValueError(f"Expected 16 kHz pinned audio for {row.sample_id}; got {sample_rate}")
        if waveform.size == 0:
            raise ValueError(f"Pinned audio is empty: {row.sample_id}")
        ordered.append(
            {
                "sample_id": row.sample_id,
                "source_id": row.source_id,
                "label": int(row.label),
                "sample_rate": sample_rate,
                "source_samples": int(waveform.size),
                "original_wave_sha256": sha256_bytes(waveform),
                "waveform": waveform,
            }
        )
    return ordered


def rank_correlation(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def score_variant(
    *,
    model_name: str,
    model_path: Path,
    model_sha256: str,
    dataset_name: str,
    dataset_revision: str,
    records: list[dict[str, Any]],
    arena_scores: pd.DataFrame,
    preprocessing: str,
    raw_scalar_index: int,
    cuda_device_id: int,
    force_cpu: bool,
    batch_size: int,
    min_spearman: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one predeclared preprocessing candidate against Arena baselines."""
    runner = FixedWindowOnnxRunner(
        model_path,
        preprocessing=preprocessing,
        raw_scalar_index=raw_scalar_index,
        cuda_device_id=cuda_device_id,
        force_cpu=force_cpu,
    )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        output = runner.score([item["waveform"] for item in chunk])
        for item, detector_waveform, logits, raw_score in zip(
            chunk, output.detector_waveforms, output.logits, output.raw_score, strict=True
        ):
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "source_id": item["source_id"],
                    "label": item["label"],
                    "dataset": dataset_name,
                    "dataset_revision": dataset_revision,
                    "model": model_name,
                    "model_path": str(model_path),
                    "model_sha256": model_sha256,
                    "preprocessing": preprocessing,
                    "raw_scalar_index": raw_scalar_index,
                    "requested_cuda_device": cuda_device_id,
                    "batch_size": batch_size,
                    "provider": runner.provider,
                    "used_cpu_fallback": runner.used_cpu_fallback,
                    "sample_rate": item["sample_rate"],
                    "source_samples": item["source_samples"],
                    "detector_samples": WINDOW_SAMPLES,
                    "window_rule": "preprocess_then_first_64600_or_tile_repeat",
                    "original_wave_sha256": item["original_wave_sha256"],
                    "detector_input_sha256": sha256_bytes(detector_waveform),
                    "logit_0": float(logits[0]),
                    "logit_1": float(logits[1]),
                    "selected_raw_score": float(raw_score),
                }
            )
    output = pd.DataFrame(rows).merge(
        arena_scores[["sample_id", "raw_score", "score_spoof", "orientation"]].rename(
            columns={"raw_score": "arena_raw_score", "score_spoof": "arena_score_spoof", "orientation": "arena_orientation"}
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(records):
        raise ValueError(f"Only {len(output)}/{len(records)} calibration clips joined to the Arena score artifact")
    orientations = output["arena_orientation"].unique().tolist()
    if len(orientations) != 1:
        raise ValueError(f"Ambiguous Arena orientation for {model_name}: {orientations}")
    arena_orientation = str(orientations[0])
    output["score_spoof"] = canonicalize_raw_score(output["selected_raw_score"].to_numpy(), arena_orientation)
    raw_label_rho = rank_correlation(output["selected_raw_score"], output["label"])
    expected_raw_label_sign = 1 if arena_orientation == "raw_is_spoof" else -1
    observed_sign = int(np.sign(raw_label_rho)) if np.isfinite(raw_label_rho) else 0
    raw_vs_arena_rho = rank_correlation(output["selected_raw_score"], output["arena_raw_score"])
    spoof_vs_arena_rho = rank_correlation(output["score_spoof"], output["arena_score_spoof"])
    orientation_gate_pass = observed_sign == expected_raw_label_sign
    ordering_gate_pass = bool(np.isfinite(raw_vs_arena_rho) and raw_vs_arena_rho >= min_spearman)
    summary = {
        "dataset": dataset_name,
        "dataset_revision": dataset_revision,
        "model": model_name,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "preprocessing": preprocessing,
        "raw_scalar_index": raw_scalar_index,
        "requested_cuda_device": cuda_device_id,
        "batch_size": batch_size,
        "provider": runner.provider,
        "used_cpu_fallback": runner.used_cpu_fallback,
        "n_clips": len(output),
        "n_bonafide": int((output["label"] == 0).sum()),
        "n_spoof": int((output["label"] == 1).sum()),
        "arena_orientation": arena_orientation,
        "expected_raw_label_sign": expected_raw_label_sign,
        "observed_raw_label_spearman": raw_label_rho,
        "raw_vs_arena_spearman": raw_vs_arena_rho,
        "score_spoof_vs_arena_spearman": spoof_vs_arena_rho,
        "min_spearman": min_spearman,
        "orientation_gate_pass": orientation_gate_pass,
        "ordering_gate_pass": ordering_gate_pass,
        "eligible": bool(orientation_gate_pass and ordering_gate_pass),
    }
    return output, summary


def choose_preprocessing(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a predeclared least-assumption tie break without treatment data."""
    eligible = [summary for summary in summaries if summary["eligible"]]
    raw = next((summary for summary in eligible if summary["preprocessing"] == "raw"), None)
    if raw is not None:
        return {
            "status": "eligible",
            "chosen_preprocessing": "raw",
            "selection_rule": "predeclared_prefer_raw_when_it_passes_all_parity_gates",
            "chosen_provider": raw["provider"],
        }
    if len(eligible) == 1:
        selected = eligible[0]
        return {
            "status": "eligible",
            "chosen_preprocessing": selected["preprocessing"],
            "selection_rule": "predeclared_only_nonraw_candidate_passing_all_parity_gates",
            "chosen_provider": selected["provider"],
        }
    return {
        "status": "blocked",
        "chosen_preprocessing": None,
        "selection_rule": "no_candidate_passed_all_parity_gates",
        "chosen_provider": None,
    }


def merge_resumable_summaries(existing: pd.DataFrame, current: pd.DataFrame, rerun_models: tuple[str, ...]) -> pd.DataFrame:
    """Retain already-calibrated models when a resource-safe rerun is partial."""
    retained = existing.loc[~existing["model"].isin(rerun_models)].copy() if not existing.empty else existing
    merged = pd.concat([retained, current], ignore_index=True)
    if merged.duplicated(["model", "preprocessing"]).any():
        raise ValueError("Parity summary would contain duplicate model/preprocessing rows")
    return merged.sort_values(["model", "preprocessing"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/arena-index.yaml")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output-dir", default="experiments/h2_causal_interventions/results/parity_calibration")
    parser.add_argument("--seed", type=int, default=2609)
    parser.add_argument("--per-label", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--raw-scalar-index", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--min-spearman", type=float, default=0.999)
    parser.add_argument(
        "--preprocessing",
        action="append",
        choices=PREPROCESSING_CHOICES,
        help="Repeat to override the audited raw/pre-emphasis candidate set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < args.min_spearman <= 1:
        raise ValueError("min_spearman must be in (0, 1]")
    index = yaml.safe_load(Path(args.index).read_text())
    if args.dataset not in index["datasets"]:
        raise KeyError(f"Unknown dataset: {args.dataset}")
    dataset = index["datasets"][args.dataset]
    models = tuple(args.models or DEFAULT_MODELS)
    for model_name in models:
        if model_name not in ONNX_FILENAMES:
            raise ValueError(f"No locally audited fixed-window ONNX contract for {model_name}")
        if model_name not in index["models"]:
            raise KeyError(f"Unknown model: {model_name}")
    output_dir = Path(args.output_dir) / args.dataset
    manifest_path = output_dir / "calibration_manifest.parquet"
    manifest = load_or_create_manifest(manifest_path, dataset, seed=args.seed, per_label=args.per_label)
    records = load_calibration_audio(dataset, manifest)
    preprocessing_choices = tuple(args.preprocessing or PREPROCESSING_CHOICES)
    all_summaries: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for model_offset, model_name in enumerate(models):
        model = index["models"][model_name]
        model_path = Path(model["local_dir"]) / ONNX_FILENAMES[model_name]
        model_sha256 = sha256_file(model_path)
        arena_scores = load_model_scores(index, args.dataset, model_name)
        model_summaries: list[dict[str, Any]] = []
        for preprocessing in preprocessing_choices:
            device_id = args.cuda_device if args.cuda_device is not None else model_offset
            scores, summary = score_variant(
                model_name=model_name,
                model_path=model_path,
                model_sha256=model_sha256,
                dataset_name=args.dataset,
                dataset_revision=str(dataset["revision"]),
                records=records,
                arena_scores=arena_scores,
                preprocessing=preprocessing,
                raw_scalar_index=args.raw_scalar_index,
                cuda_device_id=device_id,
                force_cpu=args.force_cpu,
                batch_size=args.batch_size,
                min_spearman=args.min_spearman,
            )
            score_path = output_dir / f"{model_name}__{preprocessing}__scores.parquet"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            scores.to_parquet(score_path, index=False)
            model_summaries.append(summary)
            all_summaries.append(summary)
            print(f"{model_name} {preprocessing}: eligible={summary['eligible']} rho={summary['raw_vs_arena_spearman']:.6f}")
        decisions[model_name] = choose_preprocessing(model_summaries)
    current_summary = pd.DataFrame(all_summaries)
    summary_path = output_dir / "parity_summary.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    summary = merge_resumable_summaries(existing_summary, current_summary, models)
    summary.to_csv(summary_path, index=False)
    for model_name, group in summary.groupby("model", sort=True):
        decisions[str(model_name)] = choose_preprocessing(group.to_dict("records"))
    report = {
        "report_schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "H2 pre-intervention ONNX baseline parity gate; not an intervention result",
        "dataset": args.dataset,
        "dataset_revision": dataset["revision"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": {"seed": args.seed, "per_label": args.per_label, "n_total": len(manifest)},
        "predeclared_preprocessing_policy": "prefer raw if it passes; otherwise accept exactly one passing nonraw candidate",
        "decisions": decisions,
    }
    (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
