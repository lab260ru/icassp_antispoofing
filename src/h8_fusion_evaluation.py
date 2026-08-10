"""Final, hash-validated target evaluation for the locked H8-SF study.

Unlike the preceding H8 stages, this is the one module allowed to open target
labels.  It has no model-fitting code and verifies every source-only prediction
and input hash before joining labels or calculating a target metric.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.h8_fusion_training import FUSION_METHODS, _eer
from src.h8_score_fusion import (
    H8_VERSION,
    LABELS,
    TARGET_DATASETS,
    _directory_for_new_outputs,
    _labels_path,
    _read_labels,
    canonical_json,
    sha256_file,
)


BOOTSTRAP_SEED = 2608
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_CAP_PER_TARGET_LABEL = 10_000
TIE_EER_ABSOLUTE = 0.0005
RELATIVE_MEAN_EER_REDUCTION = 0.05


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"H8 {description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H8 {description} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"H8 {description} must be a JSON object")
    return value


def _load_index(index_path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"H8 cannot read Arena index: {index_path}") from error
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), Mapping):
        raise ValueError("H8 Arena index must contain a dataset mapping")
    if not set(TARGET_DATASETS).issubset(value["datasets"]):
        raise ValueError("H8 Arena index lacks a locked target")
    return value


def _pinned_label_hash(dataset_record: Mapping[str, Any]) -> str:
    files = dataset_record.get("files")
    pinned = dataset_record.get("pinned_files")
    if not isinstance(files, Mapping) or not isinstance(files.get("labels"), str) or not isinstance(pinned, list):
        raise ValueError("H8 target record lacks label pin metadata")
    for record in pinned:
        if isinstance(record, Mapping) and record.get("path") == files["labels"] and isinstance(record.get("sha256"), str):
            return str(record["sha256"])
    raise ValueError("H8 target label pin lacks SHA-256")


def _prediction_inputs(
    *,
    target_predictions_path: Path,
    fit_provenance_path: Path,
    model_record_path: Path,
    target_feature_provenance_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_provenance = _read_json(fit_provenance_path, description="source-only fit provenance")
    if fit_provenance.get("artifact_kind") != "h8sf_source_only_fit" or fit_provenance.get("version") != H8_VERSION:
        raise ValueError("H8 source-only fit provenance contract drift")
    if fit_provenance.get("target_predictions_sha256") != sha256_file(target_predictions_path):
        raise ValueError("H8 target prediction hash mismatch")
    if fit_provenance.get("model_record_sha256") != sha256_file(model_record_path):
        raise ValueError("H8 source fuser model-record hash mismatch")
    if fit_provenance.get("target_feature_provenance_sha256") != sha256_file(target_feature_provenance_path):
        raise ValueError("H8 target feature provenance hash mismatch")
    if fit_provenance.get("target_labels_read") is not False or fit_provenance.get("target_metrics_read") is not False or fit_provenance.get("source_only_selection") is not True:
        raise ValueError("H8 predictions were not produced under the source-only firewall")
    target_provenance = _read_json(target_feature_provenance_path, description="target feature provenance")
    if target_provenance.get("artifact_kind") != "h8sf_label_free_target_rank_probit_features" or target_provenance.get("version") != H8_VERSION:
        raise ValueError("H8 target feature provenance contract drift")
    if target_provenance.get("target_labels_read") is not False or target_provenance.get("target_metrics_read") is not False:
        raise ValueError("H8 target feature provenance was not label-free")
    predictions = pd.read_parquet(target_predictions_path)
    expected_columns = ["dataset", "sample_id", *FUSION_METHODS]
    if list(predictions.columns) != expected_columns or predictions.empty:
        raise ValueError("H8 target prediction schema drift")
    if set(predictions["dataset"].astype(str)) != set(TARGET_DATASETS) or predictions.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H8 target prediction identities drift")
    if not np.isfinite(predictions.loc[:, FUSION_METHODS].to_numpy(dtype=float)).all():
        raise ValueError("H8 target prediction values are non-finite")
    return predictions, fit_provenance


def _read_target_labels(index: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read all target labels exactly once and prove their immutable identity."""
    frames: list[pd.DataFrame] = []
    provenance: dict[str, object] = {}
    for dataset in TARGET_DATASETS:
        record = index["datasets"][dataset]
        path = _labels_path(record)
        expected_hash = _pinned_label_hash(record)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"H8 target labels hash mismatch for {dataset}")
        labels = _read_labels(path)
        if len(labels) != int(record.get("n_trials", -1)):
            raise ValueError(f"H8 target label count differs from declared trials for {dataset}")
        labels.insert(0, "dataset", dataset)
        frames.append(labels[["dataset", "sample_id", "label"]])
        provenance[dataset] = {
            "path": str(path),
            "sha256": observed_hash,
            "size_bytes": path.stat().st_size,
            "n_rows": int(len(labels)),
            "label_counts": {str(label): int(labels["label"].eq(label).sum()) for label in LABELS},
            "dataset_revision": record.get("revision"),
        }
    all_labels = pd.concat(frames, ignore_index=True)
    if all_labels.duplicated(["dataset", "sample_id"]).any() or not all_labels["label"].isin(LABELS).all():
        raise ValueError("H8 target labels violate binary/identity invariants")
    return all_labels, provenance


def _bootstrap_key(dataset: str, label: int, sample_id: str) -> str:
    return hashlib.sha256(f"H8SFBOOT|{BOOTSTRAP_SEED}|{dataset}|{label}|{sample_id}".encode("utf-8")).hexdigest()


def freeze_bootstrap_panel(joined: pd.DataFrame) -> pd.DataFrame:
    """Freeze a method-independent label-stratified target uncertainty panel."""
    rows: list[pd.DataFrame] = []
    for dataset in TARGET_DATASETS:
        for label in LABELS:
            subset = joined.loc[(joined["dataset"] == dataset) & (joined["label"] == label)].copy()
            if subset.empty:
                raise ValueError(f"H8 target lacks label {label}: {dataset}")
            subset["selection_key_sha256"] = subset["sample_id"].astype(str).map(lambda value: _bootstrap_key(dataset, label, value))
            subset = subset.sort_values(["selection_key_sha256", "sample_id"], kind="mergesort").head(BOOTSTRAP_CAP_PER_TARGET_LABEL).copy()
            subset["selection_rank"] = np.arange(1, len(subset) + 1, dtype=np.int64)
            rows.append(subset)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H8 bootstrap panel has duplicate IDs")
    return result


def _metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in TARGET_DATASETS:
        data = joined.loc[joined["dataset"].eq(dataset)]
        for method in FUSION_METHODS:
            score = data[method].to_numpy(dtype=float)
            label = data["label"].to_numpy(dtype=int)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_trials": int(len(data)),
                    "n_bonafide": int((label == 0).sum()),
                    "n_spoof": int((label == 1).sum()),
                    "eer": _eer(label, score),
                    "eer_percent": 100.0 * _eer(label, score),
                    "auroc": float(roc_auc_score(label, score)),
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != len(TARGET_DATASETS) * len(FUSION_METHODS) or output.duplicated(["dataset", "method"]).any():
        raise RuntimeError("H8 metric output is incomplete")
    return output


def bootstrap_mean_eer_difference(panel: pd.DataFrame, *, replicates: int = BOOTSTRAP_REPLICATES) -> np.ndarray:
    """Return fixed stratified-bootstrap mean EER(P) - mean EER(B2) draws."""
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Production H8 bootstrap requires exactly 2,000 replicates")
    target_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] = []
    for dataset in TARGET_DATASETS:
        data = panel.loc[panel["dataset"].eq(dataset)]
        by_label = [np.flatnonzero(data["label"].to_numpy(dtype=int) == label) for label in LABELS]
        if any(len(indices) == 0 for indices in by_label):
            raise ValueError(f"H8 bootstrap panel lacks a class: {dataset}")
        labels = data["label"].to_numpy(dtype=int)
        target_data.append((labels, data["P"].to_numpy(dtype=float), data["B2"].to_numpy(dtype=float), (by_label[0], by_label[1])))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        per_target: list[float] = []
        for labels, p_scores, b2_scores, class_positions in target_data:
            selected = np.concatenate([positions[rng.integers(0, len(positions), size=len(positions))] for positions in class_positions])
            per_target.append(_eer(labels[selected], p_scores[selected]) - _eer(labels[selected], b2_scores[selected]))
        differences[replicate] = float(np.mean(per_target))
    return differences


def decision_gate(metrics: pd.DataFrame, bootstrap_differences: np.ndarray) -> dict[str, object]:
    """Evaluate the five literal H8-SF success requirements without selection."""
    if set(metrics["dataset"]) != set(TARGET_DATASETS) or set(metrics["method"]) != set(FUSION_METHODS):
        raise ValueError("H8 decision requires every locked target and method")
    by_method = metrics.pivot(index="dataset", columns="method", values="eer").reindex(index=TARGET_DATASETS, columns=FUSION_METHODS)
    means = by_method.mean(axis=0)
    worsts = by_method.max(axis=0)
    p_mean = float(means["P"])
    b1_mean = float(means["B1"])
    b2_mean = float(means["B2"])
    relative_b1 = float(1.0 - p_mean / b1_mean) if b1_mean > 0 else float("nan")
    relative_b2 = float(1.0 - p_mean / b2_mean) if b2_mean > 0 else float("nan")
    win_tie = (by_method["P"] <= by_method["B1"] + TIE_EER_ABSOLUTE) & (by_method["P"] <= by_method["B2"] + TIE_EER_ABSOLUTE)
    low, high = np.quantile(bootstrap_differences, [0.025, 0.975])
    rules = {
        "mean_relative_reduction_vs_b1_at_least_5pct": bool(relative_b1 >= RELATIVE_MEAN_EER_REDUCTION),
        "mean_relative_reduction_vs_b2_at_least_5pct": bool(relative_b2 >= RELATIVE_MEAN_EER_REDUCTION),
        "worst_eer_lower_than_b1": bool(float(worsts["P"]) < float(worsts["B1"])),
        "worst_eer_lower_than_b2": bool(float(worsts["P"]) < float(worsts["B2"])),
        "wins_or_ties_vs_b1_and_b2_on_at_least_3_of_5": bool(int(win_tie.sum()) >= 3),
        "bootstrap_mean_eer_difference_vs_b2_ci_below_zero": bool(float(high) < 0.0),
    }
    return {
        "primary_method": "P",
        "mean_eer": {method: float(means[method]) for method in FUSION_METHODS},
        "worst_eer": {method: float(worsts[method]) for method in FUSION_METHODS},
        "relative_mean_eer_reduction_p_vs_b1": relative_b1,
        "relative_mean_eer_reduction_p_vs_b2": relative_b2,
        "win_or_tie_targets_p_vs_b1_and_b2": [dataset for dataset, passed in win_tie.items() if bool(passed)],
        "win_or_tie_count": int(win_tie.sum()),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "mean_eer_difference_p_minus_b2": float(np.mean(bootstrap_differences)),
            "ci_low": float(low),
            "ci_high": float(high),
        },
        "rules": rules,
        "positive_result_gate_passed": bool(all(rules.values())),
    }


def evaluate_h8_targets(
    *,
    index_path: Path,
    target_predictions_path: Path,
    fit_provenance_path: Path,
    model_record_path: Path,
    target_feature_provenance_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Perform the one final target-label join and locked exhaustive analysis."""
    index_path = Path(index_path).resolve()
    predictions_path = Path(target_predictions_path).resolve()
    fit_provenance_path = Path(fit_provenance_path).resolve()
    model_record_path = Path(model_record_path).resolve()
    target_feature_provenance_path = Path(target_feature_provenance_path).resolve()
    index = _load_index(index_path)
    predictions, fit_provenance = _prediction_inputs(
        target_predictions_path=predictions_path,
        fit_provenance_path=fit_provenance_path,
        model_record_path=model_record_path,
        target_feature_provenance_path=target_feature_provenance_path,
    )
    labels, label_provenance = _read_target_labels(index)
    joined = predictions.merge(labels, on=["dataset", "sample_id"], how="inner", validate="one_to_one")
    if len(joined) != len(predictions) or len(joined) != len(labels):
        raise ValueError(f"H8 target prediction/label coverage mismatch: predictions={len(predictions)} labels={len(labels)} joined={len(joined)}")
    metrics = _metrics(joined)
    bootstrap_panel = freeze_bootstrap_panel(joined)
    bootstrap_differences = bootstrap_mean_eer_difference(bootstrap_panel)
    gate = decision_gate(metrics, bootstrap_differences)
    output_dir = _directory_for_new_outputs(Path(output_dir))
    metrics_path = output_dir / "h8_target_metrics.csv"
    bootstrap_panel_path = output_dir / "h8_bootstrap_panel.parquet"
    bootstrap_path = output_dir / "h8_bootstrap_mean_eer_differences.csv"
    decision_path = output_dir / "h8_decision_gate.json"
    provenance_path = output_dir / "h8_target_evaluation_provenance.json"
    metrics.to_csv(metrics_path, index=False)
    bootstrap_panel.to_parquet(bootstrap_panel_path, index=False)
    pd.DataFrame({"replicate": np.arange(BOOTSTRAP_REPLICATES, dtype=np.int64), "mean_eer_difference_p_minus_b2": bootstrap_differences}).to_csv(bootstrap_path, index=False)
    decision_path.write_text(canonical_json(gate), encoding="utf-8")
    provenance = {
        "artifact_kind": "h8sf_final_target_evaluation",
        "version": H8_VERSION,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "target_predictions_path": str(predictions_path),
        "target_predictions_sha256": sha256_file(predictions_path),
        "source_only_fit_provenance_path": str(fit_provenance_path),
        "source_only_fit_provenance_sha256": sha256_file(fit_provenance_path),
        "source_fuser_models_path": str(model_record_path),
        "source_fuser_models_sha256": sha256_file(model_record_path),
        "target_feature_provenance_path": str(target_feature_provenance_path),
        "target_feature_provenance_sha256": sha256_file(target_feature_provenance_path),
        "target_label_artifacts": label_provenance,
        "target_label_join_rows": int(len(joined)),
        "target_label_join_complete": True,
        "target_labels_read": True,
        "target_metrics_read": True,
        "bootstrap_selection": {
            "seed": BOOTSTRAP_SEED,
            "cap_per_target_label": BOOTSTRAP_CAP_PER_TARGET_LABEL,
            "key_template": "sha256('H8SFBOOT|2608|dataset|label|sample_id')",
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "metrics_sha256": sha256_file(metrics_path),
        "bootstrap_panel_sha256": sha256_file(bootstrap_panel_path),
        "bootstrap_differences_sha256": sha256_file(bootstrap_path),
        "decision_gate_sha256": sha256_file(decision_path),
        "fit_firewall_target_labels_read": fit_provenance.get("target_labels_read"),
        "fit_firewall_target_metrics_read": fit_provenance.get("target_metrics_read"),
    }
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return {
        "metrics": metrics_path,
        "bootstrap_panel": bootstrap_panel_path,
        "bootstrap_differences": bootstrap_path,
        "decision": decision_path,
        "provenance": provenance_path,
    }
