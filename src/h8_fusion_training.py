"""Source-only fitting for the H8-SF rank-space robust score fusers.

This module accepts a sealed source feature panel and a sealed *label-free*
target feature panel.  It may produce target predictions, but it has no target
label path and intentionally contains no target metric implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import roc_curve

from src.h8_score_fusion import (
    H8_VERSION,
    MODELS,
    SOURCE_DATASETS,
    TARGET_DATASETS,
    _directory_for_new_outputs,
    canonical_json,
    feature_column,
    sha256_file,
)


FUSION_METHODS = ("B0", "B1", "B2", "P", "A1")
FUSION_SEEDS = (1701, 1702, 1703)
SELECTION_SEED = 1701
FUSION_STEPS = 1_000
FUSION_LR = 1e-2
B2_L2 = 1e-2
P_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
P_ETA_GRID = (1e-2, 5e-2, 1e-1)
A1_L2_GRID = P_L2_GRID
A1_VREX_GRID = (1e-2, 1e-1, 1.0)

MethodKind = Literal["erm", "groupdro", "vrex"]


@dataclass(frozen=True)
class LinearFuser:
    """A fitted nonnegative linear fuser in the frozen feature order."""

    method: str
    seed: int
    l2: float
    groupdro_eta: float | None
    vrex_lambda: float | None
    weights: np.ndarray
    bias: float
    final_group_losses: dict[str, float]
    final_group_weights: dict[str, float] | None

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.weights + self.bias

    def jsonable(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "l2": self.l2,
            "groupdro_eta": self.groupdro_eta,
            "vrex_lambda": self.vrex_lambda,
            "weights": [float(value) for value in self.weights],
            "bias": self.bias,
            "final_group_losses": self.final_group_losses,
            "final_group_weights": self.final_group_weights,
        }


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute deterministic interpolated EER with vertical-segment handling."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise ValueError("H8 EER needs finite scores and both binary classes")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    differences = fpr - fnr
    exact = np.flatnonzero(np.isclose(differences, 0.0, atol=1e-15))
    if len(exact):
        return float(fpr[int(exact[0])])
    crossings = np.flatnonzero(np.signbit(differences[:-1]) != np.signbit(differences[1:]))
    if not len(crossings):
        nearest = int(np.argmin(np.abs(differences)))
        return float((fpr[nearest] + fnr[nearest]) / 2.0)
    index = int(crossings[0])
    fpr_low, fpr_high = float(fpr[index]), float(fpr[index + 1])
    if math.isclose(fpr_low, fpr_high, abs_tol=1e-15):
        return fpr_low
    difference_low, difference_high = float(differences[index]), float(differences[index + 1])
    interpolation = -difference_low / (difference_high - difference_low)
    return float(fpr_low + interpolation * (fpr_high - fpr_low))


def _feature_columns() -> tuple[str, ...]:
    return tuple(feature_column(model) for model in MODELS)


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    group_names = tuple(f"{dataset}|{label}" for dataset in sorted(frame["dataset"].unique()) for label in (0, 1))
    row_groups = (frame["dataset"].astype(str) + "|" + frame["label"].astype(int).astype(str)).to_numpy()
    if set(row_groups) != set(group_names):
        raise ValueError("H8 source panel lacks a complete dataset-by-class group")
    mapping = {name: index for index, name in enumerate(group_names)}
    return np.asarray([mapping[name] for name in row_groups], dtype=np.int64), group_names


def _softplus_inverse(value: float) -> float:
    return float(math.log(math.expm1(value)))


def fit_nonnegative_fuser(
    frame: pd.DataFrame,
    *,
    kind: MethodKind,
    l2: float,
    seed: int,
    groupdro_eta: float | None = None,
    vrex_lambda: float | None = None,
    steps: int = FUSION_STEPS,
) -> LinearFuser:
    """Fit a full-batch, corpus-by-class-balanced rank-space logistic fuser."""
    if kind not in ("erm", "groupdro", "vrex") or not np.isfinite(l2) or l2 < 0 or steps <= 0:
        raise ValueError("H8 fuser received an invalid fixed training configuration")
    if kind == "groupdro" and (groupdro_eta is None or groupdro_eta <= 0):
        raise ValueError("H8 GroupDRO requires a positive fixed eta")
    if kind == "vrex" and (vrex_lambda is None or vrex_lambda <= 0):
        raise ValueError("H8 V-REx requires a positive fixed coefficient")
    columns = _feature_columns()
    if any(column not in frame.columns for column in ("dataset", "label", *columns)):
        raise ValueError("H8 fuser source frame schema is incomplete")
    features = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    labels = frame["label"].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all() or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("H8 fuser source features/labels are invalid")
    group_ids, group_names = _groups(frame)
    torch.manual_seed(seed)
    raw_weights = torch.nn.Parameter(torch.full((len(columns),), _softplus_inverse(1.0 / len(columns)), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.AdamW([raw_weights, bias], lr=FUSION_LR, weight_decay=0.0)
    x = torch.as_tensor(features)
    y = torch.as_tensor(labels)
    group_tensor = torch.as_tensor(group_ids)
    group_masks = [group_tensor.eq(index) for index in range(len(group_names))]
    adversarial = torch.full((len(group_names),), 1.0 / len(group_names), dtype=torch.float32)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = functional.softplus(raw_weights)
        logits = x @ weights + bias
        per_example = functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        group_losses = torch.stack([per_example[mask].mean() for mask in group_masks])
        if kind == "erm":
            objective = group_losses.mean()
        elif kind == "groupdro":
            with torch.no_grad():
                updated = adversarial * torch.exp(float(groupdro_eta) * group_losses.detach())
                adversarial = updated / updated.sum()
            objective = torch.sum(adversarial * group_losses)
        else:
            objective = group_losses.mean() + float(vrex_lambda) * torch.var(group_losses, correction=0)
        objective = objective + float(l2) * torch.sum(weights.square())
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        weights = functional.softplus(raw_weights)
        logits = x @ weights + bias
        losses = functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        group_losses = [float(losses[mask].mean()) for mask in group_masks]
    return LinearFuser(
        method=kind,
        seed=seed,
        l2=float(l2),
        groupdro_eta=float(groupdro_eta) if groupdro_eta is not None else None,
        vrex_lambda=float(vrex_lambda) if vrex_lambda is not None else None,
        weights=weights.detach().cpu().numpy().astype(np.float64),
        bias=float(bias.detach().cpu()),
        final_group_losses={name: value for name, value in zip(group_names, group_losses, strict=True)},
        final_group_weights={name: float(value) for name, value in zip(group_names, adversarial.tolist(), strict=True)} if kind == "groupdro" else None,
    )


def _aggregate_eers(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, float, dict[str, float]]:
    per_dataset: dict[str, float] = {}
    for dataset in sorted(frame["dataset"].unique()):
        mask = frame["dataset"].eq(dataset).to_numpy()
        per_dataset[str(dataset)] = _eer(frame.loc[mask, "label"].to_numpy(), scores[mask])
    values = list(per_dataset.values())
    return float(np.mean(values)), float(np.max(values)), per_dataset


def select_source_hyperparameters(
    frame: pd.DataFrame,
    *,
    kind: Literal["groupdro", "vrex"],
    steps: int = FUSION_STEPS,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Choose only from the locked grids through source-corpus LOO validation."""
    datasets = tuple(sorted(frame["dataset"].unique()))
    if datasets != tuple(sorted(SOURCE_DATASETS)):
        raise ValueError("H8 source hyperparameter selection requires exactly the locked source datasets")
    if kind == "groupdro":
        candidates = [(l2, value) for l2 in P_L2_GRID for value in P_ETA_GRID]
    elif kind == "vrex":
        candidates = [(l2, value) for l2 in A1_L2_GRID for value in A1_VREX_GRID]
    else:
        raise ValueError("H8 source selection only supports GroupDRO or V-REx")
    reports: list[dict[str, object]] = []
    for l2, parameter in candidates:
        validation_eers: dict[str, float] = {}
        for held_out in datasets:
            train = frame.loc[~frame["dataset"].eq(held_out)].reset_index(drop=True)
            validation = frame.loc[frame["dataset"].eq(held_out)].reset_index(drop=True)
            model = fit_nonnegative_fuser(
                train,
                kind=kind,
                l2=l2,
                seed=SELECTION_SEED,
                groupdro_eta=parameter if kind == "groupdro" else None,
                vrex_lambda=parameter if kind == "vrex" else None,
                steps=steps,
            )
            validation_eers[held_out] = _eer(validation["label"].to_numpy(), model.predict(validation.loc[:, _feature_columns()].to_numpy()))
        reports.append(
            {
                "kind": kind,
                "l2": l2,
                "groupdro_eta": parameter if kind == "groupdro" else None,
                "vrex_lambda": parameter if kind == "vrex" else None,
                "validation_eer_by_heldout_source": validation_eers,
                "validation_worst_eer": float(max(validation_eers.values())),
                "validation_macro_eer": float(np.mean(list(validation_eers.values()))),
            }
        )
    selected = min(
        reports,
        key=lambda item: (
            float(item["validation_worst_eer"]),
            float(item["validation_macro_eer"]),
            float(item["l2"]),
            float(item["groupdro_eta"] if kind == "groupdro" else item["vrex_lambda"]),
        ),
    )
    return {
        "l2": float(selected["l2"]),
        "groupdro_eta": float(selected["groupdro_eta"]) if kind == "groupdro" else None,
        "vrex_lambda": float(selected["vrex_lambda"]) if kind == "vrex" else None,
    }, reports


def _mean_prediction(models: Sequence[LinearFuser], features: np.ndarray) -> np.ndarray:
    if not models:
        raise ValueError("H8 cannot average an empty fuser family")
    return np.mean(np.stack([model.predict(features) for model in models], axis=0), axis=0)


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


def _read_source_panel(features_path: Path, provenance_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = _read_json(provenance_path, description="source feature provenance")
    if provenance.get("artifact_kind") != "h8sf_source_rank_probit_features" or provenance.get("version") != H8_VERSION:
        raise ValueError("H8 source feature provenance contract drift")
    if provenance.get("features_sha256") != sha256_file(features_path):
        raise ValueError("H8 source feature hash mismatch")
    if tuple(provenance.get("source_datasets", [])) != SOURCE_DATASETS or tuple(provenance.get("models", [])) != MODELS:
        raise ValueError("H8 source feature datasets/model roster drift")
    frame = pd.read_parquet(features_path)
    expected = ["dataset", "sample_id", "label", *_feature_columns()]
    if list(frame.columns) != expected or frame.duplicated(["dataset", "sample_id"]).any() or not frame["label"].isin((0, 1)).all():
        raise ValueError("H8 source feature panel schema/identity drift")
    if not np.isfinite(frame.loc[:, _feature_columns()].to_numpy(dtype=float)).all():
        raise ValueError("H8 source feature panel has non-finite values")
    return frame, provenance


def _read_target_panel(features_path: Path, provenance_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = _read_json(provenance_path, description="target feature provenance")
    if provenance.get("artifact_kind") != "h8sf_label_free_target_rank_probit_features" or provenance.get("version") != H8_VERSION:
        raise ValueError("H8 target feature provenance contract drift")
    if provenance.get("features_sha256") != sha256_file(features_path):
        raise ValueError("H8 target feature hash mismatch")
    if tuple(provenance.get("target_datasets", [])) != TARGET_DATASETS or tuple(provenance.get("models", [])) != MODELS:
        raise ValueError("H8 target feature datasets/model roster drift")
    if provenance.get("target_labels_read") is not False or provenance.get("target_metrics_read") is not False:
        raise ValueError("H8 target feature artifact is not label-free")
    frame = pd.read_parquet(features_path)
    expected = ["dataset", "sample_id", *_feature_columns()]
    if list(frame.columns) != expected or frame.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H8 target feature panel schema/identity drift")
    if not np.isfinite(frame.loc[:, _feature_columns()].to_numpy(dtype=float)).all():
        raise ValueError("H8 target feature panel has non-finite values")
    return frame, provenance


def _source_selected_expert(frame: pd.DataFrame) -> tuple[str, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for model in MODELS:
        macro, worst, per_dataset = _aggregate_eers(frame, frame[feature_column(model)].to_numpy(dtype=float))
        candidates.append({"model": model, "source_macro_eer": macro, "source_worst_eer": worst, "source_eer_by_dataset": per_dataset})
    winner = min(candidates, key=lambda item: (float(item["source_macro_eer"]), str(item["model"])))
    return str(winner["model"]), candidates


def fit_h8_source_only_fusers(
    *,
    source_features_path: Path,
    source_provenance_path: Path,
    target_features_path: Path,
    target_provenance_path: Path,
    output_dir: Path,
    steps: int = FUSION_STEPS,
) -> dict[str, Path]:
    """Fit all H8 methods on sources and write label-free target predictions."""
    if steps != FUSION_STEPS:
        raise ValueError("Production H8 fusers require the locked 1,000 steps")
    source, source_provenance = _read_source_panel(Path(source_features_path), Path(source_provenance_path))
    target, target_provenance = _read_target_panel(Path(target_features_path), Path(target_provenance_path))
    columns = _feature_columns()
    b0_model, b0_candidates = _source_selected_expert(source)
    p_settings, p_selection = select_source_hyperparameters(source, kind="groupdro", steps=steps)
    a1_settings, a1_selection = select_source_hyperparameters(source, kind="vrex", steps=steps)
    b2_models = [fit_nonnegative_fuser(source, kind="erm", l2=B2_L2, seed=seed, steps=steps) for seed in FUSION_SEEDS]
    p_models = [
        fit_nonnegative_fuser(source, kind="groupdro", l2=float(p_settings["l2"]), seed=seed, groupdro_eta=float(p_settings["groupdro_eta"]), steps=steps)
        for seed in FUSION_SEEDS
    ]
    a1_models = [
        fit_nonnegative_fuser(source, kind="vrex", l2=float(a1_settings["l2"]), seed=seed, vrex_lambda=float(a1_settings["vrex_lambda"]), steps=steps)
        for seed in FUSION_SEEDS
    ]

    def prediction_table(frame: pd.DataFrame, *, include_label: bool) -> pd.DataFrame:
        features = frame.loc[:, columns].to_numpy(dtype=float)
        data: dict[str, object] = {"dataset": frame["dataset"].astype(str), "sample_id": frame["sample_id"].astype(str)}
        if include_label:
            data["label"] = frame["label"].astype("int8")
        data["B0"] = frame[feature_column(b0_model)].to_numpy(dtype=float)
        data["B1"] = features.mean(axis=1)
        data["B2"] = _mean_prediction(b2_models, features)
        data["P"] = _mean_prediction(p_models, features)
        data["A1"] = _mean_prediction(a1_models, features)
        result = pd.DataFrame(data)
        if not np.isfinite(result.loc[:, FUSION_METHODS].to_numpy(dtype=float)).all():
            raise ValueError("H8 fuser emitted a non-finite prediction")
        return result

    source_predictions = prediction_table(source, include_label=True)
    target_predictions = prediction_table(target, include_label=False)
    output_dir = _directory_for_new_outputs(Path(output_dir))
    source_predictions_path = output_dir / "source_predictions.parquet"
    target_predictions_path = output_dir / "target_predictions.parquet"
    model_path = output_dir / "source_fuser_models.json"
    provenance_path = output_dir / "source_only_fit_provenance.json"
    source_predictions.to_parquet(source_predictions_path, index=False)
    target_predictions.to_parquet(target_predictions_path, index=False)
    model_record = {
        "artifact_kind": "h8sf_source_only_fusers",
        "version": H8_VERSION,
        "feature_columns": list(columns),
        "source_selected_expert": b0_model,
        "source_single_expert_candidates": b0_candidates,
        "b2": {"l2": B2_L2, "seeds": [model.jsonable() for model in b2_models]},
        "P": {"selection": p_settings, "inner_loo": p_selection, "seeds": [model.jsonable() for model in p_models]},
        "A1": {"selection": a1_settings, "inner_loo": a1_selection, "seeds": [model.jsonable() for model in a1_models]},
        "optimizer": {"name": "AdamW", "steps": FUSION_STEPS, "lr": FUSION_LR, "weight_decay": 0.0, "full_batch": True},
    }
    model_path.write_text(canonical_json(model_record), encoding="utf-8")
    provenance = {
        "artifact_kind": "h8sf_source_only_fit",
        "version": H8_VERSION,
        "source_features_path": str(Path(source_features_path).resolve()),
        "source_features_sha256": sha256_file(Path(source_features_path)),
        "source_feature_provenance_path": str(Path(source_provenance_path).resolve()),
        "source_feature_provenance_sha256": sha256_file(Path(source_provenance_path)),
        "target_features_path": str(Path(target_features_path).resolve()),
        "target_features_sha256": sha256_file(Path(target_features_path)),
        "target_feature_provenance_path": str(Path(target_provenance_path).resolve()),
        "target_feature_provenance_sha256": sha256_file(Path(target_provenance_path)),
        "source_feature_provenance_hash_from_input": sha256_file(Path(source_provenance_path)),
        "target_feature_provenance_hash_from_input": sha256_file(Path(target_provenance_path)),
        "model_record_sha256": sha256_file(model_path),
        "source_predictions_sha256": sha256_file(source_predictions_path),
        "target_predictions_sha256": sha256_file(target_predictions_path),
        "n_source_predictions": int(len(source_predictions)),
        "n_target_predictions": int(len(target_predictions)),
        "target_labels_read": False,
        "target_metrics_read": False,
        "source_only_selection": True,
        "source_provenance_contract": source_provenance.get("artifact_kind"),
        "target_provenance_contract": target_provenance.get("artifact_kind"),
    }
    provenance_path.write_text(canonical_json(provenance), encoding="utf-8")
    return {
        "source_predictions": source_predictions_path,
        "target_predictions": target_predictions_path,
        "models": model_path,
        "provenance": provenance_path,
    }
