"""Terminal-only H9-PCR target evaluation with provenance hard stops.

This module is deliberately the *only* H9 component which decodes target
labels.  It is not a model-selection tool: before it opens a SONAR or ArAD
label file it validates two canonical target manifests, an exhaustive frozen
``B1``/``B2``/``P`` by seed checkpoint ledger, every checkpoint payload, the
source pairing hashes, and the source--target canonical-fingerprint audit.

The expected input contracts are documented in
``experiments/h9_paired_counterfactual/TERMINAL_EVALUATION.md``.  They keep
target labels out of every source-development artifact and make the terminal
call fail closed if a result is missing, duplicated, or not fresh initialized.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.h9_pcr_training import (
    H9_LAMBDA_GRID,
    H9_SEEDS,
    H9_SEED_DEVICES,
    H9_MARGIN,
    _eer,
    _load_audio_16k,
    _spoof_logit,
    build_fresh_res2tcn_guard,
)
from src.res2tcn_pytorch import sha256_file


H9_EVALUATION_VERSION = "h9-pcr-terminal-evaluation-v1"
TARGET_REVISIONS: Mapping[str, str] = {
    "SONAR": "eca7c72ebdf0f7936a644605a56735ac8564dbd9",
    "ArAD": "350184966eeb5b46ff2acdabd8f4d12e41e582da",
}
METHODS: tuple[str, ...] = ("B1", "B2", "P")
TERMINAL_BOOTSTRAP_SEED = 2909
TERMINAL_BOOTSTRAP_REPLICATES = 2_000
EVALUATION_BATCH_SIZE = 24
REQUIRED_TARGET_RECORD_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "audio_path",
    "audio_sha256",
    "audio_bytes",
    "canonical_fingerprint",
)
_SHA256_HEX_LENGTH = 64


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H9 {description} is not a valid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"H9 {description} must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH and all(character in "0123456789abcdef" for character in value.lower())


def _require_sha256(value: object, *, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"H9 {description} must be a lowercase SHA-256 hex digest")
    return str(value)


def _hash_matches(path: Path, expected: object, *, description: str) -> str:
    expected_digest = _require_sha256(expected, description=f"{description} expected hash")
    if not path.is_file():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    observed = sha256_file(path)
    if observed != expected_digest:
        raise ValueError(f"H9 {description} SHA-256 mismatch")
    return observed


def _read_table(path: Path, *, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"H9 {description} is unavailable: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"H9 {description} must be CSV or Parquet")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"H9 {description} did not decode to a table")
    return frame


def _nonempty_text(frame: pd.DataFrame, column: str, *, description: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise ValueError(f"H9 {description} requires nonempty {column} values")


@dataclass(frozen=True)
class TargetManifest:
    """A canonical target trial list that contains no label values."""

    dataset: str
    revision: str
    manifest_path: Path
    manifest_sha256: str
    records_path: Path
    records_sha256: str
    records: pd.DataFrame
    labels_path: Path
    labels_sha256: str


@dataclass(frozen=True)
class FrozenCheckpoint:
    """One source-selected H9 checkpoint plus its validated H9 identity."""

    method: str
    seed: int
    path: Path
    sha256: str
    state_dict: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenCheckpointLedger:
    """The exhaustive source-only checkpoint matrix and all its bindings."""

    ledger_path: Path
    ledger_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    source_p_pairs_path: Path
    source_p_pairs_sha256: str
    source_b2_pairs_path: Path
    source_b2_pairs_sha256: str
    architecture_bundle: Path
    architecture_sha256: str
    selected_lambda_rank: float
    checkpoints: Mapping[tuple[str, int], FrozenCheckpoint]


def _path_from(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"H9 {description} requires a nonempty path")
    return Path(value).expanduser().resolve()


def _fingerprint_set(frame: pd.DataFrame, *, description: str) -> set[str]:
    if "canonical_fingerprint" not in frame.columns or "sample_id" not in frame.columns:
        raise ValueError(f"H9 {description} lacks sample_id/canonical_fingerprint columns")
    _nonempty_text(frame, "sample_id", description=description)
    _nonempty_text(frame, "canonical_fingerprint", description=description)
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"H9 {description} has duplicate sample IDs")
    values = frame["canonical_fingerprint"].astype(str)
    if not values.map(_is_sha256).all():
        raise ValueError(f"H9 {description} has an invalid canonical audio fingerprint")
    return set(values.tolist())


def load_target_manifest(path: str | Path, *, expected_dataset: str) -> TargetManifest:
    """Validate one metadata-only target manifest without reading its labels."""
    if expected_dataset not in TARGET_REVISIONS:
        raise ValueError(f"H9 unknown locked target: {expected_dataset}")
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path, description=f"{expected_dataset} target manifest")
    if manifest.get("artifact_kind") != "h9_pcr_canonical_target_manifest":
        raise ValueError("H9 target manifest artifact kind drift")
    if manifest.get("version") != H9_EVALUATION_VERSION:
        raise ValueError("H9 target manifest version drift")
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("H9 target manifest dataset does not match its fixed CLI role")
    if manifest.get("dataset_revision") != TARGET_REVISIONS[expected_dataset]:
        raise ValueError("H9 target manifest revision differs from the locked data contract")
    if manifest.get("target_labels_read") is not False or manifest.get("target_metrics_read") is not False:
        raise ValueError("H9 target manifest was not created under the target-label firewall")
    records_path = _path_from(manifest.get("records_path"), description="target manifest records_path")
    records_sha256 = _hash_matches(records_path, manifest.get("records_sha256"), description="target records")
    records = _read_table(records_path, description="target records")
    if tuple(records.columns.astype(str)) != REQUIRED_TARGET_RECORD_COLUMNS:
        raise ValueError(f"H9 target records schema drift; expected exactly {list(REQUIRED_TARGET_RECORD_COLUMNS)}")
    if records.empty:
        raise ValueError("H9 target records cannot be empty")
    for column in REQUIRED_TARGET_RECORD_COLUMNS:
        _nonempty_text(records, column, description="target records")
    if records["sample_id"].astype(str).duplicated().any():
        raise ValueError("H9 target records have duplicate sample IDs")
    if not records["audio_sha256"].astype(str).map(_is_sha256).all():
        raise ValueError("H9 target records have invalid audio SHA-256 values")
    if not records["canonical_fingerprint"].astype(str).map(_is_sha256).all():
        raise ValueError("H9 target records have invalid canonical audio fingerprints")
    try:
        byte_counts = records["audio_bytes"].astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("H9 target records audio_bytes must be positive integers") from error
    if (byte_counts <= 0).any():
        raise ValueError("H9 target records audio_bytes must be positive")
    labels = manifest.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError("H9 target manifest requires a separate label-artifact mapping")
    labels_path = _path_from(labels.get("path"), description="target label artifact")
    labels_sha256 = _hash_matches(labels_path, labels.get("sha256"), description="target label artifact")
    return TargetManifest(
        dataset=expected_dataset,
        revision=TARGET_REVISIONS[expected_dataset],
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        records_path=records_path,
        records_sha256=records_sha256,
        records=records,
        labels_path=labels_path,
        labels_sha256=labels_sha256,
    )


def _read_source_fingerprints(path: Path, expected_hash: str) -> set[str]:
    _hash_matches(path, expected_hash, description="source materialized manifest")
    frame = _read_table(path, description="source materialized manifest")
    return _fingerprint_set(frame, description="source materialized manifest")


def _strict_float(value: object, *, description: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"H9 {description} must be numeric") from error
    if not math.isfinite(output):
        raise ValueError(f"H9 {description} must be finite")
    return output


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch releases
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"H9 cannot read checkpoint payload: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("H9 checkpoint payload must be a mapping")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("H9 checkpoint lacks model_state_dict")
    return payload


def _expect_identity(value: Mapping[str, Any], key: str, expected: object, *, description: str) -> None:
    if value.get(key) != expected:
        raise ValueError(f"H9 {description} drift in {key}")


def _validate_training_config(
    config: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    selected_lambda_rank: float,
    envelope: Mapping[str, Any],
) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("H9 checkpoint training_config must be a mapping")
    _expect_identity(config, "method", method, description="checkpoint training config")
    _expect_identity(config, "seed", seed, description="checkpoint training config")
    _expect_identity(config, "device", H9_SEED_DEVICES[seed], description="checkpoint training config")
    expected_lambda = 0.0 if method == "B1" else selected_lambda_rank
    if _strict_float(config.get("lambda_rank"), description="checkpoint lambda_rank") != expected_lambda:
        raise ValueError("H9 checkpoint lambda rank differs from the locked condition")
    for key in ("batch_size", "num_workers", "max_epochs", "learning_rate", "weight_decay", "margin", "require_cuda"):
        if key not in envelope:
            raise ValueError(f"H9 frozen training envelope lacks {key}")
        observed = config.get(key)
        expected = envelope[key]
        if isinstance(expected, float):
            if _strict_float(observed, description=f"checkpoint {key}") != expected:
                raise ValueError(f"H9 checkpoint training config drift in {key}")
        elif observed != expected:
            raise ValueError(f"H9 checkpoint training config drift in {key}")
    if config.get("batch_size") != EVALUATION_BATCH_SIZE:
        raise ValueError("H9 checkpoint batch size does not match the protocol-frozen value 24")
    if config.get("num_workers") != 4 or config.get("max_epochs") != 6 or config.get("require_cuda") is not True:
        raise ValueError("H9 checkpoint violates the locked worker/epoch/CUDA envelope")
    if _strict_float(config.get("margin"), description="checkpoint margin") != H9_MARGIN:
        raise ValueError("H9 checkpoint margin differs from the locked PCR objective")


def _validate_sidecar(
    sidecar: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    method: str,
    seed: int,
    checkpoint_hash: str,
    source_hashes: Mapping[str, str],
    architecture_sha256: str,
) -> None:
    _expect_identity(sidecar, "method", method, description="training result sidecar")
    _expect_identity(sidecar, "seed", seed, description="training result sidecar")
    _expect_identity(sidecar, "checkpoint_sha256", checkpoint_hash, description="training result sidecar")
    if sidecar.get("target_labels_read") is not False or sidecar.get("target_audio_read") is not False:
        raise ValueError("H9 checkpoint sidecar breached the target firewall")
    _expect_identity(sidecar, "precision", "cuda_bfloat16_autocast", description="training precision provenance")
    architecture = sidecar.get("architecture_provenance")
    if not isinstance(architecture, Mapping):
        raise ValueError("H9 checkpoint sidecar lacks architecture provenance")
    _expect_identity(architecture, "architecture_sha256", architecture_sha256, description="sidecar architecture provenance")
    _expect_identity(architecture, "initialization", "fresh_seeded", description="sidecar initialization provenance")
    _expect_identity(architecture, "initialization_seed", seed, description="sidecar initialization provenance")
    _expect_identity(architecture, "checkpoint_loaded", False, description="sidecar initialization provenance")
    run_hashes = sidecar.get("source_artifact_hashes")
    if not isinstance(run_hashes, Mapping) or dict(run_hashes) != dict(source_hashes):
        raise ValueError("H9 checkpoint sidecar source manifest/pair hash binding drift")
    if record.get("source_artifact_hashes") != run_hashes:
        raise ValueError("H9 ledger record does not match checkpoint sidecar source hash binding")


def load_frozen_checkpoint_ledger(
    path: str | Path,
    *,
    plan_path: str | Path,
    data_contract_path: str | Path,
) -> FrozenCheckpointLedger:
    """Validate the complete source-only H9 checkpoint matrix before inference."""
    ledger_path = Path(path).expanduser().resolve()
    ledger = _read_json(ledger_path, description="frozen checkpoint ledger")
    if ledger.get("artifact_kind") != "h9_pcr_frozen_checkpoint_ledger" or ledger.get("version") != H9_EVALUATION_VERSION:
        raise ValueError("H9 frozen checkpoint ledger contract drift")
    if ledger.get("target_labels_read") is not False or ledger.get("target_audio_read") is not False or ledger.get("target_metrics_read") is not False:
        raise ValueError("H9 frozen checkpoint ledger breached the target firewall")
    protocol = ledger.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("H9 frozen checkpoint ledger lacks protocol hashes")
    plan = Path(plan_path).expanduser().resolve()
    data_contract = Path(data_contract_path).expanduser().resolve()
    _hash_matches(plan, protocol.get("plan_sha256"), description="locked H9 plan")
    _hash_matches(data_contract, protocol.get("data_contract_sha256"), description="locked H9 data contract")
    source = ledger.get("source_artifacts")
    if not isinstance(source, Mapping):
        raise ValueError("H9 frozen checkpoint ledger lacks source artifact bindings")
    source_manifest_path = _path_from(source.get("source_manifest_path"), description="source manifest")
    source_manifest_sha256 = _hash_matches(source_manifest_path, source.get("source_manifest_sha256"), description="source manifest")
    source_p_pairs_path = _path_from(source.get("p_pairs_path"), description="source P pairs")
    source_p_pairs_sha256 = _hash_matches(source_p_pairs_path, source.get("p_pairs_sha256"), description="source P pairs")
    source_b2_pairs_path = _path_from(source.get("b2_pairs_path"), description="source B2 pairs")
    source_b2_pairs_sha256 = _hash_matches(source_b2_pairs_path, source.get("b2_pairs_sha256"), description="source B2 pairs")
    source_hashes = {
        "source_manifest_sha256": source_manifest_sha256,
        "p_pairs_sha256": source_p_pairs_sha256,
        "b2_pairs_sha256": source_b2_pairs_sha256,
    }
    architecture = ledger.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("H9 frozen checkpoint ledger lacks architecture binding")
    architecture_bundle = _path_from(architecture.get("bundle_dir"), description="architecture bundle")
    architecture_path = architecture_bundle / "_net.py"
    architecture_sha256 = _hash_matches(architecture_path, architecture.get("architecture_sha256"), description="architecture definition")
    selection = ledger.get("source_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("H9 frozen checkpoint ledger lacks source-only lambda selection")
    selected_lambda_rank = _strict_float(selection.get("selected_lambda_rank"), description="selected lambda rank")
    if selected_lambda_rank not in H9_LAMBDA_GRID:
        raise ValueError("H9 selected lambda is outside the locked source-only grid")
    selection_path = _path_from(selection.get("path"), description="source-only lambda selection")
    _hash_matches(selection_path, selection.get("sha256"), description="source-only lambda selection")
    selection_value = _read_json(selection_path, description="source-only lambda selection")
    _expect_identity(selection_value, "selected_lambda_rank", selected_lambda_rank, description="source-only lambda selection")
    if selection_value.get("target_labels_read") is not False or selection_value.get("target_audio_read") is not False:
        raise ValueError("H9 lambda selection breached the target firewall")
    envelope = ledger.get("training_envelope")
    if not isinstance(envelope, Mapping):
        raise ValueError("H9 frozen checkpoint ledger lacks training envelope")
    runs = ledger.get("checkpoints")
    if not isinstance(runs, list):
        raise ValueError("H9 frozen checkpoint ledger checkpoints must be a list")
    expected_keys = {(method, seed) for method in METHODS for seed in H9_SEEDS}
    checkpoints: dict[tuple[str, int], FrozenCheckpoint] = {}
    for record in runs:
        if not isinstance(record, Mapping):
            raise ValueError("H9 checkpoint ledger record must be an object")
        method, seed = record.get("method"), record.get("seed")
        if method not in METHODS or not isinstance(seed, int) or seed not in H9_SEEDS:
            raise ValueError("H9 checkpoint ledger has an invalid method/seed identity")
        key = (str(method), seed)
        if key in checkpoints:
            raise ValueError("H9 checkpoint ledger has duplicate method/seed records")
        checkpoint_path = _path_from(record.get("checkpoint_path"), description="checkpoint")
        checkpoint_sha256 = _hash_matches(checkpoint_path, record.get("checkpoint_sha256"), description="checkpoint")
        sidecar_path = _path_from(record.get("training_record_path"), description="checkpoint training record")
        _hash_matches(sidecar_path, record.get("training_record_sha256"), description="checkpoint training record")
        sidecar = _read_json(sidecar_path, description="checkpoint training record")
        _validate_sidecar(
            sidecar,
            record=record,
            method=str(method),
            seed=seed,
            checkpoint_hash=checkpoint_sha256,
            source_hashes=source_hashes,
            architecture_sha256=architecture_sha256,
        )
        payload = _checkpoint_payload(checkpoint_path)
        _expect_identity(payload, "source_manifest_sha256", source_manifest_sha256, description="checkpoint source manifest")
        _expect_identity(payload, "selection_rule", "lowest_source_dev_eer_then_lower_epoch", description="checkpoint selection rule")
        best_epoch = payload.get("best_epoch")
        if not isinstance(best_epoch, int) or not 1 <= best_epoch <= 6:
            raise ValueError("H9 checkpoint best epoch violates the fixed 6-epoch source stop")
        _strict_float(payload.get("source_dev_eer"), description="checkpoint source development EER")
        checkpoint_architecture = payload.get("architecture_provenance")
        if not isinstance(checkpoint_architecture, Mapping):
            raise ValueError("H9 checkpoint lacks architecture provenance")
        _expect_identity(checkpoint_architecture, "architecture_sha256", architecture_sha256, description="checkpoint architecture provenance")
        _expect_identity(checkpoint_architecture, "initialization", "fresh_seeded", description="checkpoint initialization provenance")
        _expect_identity(checkpoint_architecture, "initialization_seed", seed, description="checkpoint initialization provenance")
        _expect_identity(checkpoint_architecture, "checkpoint_loaded", False, description="checkpoint initialization provenance")
        _validate_training_config(
            payload.get("training_config"),
            method=str(method),
            seed=seed,
            selected_lambda_rank=selected_lambda_rank,
            envelope=envelope,
        )
        checkpoints[key] = FrozenCheckpoint(str(method), seed, checkpoint_path, checkpoint_sha256, payload["model_state_dict"])
    if set(checkpoints) != expected_keys:
        missing = sorted(expected_keys - set(checkpoints))
        extra = sorted(set(checkpoints) - expected_keys)
        raise ValueError(f"H9 frozen checkpoint ledger must contain exactly all B1/B2/P × four seeds; missing={missing}, extra={extra}")
    return FrozenCheckpointLedger(
        ledger_path=ledger_path,
        ledger_sha256=sha256_file(ledger_path),
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        source_p_pairs_path=source_p_pairs_path,
        source_p_pairs_sha256=source_p_pairs_sha256,
        source_b2_pairs_path=source_b2_pairs_path,
        source_b2_pairs_sha256=source_b2_pairs_sha256,
        architecture_bundle=architecture_bundle,
        architecture_sha256=architecture_sha256,
        selected_lambda_rank=selected_lambda_rank,
        checkpoints=checkpoints,
    )


def _verify_target_audio(records: pd.DataFrame) -> None:
    """Byte-validate every declared target waveform before model inference."""
    for row in records.itertuples(index=False):
        path = Path(str(row.audio_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"H9 target audio is unavailable: {path}")
        if path.stat().st_size != int(row.audio_bytes):
            raise ValueError(f"H9 target audio byte count mismatch: {path}")
        if sha256_file(path) != str(row.audio_sha256):
            raise ValueError(f"H9 target audio SHA-256 mismatch: {path}")


def _predict_checkpoint(
    target: TargetManifest,
    checkpoint: FrozenCheckpoint,
    *,
    architecture_bundle: Path,
    device: str,
    waveform_loader: Callable[[str], np.ndarray],
    model_factory: Callable[..., tuple[torch.nn.Module, Mapping[str, Any]]],
) -> pd.DataFrame:
    model, provenance = model_factory(architecture_bundle, seed=checkpoint.seed, device=device)
    if provenance.get("checkpoint_loaded") is not False or provenance.get("initialization") != "fresh_seeded":
        raise ValueError("H9 terminal evaluator requires a fresh architecture construction before checkpoint load")
    model.load_state_dict(checkpoint.state_dict, strict=True)
    model.eval()
    sample_ids: list[str] = []
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    device_type = torch.device(device).type
    autocast: contextlib.AbstractContextManager[Any]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else contextlib.nullcontext()
    records = target.records
    with torch.no_grad():
        for start in range(0, len(records), EVALUATION_BATCH_SIZE):
            chunk = records.iloc[start : start + EVALUATION_BATCH_SIZE]
            waveforms = [np.asarray(waveform_loader(str(path)), dtype=np.float32) for path in chunk["audio_path"].tolist()]
            batch = np.stack(waveforms, axis=0).astype(np.float32, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            with autocast:
                full_logits = model(tensor)
                spoof_logits = _spoof_logit(full_logits)
                spoof_probability = torch.softmax(full_logits[1] if isinstance(full_logits, tuple) else full_logits, dim=1)[:, 1]
            sample_ids.extend(chunk["sample_id"].astype(str).tolist())
            logits.append(spoof_logits.float().detach().cpu().numpy())
            probabilities.append(spoof_probability.float().detach().cpu().numpy())
    return pd.DataFrame(
        {
            "dataset": target.dataset,
            "sample_id": sample_ids,
            "method": checkpoint.method,
            "seed": checkpoint.seed,
            "checkpoint_sha256": checkpoint.sha256,
            "spoof_logit": np.concatenate(logits),
            "spoof_probability": np.concatenate(probabilities),
        }
    )


def _read_target_labels(target: TargetManifest) -> pd.DataFrame:
    """Open a label artifact only after predictions have been written."""
    if sha256_file(target.labels_path) != target.labels_sha256:
        raise ValueError(f"H9 target label artifact changed before terminal label load: {target.dataset}")
    labels = _read_table(target.labels_path, description=f"{target.dataset} target labels")
    if tuple(labels.columns.astype(str)) != ("sample_id", "label"):
        raise ValueError("H9 target labels schema must be exactly sample_id,label")
    _nonempty_text(labels, "sample_id", description="target labels")
    if labels["sample_id"].astype(str).duplicated().any():
        raise ValueError("H9 target labels have duplicate sample IDs")
    try:
        numeric = labels["label"].astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("H9 target labels must be binary integers") from error
    if set(numeric.unique()) != {0, 1}:
        raise ValueError("H9 target labels require both binary classes")
    return pd.DataFrame({"dataset": target.dataset, "sample_id": labels["sample_id"].astype(str), "label": numeric})


def _aggregate_prediction_scores(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    counts = raw_predictions.groupby(["dataset", "sample_id", "method"], sort=False)["seed"].nunique()
    if not counts.eq(len(H9_SEEDS)).all():
        raise ValueError("H9 raw predictions do not contain exactly four seeds per trial/method")
    return (
        raw_predictions.groupby(["dataset", "sample_id", "method"], as_index=False, sort=False)["spoof_probability"]
        .mean()
        .rename(columns={"spoof_probability": "mean_spoof_probability"})
    )


def _metrics_from_labels(raw_predictions: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = _aggregate_prediction_scores(raw_predictions)
    aggregate_join = aggregate.merge(labels, on=["dataset", "sample_id"], how="inner", validate="many_to_one")
    if len(aggregate_join) != len(aggregate):
        raise ValueError("H9 aggregate prediction/label coverage mismatch")
    seed_join = raw_predictions.merge(labels, on=["dataset", "sample_id"], how="inner", validate="many_to_one")
    if len(seed_join) != len(raw_predictions):
        raise ValueError("H9 seed prediction/label coverage mismatch")
    metrics: list[dict[str, object]] = []
    seed_metrics: list[dict[str, object]] = []
    for dataset in TARGET_REVISIONS:
        for method in METHODS:
            data = aggregate_join.loc[(aggregate_join["dataset"] == dataset) & (aggregate_join["method"] == method)]
            label = data["label"].to_numpy(dtype=int)
            score = data["mean_spoof_probability"].to_numpy(dtype=float)
            metrics.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_trials": int(len(data)),
                    "n_bonafide": int((label == 0).sum()),
                    "n_spoof": int((label == 1).sum()),
                    "eer": _eer(label, score),
                    "eer_percent": 100.0 * _eer(label, score),
                    "auroc": float(roc_auc_score(label, score)),
                    "score_aggregation": "mean_spoof_probability_over_four_frozen_seeds",
                }
            )
            for seed in H9_SEEDS:
                seed_data = seed_join.loc[
                    (seed_join["dataset"] == dataset) & (seed_join["method"] == method) & (seed_join["seed"] == seed)
                ]
                seed_label = seed_data["label"].to_numpy(dtype=int)
                seed_score = seed_data["spoof_probability"].to_numpy(dtype=float)
                seed_metrics.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "n_trials": int(len(seed_data)),
                        "eer": _eer(seed_label, seed_score),
                        "eer_percent": 100.0 * _eer(seed_label, seed_score),
                        "auroc": float(roc_auc_score(seed_label, seed_score)),
                    }
                )
    metrics_frame = pd.DataFrame(metrics)
    seed_metrics_frame = pd.DataFrame(seed_metrics)
    expected_metrics = len(TARGET_REVISIONS) * len(METHODS)
    expected_seed_metrics = expected_metrics * len(H9_SEEDS)
    if len(metrics_frame) != expected_metrics or len(seed_metrics_frame) != expected_seed_metrics:
        raise RuntimeError("H9 terminal metrics are incomplete")
    wide = aggregate_join.pivot(index=["dataset", "sample_id", "label"], columns="method", values="mean_spoof_probability").reset_index()
    if tuple(wide.columns) != ("dataset", "sample_id", "label", *METHODS):
        raise ValueError("H9 labeled prediction reconstruction is incomplete")
    return metrics_frame, seed_metrics_frame, wide


def bootstrap_macro_eer_differences(labeled_predictions: pd.DataFrame, *, replicates: int = TERMINAL_BOOTSTRAP_REPLICATES) -> pd.DataFrame:
    """Fixed shared-ID, label-stratified bootstrap for both H9 comparators."""
    if replicates != TERMINAL_BOOTSTRAP_REPLICATES:
        raise ValueError("H9 production terminal bootstrap requires exactly 2,000 replicates")
    target_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]] = []
    for dataset in TARGET_REVISIONS:
        data = labeled_predictions.loc[labeled_predictions["dataset"] == dataset].reset_index(drop=True)
        if data["sample_id"].astype(str).duplicated().any():
            raise ValueError("H9 bootstrap requires unique target sample IDs")
        labels = data["label"].to_numpy(dtype=int)
        locations = tuple(np.flatnonzero(labels == label) for label in (0, 1))
        if any(len(values) == 0 for values in locations):
            raise ValueError("H9 bootstrap requires both classes for every target")
        target_arrays.append(
            (
                labels,
                data["P"].to_numpy(dtype=float),
                data["B1"].to_numpy(dtype=float),
                data["B2"].to_numpy(dtype=float),
                locations,
            )
        )
    rng = np.random.default_rng(TERMINAL_BOOTSTRAP_SEED)
    p_minus_b1 = np.empty(replicates, dtype=np.float64)
    p_minus_b2 = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        b1_values: list[float] = []
        b2_values: list[float] = []
        for labels, p_score, b1_score, b2_score, locations in target_arrays:
            selected = np.concatenate([positions[rng.integers(0, len(positions), size=len(positions))] for positions in locations])
            b1_values.append(_eer(labels[selected], p_score[selected]) - _eer(labels[selected], b1_score[selected]))
            b2_values.append(_eer(labels[selected], p_score[selected]) - _eer(labels[selected], b2_score[selected]))
        p_minus_b1[replicate] = float(np.mean(b1_values))
        p_minus_b2[replicate] = float(np.mean(b2_values))
    return pd.DataFrame(
        {
            "replicate": np.arange(replicates, dtype=np.int64),
            "macro_eer_difference_p_minus_b1": p_minus_b1,
            "macro_eer_difference_p_minus_b2": p_minus_b2,
        }
    )


def decision_gate(metrics: pd.DataFrame, bootstrap: pd.DataFrame, *, fingerprint_audit_passed: bool, reconstruction_audit_passed: bool) -> dict[str, Any]:
    """Evaluate the literal locked H9-PCR success requirements without tuning."""
    required = {(dataset, method) for dataset in TARGET_REVISIONS for method in METHODS}
    observed = set(zip(metrics["dataset"].astype(str), metrics["method"].astype(str), strict=True))
    if observed != required or metrics.duplicated(["dataset", "method"]).any():
        raise ValueError("H9 decision gate requires exactly every target/method metric")
    if len(bootstrap) != TERMINAL_BOOTSTRAP_REPLICATES:
        raise ValueError("H9 decision gate requires the fixed 2,000 bootstrap replicates")
    by_target = metrics.pivot(index="dataset", columns="method", values="eer").reindex(index=list(TARGET_REVISIONS), columns=list(METHODS))
    macro = by_target.mean(axis=0)
    p_macro = float(macro["P"])
    reductions = {
        "b1": float(1.0 - p_macro / float(macro["B1"])) if float(macro["B1"]) > 0 else float("nan"),
        "b2": float(1.0 - p_macro / float(macro["B2"])) if float(macro["B2"]) > 0 else float("nan"),
    }
    ci_b1 = np.quantile(bootstrap["macro_eer_difference_p_minus_b1"].to_numpy(dtype=float), [0.025, 0.975])
    ci_b2 = np.quantile(bootstrap["macro_eer_difference_p_minus_b2"].to_numpy(dtype=float), [0.025, 0.975])
    rules = {
        "relative_macro_eer_reduction_vs_b1_at_least_10pct": bool(reductions["b1"] >= 0.10),
        "relative_macro_eer_reduction_vs_b2_at_least_10pct": bool(reductions["b2"] >= 0.10),
        "p_lower_eer_than_b1_on_both_targets": bool((by_target["P"] < by_target["B1"]).all()),
        "p_lower_eer_than_b2_on_both_targets": bool((by_target["P"] < by_target["B2"]).all()),
        "bootstrap_macro_eer_difference_p_minus_b1_ci_below_zero": bool(float(ci_b1[1]) < 0.0),
        "bootstrap_macro_eer_difference_p_minus_b2_ci_below_zero": bool(float(ci_b2[1]) < 0.0),
        "no_exact_source_target_canonical_fingerprint_collision": bool(fingerprint_audit_passed),
        "source_target_manifest_output_reconstruction_audit_passed": bool(reconstruction_audit_passed),
    }
    return {
        "artifact_kind": "h9_pcr_terminal_decision_gate",
        "version": H9_EVALUATION_VERSION,
        "primary_method": "P",
        "macro_eer": {method: float(macro[method]) for method in METHODS},
        "relative_macro_eer_reduction_p_vs_b1": reductions["b1"],
        "relative_macro_eer_reduction_p_vs_b2": reductions["b2"],
        "per_target_eer": {
            dataset: {method: float(by_target.loc[dataset, method]) for method in METHODS}
            for dataset in TARGET_REVISIONS
        },
        "bootstrap": {
            "replicates": TERMINAL_BOOTSTRAP_REPLICATES,
            "seed": TERMINAL_BOOTSTRAP_SEED,
            "resampling": "shared_sample_id_label_stratified_indices_across_P_B1_B2",
            "p_minus_b1": {
                "mean": float(bootstrap["macro_eer_difference_p_minus_b1"].mean()),
                "ci_low": float(ci_b1[0]),
                "ci_high": float(ci_b1[1]),
            },
            "p_minus_b2": {
                "mean": float(bootstrap["macro_eer_difference_p_minus_b2"].mean()),
                "ci_low": float(ci_b2[0]),
                "ci_high": float(ci_b2[1]),
            },
        },
        "rules": rules,
        "positive_result_gate_passed": bool(all(rules.values())),
    }


def _write_new_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"H9 terminal output directory already exists and will not be overwritten: {target}") from error
    return target


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def terminal_evaluate(
    *,
    sonar_manifest_path: str | Path,
    arad_manifest_path: str | Path,
    checkpoint_ledger_path: str | Path,
    plan_path: str | Path,
    data_contract_path: str | Path,
    output_dir: str | Path,
    device: str,
    terminal_evaluation: bool,
    synthetic_test_mode: bool = False,
    waveform_loader: Callable[[str], np.ndarray] = _load_audio_16k,
    model_factory: Callable[..., tuple[torch.nn.Module, Mapping[str, Any]]] = build_fresh_res2tcn_guard,
) -> dict[str, Path]:
    """Execute the one permitted H9 target-label join and all frozen analyses.

    ``terminal_evaluation`` is an intentional in-code interlock in addition to
    the CLI switch.  When false, this function returns before target manifests,
    target audio, and target labels are read.
    """
    if terminal_evaluation is not True:
        raise PermissionError("H9 target labels are sealed; pass --terminal-evaluation for the one terminal analysis")
    if not device.startswith("cuda:") and not synthetic_test_mode:
        raise ValueError("H9 production terminal evaluation requires a CUDA device")
    if not torch.cuda.is_available() and not synthetic_test_mode:
        raise RuntimeError("H9 production terminal evaluation requires CUDA")
    targets = {
        "SONAR": load_target_manifest(sonar_manifest_path, expected_dataset="SONAR"),
        "ArAD": load_target_manifest(arad_manifest_path, expected_dataset="ArAD"),
    }
    if targets["SONAR"].manifest_path == targets["ArAD"].manifest_path:
        raise ValueError("H9 SONAR and ArAD must use distinct canonical target manifests")
    ledger = load_frozen_checkpoint_ledger(checkpoint_ledger_path, plan_path=plan_path, data_contract_path=data_contract_path)
    source_fingerprints = _read_source_fingerprints(ledger.source_manifest_path, ledger.source_manifest_sha256)
    target_fingerprints = set().union(*(_fingerprint_set(target.records, description=f"{name} target records") for name, target in targets.items()))
    collisions = sorted(source_fingerprints.intersection(target_fingerprints))
    if collisions:
        raise ValueError(f"H9 exact source--target canonical fingerprint collision; terminal metrics are prohibited ({len(collisions)} collisions)")
    for target in targets.values():
        _verify_target_audio(target.records)
    output = _write_new_directory(Path(output_dir))
    raw_frames: list[pd.DataFrame] = []
    for target in targets.values():
        for method in METHODS:
            for seed in H9_SEEDS:
                raw_frames.append(
                    _predict_checkpoint(
                        target,
                        ledger.checkpoints[(method, seed)],
                        architecture_bundle=ledger.architecture_bundle,
                        device=device,
                        waveform_loader=waveform_loader,
                        model_factory=model_factory,
                    )
                )
    raw_predictions = pd.concat(raw_frames, ignore_index=True)
    expected_raw_rows = sum(len(target.records) for target in targets.values()) * len(METHODS) * len(H9_SEEDS)
    if len(raw_predictions) != expected_raw_rows or raw_predictions.duplicated(["dataset", "sample_id", "method", "seed"]).any():
        raise RuntimeError("H9 raw target prediction reconstruction is incomplete or duplicated")
    if not np.isfinite(raw_predictions[["spoof_logit", "spoof_probability"]].to_numpy(dtype=float)).all():
        raise FloatingPointError("H9 terminal predictions contain non-finite values")
    raw_path = output / "h9_terminal_raw_predictions.parquet"
    raw_predictions.to_parquet(raw_path, index=False)
    raw_sha256 = sha256_file(raw_path)
    # The raw predictions are now immutable and complete.  This is the first
    # permitted label load, and it necessarily evaluates both locked targets.
    labels = pd.concat([_read_target_labels(target) for target in targets.values()], ignore_index=True)
    if labels.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("H9 terminal labels have duplicate target identities")
    expected_label_rows = sum(len(target.records) for target in targets.values())
    if len(labels) != expected_label_rows:
        raise ValueError("H9 target labels do not cover the canonical target trial manifests")
    metrics, seed_metrics, labeled_predictions = _metrics_from_labels(raw_predictions, labels)
    bootstrap = bootstrap_macro_eer_differences(labeled_predictions)
    reconstruction_passed = bool(
        len(labeled_predictions) == expected_label_rows
        and not labeled_predictions.duplicated(["dataset", "sample_id"]).any()
        and set(labeled_predictions["dataset"].astype(str)) == set(TARGET_REVISIONS)
    )
    gate = decision_gate(metrics, bootstrap, fingerprint_audit_passed=True, reconstruction_audit_passed=reconstruction_passed)
    metrics_path = output / "h9_terminal_target_metrics.csv"
    seed_metrics_path = output / "h9_terminal_seed_metrics.csv"
    labels_path = output / "h9_terminal_labeled_predictions.parquet"
    bootstrap_path = output / "h9_terminal_bootstrap_macro_eer_differences.csv"
    gate_path = output / "h9_terminal_decision_gate.json"
    provenance_path = output / "h9_terminal_evaluation_provenance.json"
    metrics.to_csv(metrics_path, index=False)
    seed_metrics.to_csv(seed_metrics_path, index=False)
    labeled_predictions.to_parquet(labels_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    _atomic_write_json(gate_path, gate)
    provenance = {
        "artifact_kind": "h9_pcr_terminal_target_evaluation",
        "version": H9_EVALUATION_VERSION,
        "terminal_evaluation_switch": True,
        "target_labels_read": True,
        "target_metrics_read": True,
        "evaluation_device": device,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "evaluation_precision": "cuda_bfloat16_autocast",
        "target_manifests": {
            name: {
                "path": str(target.manifest_path),
                "sha256": target.manifest_sha256,
                "dataset_revision": target.revision,
                "records_path": str(target.records_path),
                "records_sha256": target.records_sha256,
                "labels_path": str(target.labels_path),
                "labels_sha256": target.labels_sha256,
                "n_trials": int(len(target.records)),
            }
            for name, target in targets.items()
        },
        "checkpoint_ledger_path": str(ledger.ledger_path),
        "checkpoint_ledger_sha256": ledger.ledger_sha256,
        "source_manifest_path": str(ledger.source_manifest_path),
        "source_manifest_sha256": ledger.source_manifest_sha256,
        "source_p_pairs_sha256": ledger.source_p_pairs_sha256,
        "source_b2_pairs_sha256": ledger.source_b2_pairs_sha256,
        "architecture_sha256": ledger.architecture_sha256,
        "source_target_canonical_fingerprint_collision_count": len(collisions),
        "raw_predictions_path": str(raw_path),
        "raw_predictions_sha256": raw_sha256,
        "target_metrics_sha256": sha256_file(metrics_path),
        "seed_metrics_sha256": sha256_file(seed_metrics_path),
        "labeled_predictions_sha256": sha256_file(labels_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "decision_gate_sha256": sha256_file(gate_path),
    }
    _atomic_write_json(provenance_path, provenance)
    return {
        "raw_predictions": raw_path,
        "metrics": metrics_path,
        "seed_metrics": seed_metrics_path,
        "labeled_predictions": labels_path,
        "bootstrap": bootstrap_path,
        "decision": gate_path,
        "provenance": provenance_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one sealed H9-PCR target evaluation.")
    parser.add_argument("--sonar-manifest", required=True, type=Path)
    parser.add_argument("--arad-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-ledger", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="A new HDD output directory; existing paths are refused.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for fixed terminal inference; this is not a selection flag.")
    parser.add_argument("--terminal-evaluation", action="store_true", help="Required acknowledgement before target labels may be read.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = terminal_evaluate(
        sonar_manifest_path=args.sonar_manifest,
        arad_manifest_path=args.arad_manifest,
        checkpoint_ledger_path=args.checkpoint_ledger,
        plan_path=args.plan,
        data_contract_path=args.data_contract,
        output_dir=args.output_dir,
        device=args.device,
        terminal_evaluation=bool(args.terminal_evaluation),
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via script wrapper
    raise SystemExit(cli_main())
