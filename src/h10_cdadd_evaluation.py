"""Terminal-only H10 CD-ADD evaluation for the sealed H9-PCR checkpoint panel.

The public entry point has exactly one target role and no knobs for model,
checkpoint, seed, source split, target subset, threshold, or bootstrap.  It
validates the source ledger and materialization provenance, writes a complete
label-free 12-checkpoint prediction table, and only then opens the separately
sealed labels once for metrics and the locked shared-ID bootstrap.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from src.h10_cdadd_target_materialize import (
    H10_CDADD_DATASET,
    H10_CDADD_REPOSITORY,
    H10_CDADD_REVISION,
    H10_CDADD_VERSION,
    H10_TARGET_MATERIALIZATION_VERSION,
    H9_FROZEN_CHECKPOINT_LEDGER_SHA256,
    H9_DATA_CONTRACT_PATH,
    H9_PLAN_PATH,
    TARGET_AUDIO_AUDIT_COLUMNS,
    TARGET_LABEL_COLUMNS,
    TARGET_RECORD_COLUMNS,
    _validate_h9_source_ledger,
)
from src.h9_odss_materialize import HDD_ROOT
from src.h9_pcr_evaluation import EVALUATION_BATCH_SIZE, FrozenCheckpoint, FrozenCheckpointLedger, METHODS
from src.h9_pcr_training import H9_SEEDS, _eer, _load_audio_16k, _spoof_logit, build_fresh_res2tcn_guard
from src.res2tcn_pytorch import sha256_file


TERMINAL_BOOTSTRAP_SEED = 2909
TERMINAL_BOOTSTRAP_REPLICATES = 2_000


class H10TargetManifest:
    """One label-fenced H10 CD-ADD target manifest and validated records."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        records_path: Path,
        records_sha256: str,
        records: pd.DataFrame,
        labels_path: Path,
        labels_sha256: str,
        materialization_provenance_path: Path,
        materialization_provenance_sha256: str,
    ) -> None:
        self.dataset = H10_CDADD_DATASET
        self.revision = H10_CDADD_REVISION
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.records_path = records_path
        self.records_sha256 = records_sha256
        self.records = records
        self.labels_path = labels_path
        self.labels_sha256 = labels_sha256
        self.materialization_provenance_path = materialization_provenance_path
        self.materialization_provenance_sha256 = materialization_provenance_sha256


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _read_json(path: str | Path, *, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H10 {description} is unavailable: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H10 {description} is not valid JSON: {resolved}") from error
    if not isinstance(value, dict):
        raise ValueError(f"H10 {description} must be a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, *, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"H10 {description} must be a lowercase SHA-256 digest")
    return str(value)


def _hash_matches(path: str | Path, expected: object, *, description: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H10 {description} is unavailable: {resolved}")
    expected_digest = _require_sha256(expected, description=f"{description} expected hash")
    observed = sha256_file(resolved)
    if observed != expected_digest:
        raise ValueError(f"H10 {description} SHA-256 mismatch")
    return observed


def _path_from(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"H10 {description} requires a nonempty path")
    return Path(value).expanduser().resolve()


def _read_table(path: str | Path, *, description: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H10 {description} is unavailable: {resolved}")
    if resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved, dtype=str, keep_default_na=False)
    elif resolved.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(resolved)
    else:
        raise ValueError(f"H10 {description} must be CSV or Parquet")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"H10 {description} did not decode to a table")
    return frame


def _nonempty_text(frame: pd.DataFrame, column: str, *, description: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise ValueError(f"H10 {description} requires nonempty {column} values")


def _fingerprint_set(frame: pd.DataFrame, *, description: str) -> set[str]:
    if "sample_id" not in frame.columns or "canonical_fingerprint" not in frame.columns:
        raise ValueError(f"H10 {description} lacks sample_id/canonical_fingerprint")
    _nonempty_text(frame, "sample_id", description=description)
    _nonempty_text(frame, "canonical_fingerprint", description=description)
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"H10 {description} has duplicate sample IDs")
    values = frame["canonical_fingerprint"].astype(str)
    if not values.map(_is_sha256).all():
        raise ValueError(f"H10 {description} has invalid canonical fingerprints")
    return set(values.tolist())


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_materialization_provenance(
    provenance_path: Path,
    provenance_sha256: str,
    *,
    records_path: Path,
    records_sha256: str,
    labels_path: Path,
    labels_sha256: str,
    n_records: int,
) -> None:
    """Replay the target's shard, card, and output integrity binding."""
    _hash_matches(provenance_path, provenance_sha256, description="CD-ADD materialization provenance")
    provenance = _read_json(provenance_path, description="CD-ADD materialization provenance")
    if provenance.get("artifact_kind") != "h10_cdadd_terminal_target_materialization":
        raise ValueError("H10 CD-ADD materialization provenance artifact kind drift")
    if provenance.get("version") != H10_TARGET_MATERIALIZATION_VERSION:
        raise ValueError("H10 CD-ADD materialization provenance version drift")
    dataset = provenance.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("H10 CD-ADD materialization provenance lacks dataset binding")
    required_dataset = {
        "name": H10_CDADD_DATASET,
        "repository": H10_CDADD_REPOSITORY,
        "revision": H10_CDADD_REVISION,
    }
    if any(dataset.get(key) != value for key, value in required_dataset.items()):
        raise ValueError("H10 CD-ADD materialization dataset binding drift")
    source = provenance.get("source_handoff")
    if not isinstance(source, Mapping):
        raise ValueError("H10 CD-ADD materialization provenance lacks H9 source handoff")
    if source.get("checkpoint_ledger_sha256") != H9_FROZEN_CHECKPOINT_LEDGER_SHA256:
        raise ValueError("H10 CD-ADD materialization does not bind the exact H9 ledger")
    if source.get("required_h9_checkpoint_ledger_sha256") != H9_FROZEN_CHECKPOINT_LEDGER_SHA256:
        raise ValueError("H10 CD-ADD materialization has a changed required H9 ledger hash")
    if source.get("validated_before_target_access") is not True:
        raise ValueError("H10 CD-ADD materialization lacks pre-target H9-ledger validation")
    metadata = provenance.get("metadata_audit")
    if not isinstance(metadata, Mapping):
        raise ValueError("H10 CD-ADD materialization lacks card/lineage audit binding")
    audit_path = _path_from(metadata.get("path"), description="CD-ADD card/lineage audit")
    _hash_matches(audit_path, metadata.get("sha256"), description="CD-ADD card/lineage audit")
    _require_sha256(metadata.get("card_sha256"), description="CD-ADD card SHA-256")
    boundary = provenance.get("access_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("H10 CD-ADD materialization lacks access-boundary provenance")
    if boundary.get("record_phase_opened_columns") != ["path", "audio"]:
        raise ValueError("H10 CD-ADD record-phase column firewall drift")
    if boundary.get("record_phase_raw_label_values_read") is not False:
        raise ValueError("H10 CD-ADD record phase read label values")
    if boundary.get("label_artifact_phase_opened_columns") != ["path", "label"]:
        raise ValueError("H10 CD-ADD label-phase column firewall drift")
    if boundary.get("label_artifact_phase_raw_label_values_read") is not True:
        raise ValueError("H10 CD-ADD provenance does not truthfully record its label-artifact pass")
    outputs = provenance.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("H10 CD-ADD materialization lacks output bindings")
    for name, expected_path, expected_hash, expected_rows in (
        ("records_csv", records_path, records_sha256, n_records),
        ("labels_csv", labels_path, labels_sha256, n_records),
    ):
        value = outputs.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"H10 CD-ADD materialization lacks {name}")
        if _path_from(value.get("path"), description=f"materialization {name}") != expected_path:
            raise ValueError(f"H10 CD-ADD materialization {name} path drift")
        if value.get("sha256") != expected_hash or value.get("n_rows") != expected_rows:
            raise ValueError(f"H10 CD-ADD materialization {name} hash/count drift")
    audio_audit = outputs.get("audio_audit_csv")
    if not isinstance(audio_audit, Mapping):
        raise ValueError("H10 CD-ADD materialization lacks audio audit")
    audio_audit_path = _path_from(audio_audit.get("path"), description="materialization audio audit")
    _hash_matches(audio_audit_path, audio_audit.get("sha256"), description="materialization audio audit")
    table = _read_table(audio_audit_path, description="materialization audio audit")
    if tuple(table.columns.astype(str)) != TARGET_AUDIO_AUDIT_COLUMNS or len(table) != n_records:
        raise ValueError("H10 CD-ADD audio audit schema/count drift")
    shard_audits = provenance.get("raw_target_shards")
    if not isinstance(shard_audits, list) or not shard_audits:
        raise ValueError("H10 CD-ADD materialization lacks raw shard audits")
    audited_rows = 0
    for shard_record in shard_audits:
        if not isinstance(shard_record, Mapping):
            raise ValueError("H10 CD-ADD raw shard audit must be an object")
        shard = _path_from(shard_record.get("path"), description="raw target shard")
        _hash_matches(shard, shard_record.get("sha256"), description="raw target shard")
        if shard.stat().st_size != shard_record.get("bytes"):
            raise ValueError("H10 CD-ADD raw target shard byte count drift")
        parquet = pq.ParquetFile(shard)
        schema = parquet.schema_arrow
        if _schema_sha256(schema) != shard_record.get("arrow_schema_sha256"):
            raise ValueError("H10 CD-ADD raw target shard Arrow schema drift")
        if int(parquet.metadata.num_rows) != shard_record.get("rows"):
            raise ValueError("H10 CD-ADD raw target shard row count drift")
        if list(schema.names) != shard_record.get("columns"):
            raise ValueError("H10 CD-ADD raw target shard column list drift")
        if not {"path", "audio", "label"}.issubset(set(schema.names)):
            raise ValueError("H10 CD-ADD raw target shard no longer has path/audio/label")
        audited_rows += int(parquet.metadata.num_rows)
    if audited_rows != n_records:
        raise ValueError("H10 CD-ADD raw shard audits do not reconstruct canonical trial count")


def load_h10_cdadd_target_manifest(path: str | Path) -> H10TargetManifest:
    """Validate metadata, records, and materialization bindings without labels."""
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path, description="CD-ADD target manifest")
    if manifest.get("artifact_kind") != "h10_cdadd_canonical_target_manifest" or manifest.get("version") != H10_CDADD_VERSION:
        raise ValueError("H10 CD-ADD target manifest contract drift")
    if manifest.get("dataset") != H10_CDADD_DATASET or manifest.get("dataset_repository") != H10_CDADD_REPOSITORY:
        raise ValueError("H10 CD-ADD target manifest dataset binding drift")
    if manifest.get("dataset_revision") != H10_CDADD_REVISION:
        raise ValueError("H10 CD-ADD target manifest revision drift")
    if manifest.get("target_labels_read") is not False or manifest.get("target_metrics_read") is not False:
        raise ValueError("H10 CD-ADD manifest was not created under the label firewall")
    records_path = _path_from(manifest.get("records_path"), description="CD-ADD target records")
    records_sha256 = _hash_matches(records_path, manifest.get("records_sha256"), description="CD-ADD target records")
    records = _read_table(records_path, description="CD-ADD target records")
    if tuple(records.columns.astype(str)) != TARGET_RECORD_COLUMNS or records.empty:
        raise ValueError("H10 CD-ADD target records schema drift or empty target")
    for column in TARGET_RECORD_COLUMNS:
        _nonempty_text(records, column, description="CD-ADD target records")
    if records["sample_id"].astype(str).duplicated().any():
        raise ValueError("H10 CD-ADD target records have duplicate sample IDs")
    if not records["audio_sha256"].astype(str).map(_is_sha256).all():
        raise ValueError("H10 CD-ADD target records have invalid audio SHA-256")
    _fingerprint_set(records, description="CD-ADD target records")
    try:
        byte_counts = records["audio_bytes"].astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("H10 CD-ADD target records audio_bytes must be integers") from error
    if (byte_counts <= 0).any():
        raise ValueError("H10 CD-ADD target records audio_bytes must be positive")
    labels = manifest.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError("H10 CD-ADD target manifest lacks a separate label artifact")
    labels_path = _path_from(labels.get("path"), description="CD-ADD target label artifact")
    labels_sha256 = _hash_matches(labels_path, labels.get("sha256"), description="CD-ADD target label artifact")
    materialization = manifest.get("materialization_provenance")
    if not isinstance(materialization, Mapping):
        raise ValueError("H10 CD-ADD target manifest lacks materialization provenance")
    materialization_path = _path_from(materialization.get("path"), description="CD-ADD materialization provenance")
    materialization_sha256 = _require_sha256(materialization.get("sha256"), description="CD-ADD materialization provenance hash")
    _validate_materialization_provenance(
        materialization_path,
        materialization_sha256,
        records_path=records_path,
        records_sha256=records_sha256,
        labels_path=labels_path,
        labels_sha256=labels_sha256,
        n_records=len(records),
    )
    return H10TargetManifest(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        records_path=records_path,
        records_sha256=records_sha256,
        records=records,
        labels_path=labels_path,
        labels_sha256=labels_sha256,
        materialization_provenance_path=materialization_path,
        materialization_provenance_sha256=materialization_sha256,
    )


def _read_source_fingerprints(ledger: FrozenCheckpointLedger) -> set[str]:
    _hash_matches(ledger.source_manifest_path, ledger.source_manifest_sha256, description="H9 source materialized manifest")
    source = _read_table(ledger.source_manifest_path, description="H9 source materialized manifest")
    return _fingerprint_set(source, description="H9 source materialized manifest")


def _verify_target_audio(records: pd.DataFrame) -> None:
    for row in records.itertuples(index=False):
        audio_path = Path(str(row.audio_path)).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"H10 CD-ADD copied waveform is unavailable: {audio_path}")
        if audio_path.stat().st_size != int(row.audio_bytes):
            raise ValueError(f"H10 CD-ADD copied waveform byte count mismatch: {audio_path}")
        if sha256_file(audio_path) != str(row.audio_sha256):
            raise ValueError(f"H10 CD-ADD copied waveform SHA-256 mismatch: {audio_path}")


def _predict_checkpoint(
    target: H10TargetManifest,
    checkpoint: FrozenCheckpoint,
    *,
    architecture_bundle: Path,
    device: str,
    waveform_loader: Callable[[str], np.ndarray],
    model_factory: Callable[..., tuple[torch.nn.Module, Mapping[str, Any]]],
) -> pd.DataFrame:
    model, provenance = model_factory(architecture_bundle, seed=checkpoint.seed, device=device)
    if provenance.get("checkpoint_loaded") is not False or provenance.get("initialization") != "fresh_seeded":
        raise ValueError("H10 requires a fresh architecture construction before frozen checkpoint load")
    model.load_state_dict(checkpoint.state_dict, strict=True)
    model.eval()
    sample_ids: list[str] = []
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    device_type = torch.device(device).type
    autocast: contextlib.AbstractContextManager[Any]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else contextlib.nullcontext()
    with torch.no_grad():
        for start in range(0, len(target.records), EVALUATION_BATCH_SIZE):
            chunk = target.records.iloc[start : start + EVALUATION_BATCH_SIZE]
            waveforms = [np.asarray(waveform_loader(str(value)), dtype=np.float32) for value in chunk["audio_path"].tolist()]
            tensor = torch.from_numpy(np.ascontiguousarray(np.stack(waveforms, axis=0))).to(device)
            with autocast:
                model_output = model(tensor)
                spoof_logits = _spoof_logit(model_output)
                classifier_logits = model_output[1] if isinstance(model_output, tuple) else model_output
                spoof_probability = torch.softmax(classifier_logits, dim=1)[:, 1]
            sample_ids.extend(chunk["sample_id"].astype(str).tolist())
            logits.append(spoof_logits.float().detach().cpu().numpy())
            probabilities.append(spoof_probability.float().detach().cpu().numpy())
    return pd.DataFrame(
        {
            "dataset": H10_CDADD_DATASET,
            "sample_id": sample_ids,
            "method": checkpoint.method,
            "seed": checkpoint.seed,
            "checkpoint_sha256": checkpoint.sha256,
            "spoof_logit": np.concatenate(logits),
            "spoof_probability": np.concatenate(probabilities),
        }
    )


def _read_target_labels(target: H10TargetManifest) -> pd.DataFrame:
    if sha256_file(target.labels_path) != target.labels_sha256:
        raise ValueError("H10 CD-ADD label artifact changed before its terminal load")
    labels = _read_table(target.labels_path, description="CD-ADD target labels")
    if tuple(labels.columns.astype(str)) != TARGET_LABEL_COLUMNS:
        raise ValueError("H10 CD-ADD labels schema must be exactly sample_id,label")
    _nonempty_text(labels, "sample_id", description="CD-ADD target labels")
    if labels["sample_id"].astype(str).duplicated().any():
        raise ValueError("H10 CD-ADD labels have duplicate sample IDs")
    try:
        numeric = labels["label"].astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("H10 CD-ADD labels must be binary integers") from error
    if set(numeric.unique()) != {0, 1}:
        raise ValueError("H10 CD-ADD labels require both binary classes")
    return pd.DataFrame({"dataset": H10_CDADD_DATASET, "sample_id": labels["sample_id"].astype(str), "label": numeric})


def _metrics_from_labels(raw_predictions: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = raw_predictions.groupby(["sample_id", "method"], sort=False)["seed"].nunique()
    if not counts.eq(len(H9_SEEDS)).all():
        raise ValueError("H10 raw predictions do not contain exactly four seeds per trial/method")
    aggregate = (
        raw_predictions.groupby(["dataset", "sample_id", "method"], as_index=False, sort=False)["spoof_probability"]
        .mean()
        .rename(columns={"spoof_probability": "mean_spoof_probability"})
    )
    aggregate_join = aggregate.merge(labels, on=["dataset", "sample_id"], how="inner", validate="many_to_one")
    seed_join = raw_predictions.merge(labels, on=["dataset", "sample_id"], how="inner", validate="many_to_one")
    if len(aggregate_join) != len(aggregate) or len(seed_join) != len(raw_predictions):
        raise ValueError("H10 CD-ADD prediction/label coverage mismatch")
    metrics: list[dict[str, object]] = []
    seed_metrics: list[dict[str, object]] = []
    for method in METHODS:
        values = aggregate_join.loc[aggregate_join["method"] == method]
        label = values["label"].to_numpy(dtype=int)
        score = values["mean_spoof_probability"].to_numpy(dtype=float)
        metrics.append(
            {
                "dataset": H10_CDADD_DATASET,
                "method": method,
                "n_trials": int(len(values)),
                "n_bonafide": int((label == 0).sum()),
                "n_spoof": int((label == 1).sum()),
                "eer": _eer(label, score),
                "eer_percent": 100.0 * _eer(label, score),
                "auroc": float(roc_auc_score(label, score)),
                "score_aggregation": "mean_spoof_probability_over_four_frozen_h9_seeds",
            }
        )
        for seed in H9_SEEDS:
            seed_values = seed_join.loc[(seed_join["method"] == method) & (seed_join["seed"] == seed)]
            seed_label = seed_values["label"].to_numpy(dtype=int)
            seed_score = seed_values["spoof_probability"].to_numpy(dtype=float)
            seed_metrics.append(
                {
                    "dataset": H10_CDADD_DATASET,
                    "method": method,
                    "seed": seed,
                    "n_trials": int(len(seed_values)),
                    "eer": _eer(seed_label, seed_score),
                    "eer_percent": 100.0 * _eer(seed_label, seed_score),
                    "auroc": float(roc_auc_score(seed_label, seed_score)),
                }
            )
    metrics_frame = pd.DataFrame(metrics)
    seed_metrics_frame = pd.DataFrame(seed_metrics)
    if len(metrics_frame) != len(METHODS) or len(seed_metrics_frame) != len(METHODS) * len(H9_SEEDS):
        raise RuntimeError("H10 CD-ADD terminal metrics are incomplete")
    wide = aggregate_join.pivot(index=["dataset", "sample_id", "label"], columns="method", values="mean_spoof_probability").reset_index()
    if tuple(wide.columns) != ("dataset", "sample_id", "label", *METHODS):
        raise ValueError("H10 CD-ADD labeled prediction reconstruction is incomplete")
    return metrics_frame, seed_metrics_frame, wide


def bootstrap_eer_differences(labeled_predictions: pd.DataFrame, *, replicates: int = TERMINAL_BOOTSTRAP_REPLICATES) -> pd.DataFrame:
    """Fixed shared sample-ID, label-stratified bootstrap for P-B1 and P-B2."""
    if replicates != TERMINAL_BOOTSTRAP_REPLICATES:
        raise ValueError("H10 production terminal bootstrap requires exactly 2,000 replicates")
    if set(labeled_predictions["dataset"].astype(str)) != {H10_CDADD_DATASET}:
        raise ValueError("H10 bootstrap requires exactly the pinned CD-ADD panel")
    if labeled_predictions["sample_id"].astype(str).duplicated().any():
        raise ValueError("H10 bootstrap requires unique CD-ADD sample IDs")
    labels = labeled_predictions["label"].to_numpy(dtype=int)
    locations = tuple(np.flatnonzero(labels == value) for value in (0, 1))
    if any(len(values) == 0 for values in locations):
        raise ValueError("H10 bootstrap requires both binary classes")
    p_score = labeled_predictions["P"].to_numpy(dtype=float)
    b1_score = labeled_predictions["B1"].to_numpy(dtype=float)
    b2_score = labeled_predictions["B2"].to_numpy(dtype=float)
    rng = np.random.default_rng(TERMINAL_BOOTSTRAP_SEED)
    p_minus_b1 = np.empty(replicates, dtype=np.float64)
    p_minus_b2 = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = np.concatenate([positions[rng.integers(0, len(positions), size=len(positions))] for positions in locations])
        p_minus_b1[replicate] = _eer(labels[selected], p_score[selected]) - _eer(labels[selected], b1_score[selected])
        p_minus_b2[replicate] = _eer(labels[selected], p_score[selected]) - _eer(labels[selected], b2_score[selected])
    return pd.DataFrame(
        {
            "replicate": np.arange(replicates, dtype=np.int64),
            "eer_difference_p_minus_b1": p_minus_b1,
            "eer_difference_p_minus_b2": p_minus_b2,
        }
    )


def decision_gate(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    source_ledger_replay_passed: bool,
    target_materialization_audit_passed: bool,
    collision_audit_passed: bool,
    raw_prediction_before_labels_passed: bool,
    reconstruction_audit_passed: bool,
) -> dict[str, Any]:
    """Apply the literal H10 single-target decision rule without a rescue path."""
    required = {(H10_CDADD_DATASET, method) for method in METHODS}
    observed = set(zip(metrics["dataset"].astype(str), metrics["method"].astype(str), strict=True))
    if observed != required or metrics.duplicated(["dataset", "method"]).any():
        raise ValueError("H10 decision gate requires exactly one CD-ADD metric per method")
    if len(bootstrap) != TERMINAL_BOOTSTRAP_REPLICATES:
        raise ValueError("H10 decision gate requires exactly 2,000 bootstrap replicates")
    by_method = metrics.set_index("method")["eer"].reindex(METHODS)
    reductions = {
        "b1": float(1.0 - float(by_method["P"]) / float(by_method["B1"])) if float(by_method["B1"]) > 0 else float("nan"),
        "b2": float(1.0 - float(by_method["P"]) / float(by_method["B2"])) if float(by_method["B2"]) > 0 else float("nan"),
    }
    ci_b1 = np.quantile(bootstrap["eer_difference_p_minus_b1"].to_numpy(dtype=float), [0.025, 0.975])
    ci_b2 = np.quantile(bootstrap["eer_difference_p_minus_b2"].to_numpy(dtype=float), [0.025, 0.975])
    rules = {
        "relative_eer_reduction_vs_b1_at_least_10pct": bool(reductions["b1"] >= 0.10),
        "relative_eer_reduction_vs_b2_at_least_10pct": bool(reductions["b2"] >= 0.10),
        "bootstrap_eer_difference_p_minus_b1_ci_below_zero": bool(float(ci_b1[1]) < 0.0),
        "bootstrap_eer_difference_p_minus_b2_ci_below_zero": bool(float(ci_b2[1]) < 0.0),
        "h9_source_ledger_replay_passed": bool(source_ledger_replay_passed),
        "target_input_reconstruction_audit_passed": bool(target_materialization_audit_passed),
        "no_exact_source_target_canonical_fingerprint_collision": bool(collision_audit_passed),
        "complete_label_free_prediction_matrix_written_before_labels": bool(raw_prediction_before_labels_passed),
        "target_manifest_output_reconstruction_audit_passed": bool(reconstruction_audit_passed),
    }
    return {
        "artifact_kind": "h10_cdadd_terminal_decision_gate",
        "version": H10_CDADD_VERSION,
        "primary_method": "P",
        "eer": {method: float(by_method[method]) for method in METHODS},
        "relative_eer_reduction_p_vs_b1": reductions["b1"],
        "relative_eer_reduction_p_vs_b2": reductions["b2"],
        "bootstrap": {
            "replicates": TERMINAL_BOOTSTRAP_REPLICATES,
            "seed": TERMINAL_BOOTSTRAP_SEED,
            "resampling": "shared_sample_id_label_stratified_indices_across_P_B1_B2",
            "p_minus_b1": {"mean": float(bootstrap["eer_difference_p_minus_b1"].mean()), "ci_low": float(ci_b1[0]), "ci_high": float(ci_b1[1])},
            "p_minus_b2": {"mean": float(bootstrap["eer_difference_p_minus_b2"].mean()), "ci_low": float(ci_b2[0]), "ci_high": float(ci_b2[1])},
        },
        "rules": rules,
        "positive_result_gate_passed": bool(all(rules.values())),
    }


def _write_new_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"H10 CD-ADD terminal output exists and will not be overwritten: {output}") from error
    return output


def _require_hdd_output(path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    root = Path(HDD_ROOT).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError(f"H10 CD-ADD production terminal output must be below {root}: {output}") from error


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", mode="w", encoding="utf-8", delete=False) as handle:
        handle.write(_canonical_json(value) + "\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def terminal_evaluate_h10_cdadd(
    *,
    target_manifest_path: str | Path,
    checkpoint_ledger_path: str | Path,
    output_dir: str | Path,
    device: str,
    terminal_evaluation: bool,
    synthetic_test_mode: bool = False,
    waveform_loader: Callable[[str], np.ndarray] = _load_audio_16k,
    model_factory: Callable[..., tuple[torch.nn.Module, Mapping[str, Any]]] = build_fresh_res2tcn_guard,
    _source_ledger_loader: Callable[[str | Path], object] = _validate_h9_source_ledger,
    _prediction_runner: Callable[[H10TargetManifest, FrozenCheckpoint], pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Run the single H10 CD-ADD terminal label join and fixed analyses once."""
    if terminal_evaluation is not True:
        raise PermissionError("H10 CD-ADD labels are sealed; pass --terminal-evaluation for the one terminal analysis")
    if not device.startswith("cuda:") and not synthetic_test_mode:
        raise ValueError("H10 CD-ADD production terminal evaluation requires a CUDA device")
    if not torch.cuda.is_available() and not synthetic_test_mode:
        raise RuntimeError("H10 CD-ADD production terminal evaluation requires CUDA")
    if _prediction_runner is not None and not synthetic_test_mode:
        raise ValueError("H10 CD-ADD production evaluation does not permit a synthetic prediction runner")
    if not synthetic_test_mode:
        _require_hdd_output(output_dir)
    # Replay source first, before any target manifest/record/label is opened.
    ledger = _source_ledger_loader(checkpoint_ledger_path)
    if not isinstance(ledger, FrozenCheckpointLedger):
        if not synthetic_test_mode:
            raise ValueError("H10 source-ledger validator did not return the H9 frozen ledger type")
    if sha256_file(Path(checkpoint_ledger_path).expanduser().resolve()) != H9_FROZEN_CHECKPOINT_LEDGER_SHA256 and not synthetic_test_mode:
        raise ValueError("H10 terminal evaluator requires the exact H9 frozen checkpoint ledger")
    target = load_h10_cdadd_target_manifest(target_manifest_path)
    if not hasattr(ledger, "source_manifest_path") or not hasattr(ledger, "checkpoints"):
        raise ValueError("H10 synthetic source-ledger seam lacks H9 source/checkpoint fields")
    source_fingerprints = _read_source_fingerprints(ledger)  # type: ignore[arg-type]
    collisions = sorted(source_fingerprints.intersection(_fingerprint_set(target.records, description="CD-ADD target records")))
    if collisions:
        raise ValueError(f"H10 exact H9-source/CD-ADD canonical fingerprint collision ({len(collisions)} collisions)")
    _verify_target_audio(target.records)
    expected_keys = {(method, seed) for method in METHODS for seed in H9_SEEDS}
    if set(ledger.checkpoints) != expected_keys:  # type: ignore[union-attr]
        raise ValueError("H10 source ledger no longer contains the complete B1/B2/P × four-seed matrix")
    output = _write_new_directory(output_dir)
    raw_frames: list[pd.DataFrame] = []
    for method in METHODS:
        for seed in H9_SEEDS:
            checkpoint = ledger.checkpoints[(method, seed)]  # type: ignore[union-attr]
            if _prediction_runner is not None:
                frame = _prediction_runner(target, checkpoint)
            else:
                frame = _predict_checkpoint(
                    target,
                    checkpoint,
                    architecture_bundle=ledger.architecture_bundle,  # type: ignore[union-attr]
                    device=device,
                    waveform_loader=waveform_loader,
                    model_factory=model_factory,
                )
            raw_frames.append(frame)
    raw_predictions = pd.concat(raw_frames, ignore_index=True)
    expected_raw_rows = len(target.records) * len(METHODS) * len(H9_SEEDS)
    expected_columns = ("dataset", "sample_id", "method", "seed", "checkpoint_sha256", "spoof_logit", "spoof_probability")
    if tuple(raw_predictions.columns.astype(str)) != expected_columns:
        raise ValueError("H10 raw predictions schema drift")
    if len(raw_predictions) != expected_raw_rows or raw_predictions.duplicated(["dataset", "sample_id", "method", "seed"]).any():
        raise RuntimeError("H10 raw prediction matrix is incomplete or duplicated")
    if set(raw_predictions["dataset"].astype(str)) != {H10_CDADD_DATASET}:
        raise ValueError("H10 raw prediction dataset drift")
    if not np.isfinite(raw_predictions[["spoof_logit", "spoof_probability"]].to_numpy(dtype=float)).all():
        raise FloatingPointError("H10 raw predictions contain non-finite values")
    raw_path = output / "h10_cdadd_raw_predictions.parquet"
    raw_predictions.to_parquet(raw_path, index=False)
    raw_sha256 = sha256_file(raw_path)
    # This is intentionally the first terminal label load: the complete,
    # immutable raw matrix above is already on disk and hash-bound.
    labels = _read_target_labels(target)
    if len(labels) != len(target.records):
        raise ValueError("H10 CD-ADD labels do not cover the canonical record panel")
    metrics, seed_metrics, labeled_predictions = _metrics_from_labels(raw_predictions, labels)
    bootstrap = bootstrap_eer_differences(labeled_predictions)
    reconstruction_passed = bool(
        len(labeled_predictions) == len(target.records)
        and not labeled_predictions.duplicated(["dataset", "sample_id"]).any()
        and set(labeled_predictions["dataset"].astype(str)) == {H10_CDADD_DATASET}
    )
    gate = decision_gate(
        metrics,
        bootstrap,
        source_ledger_replay_passed=True,
        target_materialization_audit_passed=True,
        collision_audit_passed=True,
        raw_prediction_before_labels_passed=True,
        reconstruction_audit_passed=reconstruction_passed,
    )
    metrics_path = output / "h10_cdadd_target_metrics.csv"
    seed_metrics_path = output / "h10_cdadd_seed_metrics.csv"
    labeled_path = output / "h10_cdadd_labeled_predictions.parquet"
    bootstrap_path = output / "h10_cdadd_bootstrap_eer_differences.csv"
    decision_path = output / "h10_cdadd_decision_gate.json"
    provenance_path = output / "h10_cdadd_terminal_evaluation_provenance.json"
    metrics.to_csv(metrics_path, index=False)
    seed_metrics.to_csv(seed_metrics_path, index=False)
    labeled_predictions.to_parquet(labeled_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    _atomic_write_json(decision_path, gate)
    provenance = {
        "artifact_kind": "h10_cdadd_terminal_target_evaluation",
        "version": H10_CDADD_VERSION,
        "terminal_evaluation_switch": True,
        "target_labels_read": True,
        "target_metrics_read": True,
        "evaluation_device": device,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "evaluation_precision": "cuda_bfloat16_autocast",
        "target_manifest": {
            "path": str(target.manifest_path),
            "sha256": target.manifest_sha256,
            "dataset_revision": target.revision,
            "records_path": str(target.records_path),
            "records_sha256": target.records_sha256,
            "labels_path": str(target.labels_path),
            "labels_sha256": target.labels_sha256,
            "materialization_provenance_path": str(target.materialization_provenance_path),
            "materialization_provenance_sha256": target.materialization_provenance_sha256,
            "n_trials": int(len(target.records)),
        },
        "checkpoint_ledger_path": str(ledger.ledger_path),  # type: ignore[union-attr]
        "checkpoint_ledger_sha256": sha256_file(Path(checkpoint_ledger_path).expanduser().resolve()),
        "required_h9_checkpoint_ledger_sha256": H9_FROZEN_CHECKPOINT_LEDGER_SHA256,
        "source_manifest_path": str(ledger.source_manifest_path),  # type: ignore[union-attr]
        "source_manifest_sha256": ledger.source_manifest_sha256,  # type: ignore[union-attr]
        "source_target_canonical_fingerprint_collision_count": len(collisions),
        "raw_predictions_path": str(raw_path),
        "raw_predictions_sha256": raw_sha256,
        "target_metrics_sha256": sha256_file(metrics_path),
        "seed_metrics_sha256": sha256_file(seed_metrics_path),
        "labeled_predictions_sha256": sha256_file(labeled_path),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "decision_gate_sha256": sha256_file(decision_path),
    }
    _atomic_write_json(provenance_path, provenance)
    return {
        "raw_predictions": raw_path,
        "metrics": metrics_path,
        "seed_metrics": seed_metrics_path,
        "labeled_predictions": labeled_path,
        "bootstrap": bootstrap_path,
        "decision": decision_path,
        "provenance": provenance_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one sealed H10 CD-ADD terminal evaluation.")
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New HDD output directory; existing output is refused.")
    parser.add_argument("--device", default="cuda:0", help="Fixed CUDA device for BF16 inference; not a selection flag.")
    parser.add_argument("--terminal-evaluation", action="store_true", help="Required acknowledgement before the sealed label join.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = terminal_evaluate_h10_cdadd(
        target_manifest_path=args.target_manifest,
        checkpoint_ledger_path=args.checkpoint_ledger,
        output_dir=args.output_dir,
        device=args.device,
        terminal_evaluation=bool(args.terminal_evaluation),
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
