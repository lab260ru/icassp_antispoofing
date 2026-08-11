"""Fail-closed terminal-target materialization for the locked H9-PCR study.

This module deliberately does not train a model, calculate a metric, or make
a prediction.  It runs only after the immutable source checkpoint ledger has
been validated, and produces the *separate* score-free record and label
artifacts consumed by :mod:`src.h9_pcr_evaluation`.

The implementation has two explicit passes over the pinned raw Parquet
shards.  The first opens only ``path`` and ``audio`` to copy and fingerprint
waveforms, then writes the label-free record table.  The second opens only
``path`` and ``label`` to publish the separate label artifact.  No label value
is available to the record/prediction-manifest phase, and neither phase has a
model or metric dependency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.h9_odss_materialize import CANONICAL_AUDIO_POLICY, HDD_ROOT, canonical_mono16k_pcm_fingerprint
from src.h9_pcr_evaluation import H9_EVALUATION_VERSION, TARGET_REVISIONS, load_frozen_checkpoint_ledger
from src.res2tcn_pytorch import sha256_file


H9_TARGET_MATERIALIZATION_VERSION = "h9_pcr_terminal_target_materialization_v1"
TARGET_RECORD_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "audio_path",
    "audio_sha256",
    "audio_bytes",
    "canonical_fingerprint",
)
TARGET_LABEL_COLUMNS: tuple[str, ...] = ("sample_id", "label")
TARGET_AUDIO_AUDIT_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "raw_relative_path",
    "raw_shard",
    "raw_audio_sha256",
    "raw_audio_bytes",
    "copied_audio_path",
    "copied_audio_sha256",
    "copied_audio_bytes",
    "canonical_fingerprint",
    "canonical_frames",
    "canonical_pcm_bytes",
)


@dataclass(frozen=True)
class TargetMaterialization:
    """Paths to one newly published H9 target materialization."""

    dataset: str
    output_dir: Path
    manifest: Path
    records: Path
    labels: Path
    audio_audit: Path
    provenance: Path
    waveforms_dir: Path


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _path_below(path: Path, root: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as error:
        raise ValueError(f"H9 {description} must be below {root.expanduser().resolve()}: {resolved}") from error
    return resolved


def _raw_target_shards(raw_shard_dir: str | Path, *, dataset: str) -> list[Path]:
    """Accept only the locked target's ``<revision>/raw/data`` directory."""
    if dataset not in TARGET_REVISIONS:
        raise ValueError(f"H9 target dataset must be one of {tuple(TARGET_REVISIONS)}, got {dataset!r}")
    directory = Path(raw_shard_dir).expanduser().resolve()
    revision = TARGET_REVISIONS[dataset]
    if (
        not directory.is_dir()
        or directory.name != "data"
        or directory.parent.name != "raw"
        or directory.parent.parent.name != revision
    ):
        raise ValueError(
            "H9 target raw-shard input must be the pinned target revision's "
            f"<revision>/raw/data directory for {dataset}"
        )
    # Hugging Face dataset snapshots place a separate ``labels.parquet`` next
    # to the audio-bearing dataset shards.  It is deliberately *not* an input
    # to this adapter: each audio shard must carry its own path/audio/label
    # columns, while the standalone label index is outside the terminal
    # materialization contract.  Excluding only this exact conventional file
    # keeps any unexpected Parquet file fail-closed in schema validation.
    shards = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".parquet" and path.name != "labels.parquet"
    )
    if not shards:
        raise FileNotFoundError(f"H9 target raw-shard directory has no Parquet files: {directory}")
    return shards


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_target_shard_schema(shard: Path) -> dict[str, Any]:
    """Validate the raw source shape before either audio or labels are read."""
    parquet = pq.ParquetFile(shard)
    schema = parquet.schema_arrow
    required = {"path", "audio", "label"}
    if not required.issubset(set(schema.names)):
        raise ValueError(f"H9 target shard lacks required path/audio/label fields: {shard}")
    path_type = schema.field("path").type
    if not (pa.types.is_string(path_type) or pa.types.is_large_string(path_type)):
        raise ValueError(f"H9 target shard path field must be string: {shard}")
    audio_type = schema.field("audio").type
    if not pa.types.is_struct(audio_type) or "bytes" not in audio_type.names:
        raise ValueError(f"H9 target shard audio field must be a struct with binary bytes: {shard}")
    payload_type = audio_type.field("bytes").type
    if not (pa.types.is_binary(payload_type) or pa.types.is_large_binary(payload_type)):
        raise ValueError(f"H9 target shard audio.bytes field must be binary: {shard}")
    if not pa.types.is_integer(schema.field("label").type):
        raise ValueError(f"H9 target shard label field must be integer: {shard}")
    return {
        "path": str(shard),
        "sha256": sha256_file(shard),
        "bytes": int(shard.stat().st_size),
        "arrow_schema_sha256": _schema_sha256(schema),
        "rows": int(parquet.metadata.num_rows),
        "columns": list(schema.names),
    }


def _sample_id(dataset: str, raw_path: str) -> str:
    """Derive an opaque, deterministic target ID without consulting labels."""
    digest = hashlib.sha256(f"h9-pcr-target-v1|{dataset}|{raw_path}".encode("utf-8")).hexdigest()
    return f"{dataset.lower()}_{digest}"


def _extract_audio_payload(row: Mapping[str, object], *, shard: Path) -> tuple[str, bytes]:
    raw_path = row.get("path")
    audio = row.get("audio")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"H9 target shard has an empty raw path: {shard}")
    if not isinstance(audio, Mapping):
        raise ValueError(f"H9 target shard audio cell is not a struct: {shard}")
    payload = audio.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError(f"H9 target shard has empty/non-binary audio bytes: {shard}")
    return raw_path, bytes(payload)


def _read_record_phase(
    shards: Sequence[Path], *, dataset: str, stage: Path, published_output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy/audio-audit target trials without opening the raw label column."""
    waveforms = stage / "waveforms"
    waveforms.mkdir()
    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    raw_paths: set[str] = set()
    sample_ids: set[str] = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["path", "audio"], batch_size=128):
            for row in batch.to_pylist():
                raw_path, payload = _extract_audio_payload(row, shard=shard)
                if raw_path in raw_paths:
                    raise ValueError(f"H9 target raw path is duplicated across shards: {raw_path}")
                raw_paths.add(raw_path)
                sample_id = _sample_id(dataset, raw_path)
                if sample_id in sample_ids:
                    raise ValueError("H9 target deterministic sample ID collision")
                sample_ids.add(sample_id)
                fingerprint, frames, pcm_bytes = canonical_mono16k_pcm_fingerprint(payload)
                payload_hash = hashlib.sha256(payload).hexdigest()
                copied = waveforms / f"{sample_id}.wav"
                copied.write_bytes(payload)
                copied_hash = sha256_file(copied)
                if copied_hash != payload_hash or copied.stat().st_size != len(payload):
                    raise RuntimeError("H9 target copied audio byte audit failed")
                published_audio = (published_output / "waveforms" / copied.name).resolve()
                records.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": str(published_audio),
                        "audio_sha256": payload_hash,
                        "audio_bytes": len(payload),
                        "canonical_fingerprint": fingerprint,
                    }
                )
                audits.append(
                    {
                        "sample_id": sample_id,
                        "raw_relative_path": raw_path,
                        "raw_shard": str(shard.resolve()),
                        "raw_audio_sha256": payload_hash,
                        "raw_audio_bytes": len(payload),
                        "copied_audio_path": str(published_audio),
                        "copied_audio_sha256": copied_hash,
                        "copied_audio_bytes": int(copied.stat().st_size),
                        "canonical_fingerprint": fingerprint,
                        "canonical_frames": frames,
                        "canonical_pcm_bytes": pcm_bytes,
                    }
                )
    frame = pd.DataFrame(records, columns=TARGET_RECORD_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    audit = pd.DataFrame(audits, columns=TARGET_AUDIO_AUDIT_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if frame.empty or frame["sample_id"].duplicated().any() or len(frame) != len(audit):
        raise ValueError("H9 target record phase did not produce one unique record/audit per raw trial")
    return frame, audit


def _read_label_phase(shards: Sequence[Path], *, dataset: str, records: pd.DataFrame) -> pd.DataFrame:
    """Read only path/label after the label-free record table is published."""
    # The record table intentionally excludes raw paths; the caller passes a
    # temporary map through ``records.attrs`` so that no labels can influence
    # the published score-free schema.
    raw_path_to_sample = records.attrs.get("raw_path_to_sample")
    if not isinstance(raw_path_to_sample, dict) or not raw_path_to_sample:
        raise RuntimeError("H9 target label phase lacks its label-free raw-path mapping")
    labels: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["path", "label"], batch_size=256):
            for row in batch.to_pylist():
                raw_path = row.get("path")
                value = row.get("label")
                if not isinstance(raw_path, str) or raw_path not in raw_path_to_sample or raw_path in seen_paths:
                    raise ValueError("H9 target label phase raw paths disagree with the label-free record phase")
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in {0, 1}:
                    raise ValueError("H9 target labels must be exactly binary integers 0/1")
                seen_paths.add(raw_path)
                labels.append({"sample_id": raw_path_to_sample[raw_path], "label": int(value)})
    if seen_paths != set(raw_path_to_sample):
        raise ValueError("H9 target label phase is missing a label-free record identity")
    frame = pd.DataFrame(labels, columns=TARGET_LABEL_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if frame.empty or frame["sample_id"].duplicated().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("H9 target label artifact requires unique IDs and both binary classes")
    return frame


def _new_stage(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"H9 target materialization refuses to overwrite existing output: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"H9 target materialization output parent does not exist: {output_dir.parent}")
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))


def _publish_stage(stage: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"H9 target materialization output appeared during publication: {output_dir}")
    os.replace(stage, output_dir)


def materialize_h9_target(
    *,
    dataset: str,
    raw_shard_dir: str | Path,
    checkpoint_ledger: str | Path,
    plan: str | Path,
    data_contract: str | Path,
    output_dir: str | Path,
    allowed_output_root: str | Path = HDD_ROOT,
    _source_handoff_validator: Callable[..., object] = load_frozen_checkpoint_ledger,
) -> TargetMaterialization:
    """Build one immutable target handoff after all source fits are frozen.

    The default source-handoff validator is the same strict loader the terminal
    evaluator uses.  Tests may inject a harmless validator; the production CLI
    exposes no bypass.
    """
    if dataset not in TARGET_REVISIONS:
        raise ValueError(f"H9 target dataset must be one of {tuple(TARGET_REVISIONS)}")
    output = _path_below(Path(output_dir), Path(allowed_output_root), description="target output")
    # The source handoff must be frozen before this function touches a target
    # path or Parquet schema.  This enforces PLAN.md's terminal sequencing.
    _source_handoff_validator(checkpoint_ledger, plan_path=plan, data_contract_path=data_contract)
    shards = _raw_target_shards(raw_shard_dir, dataset=dataset)
    shard_audits = [_validate_target_shard_schema(shard) for shard in shards]
    stage = _new_stage(output)
    stem = dataset.lower()
    try:
        records, audio_audit = _read_record_phase(
            shards, dataset=dataset, stage=stage, published_output=output
        )
        raw_path_to_sample = {
            str(row.raw_relative_path): str(row.sample_id)
            for row in audio_audit.loc[:, ["raw_relative_path", "sample_id"]].itertuples(index=False)
        }
        if len(raw_path_to_sample) != len(records):
            raise ValueError("H9 target record phase raw-path mapping is not one-to-one")
        records.attrs["raw_path_to_sample"] = raw_path_to_sample
        records_path = stage / f"h9_{stem}_target_records.csv"
        audit_path = stage / f"h9_{stem}_target_audio_audit.csv"
        records.to_csv(records_path, index=False)
        audio_audit.to_csv(audit_path, index=False)
        records_sha256_before_label_access = sha256_file(records_path)
        # This is the first and only raw-label pass.  It cannot alter the
        # already-written score-free record CSV or its recorded byte hash.
        labels = _read_label_phase(shards, dataset=dataset, records=records)
        labels_path = stage / f"h9_{stem}_target_labels.csv"
        labels.to_csv(labels_path, index=False)
        manifest_path = stage / f"h9_{stem}_target_manifest.json"
        manifest = {
            "artifact_kind": "h9_pcr_canonical_target_manifest",
            "version": H9_EVALUATION_VERSION,
            "dataset": dataset,
            "dataset_revision": TARGET_REVISIONS[dataset],
            "records_path": str((output / records_path.name).resolve()),
            "records_sha256": records_sha256_before_label_access,
            "labels": {
                "path": str((output / labels_path.name).resolve()),
                "sha256": sha256_file(labels_path),
            },
            # This flag is the terminal evaluator's label firewall: labels
            # have not been supplied to a trainer, predictor, selector, or
            # metric.  The provenance below precisely records their separate
            # artifact-materialization read.
            "target_labels_read": False,
            "target_metrics_read": False,
        }
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        provenance_path = stage / f"h9_{stem}_target_materialization.json"
        provenance = {
            "artifact_kind": "h9_pcr_terminal_target_materialization",
            "version": H9_TARGET_MATERIALIZATION_VERSION,
            "dataset": {"name": dataset, "revision": TARGET_REVISIONS[dataset]},
            "source_handoff": {
                "checkpoint_ledger_path": str(Path(checkpoint_ledger).expanduser().resolve()),
                "validated_before_target_access": True,
            },
            "raw_target_shards": shard_audits,
            "canonical_audio_policy": CANONICAL_AUDIO_POLICY,
            "access_boundary": {
                "record_phase_opened_columns": ["path", "audio"],
                "record_phase_raw_label_values_read": False,
                "record_phase_model_predictions_read": False,
                "record_phase_target_metrics_read": False,
                "label_artifact_phase_opened_columns": ["path", "label"],
                "label_artifact_phase_raw_label_values_read": True,
                "label_artifact_phase_purpose": "write_separate_terminal_label_artifact_only",
                "label_artifact_phase_model_predictions_read": False,
                "label_artifact_phase_target_metrics_read": False,
                "terminal_evaluator_label_metrics_read": False,
            },
            "outputs": {
                "target_manifest_json": {"path": str((output / manifest_path.name).resolve()), "sha256": sha256_file(manifest_path)},
                "records_csv": {"path": str((output / records_path.name).resolve()), "sha256": records_sha256_before_label_access, "n_rows": int(len(records))},
                "labels_csv": {"path": str((output / labels_path.name).resolve()), "sha256": sha256_file(labels_path), "n_rows": int(len(labels))},
                "audio_audit_csv": {"path": str((output / audit_path.name).resolve()), "sha256": sha256_file(audit_path), "n_rows": int(len(audio_audit))},
                "waveforms_dir": str((output / "waveforms").resolve()),
            },
        }
        provenance_path.write_text(_canonical_json(provenance) + "\n", encoding="utf-8")
        _publish_stage(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return TargetMaterialization(
        dataset=dataset,
        output_dir=output,
        manifest=output / manifest_path.name,
        records=output / records_path.name,
        labels=output / labels_path.name,
        audio_audit=output / audit_path.name,
        provenance=output / provenance_path.name,
        waveforms_dir=output / "waveforms",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one sealed H9-PCR terminal target without predictions or metrics.")
    parser.add_argument("--dataset", required=True, choices=tuple(TARGET_REVISIONS))
    parser.add_argument("--raw-shard-dir", required=True, type=Path, help="Pinned target <revision>/raw/data directory.")
    parser.add_argument("--checkpoint-ledger", required=True, type=Path, help="Immutable complete source-only checkpoint ledger.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New HDD output directory; existing output is refused.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = materialize_h9_target(
        dataset=args.dataset,
        raw_shard_dir=args.raw_shard_dir,
        checkpoint_ledger=args.checkpoint_ledger,
        plan=args.plan,
        data_contract=args.data_contract,
        output_dir=args.output_dir,
    )
    print(_canonical_json({"dataset": result.dataset, "manifest": str(result.manifest), "provenance": str(result.provenance)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through scripts/materialize_h9_pcr_target.py
    raise SystemExit(cli_main())
