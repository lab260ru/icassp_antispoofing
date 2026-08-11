"""Two-pass, fail-closed CD-ADD materialization for the prospective H10 run.

This module is deliberately namespaced separately from H9.  It may replay the
already sealed H9 source checkpoint ledger, but it never changes that ledger
or lets CD-ADD data select any source-side decision.  The record pass opens
only ``path,audio`` and publishes a label-free record table before the second
``path,label`` pass creates a separately sealed label artifact.

The production CLI has no synthetic/test bypass.  The narrow dependency
injection points below exist only for synthetic contract tests, which must run
before a real CD-ADD directory is accessed.
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
from src.h9_pcr_evaluation import load_frozen_checkpoint_ledger
from src.res2tcn_pytorch import sha256_file


H10_CDADD_VERSION = "h10-cdadd-external-replication-v1"
H10_CDADD_DATASET = "CD-ADD"
H10_CDADD_REPOSITORY = "SpeechAntiSpoofingBenchmarks/CD-ADD"
H10_CDADD_REVISION = "b03c6cf3463d67c1525ba20738c495d120675876"
H10_CDADD_LICENSE = "CC-BY-4.0"
H10_CDADD_CARD_URL = (
    "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD/"
    "blob/b03c6cf3463d67c1525ba20738c495d120675876/README.md"
)
H9_FROZEN_CHECKPOINT_LEDGER_SHA256 = "3e0530e2b850cbb08329d14a8982a7448a319a4a30e75ee2d51dc4ecb7f2325d"
H9_PLAN_PATH = Path("experiments/h9_paired_counterfactual/PLAN.md")
H9_DATA_CONTRACT_PATH = Path("experiments/h9_paired_counterfactual/DATA.md")
H10_PROTOCOL_PATH = Path("experiments/future_directions/H10_CDADD_EXTERNAL_REPLICATION_PROTOCOL.md")
H10_TARGET_MATERIALIZATION_VERSION = "h10_cdadd_target_materialization_v1"

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
class H10TargetMaterialization:
    """Published, immutable paths for one H10 CD-ADD target handoff."""

    output_dir: Path
    manifest: Path
    records: Path
    labels: Path
    audio_audit: Path
    provenance: Path
    waveforms_dir: Path


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


def _require_hash(value: object, *, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"H10 {description} must be a lowercase SHA-256 digest")
    return str(value)


def _path_below(path: str | Path, root: str | Path, *, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"H10 {description} must be below {allowed}: {resolved}") from error
    return resolved


def _validate_h9_source_ledger(path: str | Path) -> object:
    """Replay the source-only H9 ledger before touching a target path."""
    ledger_path = Path(path).expanduser().resolve()
    if sha256_file(ledger_path) != H9_FROZEN_CHECKPOINT_LEDGER_SHA256:
        raise ValueError("H10 requires the exact H9 frozen checkpoint ledger SHA-256")
    return load_frozen_checkpoint_ledger(
        ledger_path,
        plan_path=H9_PLAN_PATH,
        data_contract_path=H9_DATA_CONTRACT_PATH,
    )


def locked_protocol_bindings() -> dict[str, dict[str, str]]:
    """Hash the three fixed protocol documents before target materialization.

    The H9 loader independently replays the H9 PLAN/DATA values while it
    validates the source ledger.  Persisting the same hashes in H10 artifacts
    makes the cross-study dependency explicit and gives the terminal evaluator
    a second, direct provenance check.
    """
    bindings: dict[str, dict[str, str]] = {}
    for name, path in (("h10_protocol", H10_PROTOCOL_PATH), ("h9_plan", H9_PLAN_PATH), ("h9_data", H9_DATA_CONTRACT_PATH)):
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"H10 locked {name.replace('_', ' ')} is unavailable: {resolved}")
        bindings[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return bindings


def _raw_target_shards(raw_shard_dir: str | Path) -> list[Path]:
    """Accept only the H10 pinned CD-ADD revision's explicit ``data`` root."""
    directory = Path(raw_shard_dir).expanduser().resolve()
    if not directory.is_dir() or directory.name != "data" or directory.parent.name != H10_CDADD_REVISION:
        raise ValueError(
            "H10 CD-ADD raw-shard input must be the pinned "
            "<revision>/data directory; do not infer another target root"
        )
    # The Arena layout may carry a conventional standalone ``labels.parquet``
    # beside audio-bearing shards.  It is not a target trial shard and is
    # deliberately excluded; each admitted audio shard must still itself
    # expose the complete path/audio/label schema below.
    shards = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".parquet" and path.name != "labels.parquet"
    )
    if not shards:
        raise FileNotFoundError(f"H10 CD-ADD raw-shard directory has no Parquet files: {directory}")
    return shards


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_target_shard_schema(shard: Path) -> dict[str, Any]:
    """Check the full audio-bearing raw schema without opening label values."""
    parquet = pq.ParquetFile(shard)
    schema = parquet.schema_arrow
    required = {"path", "audio", "label"}
    if not required.issubset(set(schema.names)):
        raise ValueError(f"H10 CD-ADD target shard lacks required path/audio/label fields: {shard}")
    path_type = schema.field("path").type
    if not (pa.types.is_string(path_type) or pa.types.is_large_string(path_type)):
        raise ValueError(f"H10 CD-ADD target path field must be string: {shard}")
    audio_type = schema.field("audio").type
    if not pa.types.is_struct(audio_type) or "bytes" not in audio_type.names:
        raise ValueError(f"H10 CD-ADD target audio field must be a struct with binary bytes: {shard}")
    payload_type = audio_type.field("bytes").type
    if not (pa.types.is_binary(payload_type) or pa.types.is_large_binary(payload_type)):
        raise ValueError(f"H10 CD-ADD target audio.bytes field must be binary: {shard}")
    if not pa.types.is_integer(schema.field("label").type):
        raise ValueError(f"H10 CD-ADD target label field must be integer: {shard}")
    return {
        "path": str(shard.resolve()),
        "sha256": sha256_file(shard),
        "bytes": int(shard.stat().st_size),
        "arrow_schema_sha256": _schema_sha256(schema),
        "rows": int(parquet.metadata.num_rows),
        "columns": list(schema.names),
    }


def _validate_metadata_audit(path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    """Fail unless the pre-raw-data card/lineage audit declares no known overlap."""
    audit_path = Path(path).expanduser().resolve()
    audit = _read_json(audit_path, description="CD-ADD card and metadata audit")
    if audit.get("artifact_kind") != "h10_cdadd_target_metadata_audit":
        raise ValueError("H10 CD-ADD metadata audit artifact kind drift")
    if audit.get("repository") != H10_CDADD_REPOSITORY or audit.get("revision") != H10_CDADD_REVISION:
        raise ValueError("H10 CD-ADD metadata audit repository/revision drift")
    if audit.get("license") != H10_CDADD_LICENSE or audit.get("card_url") != H10_CDADD_CARD_URL:
        raise ValueError("H10 CD-ADD metadata audit card/license provenance drift")
    _require_hash(audit.get("card_sha256"), description="CD-ADD card SHA-256")
    for key in ("target_rows_read", "target_audio_read", "target_labels_read", "target_scores_read", "target_predictions_read"):
        if audit.get(key) is not False:
            raise ValueError(f"H10 CD-ADD metadata audit breached the target firewall in {key}")
    overlap = audit.get("odss_overlap_audit")
    if not isinstance(overlap, Mapping):
        raise ValueError("H10 CD-ADD metadata audit lacks an ODSS overlap audit")
    for key in ("known_recording_source_overlap", "known_speaker_list_overlap", "known_released_item_overlap"):
        if overlap.get(key) is not False:
            raise ValueError(f"H10 CD-ADD metadata audit has unresolved {key}")
    return audit_path, sha256_file(audit_path), audit


def _sample_id(raw_path: str) -> str:
    digest = hashlib.sha256(f"h10-cdadd-target-v1|{H10_CDADD_REVISION}|{raw_path}".encode("utf-8")).hexdigest()
    return f"cdadd_{digest}"


def _extract_audio_payload(row: Mapping[str, object], *, shard: Path) -> tuple[str, bytes]:
    raw_path = row.get("path")
    audio = row.get("audio")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"H10 CD-ADD target shard has an empty raw path: {shard}")
    if not isinstance(audio, Mapping):
        raise ValueError(f"H10 CD-ADD target audio cell is not a struct: {shard}")
    payload = audio.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError(f"H10 CD-ADD target shard has empty/non-binary audio bytes: {shard}")
    return raw_path, bytes(payload)


def _target_payload_suffix(payload: bytes) -> str:
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        return ".wav"
    if payload.startswith(b"fLaC"):
        return ".flac"
    raise ValueError("H10 CD-ADD target payload must be RIFF/WAVE or FLAC")


def _read_record_phase(
    shards: Sequence[Path], *, stage: Path, published_output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open only ``path,audio``, copy byte-identical payloads, then fingerprint."""
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
                    raise ValueError(f"H10 CD-ADD target raw path is duplicated: {raw_path}")
                raw_paths.add(raw_path)
                sample_id = _sample_id(raw_path)
                if sample_id in sample_ids:
                    raise ValueError("H10 CD-ADD deterministic sample-ID collision")
                sample_ids.add(sample_id)
                suffix = _target_payload_suffix(payload)
                fingerprint, frames, pcm_bytes = canonical_mono16k_pcm_fingerprint(payload, require_riff_wave=False)
                payload_hash = hashlib.sha256(payload).hexdigest()
                copied = waveforms / f"{sample_id}{suffix}"
                copied.write_bytes(payload)
                copied_hash = sha256_file(copied)
                if copied_hash != payload_hash or copied.stat().st_size != len(payload):
                    raise RuntimeError("H10 CD-ADD copied audio byte audit failed")
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
    records_frame = pd.DataFrame(records, columns=TARGET_RECORD_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    audit_frame = pd.DataFrame(audits, columns=TARGET_AUDIO_AUDIT_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if records_frame.empty or records_frame["sample_id"].duplicated().any() or len(records_frame) != len(audit_frame):
        raise ValueError("H10 CD-ADD record phase did not produce one unique record/audit per raw trial")
    return records_frame, audit_frame


def _read_label_phase(shards: Sequence[Path], *, raw_path_to_sample: Mapping[str, str]) -> pd.DataFrame:
    """The only raw label pass, after label-free records are written and hashed."""
    labels: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["path", "label"], batch_size=256):
            for row in batch.to_pylist():
                raw_path = row.get("path")
                value = row.get("label")
                if not isinstance(raw_path, str) or raw_path not in raw_path_to_sample or raw_path in seen_paths:
                    raise ValueError("H10 CD-ADD label phase disagrees with the label-free record phase")
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in {0, 1}:
                    raise ValueError("H10 CD-ADD labels must be exactly binary integers 0/1")
                seen_paths.add(raw_path)
                labels.append({"sample_id": raw_path_to_sample[raw_path], "label": int(value)})
    if seen_paths != set(raw_path_to_sample):
        raise ValueError("H10 CD-ADD label phase is missing a label-free record identity")
    frame = pd.DataFrame(labels, columns=TARGET_LABEL_COLUMNS).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if frame.empty or frame["sample_id"].duplicated().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("H10 CD-ADD label artifact requires unique IDs and both binary classes")
    return frame


def _new_stage(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"H10 CD-ADD materialization refuses to overwrite existing output: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"H10 CD-ADD output parent does not exist: {output_dir.parent}")
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))


def materialize_h10_cdadd_target(
    *,
    raw_shard_dir: str | Path,
    checkpoint_ledger: str | Path,
    metadata_audit: str | Path,
    output_dir: str | Path,
    allowed_output_root: str | Path = HDD_ROOT,
    _source_ledger_validator: Callable[[str | Path], object] = _validate_h9_source_ledger,
) -> H10TargetMaterialization:
    """Create the H10 CD-ADD target handoff after H9 source replay succeeds."""
    output = _path_below(output_dir, allowed_output_root, description="CD-ADD target output")
    # This must be first: no target directory, metadata-audit, shard schema,
    # row, audio, or label is opened before the immutable H9 source handoff.
    _source_ledger_validator(checkpoint_ledger)
    protocol_bindings = locked_protocol_bindings()
    audit_path, audit_sha256, audit = _validate_metadata_audit(metadata_audit)
    shards = _raw_target_shards(raw_shard_dir)
    shard_audits = [_validate_target_shard_schema(shard) for shard in shards]
    stage = _new_stage(output)
    try:
        records, audio_audit = _read_record_phase(shards, stage=stage, published_output=output)
        raw_path_to_sample = {
            str(row.raw_relative_path): str(row.sample_id)
            for row in audio_audit.loc[:, ["raw_relative_path", "sample_id"]].itertuples(index=False)
        }
        if len(raw_path_to_sample) != len(records):
            raise ValueError("H10 CD-ADD record phase raw-path mapping is not one-to-one")
        records_path = stage / "h10_cdadd_target_records.csv"
        audio_audit_path = stage / "h10_cdadd_target_audio_audit.csv"
        records.to_csv(records_path, index=False)
        audio_audit.to_csv(audio_audit_path, index=False)
        records_sha256_before_label_access = sha256_file(records_path)
        labels = _read_label_phase(shards, raw_path_to_sample=raw_path_to_sample)
        labels_path = stage / "h10_cdadd_target_labels.csv"
        labels.to_csv(labels_path, index=False)
        provenance_path = stage / "h10_cdadd_target_materialization.json"
        manifest_path = stage / "h10_cdadd_target_manifest.json"
        manifest = {
            "artifact_kind": "h10_cdadd_canonical_target_manifest",
            "version": H10_CDADD_VERSION,
            "dataset": H10_CDADD_DATASET,
            "dataset_repository": H10_CDADD_REPOSITORY,
            "dataset_revision": H10_CDADD_REVISION,
            "locked_protocols": protocol_bindings,
            "target_construction": {
                "all_audio_bearing_parquet_trials": True,
                "target_subset": None,
                "generator_stratum": None,
                "path_based_exclusions": None,
                "raw_shard_count": len(shard_audits),
                "raw_trial_count": int(len(records)),
            },
            "records_path": str((output / records_path.name).resolve()),
            "records_sha256": records_sha256_before_label_access,
            "labels": {"path": str((output / labels_path.name).resolve()), "sha256": sha256_file(labels_path)},
            "materialization_provenance": {"path": str((output / provenance_path.name).resolve())},
            "target_labels_read": False,
            "target_metrics_read": False,
        }
        provenance = {
            "artifact_kind": "h10_cdadd_terminal_target_materialization",
            "version": H10_TARGET_MATERIALIZATION_VERSION,
            "dataset": {
                "name": H10_CDADD_DATASET,
                "repository": H10_CDADD_REPOSITORY,
                "revision": H10_CDADD_REVISION,
                "license": H10_CDADD_LICENSE,
                "public_card_url": H10_CDADD_CARD_URL,
            },
            "source_handoff": {
                "checkpoint_ledger_path": str(Path(checkpoint_ledger).expanduser().resolve()),
                # The production validator above independently byte-checks
                # the supplied file against this literal sealed H9 identity.
                # Record the required identity, not a caller-controlled
                # replacement value.
                "checkpoint_ledger_sha256": H9_FROZEN_CHECKPOINT_LEDGER_SHA256,
                "required_h9_checkpoint_ledger_sha256": H9_FROZEN_CHECKPOINT_LEDGER_SHA256,
                "validated_before_target_access": True,
            },
            "locked_protocols": protocol_bindings,
            "target_construction": {
                "all_audio_bearing_parquet_trials": True,
                "target_subset": None,
                "generator_stratum": None,
                "path_based_exclusions": None,
                "raw_shard_count": len(shard_audits),
                "raw_trial_count": int(len(records)),
            },
            "metadata_audit": {"path": str(audit_path), "sha256": audit_sha256, "card_sha256": audit["card_sha256"]},
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
                "target_manifest_json": {"path": str((output / manifest_path.name).resolve())},
                "records_csv": {"path": str((output / records_path.name).resolve()), "sha256": records_sha256_before_label_access, "n_rows": int(len(records))},
                "labels_csv": {"path": str((output / labels_path.name).resolve()), "sha256": sha256_file(labels_path), "n_rows": int(len(labels))},
                "audio_audit_csv": {"path": str((output / audio_audit_path.name).resolve()), "sha256": sha256_file(audio_audit_path), "n_rows": int(len(audio_audit))},
                "waveforms_dir": str((output / "waveforms").resolve()),
            },
        }
        provenance_path.write_text(_canonical_json(provenance) + "\n", encoding="utf-8")
        manifest["materialization_provenance"]["sha256"] = sha256_file(provenance_path)
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return H10TargetMaterialization(
        output_dir=output,
        manifest=output / manifest_path.name,
        records=output / records_path.name,
        labels=output / labels_path.name,
        audio_audit=output / audio_audit_path.name,
        provenance=output / provenance_path.name,
        waveforms_dir=output / "waveforms",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the locked H10 CD-ADD target without predictions or metrics.")
    parser.add_argument("--raw-shard-dir", required=True, type=Path, help="Pinned CD-ADD <revision>/data directory.")
    parser.add_argument("--checkpoint-ledger", required=True, type=Path, help="Exact immutable H9 12-checkpoint ledger.")
    parser.add_argument("--metadata-audit", required=True, type=Path, help="Pre-raw-data CD-ADD card/ODSS-overlap audit JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New HDD output directory; existing output is refused.")
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = materialize_h10_cdadd_target(
        raw_shard_dir=args.raw_shard_dir,
        checkpoint_ledger=args.checkpoint_ledger,
        metadata_audit=args.metadata_audit,
        output_dir=args.output_dir,
    )
    print(_canonical_json({"manifest": str(result.manifest), "provenance": str(result.provenance)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
