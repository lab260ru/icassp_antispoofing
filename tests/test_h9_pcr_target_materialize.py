"""Synthetic-only tests for the sealed H9-PCR terminal target materializer."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from src.h9_pcr_evaluation import TARGET_REVISIONS, load_target_manifest
from src.h9_pcr_target_materialize import (
    H9_TARGET_MATERIALIZATION_VERSION,
    TARGET_AUDIO_AUDIT_COLUMNS,
    TARGET_LABEL_COLUMNS,
    TARGET_RECORD_COLUMNS,
    materialize_h9_target,
)
from src.res2tcn_pytorch import sha256_file


def _audio_bytes(value: float, *, format: str = "WAV") -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.full(32, value, dtype=np.float32), 16_000, format=format, subtype="PCM_16")
    return stream.getvalue()


def _raw_target(
    tmp_path: Path,
    *,
    dataset: str = "SONAR",
    labels: list[int] | None = None,
    audio_format: str = "WAV",
) -> Path:
    labels = [0, 0, 1, 1] if labels is None else labels
    raw = tmp_path / "datasets" / dataset.lower() / TARGET_REVISIONS[dataset] / "raw" / "data"
    raw.mkdir(parents=True)
    rows = [
        {
            "path": f"synthetic/{dataset.lower()}_{index}.wav",
            "audio": {"bytes": _audio_bytes(float(index) / 8.0, format=audio_format), "path": None},
            "label": label,
        }
        for index, label in enumerate(labels)
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw / "data-00000.parquet")
    # Match the Hub layout: this standalone label index must never be
    # mistaken for an audio-bearing target shard.
    pq.write_table(
        pa.Table.from_pydict({"path": [row["path"] for row in rows], "label": labels}),
        raw / "labels.parquet",
    )
    return raw


def _validator(*_args: object, **_kwargs: object) -> object:
    """Synthetic test seam: production CLI always uses the strict loader."""
    return object()


def test_materializes_a_separated_synthetic_target_contract(tmp_path: Path) -> None:
    raw = _raw_target(tmp_path)
    output = tmp_path / "hdd" / "h9_sonar_target_001"
    output.parent.mkdir(parents=True)
    result = materialize_h9_target(
        dataset="SONAR",
        raw_shard_dir=raw,
        checkpoint_ledger=tmp_path / "synthetic-ledger.json",
        plan=tmp_path / "synthetic-plan.md",
        data_contract=tmp_path / "synthetic-data.md",
        output_dir=output,
        allowed_output_root=tmp_path / "hdd",
        _source_handoff_validator=_validator,
    )
    records = pd.read_csv(result.records)
    labels = pd.read_csv(result.labels)
    audio_audit = pd.read_csv(result.audio_audit)
    assert tuple(records.columns) == TARGET_RECORD_COLUMNS
    assert tuple(labels.columns) == TARGET_LABEL_COLUMNS
    assert tuple(audio_audit.columns) == TARGET_AUDIO_AUDIT_COLUMNS
    assert "label" not in records.columns
    assert len(records) == len(labels) == len(audio_audit) == 4
    assert set(labels["label"]) == {0, 1}
    assert records["sample_id"].tolist() == labels["sample_id"].tolist()
    assert records["canonical_fingerprint"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert all(Path(path).is_file() for path in records["audio_path"])
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["target_labels_read"] is False
    assert manifest["target_metrics_read"] is False
    assert manifest["records_sha256"] == sha256_file(result.records)
    assert manifest["labels"]["sha256"] == sha256_file(result.labels)
    parsed = load_target_manifest(result.manifest, expected_dataset="SONAR")
    assert len(parsed.records) == 4
    provenance = json.loads(result.provenance.read_text(encoding="utf-8"))
    assert provenance["version"] == H9_TARGET_MATERIALIZATION_VERSION
    assert provenance["access_boundary"]["record_phase_raw_label_values_read"] is False
    assert provenance["access_boundary"]["label_artifact_phase_raw_label_values_read"] is True
    assert provenance["access_boundary"]["label_artifact_phase_target_metrics_read"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_h9_target(
            dataset="SONAR",
            raw_shard_dir=raw,
            checkpoint_ledger=tmp_path / "synthetic-ledger.json",
            plan=tmp_path / "synthetic-plan.md",
            data_contract=tmp_path / "synthetic-data.md",
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _source_handoff_validator=_validator,
        )


def test_rejects_nonbinary_label_without_publishing_output(tmp_path: Path) -> None:
    raw = _raw_target(tmp_path, labels=[0, 0, 1, 2])
    output = tmp_path / "hdd" / "bad_labels"
    output.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="binary integers"):
        materialize_h9_target(
            dataset="SONAR",
            raw_shard_dir=raw,
            checkpoint_ledger=tmp_path / "synthetic-ledger.json",
            plan=tmp_path / "synthetic-plan.md",
            data_contract=tmp_path / "synthetic-data.md",
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _source_handoff_validator=_validator,
        )
    assert not output.exists()


def test_materializes_pinned_flac_payloads_with_their_true_extension(tmp_path: Path) -> None:
    raw = _raw_target(tmp_path, audio_format="FLAC")
    output = tmp_path / "hdd" / "flac_target"
    output.parent.mkdir(parents=True)
    result = materialize_h9_target(
        dataset="SONAR",
        raw_shard_dir=raw,
        checkpoint_ledger=tmp_path / "synthetic-ledger.json",
        plan=tmp_path / "synthetic-plan.md",
        data_contract=tmp_path / "synthetic-data.md",
        output_dir=output,
        allowed_output_root=tmp_path / "hdd",
        _source_handoff_validator=_validator,
    )
    records = pd.read_csv(result.records)
    assert records["audio_path"].str.endswith(".flac").all()
    assert all(Path(path).read_bytes().startswith(b"fLaC") for path in records["audio_path"])


def test_source_handoff_is_validated_before_a_target_path_is_inspected(tmp_path: Path) -> None:
    calls: list[tuple[object, object, object]] = []

    def validator(ledger: object, *, plan_path: object, data_contract_path: object) -> object:
        calls.append((ledger, plan_path, data_contract_path))
        raise RuntimeError("synthetic source handoff rejection")

    output = tmp_path / "hdd" / "must_not_exist"
    output.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="source handoff rejection"):
        materialize_h9_target(
            dataset="SONAR",
            raw_shard_dir=tmp_path / "not_a_real_target_root",
            checkpoint_ledger=tmp_path / "ledger.json",
            plan=tmp_path / "plan.md",
            data_contract=tmp_path / "data.md",
            output_dir=output,
            allowed_output_root=tmp_path / "hdd",
            _source_handoff_validator=validator,
        )
    assert len(calls) == 1
    assert not output.exists()
