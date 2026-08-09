from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.h2_input_csv import (
    assert_score_free_columns,
    build_score_free_input_from_index,
    build_score_free_input_rows,
    write_score_free_input_artifacts,
)


def _write_synthetic_index(tmp_path: Path) -> tuple[Path, Path, str]:
    dataset_root = tmp_path / "dataset"
    labels_path = dataset_root / "data" / "labels.parquet"
    labels_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "utterance_id": ["b.wav", "a.wav", "c.wav", "d.wav"],
                "label": [1, 0, 1, 0],
            }
        ),
        labels_path,
    )
    labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    index_path = tmp_path / "arena-index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "Synthetic": {
                        "repo_id": "example/Synthetic",
                        "revision": "abc123",
                        "source_revision_resolved": "abc123",
                        "local_dir": str(dataset_root),
                        "files": {"labels": "data/labels.parquet"},
                        "pinned_files": [{"path": "data/labels.parquet", "sha256": labels_hash}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return index_path, labels_path, labels_hash


def test_builds_sorted_score_free_rows_via_synthetic_pinned_labels(tmp_path: Path) -> None:
    index_path, labels_path, labels_hash = _write_synthetic_index(tmp_path)
    rows, provenance = build_score_free_input_from_index(index_path, "Synthetic")

    assert rows.columns.tolist() == ["dataset", "sample_id", "source_id", "label", "dataset_revision"]
    assert rows.to_dict(orient="records") == [
        {"dataset": "Synthetic", "sample_id": "a", "source_id": "a.wav", "label": 0, "dataset_revision": "abc123"},
        {"dataset": "Synthetic", "sample_id": "d", "source_id": "d.wav", "label": 0, "dataset_revision": "abc123"},
        {"dataset": "Synthetic", "sample_id": "b", "source_id": "b.wav", "label": 1, "dataset_revision": "abc123"},
        {"dataset": "Synthetic", "sample_id": "c", "source_id": "c.wav", "label": 1, "dataset_revision": "abc123"},
    ]
    assert provenance["labels"]["local_path"] == str(labels_path)
    assert provenance["labels"]["sha256"] == labels_hash
    assert provenance["labels"]["pinned_sha256_verified"] is True
    assert provenance["source_access"] == {
        "audio_read": False,
        "feature_read": False,
        "score_read": False,
        "asr_loaded": False,
        "detector_loaded": False,
    }
    assert not any("score" in column.casefold() for column in rows.columns)


def test_preserves_optional_speaker_and_rejects_unknown_dataset(tmp_path: Path) -> None:
    index_path, _labels_path, _labels_hash = _write_synthetic_index(tmp_path)
    index = yaml.safe_load(index_path.read_text())
    labels = pd.DataFrame(
        {"sample_id": ["u2", "u1"], "source_id": ["raw2", "raw1"], "label": [1, 0], "speaker_id": ["s2", "s1"]}
    )
    # This builder receives labels only through its explicit load_labels-equivalent injection.
    rows, _provenance = build_score_free_input_rows(index, "Synthetic", labels_loader=lambda _dataset: labels)
    assert rows["speaker_id"].tolist() == ["s1", "s2"]
    with pytest.raises(ValueError, match="Unknown Arena dataset"):
        build_score_free_input_rows(index, "NotInIndex", labels_loader=lambda _dataset: labels)


def test_write_refuses_overwrite_and_response_like_columns(tmp_path: Path) -> None:
    index_path, _labels_path, _labels_hash = _write_synthetic_index(tmp_path)
    rows, provenance = build_score_free_input_from_index(index_path, "Synthetic")
    output_csv = tmp_path / "out" / "h2_input.csv"
    output_json = tmp_path / "out" / "h2_input.provenance.json"
    write_score_free_input_artifacts(rows, provenance, output_csv=output_csv, output_provenance=output_json)
    assert pd.read_csv(output_csv).columns.tolist() == rows.columns.tolist()
    assert json.loads(output_json.read_text())["artifact_kind"] == "h2_score_free_input_csv"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_score_free_input_artifacts(rows, provenance, output_csv=output_csv, output_provenance=output_json)
    with pytest.raises(ValueError, match="response-like"):
        assert_score_free_columns(["dataset", "detector_logit"])
