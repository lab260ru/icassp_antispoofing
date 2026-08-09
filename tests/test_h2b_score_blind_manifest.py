from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.h2_input_csv import build_score_free_input_from_index, write_score_free_input_artifacts
from src.h2b_score_blind_manifest import freeze_h2b_q0_manifest, write_h2b_q0_artifacts


def _source_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    labels_path = root / "data" / "labels.parquet"
    labels_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "utterance_id": [f"u{label}_{idx}.wav" for label in (0, 1) for idx in range(4)],
                "label": [label for label in (0, 1) for _ in range(4)],
            }
        ),
        labels_path,
    )
    labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    index_path = tmp_path / "index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "DeepVoice": {
                        "repo_id": "example/DeepVoice",
                        "revision": "pinned-revision",
                        "local_dir": str(root),
                        "files": {"labels": "data/labels.parquet"},
                        "pinned_files": [{"path": "data/labels.parquet", "sha256": labels_hash}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows, provenance = build_score_free_input_from_index(index_path, "DeepVoice")
    input_csv = tmp_path / "h2b-input.csv"
    input_json = tmp_path / "h2b-input.json"
    write_score_free_input_artifacts(rows, provenance, output_csv=input_csv, output_provenance=input_json)
    return index_path, input_csv, input_json


def test_freezes_only_byte_bound_declared_score_blind_source(tmp_path: Path) -> None:
    _index, input_csv, input_json = _source_artifacts(tmp_path)
    frozen = freeze_h2b_q0_manifest(
        input_csv,
        input_json,
        dataset="DeepVoice",
        revision="pinned-revision",
        per_label=2,
        seed=2609,
    )
    assert len(frozen.rows) == 4
    assert frozen.rows.groupby("label").size().to_dict() == {0: 2, 1: 2}
    assert set(frozen.rows["h2b_stage"]) == {"q0_frozen_score_blind"}
    assert frozen.provenance["detector_scoring_allowed"] is False
    assert frozen.provenance["selection"]["detector_scores_read"] is False
    assert not any("score" in column.casefold() for column in frozen.rows.columns)


def test_rejects_csv_substitution_or_non_score_free_provenance(tmp_path: Path) -> None:
    _index, input_csv, input_json = _source_artifacts(tmp_path)
    input_csv.write_text(input_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        freeze_h2b_q0_manifest(
            input_csv, input_json, dataset="DeepVoice", revision="pinned-revision", per_label=2, seed=2609
        )

    _index, input_csv, input_json = _source_artifacts(tmp_path / "separate")
    provenance = json.loads(input_json.read_text(encoding="utf-8"))
    provenance["source_access"]["score_read"] = True
    input_json.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="score-free"):
        freeze_h2b_q0_manifest(
            input_csv, input_json, dataset="DeepVoice", revision="pinned-revision", per_label=2, seed=2609
        )


def test_q0_writer_refuses_overwrite_and_binds_output_bytes(tmp_path: Path) -> None:
    _index, input_csv, input_json = _source_artifacts(tmp_path)
    frozen = freeze_h2b_q0_manifest(
        input_csv, input_json, dataset="DeepVoice", revision="pinned-revision", per_label=2, seed=2609
    )
    manifest = tmp_path / "q0.csv"
    provenance = tmp_path / "q0.json"
    write_h2b_q0_artifacts(frozen, output_manifest=manifest, output_provenance=provenance)
    report = json.loads(provenance.read_text(encoding="utf-8"))
    assert report["output_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(pd.read_csv(manifest)) == 4
    with pytest.raises(FileExistsError, match="overwrite"):
        write_h2b_q0_artifacts(frozen, output_manifest=manifest, output_provenance=provenance)
