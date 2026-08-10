"""Synthetic contract tests for the source-only H8-SF freezer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.h8_score_fusion import freeze_source_inputs


def _write_scores(path: Path, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}.wav {value}\n" for key, value in values.items()), encoding="utf-8")


def _index(tmp_path: Path, *, reverse_second_model_in_second_source: bool = False) -> Path:
    datasets: dict[str, object] = {}
    models: dict[str, object] = {}
    names = ("source_a", "source_b")
    for dataset_index, dataset in enumerate(names):
        root = tmp_path / "datasets" / dataset
        (root / "data").mkdir(parents=True)
        labels = pd.DataFrame(
            {
                "utterance_id": [f"{dataset}_{index}.wav" for index in range(6)],
                "label": [0, 0, 0, 1, 1, 1],
            }
        )
        labels.to_parquet(root / "data" / "labels.parquet", index=False)
        datasets[dataset] = {"local_dir": str(root), "files": {"labels": "data/labels.parquet"}, "revision": f"rev-{dataset_index}"}
    for model_index, model in enumerate(("model_a", "model_b")):
        root = tmp_path / "models" / model
        artifacts: dict[str, object] = {}
        for dataset_index, dataset in enumerate(names):
            values = {f"{dataset}_{index}": float(index if index < 3 else index + 5) for index in range(6)}
            if model_index == 1:
                values = {key: -value for key, value in values.items()}
            if reverse_second_model_in_second_source and model_index == 1 and dataset_index == 1:
                values = {key: -value for key, value in values.items()}
            relative = f"scores/{dataset}.txt"
            _write_scores(root / relative, values)
            artifacts[dataset] = {"scores": {"path": relative, "size_bytes": (root / relative).stat().st_size}}
        models[model] = {"local_dir": str(root), "revision": f"model-rev-{model_index}", "score_artifacts": artifacts}
    index_path = tmp_path / "arena-index.yaml"
    index_path.write_text(yaml.safe_dump({"datasets": datasets, "models": models}), encoding="utf-8")
    return index_path


def test_source_freeze_writes_only_source_provenance_and_stable_orientation(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    output_dir = tmp_path / "hdd" / "freeze"
    outputs = freeze_source_inputs(
        index_path=index_path,
        output_dir=output_dir,
        datasets=("source_a", "source_b"),
        models=("model_a", "model_b"),
    )
    manifest = pd.read_csv(outputs.manifest_path)
    provenance = json.loads(outputs.provenance_path.read_text(encoding="utf-8"))
    orientation = json.loads(outputs.orientation_path.read_text(encoding="utf-8"))
    assert len(manifest) == 12
    assert provenance["target_labels_read"] is False
    assert provenance["target_scores_read"] is False
    assert provenance["target_metrics_read"] is False
    assert orientation["models"]["model_a"]["orientation_multiplier"] == 1
    assert orientation["models"]["model_b"]["orientation_multiplier"] == -1
    assert set(manifest["selection_rank"]) == {1, 2, 3}


def test_source_freeze_rejects_cross_source_orientation_flip(tmp_path: Path) -> None:
    index_path = _index(tmp_path, reverse_second_model_in_second_source=True)
    with pytest.raises(ValueError, match="orientation changes"):
        freeze_source_inputs(
            index_path=index_path,
            output_dir=tmp_path / "hdd" / "freeze",
            datasets=("source_a", "source_b"),
            models=("model_a", "model_b"),
        )


def test_source_freeze_refuses_output_overwrite(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    output_dir = tmp_path / "hdd" / "freeze"
    freeze_source_inputs(index_path=index_path, output_dir=output_dir, datasets=("source_a", "source_b"), models=("model_a", "model_b"))
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        freeze_source_inputs(index_path=index_path, output_dir=output_dir, datasets=("source_a", "source_b"), models=("model_a", "model_b"))
