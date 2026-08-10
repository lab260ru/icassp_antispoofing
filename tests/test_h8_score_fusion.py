"""Synthetic contract tests for the source-only H8-SF freezer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.h8_score_fusion import (
    freeze_source_inputs,
    materialize_label_free_target_features,
    materialize_source_features,
    normalize_sample_id,
)


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


def test_target_features_are_label_free_and_use_full_score_ranks(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    source = freeze_source_inputs(
        index_path=index_path,
        output_dir=tmp_path / "hdd" / "source",
        datasets=("source_a", "source_b"),
        models=("model_a", "model_b"),
    )
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    for model in ("model_a", "model_b"):
        root = Path(index["models"][model]["local_dir"])
        relative = "scores/target.txt"
        values = {"a": 0.0, "b": 1.0, "c": 2.0}
        if model == "model_b":
            values = {key: -value for key, value in values.items()}
        _write_scores(root / relative, values)
        index["models"][model]["score_artifacts"]["target"] = {
            "scores": {"path": relative, "size_bytes": (root / relative).stat().st_size}
        }
    index["datasets"]["target"] = {"n_trials": 3}
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    outputs = materialize_label_free_target_features(
        index_path=index_path,
        orientation_path=source.orientation_path,
        output_dir=tmp_path / "hdd" / "target",
        datasets=("target",),
        models=("model_a", "model_b"),
    )
    features = pd.read_parquet(outputs.features_path)
    provenance = json.loads(outputs.provenance_path.read_text(encoding="utf-8"))
    assert list(features["sample_id"]) == ["a", "b", "c"]
    assert provenance["target_labels_read"] is False
    assert provenance["target_metrics_read"] is False
    assert provenance["target_score_artifacts"]["target"]["common_panel"]["n_common_rows"] == 3


def test_target_features_refuse_nonexistent_orientation_model(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    orientation = tmp_path / "orientation.json"
    orientation.write_text(json.dumps({"version": "h8sf_source_freeze_v1", "models": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="orientation"):
        materialize_label_free_target_features(
            index_path=index_path,
            orientation_path=orientation,
            output_dir=tmp_path / "hdd" / "target",
            datasets=("source_a",),
            models=("model_a",),
        )


def test_normalize_sample_id_only_strips_terminal_audio_extension() -> None:
    assert normalize_sample_id("dir/CVF_de_vctk_multi_band_melgan.v2_generated_common_voice_de_1_Gen") == "CVF_de_vctk_multi_band_melgan.v2_generated_common_voice_de_1_Gen"
    assert normalize_sample_id("dir/clip.wav") == "clip"


def test_source_features_revalidate_freeze_and_write_no_target_access(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    source = freeze_source_inputs(
        index_path=index_path,
        output_dir=tmp_path / "hdd" / "source_freeze",
        datasets=("source_a", "source_b"),
        models=("model_a", "model_b"),
    )
    outputs = materialize_source_features(
        index_path=index_path,
        source_manifest_path=source.manifest_path,
        source_orientation_path=source.orientation_path,
        source_provenance_path=source.provenance_path,
        output_dir=tmp_path / "hdd" / "source_features",
    )
    features = pd.read_parquet(outputs.features_path)
    provenance = json.loads(outputs.provenance_path.read_text(encoding="utf-8"))
    assert len(features) == 12
    assert set(features["label"]) == {0, 1}
    assert provenance["target_labels_read"] is False
    assert provenance["target_scores_read"] is False
