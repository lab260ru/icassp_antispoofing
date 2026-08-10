"""Synthetic guardrails for the locked H6 raw-score agreement atlas."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.h6_score_agreement as h6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_index(
    root: Path,
    *,
    scores_missing_selected: tuple[str, str, int] | None = None,
    duplicate_score: tuple[str, str] | None = None,
    ambiguous_orientation: tuple[str, str] | None = None,
    duplicate_normalized_label: str | None = None,
) -> Path:
    """Build a complete local 5x8 fixture without reading a project artifact."""
    datasets: dict[str, object] = {}
    models: dict[str, object] = {}
    for dataset_index, dataset in enumerate(h6.DATASETS):
        dataset_root = root / "datasets" / dataset / "revision"
        labels_path = dataset_root / "data" / "labels.parquet"
        labels_path.parent.mkdir(parents=True)
        label_rows: list[dict[str, object]] = []
        for label in h6.LABELS:
            for sample_index in range(4):
                label_rows.append(
                    {
                        "utterance_id": f"{dataset}_label{label}_sample{sample_index}.wav",
                        "speaker_id": f"{dataset}_label{label}_speaker{sample_index // 2}",
                        "label": label,
                    }
                )
        if duplicate_normalized_label == dataset:
            label_rows.append(
                {
                    "utterance_id": f"{dataset}_label0_sample0.flac",
                    "speaker_id": f"{dataset}_duplicate_speaker",
                    "label": 0,
                }
            )
        pd.DataFrame(label_rows).to_parquet(labels_path, index=False)
        datasets[dataset] = {
            "repo_id": f"synthetic/{dataset}",
            "revision": f"dataset-revision-{dataset_index}",
            "local_dir": str(dataset_root),
            "files": {"labels": "data/labels.parquet"},
            "pinned_files": [{"path": "data/labels.parquet", "size_bytes": labels_path.stat().st_size, "sha256": _sha256(labels_path)}],
        }
    for model_index, model in enumerate(h6.MODELS):
        model_root = root / "models" / model / "revision"
        artifacts: dict[str, object] = {}
        for dataset_index, dataset in enumerate(h6.DATASETS):
            score_path = model_root / "scores" / dataset / "scores.txt"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            for label in h6.LABELS:
                for sample_index in range(4):
                    if scores_missing_selected == (dataset, model, label) and sample_index == 0:
                        continue
                    # The label term fixes the spoof orientation; the mixed
                    # sample/model term produces finite non-identical
                    # within-class rank relations for all model pairs.
                    raw = (
                        sample_index
                        if ambiguous_orientation == (dataset, model)
                        else 100.0 * label + ((sample_index * (model_index + 1) + dataset_index + model_index) % 11)
                    )
                    if model_index % 2:
                        raw = -raw
                    lines.append(f"{dataset}_label{label}_sample{sample_index}.wav {raw:.6f}")
            if duplicate_score == (dataset, model):
                lines.append(lines[0])
            score_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            artifacts[dataset] = {
                "scores": {
                    "path": str(score_path.relative_to(model_root)),
                    "size_bytes": score_path.stat().st_size,
                    "sha256": _sha256(score_path),
                }
            }
        models[model] = {
            "repo_id": f"synthetic/{model}",
            "revision": f"model-revision-{model_index}",
            "local_dir": str(model_root),
            "score_artifacts": artifacts,
        }
    index_path = root / "arena-index.yaml"
    index_path.write_text(yaml.safe_dump({"datasets": datasets, "models": models}, sort_keys=False), encoding="utf-8")
    return index_path


def _freeze(root: Path, **kwargs: object) -> tuple[Path, h6.FreezeArtifacts]:
    index_path = _synthetic_index(root / "inputs", **kwargs)
    freeze = h6.freeze_inputs(
        root / "freeze",
        index_path=index_path,
        allowed_index_path=index_path,
        frozen_at_utc="2026-08-10T00:00:00Z",
    )
    return index_path, freeze


def test_h6_label_only_freeze_has_independent_seed_and_never_parses_score_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = _synthetic_index(tmp_path / "inputs")
    monkeypatch.setattr(h6, "_read_raw_scores", lambda *_: (_ for _ in ()).throw(AssertionError("freeze must not parse raw scores")))
    frozen = h6.freeze_inputs(tmp_path / "freeze", index_path=index_path, allowed_index_path=index_path, frozen_at_utc="2026-08-10T00:00:00Z")
    provenance = json.loads(frozen.provenance_path.read_text())
    manifest = pd.read_parquet(frozen.manifest_path)
    assert h6.SEED == 2611
    assert provenance["raw_score_values_read_during_freeze"] is False
    assert provenance["selection_seed"] == 2611
    assert len(provenance["score_artifacts"]) == len(h6.DATASETS) * len(h6.MODELS) == 40
    assert len(manifest) == len(h6.DATASETS) * len(h6.LABELS) * 4
    assert manifest.columns.tolist() == ["dataset", "label", "sample_id", "bootstrap_cluster_id", "selection_rank", "selection_key_sha256"]
    assert (manifest.groupby(["dataset", "label"]).size() == 4).all()


def test_h6_full_synthetic_analysis_materializes_exact_280_cells_and_28_summaries(tmp_path: Path) -> None:
    index_path, frozen = _freeze(tmp_path)
    matrix_path, aggregation_path, report_path = h6.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_index_path=index_path,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    assert len(h6.MODEL_PAIRS) == 28
    assert len(matrix) == len(h6.DATASETS) * len(h6.LABELS) * len(h6.MODEL_PAIRS) == 280
    assert len(aggregation) == len(h6.MODEL_PAIRS) == 28
    assert not matrix.duplicated(["dataset", "label", "model_a", "model_b"]).any()
    assert set(matrix["cell_status"]) == {"ok"}
    assert set(matrix["bootstrap_replicates_requested"]) == {h6.BOOTSTRAP_REPLICATES}
    assert (matrix["bootstrap_replicates_valid"] > 0).all()
    assert matrix["bootstrap_seed_sha256"].nunique() == len(matrix)
    assert np.isfinite(matrix["spearman_agreement"]).all()
    assert aggregation["all_ten_cells_ok"].all()
    assert report["raw_score_only"] is True
    assert report["score_values_emitted"] is False
    assert report["complete_280_cell_matrix"] is True
    assert report["analysis_status"] == "complete"
    orientation = {(row["dataset"], row["model"]): row["orientation"] for row in report["orientation_records"]}
    assert orientation[(h6.DATASETS[0], h6.MODELS[0])] == "raw_is_spoof"
    assert orientation[(h6.DATASETS[0], h6.MODELS[1])] == "negated_raw_is_spoof"


def test_h6_missing_score_is_explicit_and_never_intersection_reselects(tmp_path: Path) -> None:
    dataset, missing_model, label = h6.DATASETS[0], h6.MODELS[0], 1
    index_path, frozen = _freeze(tmp_path, scores_missing_selected=(dataset, missing_model, label))
    matrix_path, aggregation_path, report_path = h6.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_index_path=index_path,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    affected = matrix.loc[
        (matrix["dataset"] == dataset)
        & (matrix["label"] == label)
        & ((matrix["model_a"] == missing_model) | (matrix["model_b"] == missing_model))
    ]
    assert len(matrix) == 280 and len(aggregation) == 28
    assert len(affected) == len(h6.MODELS) - 1
    assert affected["cell_status"].str.contains("missing_selected_raw_score").all()
    assert (affected["exact_joined_sample_count"] == 0).all()
    assert report["analysis_status"] == "stopped_with_explicit_unavailable_cells"
    assert report["stop_reason"] == "raw_score_validation_or_unavailable_cell"


def test_h6_duplicate_raw_ids_are_unavailable_cells_but_preserve_registry(tmp_path: Path) -> None:
    dataset, duplicate_model = h6.DATASETS[-1], h6.MODELS[-1]
    index_path, frozen = _freeze(tmp_path, duplicate_score=(dataset, duplicate_model))
    matrix_path, _, report_path = h6.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_index_path=index_path,
    )
    matrix = pd.read_csv(matrix_path)
    report = json.loads(report_path.read_text())
    affected = matrix.loc[
        (matrix["dataset"] == dataset) & ((matrix["model_a"] == duplicate_model) | (matrix["model_b"] == duplicate_model))
    ]
    assert len(matrix) == 280
    assert len(affected) == 2 * (len(h6.MODELS) - 1)
    assert affected["cell_status"].str.contains("duplicate_model_sample_id").all()
    assert report["complete_280_cell_matrix"] is True


def test_h6_ambiguous_orientation_is_a_stopped_explicit_matrix(tmp_path: Path) -> None:
    dataset, ambiguous_model = h6.DATASETS[1], h6.MODELS[2]
    index_path, frozen = _freeze(tmp_path, ambiguous_orientation=(dataset, ambiguous_model))
    matrix_path, _, report_path = h6.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_index_path=index_path,
    )
    matrix = pd.read_csv(matrix_path)
    report = json.loads(report_path.read_text())
    affected = matrix.loc[
        (matrix["dataset"] == dataset) & ((matrix["model_a"] == ambiguous_model) | (matrix["model_b"] == ambiguous_model))
    ]
    assert len(matrix) == 280
    assert len(affected) == 2 * (len(h6.MODELS) - 1)
    assert affected["cell_status"].str.contains("ambiguous_spoof_orientation").all()
    assert report["orientation_records"][h6.DATASETS.index(dataset) * len(h6.MODELS) + h6.MODELS.index(ambiguous_model)]["validation_status"] == "ambiguous_spoof_orientation"


def test_h6_invalid_label_freeze_records_all_failed_cells_without_opening_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path, frozen = _freeze(tmp_path, duplicate_normalized_label=h6.DATASETS[0])
    provenance = json.loads(frozen.provenance_path.read_text())
    assert provenance["freeze_status"] == "failed_input_integrity"
    assert h6.DATASETS[0] in provenance["failed_datasets"]
    monkeypatch.setattr(h6, "_read_raw_scores", lambda *_: (_ for _ in ()).throw(AssertionError("invalid label freeze must stop before score parsing")))
    matrix_path, aggregation_path, report_path = h6.analyze_frozen_manifest(
        frozen.provenance_path,
        tmp_path / "analysis",
        allowed_index_path=index_path,
    )
    matrix = pd.read_csv(matrix_path)
    aggregation = pd.read_csv(aggregation_path)
    report = json.loads(report_path.read_text())
    assert len(matrix) == 280 and len(aggregation) == 28
    assert set(matrix["cell_status"]) == {"failed_label_or_score_input_freeze"}
    assert report["stop_reason"] == "input_freeze_not_complete"


def test_h6_hash_and_manifest_guards_fail_before_score_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path, frozen = _freeze(tmp_path)
    provenance = json.loads(frozen.provenance_path.read_text())
    score_path = Path(provenance["score_artifacts"][0]["absolute_path"])
    score_path.write_text(score_path.read_text() + "extra.wav 1.0\n", encoding="utf-8")
    monkeypatch.setattr(h6, "_read_raw_scores", lambda *_: (_ for _ in ()).throw(AssertionError("hash guard must run before parsing")))
    with pytest.raises(ValueError, match="raw score artifact changed"):
        h6.analyze_frozen_manifest(frozen.provenance_path, tmp_path / "analysis_changed", allowed_index_path=index_path)

    # A new clean freeze lets the second branch isolate selection-manifest path
    # tampering rather than treating it as a score-data issue.
    clean_index, clean_frozen = _freeze(tmp_path / "clean")
    clean = json.loads(clean_frozen.provenance_path.read_text())
    clean["selection_manifest"]["absolute_path"] = str(tmp_path / "outside" / "manifest.parquet")
    clean_frozen.provenance_path.write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(ValueError, match="selection manifest path"):
        h6.analyze_frozen_manifest(clean_frozen.provenance_path, tmp_path / "clean_analysis", allowed_index_path=clean_index)


def test_h6_has_no_waveform_feature_or_prior_loop_imports() -> None:
    module = ast.parse((Path(h6.__file__)).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = ("audio", "feature", "h1", "h2", "h4", "arena_io", "waveform")
    assert not [name for name in imported if any(term in name.casefold() for term in forbidden)]
