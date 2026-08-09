from __future__ import annotations

import pandas as pd

from scripts.calibrate_h2_onnx_parity import choose_preprocessing, load_or_create_manifest, merge_resumable_summaries


def _dataset(tmp_path):
    labels_path = tmp_path / "data" / "labels.parquet"
    labels_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "utterance_id": [f"id-{index}" for index in range(12)],
            "label": [0] * 6 + [1] * 6,
        }
    ).to_parquet(labels_path, index=False)
    return {"revision": "pinned", "local_dir": str(tmp_path), "files": {"labels": "data/labels.parquet"}}


def test_calibration_manifest_is_frozen_and_stratified(tmp_path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    manifest_path = tmp_path / "out" / "calibration_manifest.parquet"
    first = load_or_create_manifest(manifest_path, dataset, seed=2609, per_label=4)
    second = load_or_create_manifest(manifest_path, dataset, seed=2609, per_label=4)
    assert first.equals(second)
    assert first.groupby("label").size().to_dict() == {0: 4, 1: 4}
    assert first["selection_seed"].unique().tolist() == [2609]


def test_preprocessing_tie_break_is_predeclared_and_raw_first() -> None:
    rows = [
        {"preprocessing": "raw", "eligible": True, "provider": "CUDAExecutionProvider"},
        {"preprocessing": "preemphasis_0.97", "eligible": True, "provider": "CUDAExecutionProvider"},
    ]
    choice = choose_preprocessing(rows)
    assert choice["status"] == "eligible"
    assert choice["chosen_preprocessing"] == "raw"
    assert choice["selection_rule"] == "predeclared_prefer_raw_when_it_passes_all_parity_gates"


def test_partial_model_rerun_preserves_other_calibration_summaries() -> None:
    existing = pd.DataFrame(
        [
            {"model": "AASIST", "preprocessing": "raw", "eligible": True},
            {"model": "Spectra-AASIST", "preprocessing": "raw", "eligible": False},
        ]
    )
    current = pd.DataFrame(
        [
            {"model": "Spectra-AASIST", "preprocessing": "raw", "eligible": True},
            {"model": "Spectra-AASIST", "preprocessing": "preemphasis_0.97", "eligible": False},
        ]
    )
    merged = merge_resumable_summaries(existing, current, ("Spectra-AASIST",))
    assert set(merged["model"]) == {"AASIST", "Spectra-AASIST"}
    assert len(merged) == 3
