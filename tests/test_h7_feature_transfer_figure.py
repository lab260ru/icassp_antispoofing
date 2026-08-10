from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.h7_feature_transfer import CORPORA
from src.h7_feature_transfer_figure import render, validate_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path, *, bad_header: bool = False) -> dict[str, tuple[Path, str]]:
    tmp_path.mkdir(parents=True)
    freeze_manifest = tmp_path / "freeze.csv"
    freeze_manifest.write_text("sample_id\nsynthetic\n", encoding="utf-8")
    freeze_provenance = tmp_path / "freeze.json"
    freeze_provenance.write_text(json.dumps({"reads": {"score": False, "feature_values": False, "audio": False, "asr": False, "model": False}}), encoding="utf-8")
    matrix = pd.DataFrame(
        {
            "held_out_dataset": list(CORPORA),
            "n_train": [40000] * 5,
            "n_test": [10000] * 5,
            "auroc": [0.91, 0.85, 0.78, 0.53, 0.59],
            # Explicitly declared derived field; it is allowed but not plotted.
            "eer": [0.16, 0.20, 0.29, 0.45, 0.44],
            "auroc_ci_low": [0.90, 0.84, 0.77, 0.52, 0.58],
            "auroc_ci_high": [0.92, 0.86, 0.79, 0.54, 0.60],
            "fit_converged": [True] * 5,
            "bootstrap_replicates_requested": [500] * 5,
            "bootstrap_replicates_valid": [500] * 5,
        }
    )
    if bad_header:
        matrix["score_spoof"] = 0.0
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "complete_matrix_cells": 5,
                "representation": "all_28_full_waveform_features",
                "freeze_manifest_sha256": _sha(freeze_manifest),
                "freeze_provenance_sha256": _sha(freeze_provenance),
                "claims_not_supported": ["detector_reliance", "causality", "cue_selection", "model_ranking", "mitigation_performance"],
            }
        ),
        encoding="utf-8",
    )
    return {
        "freeze_manifest": (freeze_manifest, _sha(freeze_manifest)),
        "freeze_provenance": (freeze_provenance, _sha(freeze_provenance)),
        "matrix": (matrix_path, _sha(matrix_path)),
        "analysis_provenance": (analysis, _sha(analysis)),
    }


def test_validator_and_renderer_create_vector_and_raster_outputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    table, analysis, hashes = validate_inputs(inputs)
    output = tmp_path / "figure"
    generated = render(table, output, input_hashes=hashes, analysis=analysis)
    assert (output / "h7_feature_transfer_auroc.pdf").is_file()
    assert (output / "h7_feature_transfer_auroc.png").is_file()
    metadata = json.loads((output / "h7_feature_transfer_auroc.metadata.json").read_text())
    assert metadata["chance_reference"] == 0.5
    assert metadata["output_hashes"]["pdf"] == generated["pdf"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render(table, output, input_hashes=hashes, analysis=analysis)


def test_validator_rejects_matrix_hash_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    path, digest = inputs["matrix"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hash mismatch"):
        validate_inputs(inputs)


def test_validator_rejects_response_header(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs", bad_header=True)
    with pytest.raises(ValueError, match="response-like matrix headers"):
        validate_inputs(inputs)
