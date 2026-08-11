from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.h9_pcr_terminal_figure import render, validate_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path, *, bad_metrics_hash: bool = False) -> dict[str, tuple[Path, str]]:
    tmp_path.mkdir(parents=True)
    metrics = pd.DataFrame([
        {"dataset": dataset, "method": method, "n_trials": 10, "n_bonafide": 5, "n_spoof": 5, "eer": value / 100, "eer_percent": value, "auroc": 0.5, "score_aggregation": "mean_spoof_probability_over_four_frozen_seeds"}
        for dataset, values in (("SONAR", [58.0, 55.0, 47.0]), ("ArAD", [45.0, 48.0, 43.0]))
        for method, value in zip(("B1", "B2", "P"), values, strict=True)
    ])
    metrics_path = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    bootstrap = pd.DataFrame({"replicate": range(2_000), "macro_eer_difference_p_minus_b1": [-0.07] * 2_000, "macro_eer_difference_p_minus_b2": [-0.06] * 2_000})
    bootstrap_path = tmp_path / "bootstrap.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    decision = {
        "artifact_kind": "h9_pcr_terminal_decision_gate", "positive_result_gate_passed": True,
        "bootstrap": {"replicates": 2_000, "seed": 2909, "p_minus_b1": {"mean": -0.07, "ci_low": -0.08, "ci_high": -0.05}, "p_minus_b2": {"mean": -0.06, "ci_low": -0.07, "ci_high": -0.04}},
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    provenance = {
        "artifact_kind": "h9_pcr_terminal_target_evaluation", "version": "h9-pcr-terminal-evaluation-v1",
        "target_metrics_sha256": _sha(metrics_path), "bootstrap_sha256": _sha(bootstrap_path), "decision_gate_sha256": _sha(decision_path),
        "source_target_canonical_fingerprint_collision_count": 0, "evaluation_precision": "cuda_bfloat16_autocast", "evaluation_device": "cuda:3",
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    inputs = {"metrics": (metrics_path, _sha(metrics_path)), "bootstrap": (bootstrap_path, _sha(bootstrap_path)), "decision": (decision_path, _sha(decision_path)), "provenance": (provenance_path, _sha(provenance_path))}
    if bad_metrics_hash:
        inputs["metrics"] = (metrics_path, "0" * 64)
    return inputs


def test_validator_and_renderer_emit_vector_raster_and_metadata(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    metrics, decision, provenance, hashes = validate_inputs(inputs)
    output = tmp_path / "figure"
    generated = render(metrics, decision, provenance, output, input_hashes=hashes)
    assert (output / "h9_pcr_terminal_eer.pdf").is_file()
    assert (output / "h9_pcr_terminal_eer.png").is_file()
    metadata = json.loads((output / "h9_pcr_terminal_eer.metadata.json").read_text())
    assert metadata["output_hashes"]["pdf"] == generated["pdf"]
    assert metadata["terminal_precision"] == "cuda_bfloat16_autocast"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render(metrics, decision, provenance, output, input_hashes=hashes)


def test_validator_rejects_hash_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input hash mismatch"):
        validate_inputs(_inputs(tmp_path / "inputs", bad_metrics_hash=True))


def test_validator_rejects_missing_terminal_pass(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    decision_path, _ = inputs["decision"]
    decision = json.loads(decision_path.read_text())
    decision["positive_result_gate_passed"] = False
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    inputs["decision"] = (decision_path, _sha(decision_path))
    provenance_path, _ = inputs["provenance"]
    provenance = json.loads(provenance_path.read_text())
    provenance["decision_gate_sha256"] = _sha(decision_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    inputs["provenance"] = (provenance_path, _sha(provenance_path))
    with pytest.raises(ValueError, match="passing decision gate"):
        validate_inputs(inputs)
