from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.h2b_quality_calibration import h2b_q1_quality_paths, validate_h2b_q1_inputs


def _write_q0(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    q0 = tmp_path / "q0.csv"
    rows = pd.DataFrame(
        {
            "dataset": ["DeepVoice", "DeepVoice"],
            "sample_id": ["u0", "u1"],
            "source_id": ["u0.wav", "u1.wav"],
            "label": [0, 1],
            "dataset_revision": ["rev", "rev"],
            "selection_key_sha256": ["a" * 64, "b" * 64],
            "selection_rank": [1, 1],
            "manifest_version": ["h2_pre_score_pairs_v1", "h2_pre_score_pairs_v1"],
            "selection_seed": [2609, 2609],
            "selection_method": ["sha256", "sha256"],
            "source_input_csv_sha256": ["c" * 64, "c" * 64],
            "h2b_stage": ["q0_frozen_score_blind", "q0_frozen_score_blind"],
            "h2b_manifest_version": ["h2b_q0_score_blind_manifest_v1", "h2b_q0_score_blind_manifest_v1"],
        }
    )
    rows.to_csv(q0, index=False)
    q0_provenance = tmp_path / "q0.json"
    q0_provenance.write_text(
        json.dumps(
            {
                "artifact_kind": "h2b_q0_score_blind_input_freeze",
                "h2b_stage": "q0_frozen_score_blind",
                "detector_scoring_allowed": False,
                "declared_source": {"dataset": "DeepVoice", "revision": "rev"},
                "output_manifest": {"sha256": hashlib.sha256(q0.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    arm = {
        "arm_id": "polarity",
        "transform": "polarity",
        "parameters": {},
        "negative_control": True,
        "target_feature": "crest_factor_db",
        "target_direction": "invariant",
        "target_tolerance": 1e-6,
    }
    from src.h2_pre_score_pairs import ArmDefinition

    arm["definition_sha256"] = ArmDefinition(**{key: arm[key] for key in arm if key != "definition_sha256"}).as_record()["definition_sha256"]
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps(
            {
                "artifact_kind": "h2b_q1_quality_arm_manifest",
                "version": "h2b_q1_quality_arm_manifest_v1",
                "claim_guard": "score-blinded",
                "arms": [arm],
                "negative_control_arm_ids": ["polarity"],
                "families": [],
                "selection_policy": {"wilson_confidence_level": 0.95, "minimum_lower_wilson_retention_bound": 0.9},
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.yaml"
    index.write_text(yaml.safe_dump({"datasets": {"DeepVoice": {"revision": "rev", "repo_id": "example/DeepVoice"}}}), encoding="utf-8")
    return q0, q0_provenance, arms, index


def test_rejects_empty_family_list_and_manifest_hash_drift(tmp_path: Path) -> None:
    q0, provenance, arms, index = _write_q0(tmp_path)
    with pytest.raises(ValueError, match="selection families"):
        validate_h2b_q1_inputs(q0, provenance, arms, index)
    q0.write_text(q0.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_h2b_q1_inputs(q0, provenance, arms, index)


def test_paths_are_h2b_namespaced_and_reject_invalid_id(tmp_path: Path) -> None:
    paths = h2b_q1_quality_paths(tmp_path / "hdd", "h2b_q1_001", tmp_path / "repo")
    assert paths.run_root == (tmp_path / "hdd" / "runs" / "h2b_quality_calibration" / "h2b_q1_001").resolve()
    assert "future_directions" in str(paths.repo_summary_path)
    with pytest.raises(ValueError, match="run ID"):
        h2b_q1_quality_paths(tmp_path, "bad/id", tmp_path)
