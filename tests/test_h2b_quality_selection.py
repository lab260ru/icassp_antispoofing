from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.h2b_quality_selection import select_h2b_q1_families, wilson_lower_bound, write_h2b_q1_selection


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    arms = [
        {
            "arm_id": "candidate_a",
            "transform": "polarity",
            "parameters": {},
            "negative_control": False,
            "target_feature": "feature",
            "target_direction": "increase",
            "target_tolerance": 0.0,
        },
        {
            "arm_id": "candidate_b",
            "transform": "gain",
            "parameters": {"gain_db": 0.05},
            "negative_control": False,
            "target_feature": "feature",
            "target_direction": "increase",
            "target_tolerance": 0.0,
        },
        {
            "arm_id": "control",
            "transform": "polarity",
            "parameters": {},
            "negative_control": True,
            "target_feature": "feature",
            "target_direction": "invariant",
            "target_tolerance": 1e-6,
        },
    ]
    from src.h2_pre_score_pairs import ArmDefinition

    for arm in arms:
        arm["definition_sha256"] = ArmDefinition(**{key: arm[key] for key in arm if key != "definition_sha256"}).as_record()["definition_sha256"]
    manifest = tmp_path / "arms.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "h2b_q1_quality_arm_manifest",
                "version": "h2b_q1_quality_arm_manifest_v1",
                "claim_guard": "detector-free",
                "arms": arms,
                "negative_control_arm_ids": ["control"],
                "families": [
                    {
                        "family_id": "family",
                        "candidate_arm_ids": ["candidate_a", "candidate_b"],
                        "minimum_abs_median_target_delta": 0.1,
                        "target_feature": "feature",
                    }
                ],
                "selection_policy": {
                    "minimum_lower_wilson_retention_bound": 0.9,
                    "wilson_confidence_level": 0.95,
                    "tie_break_order": ["largest_lower_wilson_retention_bound"],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for arm, delta, wer in (("candidate_a", 0.2, 0.01), ("candidate_b", 0.4, 0.03), ("control", 0.0, 0.01)):
        for index in range(100):
            rows.append(
                {
                    "pair_id": f"{arm}-{index}",
                    "arm": arm,
                    "retained": True,
                    "target_feature": "feature",
                    "target_feature_before": 1.0,
                    "target_feature_after": 1.0 + delta,
                    "wer": wer,
                }
            )
    table = tmp_path / "quality.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "artifact_kind": "h2b_q1_detector_free_quality_summary",
                "detector_scoring_allowed": False,
                "complete": True,
                "completed_pairs": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return table, summary, manifest


def test_wilson_bound_and_quality_only_selection_tie_break(tmp_path: Path) -> None:
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(256, 256) > 0.98
    table, summary, manifest = _write_inputs(tmp_path)
    report, provenance = select_h2b_q1_families(table, summary, manifest)
    selected = report.loc[report["selected"]]
    # Both candidates pass the retention bound; b wins the fixed larger-delta tie break.
    assert selected[["family_id", "arm"]].to_dict(orient="records") == [{"family_id": "family", "arm": "candidate_b"}]
    assert provenance["detector_scoring_allowed"] is False
    assert "score" not in " ".join(report.columns).casefold()


def test_selector_requires_complete_summary_and_non_overwrite_output(tmp_path: Path) -> None:
    table, summary, manifest = _write_inputs(tmp_path)
    invalid = json.loads(summary.read_text())
    invalid["complete"] = False
    summary.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        select_h2b_q1_families(table, summary, manifest)
    _table, summary, manifest = _write_inputs(tmp_path / "fresh")
    report, provenance = select_h2b_q1_families(_table, summary, manifest)
    output = tmp_path / "selection"
    write_h2b_q1_selection(report, provenance, output_dir=output)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_h2b_q1_selection(report, provenance, output_dir=output)
