from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.h2_pre_score_pairs import arm_definitions_frame, freeze_input_manifest, quality_rows_frame
from src.h2_quality_freeze import (
    ELIGIBLE_CSV_NAME,
    FREEZE_REPORT_NAME,
    MINIMUM_ARM_RETAINED_FRACTION,
    validate_completed_quality_run,
    write_score_eligible_freeze,
)
from src.h2_quality_runner import (
    QUALITY_RUNNER_VERSION,
    RUN_PROVENANCE_KIND,
    _base_failure_row,
    build_quality_summary,
    persist_pair_checkpoint,
    validate_frozen_inputs,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_frame(count: int = 10) -> pd.DataFrame:
    source = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "sample_id": f"clip_{index:02d}",
                "source_id": f"source_{index:02d}",
                "label": 0,
                "dataset_revision": "rev-toy",
            }
            for index in range(count)
        ]
    )
    return freeze_input_manifest(source, per_label=count, seed=2609, input_csv_sha256="a" * 64).rows


def _write_frozen_inputs(root: Path, count: int = 10) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "input_manifest.csv"
    ledger = root / "arm_ledger.json"
    index = root / "arena-index.yaml"
    _manifest_frame(count).to_csv(manifest, index=False)
    ledger.write_text(json.dumps(arm_definitions_frame().to_dict(orient="records"), indent=2), encoding="utf-8")
    index.write_text(
        yaml.safe_dump({"datasets": {"toy": {"repo_id": "example/toy", "revision": "rev-toy"}}}),
        encoding="utf-8",
    )
    return manifest, ledger, index


def _passed_row(manifest_row: dict[str, object], arm: object) -> dict[str, object]:
    row = _base_failure_row(manifest_row, arm, "placeholder")
    pair_id = str(row["pair_id"])
    row.update(
        {
            "retained": True,
            "quality_status": "passed",
            "failure_reasons_json": "[]",
            "pass_stoi": True,
            "pass_wer": True,
            "pass_loudness": True,
            "pass_clipping": True,
            "pass_target_direction": True,
            "sample_rate_hz": 16_000,
            "original_samples": 16_000,
            "transformed_samples": 16_000,
            "original_wave_sha256": _sha256(f"original:{pair_id}"),
            "transformed_wave_sha256": _sha256(f"transformed:{pair_id}"),
            "stoi": 0.99,
            "wer": 0.0,
            "original_lufs": -24.0,
            "transformed_lufs": -24.05,
            "loudness_delta_lu": -0.05,
            "original_clipping_fraction": 0.0,
            "transformed_clipping_fraction": 0.0,
            "added_clipping_fraction": 0.0,
            "target_feature_before": 9.0,
            "target_feature_after": 6.0 if arm.target_direction == "decrease" else 9.0,
            "all_feature_deltas_json": '{"crest_factor_db":-3.0}',
        }
    )
    return row


def _run_contract(run_id: str, validated: object, *, mode: str = "full_panel_pre_score_quality") -> dict[str, object]:
    return {
        "artifact_kind": RUN_PROVENANCE_KIND,
        "version": QUALITY_RUNNER_VERSION,
        "run_id": run_id,
        "mode": mode,
        "limit_samples": 1 if mode == "pilot_not_panel_gate" else None,
        "detector_scoring_allowed": False,
        "panel_gate_eligible": False if mode == "pilot_not_panel_gate" else None,
        "manifest_sha256": validated.manifest_sha256,
        "arm_ledger_sha256": validated.arm_ledger_sha256,
        "arena_index_sha256": validated.arena_index_sha256,
        "manifest_version": "h2_pre_score_pairs_v1",
        "arms": {arm.arm_id: arm.as_record()["definition_sha256"] for arm in validated.arms},
        "datasets": {
            name: {
                "repo_id": value.get("repo_id"),
                "revision": value.get("revision"),
                "source_revision_resolved": value.get("source_revision_resolved"),
            }
            for name, value in sorted(validated.datasets.items())
        },
    }


def _write_quality_run(
    root: Path,
    *,
    retained_per_arm: int = 9,
    mode: str = "full_panel_pre_score_quality",
    omit_one_pair: bool = False,
) -> tuple[Path, object]:
    manifest_path, ledger, index = _write_frozen_inputs(root)
    validated = validate_frozen_inputs(manifest_path, ledger, index)
    run_id = "quality_full_001" if mode == "full_panel_pre_score_quality" else "quality_pilot_001"
    run_dir = root / run_id
    pair_dir = run_dir / "pair_rows"
    pair_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for sample_index, manifest_row in enumerate(validated.manifest.to_dict(orient="records")):
        for arm in validated.arms:
            if omit_one_pair and sample_index == 0 and arm.arm_id == validated.arms[0].arm_id:
                continue
            if sample_index < retained_per_arm:
                row = _passed_row(manifest_row, arm)
            else:
                row = _base_failure_row(manifest_row, arm, "quality_gate_failed")
            persist_pair_checkpoint(pair_dir / f"{row['pair_id']}.json", row)
            rows.append(row)
    table = quality_rows_frame(rows)
    table.to_parquet(run_dir / "quality_pairs.parquet", index=False)
    contract = _run_contract(run_id, validated, mode=mode)
    (run_dir / "quality_provenance.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = build_quality_summary(table, len(validated.manifest) * len(validated.arms), contract)
    (run_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir, validated


def test_completed_run_at_exact_90_percent_writes_retained_only_non_overwriting_freeze(tmp_path: Path) -> None:
    run_dir, validated = _write_quality_run(tmp_path, retained_per_arm=9)
    completed = validate_completed_quality_run(run_dir, validated_inputs=validated)
    assert {details["retained_fraction"] for details in completed.per_arm.values()} == {MINIMUM_ARM_RETAINED_FRACTION}

    output_dir = tmp_path / "repo" / "experiments" / "quality_freezes" / "quality_full_001"
    outputs = write_score_eligible_freeze(completed, output_dir)
    assert outputs.csv_path.name == ELIGIBLE_CSV_NAME
    assert outputs.report_path.name == FREEZE_REPORT_NAME
    frozen = pd.read_csv(outputs.csv_path)
    assert len(frozen) == 9 * 4
    assert frozen["retained"].all()
    assert frozen["quality_status"].eq("passed").all()
    assert frozen["detector_stage"].eq("eligible_after_quality_freeze").all()
    assert set(frozen["original_wave_sha256"].str.len()) == {64}
    report = json.loads(outputs.report_path.read_text(encoding="utf-8"))
    assert report["detector_scoring_performed"] is False
    assert report["detector_scoring_allowed"] is True
    assert report["score_eligible_pairs"] == len(frozen)
    assert report["score_eligible_manifest_logical_sha256"]
    assert report["score_eligible_manifest_sha256"] == _sha256(outputs.csv_path.read_text(encoding="utf-8"))
    assert report["score_eligible_parquet_sha256"] == hashlib.sha256(outputs.parquet_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_score_eligible_freeze(completed, output_dir)


def test_freeze_refuses_pilot_and_incomplete_quality_runs(tmp_path: Path) -> None:
    pilot_dir, pilot_inputs = _write_quality_run(tmp_path / "pilot", mode="pilot_not_panel_gate")
    with pytest.raises(ValueError, match="Pilot"):
        validate_completed_quality_run(pilot_dir, validated_inputs=pilot_inputs)

    incomplete_dir, incomplete_inputs = _write_quality_run(tmp_path / "incomplete", omit_one_pair=True)
    with pytest.raises(ValueError, match="incomplete"):
        validate_completed_quality_run(incomplete_dir, validated_inputs=incomplete_inputs)


def test_freeze_refuses_failed_arm_gate_and_duplicate_checkpoint(tmp_path: Path) -> None:
    failed_dir, failed_inputs = _write_quality_run(tmp_path / "failed", retained_per_arm=8)
    with pytest.raises(ValueError, match="90% retained-pair gate"):
        validate_completed_quality_run(failed_dir, validated_inputs=failed_inputs)

    duplicate_dir, duplicate_inputs = _write_quality_run(tmp_path / "duplicate", retained_per_arm=9)
    source = next((duplicate_dir / "pair_rows").glob("*.json"))
    alias = duplicate_dir / "pair_rows" / "duplicate.json"
    alias.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate quality pair checkpoint"):
        validate_completed_quality_run(duplicate_dir, validated_inputs=duplicate_inputs)
