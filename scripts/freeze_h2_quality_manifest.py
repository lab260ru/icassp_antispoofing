#!/usr/bin/env python3
"""Freeze a complete detector-free H2 quality run for downstream scoring.

This command validates only committed pre-score inputs and an already-complete
quality-run directory.  It does not decode audio, invoke ASR, load a detector,
or calculate detector outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_quality_freeze import validate_completed_quality_run, write_score_eligible_freeze
from src.h2_quality_runner import assert_git_clean_and_committed, validate_frozen_inputs


DEFAULT_FROZEN_ROOT = (
    REPO_ROOT
    / "experiments"
    / "h2_causal_interventions"
    / "results"
    / "pre_score"
    / "h2_crest_pre_score_001_20260809T205114Z"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a completed H2 quality panel for detector scoring")
    parser.add_argument("--quality-run-dir", required=True, help="Completed HDD quality-run directory; never an active/pilot run")
    parser.add_argument("--input-manifest", default=str(DEFAULT_FROZEN_ROOT / "input_manifest.csv"))
    parser.add_argument("--arm-ledger", default=str(DEFAULT_FROZEN_ROOT / "arm_ledger.json"))
    parser.add_argument("--arena-index", default=str(REPO_ROOT / "data" / "arena-index.yaml"))
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New repository-relative directory for score_eligible_pairs.{csv,parquet} and quality_freeze.json",
    )
    return parser.parse_args()


def _repository_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise SystemExit(f"Output directory must be inside the repository: {resolved}") from error
    return resolved


def main() -> None:
    args = _arguments()
    manifest = Path(args.input_manifest).resolve()
    ledger = Path(args.arm_ledger).resolve()
    index = Path(args.arena_index).resolve()
    output_dir = _repository_path(args.output_dir)
    # This verifies that the freeze's source selection and arm definitions are
    # committed protocol artifacts, rather than a local post-hoc edit.
    assert_git_clean_and_committed((manifest, ledger, index), repo_root=REPO_ROOT)
    validated = validate_frozen_inputs(manifest, ledger, index)
    completed = validate_completed_quality_run(args.quality_run_dir, validated_inputs=validated)
    outputs = write_score_eligible_freeze(completed, output_dir)
    print(
        json.dumps(
            {
                "quality_run_id": completed.run_id,
                "output_dir": str(outputs.output_dir),
                "csv": str(outputs.csv_path),
                "parquet": str(outputs.parquet_path),
                "provenance": str(outputs.report_path),
                "score_eligible_pairs": outputs.report["score_eligible_pairs"],
                "detector_scoring_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
