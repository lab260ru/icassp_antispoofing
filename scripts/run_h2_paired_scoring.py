#!/usr/bin/env python3
"""Run one pinned H2 paired scorer after a committed quality freeze.

Launch this command once per model in parallel, with the exact
``CUDA_VISIBLE_DEVICES`` value shown in ``H2_PAIRED_SCORING.md``.  It does not
read Arena score outputs, select pairs, or accept an unfrozen quality table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_paired_scoring import (
    DEFAULT_HDD_ROOT,
    DEFAULT_PARITY_ROOT,
    SCORER_SPECS,
    assert_committed_score_inputs,
    build_scorer,
    run_paired_scoring,
    score_run_paths,
    validate_parity_eligibility,
    validate_pinned_runtime_assignment,
    validate_score_eligibility,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict H2 paired detector scorer after quality freeze")
    parser.add_argument("--run-id", required=True, help="Explicit durable run ID; no timestamp/default is generated")
    parser.add_argument("--model", required=True, choices=tuple(SCORER_SPECS))
    parser.add_argument("--quality-manifest", required=True, help="Committed retained-only score_eligible_pairs.csv")
    parser.add_argument("--quality-freeze", required=True, help="Committed adjacent quality_freeze.json")
    parser.add_argument("--input-manifest", required=True, help="Committed H2 pre-score input_manifest.csv")
    parser.add_argument("--arm-ledger", required=True, help="Committed H2 pre-score arm_ledger.json")
    parser.add_argument("--arena-index", default=str(REPO_ROOT / "data" / "arena-index.yaml"))
    parser.add_argument("--parity-root", default=str(REPO_ROOT / DEFAULT_PARITY_ROOT))
    parser.add_argument("--hdd-root", default=str(DEFAULT_HDD_ROOT))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    spec = SCORER_SPECS[args.model]
    validate_pinned_runtime_assignment(spec)
    assert_committed_score_inputs(
        repo_root=REPO_ROOT,
        quality_manifest_path=args.quality_manifest,
        quality_freeze_path=args.quality_freeze,
        input_manifest_path=args.input_manifest,
        arm_ledger_path=args.arm_ledger,
        arena_index_path=args.arena_index,
        parity_root=args.parity_root,
        model=args.model,
    )
    eligibility = validate_score_eligibility(
        quality_manifest_path=args.quality_manifest,
        quality_freeze_path=args.quality_freeze,
        input_manifest_path=args.input_manifest,
        arm_ledger_path=args.arm_ledger,
        arena_index_path=args.arena_index,
    )
    parity = validate_parity_eligibility(spec, parity_root=args.parity_root)
    index = yaml.safe_load(Path(args.arena_index).read_text(encoding="utf-8"))
    model_directory = Path(index["models"][args.model]["local_dir"])
    paths = score_run_paths(args.hdd_root, args.run_id, args.model)
    summary = run_paired_scoring(
        eligibility,
        paths=paths,
        run_id=args.run_id,
        spec=spec,
        parity=parity,
        model_directory=model_directory,
        scorer_factory=lambda chosen, directory: build_scorer(chosen, model_directory=directory),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
