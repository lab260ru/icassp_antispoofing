#!/usr/bin/env python3
"""Execute the detector-free, resumable H2 waveform-quality gate.

This command never imports a detector runner.  It requires committed frozen
input/arm artifacts and writes full per-pair data to HDD plus a compact
repository summary.  A bounded run is intentionally a pilot only and cannot
open the H2 panel gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_asr_wer import WhisperQualityGateConfig
from src.h2_quality_runner import (
    DEFAULT_HDD_ROOT,
    assert_git_clean_and_committed,
    quality_run_paths,
    run_quality_panel,
    validate_frozen_inputs,
)


DEFAULT_FROZEN_ROOT = REPO_ROOT / "experiments" / "h2_causal_interventions" / "results" / "pre_score" / "h2_crest_pre_score_001_20260809T205114Z"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-free H2 waveform quality runner")
    parser.add_argument("--run-id", required=True, help="Explicit durable run ID; no timestamp/default is generated")
    parser.add_argument("--input-manifest", default=str(DEFAULT_FROZEN_ROOT / "input_manifest.csv"))
    parser.add_argument("--arm-ledger", default=str(DEFAULT_FROZEN_ROOT / "arm_ledger.json"))
    parser.add_argument("--arena-index", default=str(REPO_ROOT / "data" / "arena-index.yaml"))
    parser.add_argument("--hdd-root", default=str(DEFAULT_HDD_ROOT))
    parser.add_argument(
        "--limit",
        type=int,
        help="Bounded number of deterministically first frozen samples; requires --pilot and writes a not-panel-gate artifact",
    )
    parser.add_argument("--pilot", action="store_true", help="Required with --limit; marks output pilot_not_panel_gate")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.limit is not None and not args.pilot:
        raise SystemExit("--limit requires --pilot; bounded output is not eligible for the panel gate")
    if args.pilot and args.limit is None:
        raise SystemExit("--pilot requires a bounded --limit")
    manifest = Path(args.input_manifest).resolve()
    ledger = Path(args.arm_ledger).resolve()
    index = Path(args.arena_index).resolve()
    assert_git_clean_and_committed((manifest, ledger), repo_root=REPO_ROOT)
    validated = validate_frozen_inputs(manifest, ledger, index)
    mode = "pilot_not_panel_gate" if args.pilot else "full_panel_pre_score_quality"
    paths = quality_run_paths(args.hdd_root, args.run_id, REPO_ROOT)
    summary = run_quality_panel(
        validated,
        paths=paths,
        run_id=args.run_id,
        mode=mode,
        limit=args.limit,
        whisper_config=WhisperQualityGateConfig(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
