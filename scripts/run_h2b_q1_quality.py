#!/usr/bin/env python3
"""Run the finite detector-free H2B Q1 quality calibration on frozen Q0 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_asr_wer import WhisperQualityGateConfig
from src.h2_quality_runner import DEFAULT_HDD_ROOT, assert_git_clean_and_committed
from src.h2b_quality_calibration import h2b_q1_quality_paths, h2b_q1_run_lock, run_h2b_q1_quality, validate_h2b_q1_inputs


DEFAULT_ROOT = REPO_ROOT / "experiments" / "future_directions"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute detector-free H2B Q1 waveform quality calibration")
    parser.add_argument("--run-id", required=True, help="Explicit resumable Q1 run ID")
    parser.add_argument("--q0-manifest", default=str(DEFAULT_ROOT / "results" / "h2b_q0_deepvoice_manifest.csv"))
    parser.add_argument("--q0-provenance", default=str(DEFAULT_ROOT / "results" / "h2b_q0_deepvoice_manifest.provenance.json"))
    parser.add_argument("--arm-manifest", default=str(DEFAULT_ROOT / "h2b_q1_quality_arm_manifest.json"))
    parser.add_argument("--arena-index", default=str(REPO_ROOT / "data" / "arena-index.yaml"))
    parser.add_argument("--hdd-root", default=str(DEFAULT_HDD_ROOT))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    frozen = (Path(args.q0_manifest).resolve(), Path(args.q0_provenance).resolve(), Path(args.arm_manifest).resolve())
    assert_git_clean_and_committed(frozen, repo_root=REPO_ROOT)
    validated, q1_info = validate_h2b_q1_inputs(*frozen, args.arena_index)
    paths = h2b_q1_quality_paths(args.hdd_root, args.run_id, REPO_ROOT)
    with h2b_q1_run_lock(paths, args.run_id):
        summary = run_h2b_q1_quality(
            validated,
            paths=paths,
            run_id=args.run_id,
            q1_info=q1_info,
            whisper_config=WhisperQualityGateConfig(),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
