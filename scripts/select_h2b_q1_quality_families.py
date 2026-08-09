#!/usr/bin/env python3
"""Select only quality-feasible H2B Q1 arms from a complete Q1 calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2b_quality_selection import select_h2b_q1_families, write_h2b_q1_selection


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select H2B Q1 quality-feasible transform families")
    parser.add_argument("--quality-table", required=True, help="Complete detector-free Q1 Parquet table")
    parser.add_argument("--quality-summary", required=True, help="Finalized H2B Q1 summary JSON")
    parser.add_argument("--arm-manifest", default=str(REPO_ROOT / "experiments" / "future_directions" / "h2b_q1_quality_arm_manifest.json"))
    parser.add_argument("--output-dir", required=True, help="New non-overwriting selection output directory")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    report, provenance = select_h2b_q1_families(args.quality_table, args.quality_summary, args.arm_manifest)
    report_path, provenance_path = write_h2b_q1_selection(report, provenance, output_dir=args.output_dir)
    print(f"wrote {report_path} and {provenance_path}")


if __name__ == "__main__":
    main()
