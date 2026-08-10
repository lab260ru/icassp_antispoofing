#!/usr/bin/env python3
"""Materialise the frozen 280-cell H6 within-class score-agreement atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h6_score_agreement import analyze_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-freeze-provenance",
        required=True,
        type=Path,
        help="Immutable h6_input_freeze_provenance.json from freeze_h6_inputs.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="New, non-existent HDD directory for H6 outputs.")
    args = parser.parse_args()
    matrix, aggregation, report = analyze_frozen_manifest(args.input_freeze_provenance, args.output_dir)
    print(f"wrote {matrix} (280 locked dataset/class/model-pair cells)")
    print(f"wrote {aggregation} (28 locked model-pair summaries)")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
