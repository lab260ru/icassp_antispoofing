#!/usr/bin/env python3
"""Create a score-free H2 input CSV from one Arena dataset's pinned labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_input_csv import build_score_free_input_from_index, write_score_free_input_artifacts


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a score-free label-derived H2 input CSV")
    parser.add_argument("--dataset", required=True, help="One exact dataset key from the Arena index")
    parser.add_argument("--index", default="data/arena-index.yaml", help="Pinned Arena index YAML")
    parser.add_argument("--output-csv", required=True, help="New input CSV path")
    parser.add_argument("--output-provenance", required=True, help="New JSON provenance path")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    rows, provenance = build_score_free_input_from_index(args.index, args.dataset)
    write_score_free_input_artifacts(
        rows,
        provenance,
        output_csv=args.output_csv,
        output_provenance=args.output_provenance,
    )
    print(f"wrote {args.output_csv} ({len(rows)} score-free rows for {args.dataset})")


if __name__ == "__main__":
    main()
