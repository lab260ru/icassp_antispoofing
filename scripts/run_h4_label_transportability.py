#!/usr/bin/env python3
"""Materialise the locked 420-cell score-free H4 label--cue atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h4_label_transportability import analyze_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-freeze-provenance",
        required=True,
        type=Path,
        help="The immutable h4_input_freeze_provenance.json from freeze_h4_inputs.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="New, non-existent directory for H4 results.")
    args = parser.parse_args()
    matrix, aggregation, report = analyze_frozen_manifest(args.input_freeze_provenance, args.output_dir)
    print(f"wrote {matrix} (420 locked dataset/view/feature cells)")
    print(f"wrote {aggregation} (84 locked view/feature aggregation units)")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()

