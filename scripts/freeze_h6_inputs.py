#!/usr/bin/env python3
"""Create the immutable label-only H6 input freeze before score parsing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h6_score_agreement import DATASETS, MODELS, freeze_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New, non-existent HDD directory for the H6 freeze.")
    args = parser.parse_args()
    artifacts = freeze_inputs(args.output_dir)
    print(f"froze label-only panel for {len(DATASETS)} datasets and {len(MODELS)} score-artifact byte identities")
    print(f"wrote {artifacts.manifest_path} sha256={artifacts.manifest_sha256}")
    print(f"wrote {artifacts.provenance_path}")


if __name__ == "__main__":
    main()
