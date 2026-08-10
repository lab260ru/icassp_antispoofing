#!/usr/bin/env python3
"""Freeze the five locked H5 identity/view inputs before feature analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h5_view_invariance import DATASETS, H5_ALLOWED_INPUT_PATHS, freeze_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvspoof2019-la", required=True, type=Path, metavar="ABSOLUTE_FEATURES_WIDE_PARQUET")
    parser.add_argument("--asvspoof2021-la", required=True, type=Path, metavar="ABSOLUTE_FEATURES_WIDE_PARQUET")
    parser.add_argument("--asvspoof2021-df", required=True, type=Path, metavar="ABSOLUTE_FEATURES_WIDE_PARQUET")
    parser.add_argument("--inthe-wild", required=True, type=Path, metavar="ABSOLUTE_FEATURES_WIDE_PARQUET")
    parser.add_argument("--asvspoof5", required=True, type=Path, metavar="ABSOLUTE_FEATURES_WIDE_PARQUET")
    parser.add_argument("--output-dir", required=True, type=Path, help="New, non-existent directory for this freeze.")
    args = parser.parse_args()
    paths = {
        "ASVspoof2019_LA": args.asvspoof2019_la,
        "ASVspoof2021_LA": args.asvspoof2021_la,
        "ASVspoof2021_DF": args.asvspoof2021_df,
        "InTheWild": args.inthe_wild,
        "ASVspoof5": args.asvspoof5,
    }
    artifacts = freeze_inputs(paths, args.output_dir)
    print(f"froze datasets: {', '.join(DATASETS)}")
    print(f"locked input paths: {', '.join(str(H5_ALLOWED_INPUT_PATHS[name]) for name in DATASETS)}")
    print(f"wrote {artifacts.manifest_path} sha256={artifacts.manifest_sha256}")
    print(f"wrote {artifacts.provenance_path}")


if __name__ == "__main__":
    main()
