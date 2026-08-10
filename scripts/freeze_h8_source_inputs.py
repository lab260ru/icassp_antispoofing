#!/usr/bin/env python3
"""Freeze the H8-SF source-only score orientation and balanced IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h8_score_fusion import freeze_source_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "arena-index.yaml")
    parser.add_argument("--output-dir", type=Path, required=True, help="New absolute HDD directory; overwrites are refused.")
    args = parser.parse_args()
    outputs = freeze_source_inputs(index_path=args.index, output_dir=args.output_dir)
    print(f"wrote {outputs.manifest_path}")
    print(f"wrote {outputs.orientation_path}")
    print(f"wrote {outputs.provenance_path}")


if __name__ == "__main__":
    main()
