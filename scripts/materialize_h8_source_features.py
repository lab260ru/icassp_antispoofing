#!/usr/bin/env python3
"""Materialize the hash-revalidated H8-SF source rank/probit training panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h8_score_fusion import materialize_source_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "arena-index.yaml")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-orientation", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New absolute HDD directory; overwrites are refused.")
    args = parser.parse_args()
    outputs = materialize_source_features(
        index_path=args.index,
        source_manifest_path=args.source_manifest,
        source_orientation_path=args.source_orientation,
        source_provenance_path=args.source_provenance,
        output_dir=args.output_dir,
    )
    print(f"wrote {outputs.features_path}")
    print(f"wrote {outputs.provenance_path}")


if __name__ == "__main__":
    main()
