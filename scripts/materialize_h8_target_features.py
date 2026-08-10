#!/usr/bin/env python3
"""Create H8-SF label-free target rank/probit feature panels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h8_score_fusion import materialize_label_free_target_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "arena-index.yaml")
    parser.add_argument("--source-orientation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New absolute HDD directory; overwrites are refused.")
    args = parser.parse_args()
    outputs = materialize_label_free_target_features(
        index_path=args.index,
        orientation_path=args.source_orientation,
        output_dir=args.output_dir,
    )
    print(f"wrote {outputs.features_path}")
    print(f"wrote {outputs.provenance_path}")


if __name__ == "__main__":
    main()
