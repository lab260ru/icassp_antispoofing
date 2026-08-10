#!/usr/bin/env python3
"""Fit the locked H8-SF source-only fusers and emit label-free target predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h8_fusion_training import fit_h8_source_only_fusers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New absolute HDD directory; overwrites are refused.")
    args = parser.parse_args()
    outputs = fit_h8_source_only_fusers(
        source_features_path=args.source_features,
        source_provenance_path=args.source_provenance,
        target_features_path=args.target_features,
        target_provenance_path=args.target_provenance,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
