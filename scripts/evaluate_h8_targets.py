#!/usr/bin/env python3
"""Run H8-SF's one final target-label evaluation and locked decision gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h8_fusion_evaluation import evaluate_h8_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=REPO_ROOT / "data" / "arena-index.yaml")
    parser.add_argument("--target-predictions", type=Path, required=True)
    parser.add_argument("--fit-provenance", type=Path, required=True)
    parser.add_argument("--model-record", type=Path, required=True)
    parser.add_argument("--target-feature-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New absolute HDD directory; overwrites are refused.")
    args = parser.parse_args()
    outputs = evaluate_h8_targets(
        index_path=args.index,
        target_predictions_path=args.target_predictions,
        fit_provenance_path=args.fit_provenance,
        model_record_path=args.model_record,
        target_feature_provenance_path=args.target_feature_provenance,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
