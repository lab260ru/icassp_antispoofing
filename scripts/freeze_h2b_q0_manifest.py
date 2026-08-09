#!/usr/bin/env python3
"""Freeze a DeepVoice-style H2B Q0 score-blind calibration manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2b_score_blind_manifest import freeze_h2b_q0_manifest, write_h2b_q0_artifacts


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze an H2B Q0 score-blind, label-derived calibration manifest")
    parser.add_argument("--input-csv", required=True, help="Exact score-free CSV from build_h2_input_csv.py")
    parser.add_argument("--input-provenance", required=True, help="Byte-bound provenance JSON next to the exact input CSV")
    parser.add_argument("--dataset", required=True, help="Declared Q0 dataset identity")
    parser.add_argument("--revision", required=True, help="Declared immutable Q0 dataset revision")
    parser.add_argument("--per-label", type=int, required=True, help="Exact selected rows per label")
    parser.add_argument("--seed", type=int, required=True, help="Stable SHA-256 selection seed")
    parser.add_argument("--output-manifest", required=True, help="New Q0 CSV path")
    parser.add_argument("--output-provenance", required=True, help="New Q0 provenance JSON path")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    frozen = freeze_h2b_q0_manifest(
        args.input_csv,
        args.input_provenance,
        dataset=args.dataset,
        revision=args.revision,
        per_label=args.per_label,
        seed=args.seed,
    )
    write_h2b_q0_artifacts(
        frozen,
        output_manifest=args.output_manifest,
        output_provenance=args.output_provenance,
    )
    print(json.dumps({"rows": len(frozen.rows), "stage": "q0_frozen_score_blind"}, sort_keys=True))


if __name__ == "__main__":
    main()
