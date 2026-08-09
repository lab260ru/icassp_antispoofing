#!/usr/bin/env python3
"""Freeze a score-independent H2 input panel and its registered arm ledger.

This command deliberately does not decode audio, load ASR, load detector
weights, or emit detector responses.  It is the protocol-before-results step
for a later waveform-quality runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h2_pre_score_pairs import arm_definitions_frame, freeze_input_manifest_from_csv


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze score-independent H2 input and arm manifests")
    parser.add_argument("--input-csv", required=True, help="Explicit dataset/sample_id/label CSV; detector response columns are rejected")
    parser.add_argument("--per-label", required=True, type=int, help="Exact samples selected per dataset/label")
    parser.add_argument("--seed", required=True, type=int, help="Stable SHA-256 selection seed")
    parser.add_argument("--output-manifest", required=True, help="New CSV path for frozen input rows")
    parser.add_argument("--output-metadata", required=True, help="New JSON path for source/hash provenance")
    parser.add_argument("--output-arms", required=True, help="New JSON path for registered arm definitions")
    return parser.parse_args()


def _new_path(value: str) -> Path:
    path = Path(value).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen H2 artifact: {path}")
    return path


def main() -> None:
    args = _arguments()
    outputs = [_new_path(value) for value in (args.output_manifest, args.output_metadata, args.output_arms)]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be distinct")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    frozen = freeze_input_manifest_from_csv(args.input_csv, per_label=args.per_label, seed=args.seed)
    frozen.rows.to_csv(outputs[0], index=False)
    outputs[1].write_text(json.dumps(frozen.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arms = arm_definitions_frame().to_dict(orient="records")
    outputs[2].write_text(json.dumps(arms, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {outputs[0]} ({len(frozen.rows)} score-independent rows)")


if __name__ == "__main__":
    main()
