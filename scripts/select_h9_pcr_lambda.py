#!/usr/bin/env python3
"""Select the locked H9 PCR rank weight from source-only development ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.h9_pcr_training import select_p_lambda, write_json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dev-results", required=True, type=Path, help="JSON list of P-only source-development result objects.")
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
try:
    raw = json.loads(args.source_dev_results.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Cannot read source-only H9 P selection inputs: {error}") from error
if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
    raise SystemExit("H9 P source-development selection input must be a JSON list of objects")
selection = select_p_lambda(raw)
write_json(args.output, selection)
print(json.dumps(selection, sort_keys=True, separators=(",", ":")))
