#!/usr/bin/env python3
"""Freeze H9's P-only source-development lambda selection without targets.

The caller supplies the twelve explicit P sidecars (three locked lambdas by
four locked seeds). The script refuses a target-access record, duplicate
identity, altered source-artifact binding, or an existing output path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.h9_pcr_training import H9_LAMBDA_GRID, H9_SEEDS, select_p_lambda, sha256_file, write_json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", action="append", required=True, type=Path, help="One explicit P training sidecar; repeat exactly 12 times.")
parser.add_argument("--output", required=True, type=Path, help="Fresh source-only selection JSON.")
args = parser.parse_args()

if args.output.exists():
    raise SystemExit(f"H9 lambda-selection output already exists: {args.output}")
if len(args.result) != len(H9_LAMBDA_GRID) * len(H9_SEEDS):
    raise SystemExit("H9 lambda selection requires exactly 12 explicit P sidecars")

records: list[dict[str, object]] = []
observed: set[tuple[float, int]] = set()
source_hashes: dict[str, str] | None = None
inputs: list[dict[str, str]] = []
for path in args.result:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"H9 P sidecar is unavailable: {resolved}")
    try:
        record = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"H9 cannot parse P sidecar {resolved}: {error}") from error
    if not isinstance(record, dict) or record.get("method") != "P":
        raise SystemExit("H9 lambda selection accepts only P sidecars")
    if record.get("target_labels_read") is not False or record.get("target_audio_read") is not False:
        raise SystemExit("H9 source-only selection rejects a target-access sidecar")
    try:
        identity = (float(record["lambda_rank"]), int(record["seed"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("H9 P sidecar has invalid lambda/seed identity") from error
    if identity not in {(lam, seed) for lam in H9_LAMBDA_GRID for seed in H9_SEEDS} or identity in observed:
        raise SystemExit("H9 P sidecars do not form the locked unique lambda/seed grid")
    observed.add(identity)
    hashes = record.get("source_artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {"source_manifest_sha256", "p_pairs_sha256", "b2_pairs_sha256"}:
        raise SystemExit("H9 P sidecar lacks the complete source-artifact binding")
    normalized = {str(key): str(value) for key, value in hashes.items()}
    if source_hashes is None:
        source_hashes = normalized
    elif source_hashes != normalized:
        raise SystemExit("H9 P sidecars disagree on their source-artifact binding")
    records.append(record)
    inputs.append({"path": str(resolved), "sha256": sha256_file(resolved)})

selection = select_p_lambda(records)
selection["artifact_kind"] = "h9_pcr_frozen_source_only_lambda_selection"
selection["input_sidecars"] = sorted(inputs, key=lambda row: row["path"])
selection["source_artifact_hashes"] = source_hashes
selection["target_metrics_read"] = False
write_json(args.output, selection)
print(json.dumps(selection, sort_keys=True, separators=(",", ":")))
