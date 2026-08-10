#!/usr/bin/env python3
"""Materialize the sealed H9-PCR ODSS source WAV pool; this CLI has no target inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h9_odss_materialize import materialize_h9_odss_source  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path, help="Exact h9odss_source_pairing_001 directory.")
    parser.add_argument("--raw-shard-dir", required=True, type=Path, help="Pinned ODSS revision data/ directory containing raw Parquet shards.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory below the project HDD root.")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = materialize_h9_odss_source(
        freeze_dir=args.freeze_dir,
        raw_shard_dir=args.raw_shard_dir,
        output_dir=args.output_dir,
    )
    print(f"materialized source manifest: {result.source_manifest}")
    print(f"materialized source byte/fingerprint audit: {result.audio_audit}")
    print(f"materialized source provenance: {result.provenance}")


if __name__ == "__main__":
    main()
