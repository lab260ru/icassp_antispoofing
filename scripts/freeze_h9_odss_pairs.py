#!/usr/bin/env python3
"""Freeze target-free ODSS paired-counterfactual source manifests for H9-PCR."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h9_odss_pairs import (  # noqa: E402
    H9_ODSS_REVISION,
    H9_SPLIT_SEED,
    freeze_h9_odss_pairing_from_files,
    write_h9_odss_pairing_artifacts,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path, help="Pinned ODSS labels.parquet or metadata CSV; no audio column.")
    parser.add_argument("--odss-readme", required=True, type=Path, help="Pinned ODSS README.md defining the path/pair semantics.")
    parser.add_argument("--odss-build-script", required=True, type=Path, help="Pinned ODSS build_parquet.py defining UID construction.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory below the project HDD root.")
    parser.add_argument("--task", default=REPO_ROOT / "experiments/h9_paired_counterfactual/TASK.md", type=Path)
    parser.add_argument("--data", default=REPO_ROOT / "experiments/h9_paired_counterfactual/DATA.md", type=Path)
    parser.add_argument("--plan", default=REPO_ROOT / "experiments/h9_paired_counterfactual/PLAN.md", type=Path)
    parser.add_argument("--brainstorm", default=REPO_ROOT / "experiments/h9_paired_counterfactual/BRAINSTORM.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    freeze = freeze_h9_odss_pairing_from_files(
        metadata_path=args.metadata,
        semantics_documents={"readme": args.odss_readme, "build_script": args.odss_build_script},
        protocol_documents={
            "TASK.md": args.task,
            "DATA.md": args.data,
            "PLAN.md": args.plan,
            "BRAINSTORM.md": args.brainstorm,
        },
    )
    outputs = write_h9_odss_pairing_artifacts(freeze, output_dir=args.output_dir)
    print(f"frozen H9-PCR ODSS revision={H9_ODSS_REVISION} split_seed={H9_SPLIT_SEED}")
    print(
        f"trials={len(freeze.trials)} p_pairs={len(freeze.pairs)} b2_pairs={len(freeze.random_pairs)} "
        f"excluded_unmatched={len(freeze.excluded_unmatched)} groups={freeze.provenance['counts']['complete_groups']}"
    )
    print(f"wrote {outputs['trials']}")
    print(f"wrote {outputs['pairs']}")
    print(f"wrote {outputs['random_pairs']}")
    print(f"wrote {outputs['excluded_unmatched']}")
    print(f"wrote {outputs['provenance']}")


if __name__ == "__main__":
    main()
