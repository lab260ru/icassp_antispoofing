"""Emit the protocol-pinned, score-free H1 covariate availability disclosure."""

from __future__ import annotations

from pathlib import Path

from src.h1_covariate_availability import locked_paths, validate_and_summarize, write_outputs


FEATURE_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28")
OUTPUT_DIR = Path("experiments/h1_feature_association/results/covariate_availability_audit_001")


def main() -> None:
    table, provenance = validate_and_summarize(locked_paths(FEATURE_ROOT))
    write_outputs(OUTPUT_DIR, table, provenance)
    print(f"Wrote {len(table)} score-free covariate-availability rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
