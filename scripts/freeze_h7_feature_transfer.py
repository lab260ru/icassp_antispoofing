"""Freeze H7's exact score-free source-ID panel."""

from pathlib import Path

from src.h7_feature_transfer import freeze_inputs, locked_paths, write_freeze


FEATURE_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28")
OUTPUT_DIR = Path("experiments/h7_feature_transfer/results/h7_input_freeze_001")


def main() -> None:
    manifest, provenance = freeze_inputs(locked_paths(FEATURE_ROOT))
    write_freeze(OUTPUT_DIR, manifest, provenance)
    print(f"Wrote {len(manifest)} H7 score-free selected source rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
