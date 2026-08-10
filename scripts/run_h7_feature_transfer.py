"""Run the locked H7 five-cell feature-only transfer matrix."""

from pathlib import Path

from src.h7_feature_transfer import run_analysis, write_analysis


FREEZE_DIR = Path("experiments/h7_feature_transfer/results/h7_input_freeze_001")
OUTPUT_DIR = Path("experiments/h7_feature_transfer/results/h7_analysis_001")


def main() -> None:
    table, summary = run_analysis(FREEZE_DIR)
    write_analysis(OUTPUT_DIR, table, summary)
    print(f"Wrote {len(table)} H7 leave-one-corpus-out cells to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
