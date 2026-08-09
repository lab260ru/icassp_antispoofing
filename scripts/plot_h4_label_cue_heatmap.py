#!/usr/bin/env python3
"""Render a publication-ready heatmap from completed score-free H4 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h4_heatmap import render_h4_delta_auc_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path, help="Completed h4_label_cue_matrix.csv (absolute path).")
    parser.add_argument(
        "--aggregation", required=True, type=Path, help="Completed h4_label_cue_aggregation.csv (absolute path)."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for new PDF and PNG figure artifacts.")
    args = parser.parse_args()
    pdf_path, png_path = render_h4_delta_auc_heatmap(args.matrix, args.aggregation, args.output_dir)
    print(f"wrote vector PDF: {pdf_path}")
    print(f"wrote 300-DPI PNG: {png_path}")


if __name__ == "__main__":
    main()

