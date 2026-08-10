#!/usr/bin/env python3
"""Render the sealed H5 median-concordance supplementary heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.h5_view_invariance_heatmap import render_h5_median_concordance_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path, help="Sealed absolute h5_analysis_001 matrix path.")
    parser.add_argument("--aggregation", required=True, type=Path, help="Sealed absolute h5_analysis_001 aggregation path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for vector/raster/metadata outputs.")
    args = parser.parse_args()
    pdf_path, png_path, metadata_path = render_h5_median_concordance_heatmap(args.matrix, args.aggregation, args.output_dir)
    print(f"wrote vector PDF: {pdf_path}")
    print(f"wrote 300-DPI PNG: {png_path}")
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
