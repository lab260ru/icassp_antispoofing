#!/usr/bin/env python3
"""Render the reviewed v2 main-paper evidence boundary with explicit approval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main_crest_evidence_boundary_v2 import render_main_figure_v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-render", action="store_true", help="Required explicit authorization after v2 commit.")
    parser.add_argument("--authorization-note", default="", help="Required non-empty authorization note recorded in metadata.")
    args = parser.parse_args()
    for kind, path in render_main_figure_v2(authorized=args.authorized_render, authorization_note=args.authorization_note).items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
