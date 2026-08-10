#!/usr/bin/env python3
"""Render the prospective main-paper Fig. 1 only with explicit authorization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main_crest_evidence_boundary import render_locked_main_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorized-render",
        action="store_true",
        help="Required explicit maintainer authorization after the protocol and implementation are committed.",
    )
    parser.add_argument(
        "--authorization-note",
        default="",
        help="Required non-empty record of the maintainer's render authorization.",
    )
    args = parser.parse_args()
    outputs = render_locked_main_figure(
        authorized=args.authorized_render,
        authorization_note=args.authorization_note,
    )
    for kind, path in outputs.items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
