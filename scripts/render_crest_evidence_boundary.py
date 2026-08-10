#!/usr/bin/env python3
"""Render the locked supplementary crest-factor evidence-boundary figure."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crest_evidence_boundary import render_locked_crest_evidence_boundary


def main() -> None:
    outputs = render_locked_crest_evidence_boundary()
    for kind, path in outputs.items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
