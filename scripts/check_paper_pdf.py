#!/usr/bin/env python3
"""Run static readiness checks on the built anonymous ICASSP working draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paper_pdf_checks import audit_working_draft_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=REPO_ROOT / "paper" / "build" / "main.pdf")
    parser.add_argument("--expected-pages", type=int, default=4)
    args = parser.parse_args()
    report = audit_working_draft_pdf(args.pdf, expected_pages=args.expected_pages)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit("Paper PDF static readiness check failed")


if __name__ == "__main__":
    main()
