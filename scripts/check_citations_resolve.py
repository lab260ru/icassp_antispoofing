#!/usr/bin/env python3
"""Neither document may ship an unresolved citation.

The supplement cited eight works and had no bibliography at all, so every one of
them rendered as `[?]` -- in a document reviewers read, including inside a
sentence added specifically to credit prior art a reviewer had said we omitted.
Crediting someone with `[?]` is worse than not crediting them.

LaTeX reports an undefined citation as a warning and produces a PDF regardless,
so the build was green throughout and so was every check in the suite. It
surfaced only because I read a page I had written and saw "Loukas [?]" in my own
new sentence.

The fix was to give the supplement the paper's bibliography -- as a symlink, so
the two documents cannot drift into disagreeing about a reference. This check
guards the outcome rather than the mechanism: it reads both built PDFs and fails
if the rendered text contains a `[?]` anywhere.

Usage:  python scripts/check_citations_resolve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pypdf

REPO = Path(__file__).resolve().parent.parent
DOCS = ("paper/build/main.pdf", "paper/supplementary/build/supp.pdf")


def main() -> int:
    bad = []
    for rel in DOCS:
        p = REPO / rel
        if not p.exists():
            bad.append(f"{rel}: not built")
            continue
        pages = []
        for i, pg in enumerate(pypdf.PdfReader(str(p)).pages, 1):
            t = pg.extract_text()
            n = t.count("[?]") + t.count("[ ?]")
            if n:
                pages.append(f"p{i}x{n}")
        if pages:
            bad.append(f"{rel}: unresolved on {', '.join(pages[:6])}")
    if bad:
        print(f"  FAIL {len(bad)} document(s) with unresolved citations:")
        for b in bad:
            print(f"       {b}")
        print("       A \\cite key is missing from the bibliography the document "
              "actually loads.")
        return 1
    print("  OK   no unresolved citations in the paper or the supplement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
