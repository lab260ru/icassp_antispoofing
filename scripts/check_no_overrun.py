#!/usr/bin/env python3
"""No page may print text below the bottom of the page.

The chronology table in S16 overran its page by 173 points and LaTeX let it: an
overfull \\vbox is a warning, the PDF builds, and the text is still in the
content stream, so `pdftotext` and every check in this suite found it present.
It was not present to a reader. The last six rows -- including a pre-committed
probe launch and the adverse result it produced, which is exactly what a
chronology of decisions exists to show -- were drawn below the paper's edge.

Being in the file is not the same as being on the page, and that distinction is
invisible to any check that reads extracted text. This one reads coordinates.

The rule: on each page, find the lowest baseline any glyph is drawn at, and
compare it against the deepest baseline seen on a well-behaved page of the same
document. A page whose text runs materially deeper than the document's own
typical text block is overrunning. Using the document's own pages as the
reference avoids hardcoding a text-block height that differs between the paper
and the supplement.

Usage:  python scripts/check_no_overrun.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pypdf

REPO = Path(__file__).resolve().parent.parent
DOCS = ("paper/build/main.pdf", "paper/supplementary/build/supp.pdf")
SLACK_PT = 24.0     # a couple of lines of tolerance over the typical block


def deepest_per_page(path: Path) -> list[float]:
    out = []
    for pg in pypdf.PdfReader(str(path)).pages:
        ys: list[float] = []
        pg.extract_text(
            visitor_text=lambda t, cm, tm, f, s: ys.append(tm[5]) if t.strip() else None)
        out.append(min(ys) if ys else 0.0)
    return out


def main() -> int:
    bad = []
    for rel in DOCS:
        p = REPO / rel
        if not p.exists():
            bad.append(f"{rel}: not built")
            continue
        depths = deepest_per_page(p)
        if not depths:
            continue
        # The typical text block: the median of the deepest baselines. Pages that
        # end early (section ends, short pages) sit above it; overrunning pages
        # sit below.
        typical = statistics.median(depths)
        for i, d in enumerate(depths, 1):
            if d < typical - SLACK_PT:
                bad.append(f"{rel} p{i}: text {typical - d:.0f}pt below the "
                           f"document's own text block")
    if bad:
        print(f"  FAIL {len(bad)} page(s) print text past the bottom:")
        for b in bad[:8]:
            print(f"       {b}")
        print("       A box overran its page. Content is in the file and not on "
              "the page;\n       break the table or float across pages.")
        return 1
    print("  OK   no page prints text past the bottom of the page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
