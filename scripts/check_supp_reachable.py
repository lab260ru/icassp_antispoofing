#!/usr/bin/env python3
"""Every supplement section must be reachable from a sentence in the paper.

Two review rounds called the missing qualitative example the most damaging gap in
the paper. The supplement had answered it for days -- S25 shows a scored
generation of every outcome, chosen by a rule fixed in code. The reviewers never
found it, and one of them said why in as many words: the sensitivity analysis is
"deferred entirely to supplementary material not reproduced here".

That is the failure this catches. A supplement section the paper never names is
work that cannot be credited and an objection that cannot be answered, however
good the section is. When it was first run, six of twenty-nine sections were
unreachable -- including Reproducibility, which was the entire subject of one
reviewer's verdict.

`verify_submission.sh` already checks the other direction (every S-number the
paper cites exists, and lands on the section it means). Both directions matter
and they fail differently: a dangling pointer misleads a reader who follows it,
an unreachable section is invisible to a reader who never knows to.

Exemptions are possible in principle -- a section genuinely for the record rather
than for a reader -- but there are none today, so the rule is total. If one is
ever wanted, add it here with a reason rather than weakening the check.

Usage:  python scripts/check_supp_reachable.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUPP = REPO / "paper/supplementary/supp.tex"
BODIES = ("paper/main.tex", "paper/results_body.tex", "paper/discussion_body.tex",
          "paper/table1.tex", "paper/table2.tex")

# S-number -> why the paper does not point at it. Empty on purpose.
EXEMPT: dict[int, str] = {}


def main() -> int:
    if not SUPP.exists():
        print("  SKIP supp.tex not found")
        return 0
    titles = [m.group(1) for line in SUPP.read_text().split("\n")
              if (m := re.match(r"\\section\{(.+)\}", line))]

    text = ""
    for rel in BODIES:
        p = REPO / rel
        if p.exists():
            text += p.read_text()
    cited = {int(n) for n in re.findall(r"\bS(\d+)\b", text)}

    unreachable = [(i, t) for i, t in enumerate(titles, 1)
                   if i not in cited and i not in EXEMPT]
    if unreachable:
        print(f"  FAIL {len(unreachable)} supplement section(s) the paper never names:")
        for i, t in unreachable:
            print(f"       S{i:<3} {t[:64]}")
        print("       A reviewer reading the paper cannot reach these. Point at "
              "them or\n       exempt them with a reason in EXEMPT.")
        return 1
    print(f"  OK   all {len(titles)} supplement sections reachable from the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
