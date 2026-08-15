#!/usr/bin/env python3
"""A supplement pointer must land on the section it means, not just a valid one.

`verify_submission.sh` already checks that every S-number the paper cites is
within the supplement's section count. That check passes happily while the
pointers are scrambled, and they were: inserting the repetition-aware-sampling
section before the Spanish arm renumbered everything after it, so the paper's
S27 (Spanish) silently became RAS, S28 (period ladder) became Spanish, and S29
(RAS) became the period ladder. Three targets wrong, every number still "valid",
every existing check green. A reader following any of them lands in the wrong
section, and the paper's densest claims -- the period ladder and the ordering
control -- were the ones mis-aimed.

Section numbers are positional and therefore fragile; titles are not. This pins
each pointer the paper leans on to a keyword its section's title must contain.
A renumbering that moves a section away from its pointer now fails here by name
rather than passing silently.

The map is deliberately partial. Pointers whose target is unambiguous and stable
(S1-S2 for the Lean development, say) do not need pinning; what needs pinning is
anything a future insertion could shift, which in practice means the sections
added late and cited often.

Usage:  python scripts/check_pointer_targets.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUPP = REPO / "paper/supplementary/supp.tex"

# S-number -> a phrase that must appear in that section's title.
EXPECT = {
    5:  "negative result",
    8:  "Answering the alternatives",
    11: "Checkpoint-level",
    14: "noise floor",
    16: "Chronology",
    22: "greedy",
    25: "listen",
    26: "Causal interventions",
    27: "Spanish",
    28: "period ladder",
    29: "repetition-aware",
}


def main() -> int:
    if not SUPP.exists():
        print("  SKIP supp.tex not found")
        return 0
    titles: dict[int, str] = {}
    n = 0
    for line in SUPP.read_text().split("\n"):
        m = re.match(r"\\section\{(.+)\}", line)
        if m:
            n += 1
            titles[n] = m.group(1)

    bad = []
    for num, kw in sorted(EXPECT.items()):
        t = titles.get(num, "")
        if kw.lower() not in t.lower():
            bad.append(f"S{num} should mention {kw!r}, found {t[:56]!r}")

    if bad:
        print(f"  FAIL {len(bad)} pointer(s) no longer land on their section:")
        for b in bad:
            print(f"       {b}")
        print("       A section was inserted or moved. Either restore the order or "
              "renumber\n       every citation in the paper -- and check the other "
              "direction too.")
        return 1
    print(f"  OK   {len(EXPECT)} pinned pointers land on the section they mean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
