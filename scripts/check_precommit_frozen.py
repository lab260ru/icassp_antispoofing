#!/usr/bin/env python3
"""A pre-registration is only worth what it costs to change it.

Five scripts in this repository declare a pre-committed region: everything above
a `# POST-HOC` marker was fixed before that experiment's audio existed, and the
file says so in its own docstring. `analysis/rep_aware_sampling.py` puts it
plainly -- "Nothing above the POST-HOC line may be edited after looking at a
result; a rule that turned out to be wrong is reported as wrong, not rewritten."

Nothing enforced that. The rule lived in a comment, and comments do not stop
edits. The failure it guards against is not malice, it is the ordinary drift of
a scoring rule that turns out to be awkward once the numbers are in -- a
threshold nudged, a band widened, a caveat softened -- each change defensible on
its own and collectively the difference between a pre-registration and a
description.

So: for every file carrying the marker, this compares the region above it
against the last commit. If that region changed, the check fails and names the
file. Changing it deliberately is still possible -- you commit the change on its
own, with a message saying which rule moved and why, which is exactly the
paper-trail the discipline is for.

Two deliberate limits. It compares against HEAD, not against the commit where
the results first appeared, so it catches drift in the working tree rather than
auditing history; the history audit is in S16 and had to be done by hand.
And a file with no marker is not checked, because most scripts here have no
pre-committed region and demanding one would make the check noise.

Usage:  python scripts/check_precommit_frozen.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEARCH = ("analysis", "data/stimuli", "src/common", "src/models")
MARKER = "POST-HOC"


def frozen_region(lines: list[str]) -> list[str] | None:
    """Everything above the first commented POST-HOC marker, or None."""
    for i, line in enumerate(lines):
        if MARKER in line and line.lstrip().startswith("#"):
            return lines[:i]
    return None


def main() -> int:
    bad: list[str] = []
    checked = 0
    for d in SEARCH:
        for p in sorted((REPO / d).glob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            work = frozen_region(p.read_text().split("\n"))
            if work is None:
                continue
            r = subprocess.run(["git", "show", f"HEAD:{rel}"],
                               cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                # New file, not yet committed: nothing to compare against, and
                # its first commit is what establishes the region.
                print(f"  NOTE {rel}: not in HEAD yet, region not comparable")
                continue
            head = frozen_region(r.stdout.split("\n"))
            checked += 1
            if head is None:
                bad.append(f"{rel}: marker present now but absent in HEAD")
            elif head != work:
                n = sum(1 for a, b in zip(head, work) if a != b) + abs(len(head) - len(work))
                bad.append(f"{rel}: pre-committed region changed ({n} lines)")

    if bad:
        print(f"  FAIL {len(bad)} pre-committed region(s) edited:")
        for b in bad:
            print(f"       {b}")
        print("       If a rule genuinely has to move, commit that change alone "
              "with a message\n       saying which rule moved and why. That is "
              "the paper trail the rule exists for.")
        return 1
    print(f"  OK   {checked} pre-committed regions unchanged since HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
