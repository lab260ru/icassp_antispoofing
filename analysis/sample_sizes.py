#!/usr/bin/env python3
"""Every sample size the paper reports, and how each descends from the others.

A reviewer counted the $n$'s across Sections 3--5 --- 2052, 1559, 457, 525, 90,
30, 60, 216, 72, 12 --- and observed that the paper never reconciles them. That
is a fair complaint about a paper whose whole argument is "trust the numbers
because you can regenerate them": a reader who cannot see how 2052 becomes 457
cannot tell a defensible restriction from a convenient one.

This prints the cascade. Each row states what was removed and why, so the arrows
between the numbers are visible rather than inferred. The rows come from
`src/common/population.py`, the single definition every analysis calls, so this
cannot drift from what the analyses actually did.

Usage:  python analysis/sample_sizes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402

PAIR = ["word_rep", "control_word"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--vits", default="data/results/behavioural_vits.csv")
    ap.add_argument("--greedy", default="data/results/behavioural_greedy.csv")
    ap.add_argument("--out", default="data/results/sample_sizes.json")
    args = ap.parse_args()

    raw = pd.read_csv(args.behavioural)
    steps: list[tuple[str, int, str]] = []

    steps.append(("all generations scored", len(raw),
                  "every model, every family, every seed"))
    pair = raw[raw.family.isin(PAIR)]
    steps.append(("repeated and control families", len(pair),
                  "drops numbers, twisters, sentence_rep --- other questions"))
    noabl, drop = panel(pair, degenerate=True, bad_templates=False, cap_hits=False)
    steps.append(("panel checkpoints only", len(noabl),
                  f"drops {drop.get('ablations', 0)} ablation and baseline rows"))
    a, d1 = panel(pair, degenerate=True, cap_hits=False)
    steps.append(("judge-measurable templates", len(a),
                  f"drops {d1.get('bad_templates', 0)} rows in t2, whose fillers "
                  "the CTC judge never emits"))
    b, d2 = panel(pair, degenerate=True)
    steps.append(("stopped by the model, not our budget", len(b),
                  f"drops {d2.get('cap_hits', 0)} budget-truncated rows, whose "
                  "counts are censored downward"))
    c, _ = panel(pair)
    steps.append(("audible speech", len(c),
                  f"drops {len(b) - len(c)} empty or degenerate rows, which have "
                  "no count at all"))
    hi = c[c.k >= 6]
    steps.append(("$k\\ge6$, where the deficit lives", len(hi),
                  "below k=6 every checkpoint is essentially exact"))

    print(f"{'population':40s} {'n':>6s}  why")
    for label, n, why in steps:
        print(f"{label:40s} {n:6d}  {why}")

    leaves = {
        "word_rep at k>=6": int(len(hi[hi.family == "word_rep"])),
        "control_word at k>=6": int(len(hi[hi.family == "control_word"])),
    }
    for path, label, model in ((args.vits, "VITS per family", "vits"),
                               (args.greedy, "greedy per arm", "qwen06bgreedy")):
        p = Path(path)
        if not p.exists():
            continue
        e = pd.read_csv(p)
        e = e[(e.model == model) & (e.k >= 6) & (~e.outcome.isin(["empty", "degenerate"]))]
        for fam in PAIR:
            leaves[f"{label} ({fam})"] = int(len(e[e.family == fam]))

    print()
    for k, v in leaves.items():
        print(f"{k:40s} {v:6d}")

    res = {"cascade": [dict(population=l, n=n, why=w) for l, n, w in steps],
           "leaves": leaves}
    # The arithmetic that a reader would do by hand, done here so a future
    # change to the exclusion rules cannot silently break the reconciliation.
    res["leaves_sum_matches"] = bool(
        leaves["word_rep at k>=6"] + leaves["control_word at k>=6"] == len(hi))
    print(f"\nrepeated + control = {len(hi)}: {res['leaves_sum_matches']}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
