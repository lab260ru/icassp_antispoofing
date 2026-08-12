#!/usr/bin/env python3
"""Do our own exclusions make the effect? Re-score with each rule switched off.

About a quarter of panel generations are removed before the headline statistics
are computed --- a judge-unmeasurable template, generations cut off by our token
budget, and degenerate audio. Every one of those rules was adopted for a stated
reason and each is defensible on its own. That is exactly the situation in which
a reader should want to see the numbers without them, because three defensible
rules can still add up to a selected sample.

So this recomputes the two headline statistics -- the exactly-correct rate gap
between repeated and control items, and the median relative count error -- under
the full population and under each rule individually restored. If the gap holds
up with the excluded rows back in, the exclusions are tidying, not doing work.

Reading the arms:

  panel            what the paper reports
  +bad templates   t2 restored: the judge cannot transcribe `okay`/`hmm`, so
                   these rows score our recogniser's vocabulary and not the
                   decoder. Restoring them adds noise to both families at once.
  +cap hits        generations that ran into our own token budget. Their counts
                   are censored downward, so restoring them should *help* the
                   repeated side look worse -- this arm is the one that could
                   flatter us, and it is reported for that reason.
  +degenerate      empty or degenerate audio, which has no count at all. Scored
                   here as count 0, the harshest reading available.
  everything       all three at once.

Usage:  python analysis/exclusion_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402


def stats(d: pd.DataFrame) -> dict:
    """Exact rate and median relative error for each family, and the gap."""
    out: dict = {}
    for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
        s = d[d.family == fam]
        err = (s.count_a - s.k) / s.k
        out[tag] = dict(n=int(len(s)),
                        exact=float((err == 0).mean()) if len(s) else float("nan"),
                        median_err=float(err.median()) if len(s) else float("nan"))
    out["exact_gap"] = out["ctl"]["exact"] - out["rep"]["exact"]
    out["err_gap"] = out["ctl"]["median_err"] - out["rep"]["median_err"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/exclusion_sensitivity.json")
    args = ap.parse_args()

    raw = pd.read_csv(args.behavioural)
    raw = raw[raw.k >= args.kmin]

    arms = {
        "panel": dict(bad_templates=True, cap_hits=True, degenerate=False),
        "plus_bad_templates": dict(bad_templates=False, cap_hits=True, degenerate=False),
        "plus_cap_hits": dict(bad_templates=True, cap_hits=False, degenerate=False),
        "plus_degenerate": dict(bad_templates=True, cap_hits=True, degenerate=True),
        "everything": dict(bad_templates=False, cap_hits=False, degenerate=True),
    }

    res: dict = {"kmin": args.kmin, "arms": {}}
    print(f"{'arm':>20s} {'n':>6s} {'exact rep':>10s} {'exact ctl':>10s} "
          f"{'gap':>7s} {'med rep':>8s}")
    for name, flags in arms.items():
        d, drop = panel(raw, ablations=False, **flags)
        # Degenerate rows have no transcript to count, so scoring them at all
        # means choosing a number. Zero is the harshest available reading and
        # the only one that cannot be accused of flattering the repeated side.
        if flags["degenerate"]:
            d = d.copy()
            d.loc[d.outcome.isin(["empty", "degenerate"]), "count_a"] = 0
        st = stats(d)
        st["n"] = int(len(d))
        st["dropped"] = {k: v for k, v in drop.items()
                         if k not in {"n_input", "n_output", "excluded_templates"}}
        res["arms"][name] = st
        print(f"{name:>20s} {len(d):6d} {100*st['rep']['exact']:9.1f}% "
              f"{100*st['ctl']['exact']:9.1f}% {100*st['exact_gap']:6.1f} "
              f"{st['rep']['median_err']:+8.3f}")

    gaps = {k: v["exact_gap"] for k, v in res["arms"].items()}
    res["gap_min"] = float(min(gaps.values()))
    res["gap_max"] = float(max(gaps.values()))
    res["gap_panel"] = float(gaps["panel"])
    res["all_arms_positive"] = bool(all(g > 0 for g in gaps.values()))
    # The comparison that matters is against the arm with nothing excluded: if
    # the gap survives there, no combination of our rules can be manufacturing
    # it, whatever each rule does on its own.
    res["gap_everything"] = float(gaps["everything"])
    res["shrinkage"] = float(gaps["panel"] - gaps["everything"])

    print(f"\nexact-rate gap: panel {100*gaps['panel']:.1f} points, "
          f"nothing excluded {100*gaps['everything']:.1f} points "
          f"({100*res['shrinkage']:+.1f})")
    res["verdict"] = (
        "the gap survives every exclusion rule being switched off"
        if res["all_arms_positive"] and gaps["everything"] > 0.5 * gaps["panel"]
        else "at least one exclusion rule carries a substantial part of the gap; "
             "it must be reported with the unexcluded number beside it")
    print(f"{res['verdict']}.")
    print("\nThe cap-hit arm is the one that could flatter us -- restoring "
          "budget-truncated\ngenerations censors the repeated side downward -- "
          "so it is reported, not folded in.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
