#!/usr/bin/env python3
"""The dissociation, restricted to controls that really are aperiodic.

Two round-6 reviewers made the same fair objection: the paper calls the
repeated-vs-control contrast its sharpest test of "periodicity, not length", but
the main ladder's control fillers are drawn from a pool of eight and cycled, so
a k=32 control repeats each filler four times. That contrast is period-1 against
period-8, not against aperiodic, and the Discussion's claim about what
collapse-by-length would predict does not sit comfortably with it.

The stimulus set happens to contain two regions where the control genuinely has
no repetition at all, and they sit at opposite ends of the ladder:

  * k <= 8 on the main ladder, where the eight-word pool covers k without
    cycling; and
  * the whole extension ladder (k = 48..128), whose controls are drawn from a
    146-word pool that is never cycled and whose generator asserts it.

If the effect is really about periodicity it must survive in both, and its
absence in either would mean the main result was partly an artifact of comparing
two periodic conditions. This script reports the contrast in each region
separately, using the same population rules as everything else.

Usage:  python analysis/aperiodic_controls.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel, cap_flags  # noqa: E402

DEGENERATE = {"empty", "degenerate"}


def contrast(d: pd.DataFrame, label: str, res: dict) -> None:
    rep = d[d.family == "word_rep"]
    ctl = d[d.family == "control_word"]
    if not len(rep) or not len(ctl):
        return
    row = dict(
        n_rep=int(len(rep)), n_ctl=int(len(ctl)),
        exact_rep=float((rep.err == 0).mean()), exact_ctl=float((ctl.err == 0).mean()),
        median_rep=float(rep.err.median()), median_ctl=float(ctl.err.median()))
    row["exact_gap"] = row["exact_ctl"] - row["exact_rep"]
    res[label] = row
    print(f"{label:34s} rep n={row['n_rep']:4d} exact={100*row['exact_rep']:5.1f}% "
          f"med={row['median_rep']:+.3f} | ctl n={row['n_ctl']:4d} "
          f"exact={100*row['exact_ctl']:5.1f}% med={row['median_ctl']:+.3f} | "
          f"gap={100*row['exact_gap']:+5.1f} pts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--ext", nargs="+",
                    default=["data/results/behavioural_ext.csv",
                             "data/results/behavioural_ext_llasa.csv"])
    ap.add_argument("--pool-size", type=int, default=8,
                    help="main-ladder filler pool; controls are aperiodic at k<=this")
    ap.add_argument("--out", default="data/results/aperiodic_controls.json")
    args = ap.parse_args()

    res: dict = {"pool_size": args.pool_size}

    main_d = pd.read_csv(args.behavioural)
    main_d = main_d[main_d.family.isin(["word_rep", "control_word"])]
    main_d, _ = panel(main_d)
    main_d = main_d.assign(err=(main_d.count_a - main_d.k) / main_d.k)

    print("Controls contain no repeated filler in the two regions below; "
          "between them\nthe eight-word pool is cycled and the contrast is "
          "period-1 against period-8.\n")
    contrast(main_d[(main_d.k >= 2) & (main_d.k <= args.pool_size)],
             f"main ladder, k<={args.pool_size} (aperiodic)", res)
    contrast(main_d[main_d.k > args.pool_size],
             f"main ladder, k>{args.pool_size} (period-8 ctl)", res)

    frames = [pd.read_csv(p) for p in args.ext if Path(p).exists()]
    if frames:
        e = pd.concat(frames, ignore_index=True)
        e = e[e.family.isin(["word_rep", "control_word"])]
        flags = cap_flags()
        e = e[[not flags.get((r.model, r.item_id, r.seed), False) for r in e.itertuples()]]
        e = e[~e.outcome.isin(DEGENERATE)]
        e = e.assign(err=(e.count_a - e.k) / e.k)
        contrast(e, "extension k>=48 (146-word pool)", res)

    keys = [k for k in res if k != "pool_size"]
    ap_keys = [k for k in keys if "aperiodic" in k or "146-word" in k]
    gaps = [res[k]["exact_gap"] for k in ap_keys]
    res["aperiodic_regions"] = ap_keys
    res["aperiodic_gaps"] = gaps
    res["holds_in_all_aperiodic"] = bool(all(g > 0 for g in gaps)) if gaps else False
    print(f"\nexact-rate gap in the aperiodic regions: "
          f"{', '.join(f'{100*g:+.1f} pts' for g in gaps)}")
    print("The dissociation is not an artifact of comparing two periodic "
          "conditions."
          if res["holds_in_all_aperiodic"] else
          "The dissociation does NOT survive in a genuinely aperiodic control.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
