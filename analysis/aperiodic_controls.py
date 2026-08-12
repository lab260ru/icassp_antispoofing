#!/usr/bin/env python3
"""The dissociation, restricted to controls that really are aperiodic.

Two round-6 reviewers made the same fair objection: the paper calls the
repeated-vs-control contrast its sharpest test of "periodicity, not length", but
the main ladder's control fillers are drawn from a pool of eight and cycled, so
a k=32 control repeats each filler four times. That contrast is period-1 against
period-8, not against aperiodic, and the Discussion's claim about what
collapse-by-length would predict does not sit comfortably with it.

Three regions have controls that genuinely carry no repetition:

  * k <= 8 on the main ladder, where the eight-word pool covers k without
    cycling;
  * the whole extension ladder (k = 48..128), whose controls come from a
    146-word pool that is never cycled and whose generator asserts it; and
  * **k = 12..32, re-generated for this check** with the same 146-word pool
    (`data/stimuli/stimuli_aperiodic.jsonl`, `scripts/run_aperiodic.sh`). This
    is the range the objection was actually about, so working around it with
    sub-analyses at the two ends was the cheap answer; running it is the right
    one. These items are paired against the *same* repeated items the main
    ladder already scored, so only the control side changes.

If the effect is about periodicity it must survive in all three, and its absence
in any would mean the headline was partly an artifact of comparing two periodic
conditions.

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
    ap.add_argument("--aperiodic", default="data/results/behavioural_aperiodic.csv",
                    help="re-generated aperiodic controls for k=12..32")
    ap.add_argument("--pool-size", type=int, default=8,
                    help="main-ladder filler pool; controls are aperiodic at k<=this")
    ap.add_argument("--out", default="data/results/aperiodic_controls.json")
    args = ap.parse_args()

    res: dict = {"pool_size": args.pool_size}

    main_d = pd.read_csv(args.behavioural)
    main_d = main_d[main_d.family.isin(["word_rep", "control_word"])]
    main_d, _ = panel(main_d)
    main_d = main_d.assign(err=(main_d.count_a - main_d.k) / main_d.k)

    print("Rows marked aperiodic have controls with no repeated filler at all.\n"
          "The period-8 row is the main ladder above k=8, kept for comparison.\n")
    contrast(main_d[(main_d.k >= 2) & (main_d.k <= args.pool_size)],
             f"main ladder, k<={args.pool_size} (aperiodic)", res)
    contrast(main_d[main_d.k > args.pool_size],
             f"main ladder, k>{args.pool_size} (period-8 ctl)", res)

    # The re-generated controls for the range the objection was about. Pair them
    # with the repeated items already scored on the main ladder at the same k,
    # so the only thing that differs between the two arms is whether the control
    # repeats its own fillers.
    apc = Path(args.aperiodic)
    if apc.exists():
        a = pd.read_csv(apc)
        a, _ = panel(a)
        a = a.assign(err=(a.count_a - a.k) / a.k)
        ks = sorted(a.k.unique())
        rep_same = main_d[(main_d.family == "word_rep") & main_d.k.isin(ks)
                          & main_d.model.isin(a.model.unique())]
        contrast(pd.concat([rep_same, a[a.family == "control_word"]]),
                 f"k={min(ks)}-{max(ks)} re-generated (aperiodic)", res)

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
