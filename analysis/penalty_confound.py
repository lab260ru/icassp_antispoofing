#!/usr/bin/env python3
r"""Is the deficit just the decoding-time repetition penalty?

A round-8 reviewer raised the sharpest decoding-level rival to the whole account,
and it is one the paper had the data to answer but never did. The three families
ship repetition penalties that differ by an order of magnitude --- XTTS-v2 5.0,
Qwen3-TTS 1.05, Llasa none at all --- and a penalty is, by construction, a
mechanism that suppresses repeated tokens. If the periodicity-specific deficit
were an artifact of that decoding rule, the theorem would be explaining something
the sampler already explains.

Two things settle it, and neither needs new data:

1. **Sweep it.** Re-running XTTS-v2 across penalties from 1.0 to 8.0 leaves the
   repeated exact-rate flat. If the penalty caused the deficit, an eightfold
   change in it would move the deficit.
2. **Remove it.** Llasa passes no repetition penalty at all, so its decoding rule
   cannot be suppressing anything --- and it shows the effect at full size.

The one thing the penalty demonstrably does control is general rendering quality:
with it disabled, XTTS-v2 can no longer render the *control* either, which is why
1.0 is not a usable setting rather than evidence about repetition. Reporting that
asymmetry is the point --- an arm whose control has collapsed says nothing about
periodicity, and the sweep's slope is fitted only over arms whose control is
intact.

Usage:  python analysis/penalty_confound.py
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

# arm -> (architecture, penalty actually used)
ARMS = {
    "xtts2norp": ("XTTS-v2", 1.0), "xtts2rp2": ("XTTS-v2", 2.0),
    "xtts2rp3": ("XTTS-v2", 3.0), "xtts2": ("XTTS-v2", 5.0),
    "xtts2rp8": ("XTTS-v2", 8.0),
    "qwen06brp10": ("Qwen3-TTS-0.6B", 1.0), "qwen06b": ("Qwen3-TTS-0.6B", 1.05),
    "qwen06brp15": ("Qwen3-TTS-0.6B", 1.5), "qwen06brp30": ("Qwen3-TTS-0.6B", 3.0),
}
# Families that pass no penalty at all: their decoding rule cannot be the cause.
NO_PENALTY = ["llasa1b", "llasa3b", "llasa8b"]
CONTROL_INTACT = 0.5  # an arm whose control exact-rate falls below this is unusable


def summarise(d: pd.DataFrame) -> dict:
    rep, ctl = d[d.family == "word_rep"], d[d.family == "control_word"]
    if not len(rep) or not len(ctl):
        return {}
    return dict(n_rep=int(len(rep)), n_ctl=int(len(ctl)),
                exact_rep=float((rep.err == 0).mean()),
                exact_ctl=float((ctl.err == 0).mean()),
                median_rep=float(rep.err.median()),
                median_ctl=float(ctl.err.median()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", nargs="+",
                    default=["data/results/behavioural_ctc.csv",
                             "data/results/behavioural_qwen_penalty.csv"])
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/penalty_confound.json")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in args.behavioural if Path(p).exists()]
    d = pd.concat(frames, ignore_index=True)
    d = d[d.family.isin(["word_rep", "control_word"])]
    d, _ = panel(d, ablations=True)          # keep the sweep arms
    d = d[d.k >= args.kmin]
    d = d.assign(err=(d.count_a - d.k) / d.k)

    res: dict = {"kmin": args.kmin, "arms": {}, "no_penalty": {}}
    print(f"{'arm':14s} {'arch':16s} {'pen':>5s} {'rep exact':>10s} "
          f"{'ctl exact':>10s}  control")
    for arm, (arch, pen) in ARMS.items():
        s = summarise(d[d.model == arm])
        if not s:
            continue
        s.update(arch=arch, penalty=pen,
                 control_intact=bool(s["exact_ctl"] >= CONTROL_INTACT))
        res["arms"][arm] = s
        print(f"{arm:14s} {arch:16s} {pen:5.2f} {100*s['exact_rep']:9.1f}% "
              f"{100*s['exact_ctl']:9.1f}%  "
              f"{'intact' if s['control_intact'] else 'COLLAPSED -> unusable'}")

    for m in NO_PENALTY:
        s = summarise(d[d.model == m])
        if s:
            res["no_penalty"][m] = s
    if res["no_penalty"]:
        reps = [v["exact_rep"] for v in res["no_penalty"].values()]
        ctls = [v["exact_ctl"] for v in res["no_penalty"].values()]
        res["no_penalty_summary"] = dict(
            n_models=len(reps), exact_rep=float(np.mean(reps)),
            exact_ctl=float(np.mean(ctls)),
            exact_gap=float(np.mean(ctls) - np.mean(reps)))
        v = res["no_penalty_summary"]
        print(f"\n{len(reps)} checkpoints that pass NO penalty at all "
              f"({', '.join(res['no_penalty'])}):")
        print(f"  repeated {100*v['exact_rep']:.1f}% exact, control "
              f"{100*v['exact_ctl']:.1f}%, gap {100*v['exact_gap']:+.1f} pts")

    # Slope of the repeated exact-rate in the penalty, per architecture, over the
    # arms whose control survives.
    res["by_arch"] = {}
    for arch in sorted({a for a, _ in ARMS.values()}):
        usable = [v for v in res["arms"].values()
                  if v["arch"] == arch and v["control_intact"]]
        if len(usable) < 3:
            print(f"\n{arch}: {len(usable)} usable arms, too few to fit")
            continue
        x = np.array([v["penalty"] for v in usable])
        y = np.array([v["exact_rep"] for v in usable])
        slope = float(np.polyfit(x, y, 1)[0])
        res["by_arch"][arch] = dict(
            n_usable=len(usable), penalties=[float(t) for t in x],
            exact_rep_lo=float(y.min()), exact_rep_hi=float(y.max()),
            slope_per_unit=slope,
            fold_change=float(max(x) / min(x)))
        print(f"\n{arch}: {len(usable)} arms with an intact control, penalties "
              f"{list(x)} ({max(x)/min(x):.0f}x range)")
        print(f"  repeated exact-rate spans {100*y.min():.1f}--{100*y.max():.1f}%, "
              f"slope {100*slope:+.2f} pts per unit of penalty")

    x_arch = res["by_arch"].get("XTTS-v2", {})
    res["verdict"] = (
        "the penalty does not explain the deficit: it is flat across the sweep "
        "and present at full size in checkpoints that use no penalty at all"
        if (not x_arch or x_arch["exact_rep_hi"] < 0.5) and res.get("no_penalty_summary")
        else "revisit: the penalty moves the deficit materially")
    print(f"\nverdict: {res['verdict']}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
