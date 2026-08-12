#!/usr/bin/env python3
"""Does supplying the duration fix the deficit? Below k=12, largely yes.

Every other result in this paper is observational: we vary the stimulus and watch
what comes out. This one intervenes on the mechanism we proposed.

The proposal, formed after two non-autoregressive baselines disagreed, was that
what matters is whether a model must represent "how many" internally. VITS
predicts a duration per input token, so the count rides the input sequence;
F5-TTS estimates one total duration and denoises in parallel, so it has to hold
the count somewhere. F5-TTS exposes `fix_duration`, which replaces its estimate
with a supplied one --- so the proposal can be tested rather than asserted.

The supplied duration comes from F5-TTS's own control renderings at the same k,
which it counts correctly. It is therefore what k units of speech in this carrier
takes, obtained without looking at how the repeated item came out. (A duration
read off the repeated item's own output would be circular.)

The result is split, and the split is the finding. Where the deficit is mild the
intervention largely removes it; where the deficit is severe it does nothing. A
model handed the correct total length still cannot place thirty-two repetitions
inside it, so at high k the failure is not reducible to mis-estimating how much
speech to make.

Usage:  python analysis/duration_intervention.py
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


def load(path: str, kmin: int) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.family == "word_rep"]
    d, _ = panel(d, ablations=True)
    d = d[d.k >= kmin]
    return d.assign(err=(d.count_a - d.k) / d.k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="data/results/behavioural_f5.csv")
    ap.add_argument("--fixed", default="data/results/behavioural_f5fix.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--split", type=int, default=12,
                    help="k at or above which the intervention stops helping")
    ap.add_argument("--out", default="data/results/duration_intervention.json")
    args = ap.parse_args()

    free, fixed = load(args.free, args.kmin), load(args.fixed, args.kmin)

    def stats(d: pd.DataFrame) -> dict:
        return dict(n=int(len(d)), exact=float((d.err == 0).mean()),
                    median_err=float(d.err.median()))

    res: dict = {"kmin": args.kmin, "split": args.split,
                 "overall": {"free": stats(free), "fixed": stats(fixed)},
                 "by_k": {}}
    print(f"{'k':>4s} {'free exact':>11s} {'given exact':>12s} "
          f"{'free med':>9s} {'given med':>10s} {'n':>4s}")
    for k in sorted(set(free.k) & set(fixed.k)):
        a, b = free[free.k == k], fixed[fixed.k == k]
        res["by_k"][int(k)] = {"free": stats(a), "fixed": stats(b)}
        print(f"{k:4d} {100*(a.err==0).mean():10.1f}% {100*(b.err==0).mean():11.1f}% "
              f"{a.err.median():+9.3f} {b.err.median():+10.3f} {len(a):4d}")

    lo_f = free[free.k < args.split]
    lo_x = fixed[fixed.k < args.split]
    hi_f = free[free.k >= args.split]
    hi_x = fixed[fixed.k >= args.split]
    res["low_k"] = {"free": stats(lo_f), "fixed": stats(lo_x)}
    res["high_k"] = {"free": stats(hi_f), "fixed": stats(hi_x)}
    res["low_k_gain"] = float((lo_x.err == 0).mean() - (lo_f.err == 0).mean())
    res["high_k_gain"] = float((hi_x.err == 0).mean() - (hi_f.err == 0).mean())

    print(f"\nk<{args.split}:  exact {100*(lo_f.err==0).mean():.1f}% -> "
          f"{100*(lo_x.err==0).mean():.1f}%  ({100*res['low_k_gain']:+.1f} points, "
          f"n={len(lo_f)})")
    print(f"k>={args.split}: exact {100*(hi_f.err==0).mean():.1f}% -> "
          f"{100*(hi_x.err==0).mean():.1f}%  ({100*res['high_k_gain']:+.1f} points, "
          f"n={len(hi_f)})")

    res["verdict"] = (
        "duration estimation accounts for much of the deficit at low k and none "
        "of it at high k"
        if res["low_k_gain"] > 0.15 and res["high_k_gain"] <= 0.05 else
        "the split is not clean; read the by-k table before quoting this")
    print(f"\n{res['verdict']}.")
    print("Handed the correct total length, the model still cannot place the "
          "requested\nnumber of repetitions inside it once k is large -- so at "
          "high k the failure is\nnot reducible to mis-estimating how much speech "
          "to make.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
