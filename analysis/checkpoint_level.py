#!/usr/bin/env python3
"""Treat the checkpoint as the unit of replication, not the generation.

Round-5 reviewers objected, correctly, that the paper's evidentiary standard was
a set of vote counts --- "5 of 6", "4 of 6", "80% of pairs" --- with different
denominators, no common test, and no multiplicity control, while the pooled
confidence intervals were computed over *generations*. Those two things answer
different questions. Pooling generations asks "is the effect present in this
corpus"; the claim the paper actually makes is "this happens to models of this
kind", and for that the checkpoint is the sampling unit.

So: reduce each checkpoint to one number per contrast (its own repeated minus
its own control), then do inference across the six. Every generation-level
nuisance --- how many items a model produced, how often it went degenerate, how
long its utterances run --- is absorbed into the per-checkpoint summary, and
what comes out is an estimate that generalises in the direction the paper
claims.

Three statistics, deliberately including the least flattering:

* the mean paired difference with a bootstrap CI over checkpoints;
* the Wilcoxon signed-rank test. With n = 6 its smallest attainable two-sided
  p is 0.031, so a perfect result cannot beat that however large the effect.
  We report the value with that floor stated rather than implying more
  resolution than six checkpoints can carry;
* between-checkpoint heterogeneity (SD of the paired differences and I-squared),
  because "holds in all six" and "holds in all six by wildly different amounts"
  are different claims and the vote count could not distinguish them.

Usage:  python analysis/checkpoint_level.py
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402


def wilcoxon_signed_rank(d: np.ndarray) -> tuple[float, float]:
    """Exact two-sided signed-rank test. n is 6, so enumerate all 2^n sign
    assignments rather than lean on a normal approximation that is invalid here.
    """
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_obs = float(np.sum(ranks[d > 0]))
    total = float(ranks.sum())
    stat = min(w_obs, total - w_obs)
    count = 0
    for signs in product([0, 1], repeat=n):
        w = float(np.sum(ranks[np.array(signs, dtype=bool)]))
        if min(w, total - w) <= stat + 1e-9:
            count += 1
    return stat, count / (2 ** n)


def boot_mean(d: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = [float(np.mean(rng.choice(d, d.size, replace=True))) for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def heterogeneity(d: np.ndarray) -> dict:
    """SD of the per-checkpoint effects, and I^2 against their own sampling
    spread. A crude I^2 --- we have one estimate per checkpoint, not a variance
    each --- so it is reported as an indication, not a formal statistic."""
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    se = sd / np.sqrt(len(d)) if len(d) else float("nan")
    q = float(np.sum((d - d.mean()) ** 2) / (se ** 2)) if se and se > 0 else float("nan")
    df = len(d) - 1
    i2 = max(0.0, (q - df) / q) * 100 if np.isfinite(q) and q > 0 else float("nan")
    return dict(sd=sd, i2=i2)


def report(name: str, diffs: dict[str, float], out: dict) -> None:
    models = sorted(diffs)
    d = np.array([diffs[m] for m in models], dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        print(f"{name}: too few checkpoints")
        return
    lo, hi = boot_mean(d)
    stat, p = wilcoxon_signed_rank(d)
    het = heterogeneity(d)
    out[name] = dict(per_model={m: float(diffs[m]) for m in models},
                     mean=float(d.mean()), lo=lo, hi=hi,
                     wilcoxon_w=stat, wilcoxon_p=p, n=len(d),
                     n_positive=int((d > 0).sum()), **het)
    print(f"\n{name}  (n={len(d)} checkpoints)")
    for m in models:
        print(f"    {m:10s} {diffs[m]:+8.3f}")
    print(f"  mean {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]   "
          f"Wilcoxon W={stat:.0f}, p={p:.4f}"
          + ("  (= the exact floor for n=6)" if abs(p - 2 / 2 ** len(d)) < 1e-9 else ""))
    print(f"  between-checkpoint SD {het['sd']:.3f}, I^2 {het['i2']:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--capacity", default="data/results/capacity.json")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/checkpoint_level.json")
    args = ap.parse_args()

    out: dict = {"unit": "checkpoint", "kmin": args.kmin,
                 "note": "n=6 gives an exact two-sided Wilcoxon floor of 0.031"}

    d = pd.read_csv(args.behavioural)
    d = d[d.family.isin(["word_rep", "control_word"])]
    d, _ = panel(d)
    d = d[d.k >= args.kmin]
    d = d.assign(err=(d.count_a - d.k) / d.k)

    # Contrast 1: does a checkpoint miscount repeated text more than its own
    # length-matched control? Signed so that a *negative* difference is the
    # predicted direction; we flip it so positive means "effect present".
    diffs = {}
    for m, g in d.groupby("model"):
        r = g[g.family == "word_rep"].err.median()
        c = g[g.family == "control_word"].err.median()
        if np.isfinite(r) and np.isfinite(c):
            diffs[m] = float(c - r)
    report("count_error_gap", diffs, out)

    # Contrast 2: same question for the exactly-correct rate, which is the
    # statistic the paper now leads with and is not a median of a skewed
    # quantity.
    diffs = {}
    for m, g in d.groupby("model"):
        r = g[g.family == "word_rep"]
        c = g[g.family == "control_word"]
        if len(r) and len(c):
            diffs[m] = float((c.err == 0).mean() - (r.err == 0).mean())
    report("exact_rate_gap", diffs, out)

    # Contrast 3: the capacity mechanism, checkpoint by checkpoint.
    cap_path = Path(args.capacity)
    if cap_path.exists():
        cap = json.loads(cap_path.read_text())
        from src.common.population import ABLATIONS
        diffs = {m: float(v["control"]["gain"] - v["repeated"]["gain"])
                 for m, v in cap["models"].items()
                 if m not in ABLATIONS
                 and np.isfinite(v["control"]["gain"])
                 and np.isfinite(v["repeated"]["gain"])}
        report("capacity_gap", diffs, out)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
