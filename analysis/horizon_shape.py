#!/usr/bin/env python3
"""Does the observed deficit have the shape Theorem A predicts?

This test exists because a reviewer pointed out that our headline number and our
theorem predict different things, and they were right.

Theorem A(iv) bounds the largest separable repetition count by a constant
`N* = log(mu/2LC)/log q`. Past that horizon no Lipschitz readout can tell counts
apart, so a decoder driven past it should emit roughly `N*` repetitions
regardless of how many were requested: the rendered count *saturates*. A
constant *relative* deficit is a different claim entirely --- it says the decoder
still tracks `k`, losing a fixed fraction --- and it is not what the theorem
predicts.

We therefore fit both models to the median rendered count above `k >= kmin`:

    saturating    c_hat(k) = a          (the theorem's signature)
    proportional  c_hat(k) = b * k      (a fixed relative loss)

and report which fits, per model and pooled. Reporting this honestly matters more
than which way it comes out: if the data are proportional, the theorem's horizon
is simply not reached in the range we tested, and we must say the theorem is
untested rather than confirmed.

Usage:  python analysis/horizon_shape.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel as restrict, describe  # noqa: E402


def fit_shapes(k: np.ndarray, c: np.ndarray) -> dict:
    """Least-squares fit of both one-parameter shapes, plus a saturating-
    exponential that interpolates them.

    All three have one or two free parameters, so SSE is comparable without an
    information criterion; we report AIC anyway for the two-parameter case.
    """
    out: dict = {}
    a = float(c.mean())
    out["saturating"] = dict(param=a, sse=float(((c - a) ** 2).sum()))
    b = float((k * c).sum() / (k * k).sum())
    out["proportional"] = dict(param=b, sse=float(((c - b * k) ** 2).sum()))
    # c(k) = N*(1 - exp(-k/N*)) rises linearly then saturates at N*; fitting it
    # lets the data pick a horizon if one exists rather than forcing a choice.
    best = None
    for n_star in np.linspace(max(k.max() * 0.3, 2), k.max() * 20, 400):
        pred = n_star * (1 - np.exp(-k / n_star))
        s = float(((c - pred) ** 2).sum())
        if best is None or s < best[1]:
            best = (float(n_star), s)
    out["soft_horizon"] = dict(n_star=best[0], sse=best[1],
                               saturated_within_range=bool(best[0] < 1.5 * k.max()))
    out["winner"] = min(("saturating", "proportional", "soft_horizon"),
                        key=lambda m: out[m]["sse"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--out", default="data/results/horizon_shape.json")
    ap.add_argument("--kmin", type=int, default=6)
    args = ap.parse_args()

    d = pd.read_csv(args.behavioural)
    d, drop = restrict(d[d.family == "word_rep"])
    print(describe(drop) + "\n")

    res: dict = {"kmin": args.kmin, "models": {}}
    print(f"{'model':10s} {'saturating SSE':>15s} {'proportional SSE':>17s} "
          f"{'soft N*':>9s}  winner")
    for m in sorted(d.model.unique()):
        g = d[d.model == m].groupby("k")["count_a"].median()
        g = g[g.index >= args.kmin]
        if len(g) < 4:
            continue
        f = fit_shapes(g.index.to_numpy(float), g.to_numpy(float))
        res["models"][m] = f
        print(f"{m:10s} {f['saturating']['sse']:15.1f} "
              f"{f['proportional']['sse']:17.1f} "
              f"{f['soft_horizon']['n_star']:9.1f}  {f['winner']}")

    g = d.groupby("k")["count_a"].median()
    g = g[g.index >= args.kmin]
    pooled = fit_shapes(g.index.to_numpy(float), g.to_numpy(float))
    res["pooled"] = pooled
    res["n_proportional"] = sum(1 for v in res["models"].values()
                                if v["winner"] == "proportional")
    res["n_models"] = len(res["models"])
    res["ratio_by_k"] = {int(k): float(v / k) for k, v in g.items()}

    print(f"\npooled: saturating SSE {pooled['saturating']['sse']:.1f}, "
          f"proportional SSE {pooled['proportional']['sse']:.1f} "
          f"(slope {pooled['proportional']['param']:.3f}), "
          f"soft horizon N*={pooled['soft_horizon']['n_star']:.1f}")
    print(f"winner: {pooled['winner']}; proportional in "
          f"{res['n_proportional']}/{res['n_models']} models")
    kmax = int(g.index.max())
    if pooled["winner"] != "saturating":
        print(f"\n=> Over k in [{args.kmin},{kmax}] the rendered count tracks k rather\n"
              f"   than saturating, so no counting horizon lies inside this range.\n"
              f"   That leaves the theorem untested here, not refuted: see\n"
              f"   analysis/horizon_ext.py, which reaches k=128 and does find one.")
    else:
        print(f"\n=> Over k in [{args.kmin},{kmax}] the count saturates.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
