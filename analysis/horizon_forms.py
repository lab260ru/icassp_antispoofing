#!/usr/bin/env python3
"""Is the 3.5-fold horizon ratio a property of the data or of the curve we chose?

The extension ladder and its saturating fit were both chosen after seeing that
k<=32 was proportional. S16 dates every such decision against the commit log, but
a reviewer is right that dating *when* we chose the exponential form does not
show the form itself was not picked because it fit. Cells past k=48 hold 21-26
generations; at that size the choice of curve can move a number a long way.

So we refit with families we did not choose. All three are one-parameter, all
three have unit slope as k->0 (so none of them fights the proportional regime the
main ladder shows), and all three asymptote to N*. They differ only in how they
bend between those two ends:

    soft horizon   c(k) = N* (1 - exp(-k/N*))     the fit reported in the paper
    hyperbolic     c(k) = N* k / (N* + k)         bends earliest, heaviest tail
    tanh           c(k) = N* tanh(k/N*)           bends latest, sharpest knee

Hyperbolic and tanh bracket the exponential from both sides, so if the ratio
N*_ctl / N*_rep survives all three, it is not the exponential doing the work.

What this does and does not settle. It answers "did the functional form make the
effect", because the three families disagree about N* by construction and the
ratio divides the disagreement out. It does not answer "is a horizon the right
model at all" --- every family here is a saturating one, and the declining
repeated counts past the plateau (--report-decline in horizon_ext.py) still fit
none of them. A reader who doubts saturation itself is not answered by this
script, and we say so rather than letting three agreeing curves imply more
agreement than they carry.

Usage:  python analysis/horizon_forms.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import cap_flags  # noqa: E402

DEGENERATE = {"empty", "degenerate"}

# Each maps (k, N*) -> predicted count. Unit slope at the origin and asymptote
# N* are what make them comparable; everything else about them differs.
FORMS = {
    "soft_horizon": lambda k, n: n * (1.0 - np.exp(-k / n)),
    "hyperbolic": lambda k, n: n * k / (n + k),
    "tanh": lambda k, n: n * np.tanh(k / n),
}


def fit(form, k: np.ndarray, c: np.ndarray) -> float:
    """Least-squares N* on the same geometric grid horizon_ext.py uses."""
    if len(k) < 3:
        return float("nan")
    grid = np.geomspace(1.0, 4000.0, 3000)
    best, best_sse = float("nan"), np.inf
    for n in grid:
        sse = float(((c - form(k, n)) ** 2).sum())
        if sse < best_sse:
            best, best_sse = float(n), sse
    return best


def boot(form, df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    med = df.groupby("k").count_a.median()
    point = fit(form, med.index.to_numpy(float), med.to_numpy(float))
    vals, idx = [], np.arange(len(df))
    for _ in range(n_boot):
        s = df.iloc[rng.choice(idx, len(idx), replace=True)]
        m = s.groupby("k").count_a.median()
        if len(m) >= 3:
            v = fit(form, m.index.to_numpy(float), m.to_numpy(float))
            if np.isfinite(v):
                vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--ext", nargs="+",
                    default=["data/results/behavioural_ext.csv",
                             "data/results/behavioural_ext_llasa.csv"])
    ap.add_argument("--templates", nargs="+", default=["t1", "t3", "t4"])
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default="data/results/horizon_forms.json")
    args = ap.parse_args()

    frames = [pd.read_csv(args.main).assign(ladder="main")]
    for p in args.ext:
        if Path(p).exists():
            frames.append(pd.read_csv(p).assign(ladder="ext"))
    d = pd.concat(frames, ignore_index=True)

    flags = cap_flags()
    d["cap"] = [flags.get((r.model, r.item_id, r.seed), False) for r in d.itertuples()]
    d = d[(~d.cap) & (~d.outcome.isin(DEGENERATE)) & (d.k >= args.kmin)
          & d.template.isin(args.templates)
          & d.family.isin(["word_rep", "control_word"])]
    models = sorted(m for m in d.model.unique()
                    if set(d[d.model == m].ladder) == {"main", "ext"})
    if not models:
        raise SystemExit("no model has both ladders yet")

    res: dict = {"forms": list(FORMS), "templates": args.templates,
                 "kmin": args.kmin, "models": {}, "pooled": {}}

    print(f"{'form':13s} {'N*_rep':>8s} {'N*_ctl':>8s} {'ratio':>6s}   pooled fit")
    for name, form in FORMS.items():
        row = {}
        for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
            pt, lo, hi = boot(form, d[d.family == fam], args.n_boot, seed=1)
            row[tag] = dict(n_star=pt, lo=lo, hi=hi)
        row["ratio"] = float(row["ctl"]["n_star"] / row["rep"]["n_star"])
        res["pooled"][name] = row
        print(f"{name:13s} {row['rep']['n_star']:8.1f} {row['ctl']['n_star']:8.1f} "
              f"{row['ratio']:6.2f}")

    for m in models:
        res["models"][m] = {}
        for name, form in FORMS.items():
            row = {}
            for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
                s = d[(d.model == m) & (d.family == fam)]
                pt, lo, hi = boot(form, s, args.n_boot, seed=0)
                row[tag] = dict(n_star=pt, lo=lo, hi=hi)
            row["ratio"] = float(row["ctl"]["n_star"] / row["rep"]["n_star"])
            res["models"][m][name] = row

    ratios = [res["pooled"][f]["ratio"] for f in FORMS]
    res["pooled_ratio_min"] = float(min(ratios))
    res["pooled_ratio_max"] = float(max(ratios))
    res["pooled_ratio_spread"] = float(max(ratios) - min(ratios))
    res["n_models"] = len(models)
    # Direction and magnitude are separate questions and the answers differ, so
    # they are scored separately. A family that merely rescales every N* leaves
    # each checkpoint on the same side of 1 and changes only how far; a family
    # that flips a checkpoint changes the claim.
    per_model_signs = {m: sorted({res["models"][m][f]["ratio"] > 1
                                  for f in FORMS}) for m in models}
    res["direction_stable"] = bool(all(len(v) == 1 for v in per_model_signs.values()))
    res["n_above_one"] = {f: sum(1 for m in models
                                 if res["models"][m][f]["ratio"] > 1)
                          for f in FORMS}
    res["reversing"] = sorted(m for m in models
                              if res["models"][m]["soft_horizon"]["ratio"] < 1)

    print(f"\n{'model':9s} " + " ".join(f"{f:>13s}" for f in FORMS))
    for m in models:
        print(f"{m:9s} " + " ".join(f"{res['models'][m][f]['ratio']:13.2f}"
                                    for f in FORMS))
    print(f"\npooled ratio across families: {min(ratios):.2f}-{max(ratios):.2f} "
          f"(spread {max(ratios)-min(ratios):.2f})")
    print("checkpoints above 1: "
          + ", ".join(f"{f} {res['n_above_one'][f]}/{len(models)}" for f in FORMS))

    res["verdict"] = (
        "direction is form-independent; magnitude is not"
        if res["direction_stable"] else
        "a saturating family flips a checkpoint: the panel claim itself is "
        "form-dependent, not just its size")
    print(f"\n{res['verdict']}.")
    if res["direction_stable"]:
        print("No family moves any checkpoint across 1, so which checkpoints show a\n"
              f"lowered horizon (and which do not: {', '.join(res['reversing']) or 'none'})\n"
              "does not depend on the curve. How far it is lowered does: the pooled\n"
              f"ratio runs {min(ratios):.1f}-{max(ratios):.1f} across the three, so the\n"
              "headline number should be read as that range, not as its midpoint.")
    print("\nThis tests the choice of saturating curve, not the choice to fit a\n"
          "saturating curve at all. The decline past the plateau still fits none\n"
          "of these three.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
