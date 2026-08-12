#!/usr/bin/env python3
"""Relative count error: the behavioural measurement, done properly.

Two corrections to the first version of this analysis, both forced by evidence.

**The judge.** Whisper large-v3 is autoregressive and de-duplicates repeated
speech: on concatenative audio with known ground truth its counted/true ratio
falls to 0.27--0.65 for k >= 4 while matched distinct-word audio of the same
length scores 1.00 (`analysis/asr_reliability.py`). Scoring repetition counting
with an AR recogniser measures the phenomenon with an instrument made of the
phenomenon. We therefore score with a CTC recogniser, which has no decoder and no
language model, so its output length tracks the acoustics.

**The metric.** Exact-match correctness is the wrong resolution. A model that
renders 30 of 32 requested repetitions is doing something very different from one
that renders 3, but exact match scores both zero, and a +/-1 slop in the
recogniser flips the label. The informative quantity is the *relative count
error* `(counted - k) / k`: it is signed, so undercounting (premature stopping)
is distinguishable from overcounting (looping); it is scale-free, so k=4 and
k=32 are comparable; and it degrades gracefully under recogniser noise.

The matched control is what makes it interpretable. Both item types ask "how many
of the k requested units came out?", so a control error of zero at the same k
where the repeated error is large isolates periodicity from every property the
two share.

Usage:  python analysis/count_error.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import ABLATIONS, panel, describe  # noqa: E402


def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(x, x.size, replace=True))) for _ in range(n_boot)]
    return tuple(np.percentile(meds, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--out", default="data/results/count_error.json")
    ap.add_argument("--kmin", type=int, default=6)
    args = ap.parse_args()

    raw = pd.read_csv(args.behavioural)
    raw = raw[raw.family.isin(["word_rep", "control_word"]) & (raw.k >= 2)].copy()
    # Keep degenerate rows through the shared filter so their rate stays
    # reportable, then mask them out of the error statistics below.
    d, drop = panel(raw, degenerate=True)
    print(describe(drop))
    degen = d.outcome.isin(["empty", "degenerate"])
    d["rel_err"] = (d.count_a - d.k) / d.k

    res: dict = {"judge": "ctc", "kmin": args.kmin, "models": {}}
    print(f"{'model':11s} {'rep rel.err k>=6':>26s} {'ctl rel.err k>=6':>26s}  {'degen%':>7s}")
    for m in sorted(set(d.model.unique()) - ABLATIONS | (set(d.model.unique()) & ABLATIONS)):
        row: dict = {}
        for fam, key in (("word_rep", "rep"), ("control_word", "ctl")):
            s = d[(d.model == m) & (d.family == fam) & (d.k >= args.kmin) & ~degen]
            v = s.rel_err.to_numpy(dtype=float)
            lo, hi = boot_ci(v)
            row[key] = dict(median=float(np.median(v)) if v.size else np.nan,
                            lo=lo, hi=hi, n=int(v.size))
            row[key + "_by_k"] = {
                int(k): float(g.rel_err.median())
                for k, g in d[(d.model == m) & (d.family == fam) & ~degen].groupby("k")}
        dg = d[(d.model == m) & (d.family == "word_rep") & (d.k >= args.kmin)]
        row["degen_rate"] = float(degen[dg.index].mean()) if len(dg) else np.nan
        row["separated"] = bool(np.isfinite(row["rep"]["hi"]) and np.isfinite(row["ctl"]["lo"])
                                and row["rep"]["hi"] < row["ctl"]["lo"])
        res["models"][m] = row
        print(f"{m:11s} {row['rep']['median']:+7.3f} [{row['rep']['lo']:+.3f},{row['rep']['hi']:+.3f}] "
              f"n={row['rep']['n']:4d}  "
              f"{row['ctl']['median']:+7.3f} [{row['ctl']['lo']:+.3f},{row['ctl']['hi']:+.3f}] "
              f"n={row['ctl']['n']:4d}  {100*row['degen_rate']:6.1f}"
              f"{'  *' if row['separated'] else ''}")

    panel_models = {k: v for k, v in res["models"].items() if k not in ABLATIONS}
    pd_ = d[(d.k >= args.kmin) & ~degen]
    for fam, key in (("word_rep", "rep"), ("control_word", "ctl")):
        v = pd_[pd_.family == fam].rel_err.to_numpy(dtype=float)
        lo, hi = boot_ci(v)
        res[f"pooled_{key}"] = dict(median=float(np.median(v)), lo=lo, hi=hi, n=int(v.size))
    res["n_models"] = len(panel_models)
    res["n_separated"] = sum(1 for v in panel_models.values() if v["separated"])
    res["n_rep_below_ctl"] = sum(
        1 for v in panel_models.values()
        if np.isfinite(v["rep"]["median"]) and np.isfinite(v["ctl"]["median"])
        and v["rep"]["median"] < v["ctl"]["median"])

    r, c = res["pooled_rep"], res["pooled_ctl"]
    print(f"\npooled at k>={args.kmin}, CTC judge, panel only:")
    print(f"  repeated {r['median']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  n={r['n']}")
    print(f"  control  {c['median']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]  n={c['n']}")
    print(f"  disjoint CIs in {res['n_separated']}/{res['n_models']} models; "
          f"repeated below control in {res['n_rep_below_ctl']}/{res['n_models']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
