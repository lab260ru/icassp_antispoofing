#!/usr/bin/env python3
"""Where does the rendered count stop tracking the request, and does repetition
move that point?

The main ladder (k <= 32) cannot answer this: there the count is proportional to
k in five of six checkpoints, which is what a decoder still below its horizon
looks like. The extension ladder (k = 48..128) reaches past it, and both item
families turn out to saturate --- but at very different levels. That difference,
not the repeated saturation on its own, is the measurement.

Reporting only the repeated horizon would overclaim badly. These decoders have a
general utterance-length ceiling: their *controls* stop tracking k too, around
sixty units. If we quoted the repeated plateau alone, a reader could not tell
counting collapse from "the model will not talk for two minutes". The statistic
is therefore the ratio N*_ctl / N*_rep, which divides that ceiling out.

Model. For each family we fit

    c(k) = N* (1 - exp(-k / N*))

by least squares over the combined ladder. The form rises linearly at small k
(so it does not fight the proportional regime the main ladder shows) and
saturates at N*, so a single parameter carries "where does tracking stop". CIs
bootstrap over items.

Caveats that belong next to the number, not in a footnote:
  * Cells at high k are small; the CI reflects that and is usually wide.
  * The CTC judge undercounts long same-word runs specifically (blank-collapse,
    S9), which pushes N*_rep down and so inflates the ratio. The direction is
    known; the magnitude is not, so the ratio is an upper bound on the effect.
  * Repeated counts *decline* past the plateau rather than holding flat, which a
    pure horizon does not predict. `--report-decline` quantifies it.

Usage:  python analysis/horizon_ext.py
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


def fit_horizon(k: np.ndarray, c: np.ndarray) -> float:
    """Least-squares N* for c(k) = N*(1 - exp(-k/N*))."""
    if len(k) < 3:
        return float("nan")
    grid = np.geomspace(1.0, 4000.0, 3000)
    best, best_sse = float("nan"), np.inf
    for n in grid:
        pred = n * (1 - np.exp(-k / n))
        sse = float(((c - pred) ** 2).sum())
        if sse < best_sse:
            best, best_sse = float(n), sse
    return best


def boot_horizon(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    med = df.groupby("k").count_a.median()
    point = fit_horizon(med.index.to_numpy(float), med.to_numpy(float))
    vals = []
    idx = np.arange(len(df))
    for _ in range(n_boot):
        s = df.iloc[rng.choice(idx, len(idx), replace=True)]
        m = s.groupby("k").count_a.median()
        if len(m) >= 3:
            v = fit_horizon(m.index.to_numpy(float), m.to_numpy(float))
            if np.isfinite(v):
                vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--ext", nargs="+",
                    default=["data/results/behavioural_ext_llasa.csv"])
    ap.add_argument("--templates", nargs="+", default=["t1", "t3", "t4"],
                    help="carriers common to both ladders")
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default="data/results/horizon_ext.json")
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

    res: dict = {"models": {}, "templates": args.templates, "kmin": args.kmin}
    print(f"{'model':9s} {'N*_rep':>18s} {'N*_ctl':>18s} {'ratio':>6s}")
    for m in models:
        row: dict = {}
        for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
            s = d[(d.model == m) & (d.family == fam)]
            pt, lo, hi = boot_horizon(s, args.n_boot, seed=0)
            row[tag] = dict(n_star=pt, lo=lo, hi=hi, n=int(len(s)))
        ratio = (row["ctl"]["n_star"] / row["rep"]["n_star"]
                 if row["rep"]["n_star"] and np.isfinite(row["rep"]["n_star"]) else np.nan)
        row["ratio"] = float(ratio)
        res["models"][m] = row
        print(f"{m:9s} {row['rep']['n_star']:7.1f} [{row['rep']['lo']:5.1f},{row['rep']['hi']:6.1f}] "
              f"{row['ctl']['n_star']:7.1f} [{row['ctl']['lo']:5.1f},{row['ctl']['hi']:6.1f}] "
              f"{ratio:6.2f}")

    pooled: dict = {}
    for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
        s = d[d.family == fam]
        pt, lo, hi = boot_horizon(s, args.n_boot, seed=1)
        pooled[tag] = dict(n_star=pt, lo=lo, hi=hi, n=int(len(s)))
    pooled["ratio"] = float(pooled["ctl"]["n_star"] / pooled["rep"]["n_star"])
    res["pooled"] = pooled
    res["n_models"] = len(models)
    res["model_names"] = models
    # Reviewers asked for the sample sizes and the exclusion counts, at the k
    # values where the claim lives, rather than "few items per cell". They were
    # right that a reader cannot weigh the fit without them.
    ext = d[d.k >= 48]
    res["n_per_cell"] = {
        tag: {int(k): int(len(g))
              for k, g in ext[ext.family == fam].groupby("k")}
        for fam, tag in (("word_rep", "rep"), ("control_word", "ctl"))}
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw.k >= 48) & raw.template.isin(args.templates)
              & raw.model.isin(models)]
    raw_cap = pd.Series([flags.get((r.model, r.item_id, r.seed), False)
                         for r in raw.itertuples()], index=raw.index)
    res["excluded"] = dict(
        cap_hits=int(raw_cap.sum()),
        degenerate=int(raw.outcome.isin(DEGENERATE).sum()),
        n_generated=int(len(raw)))

    # Medians by k, both families: the raw evidence the fit summarises.
    res["median_by_k"] = {
        tag: {int(k): float(v) for k, v in
              d[d.family == fam].groupby("k").count_a.median().items()}
        for fam, tag in (("word_rep", "rep"), ("control_word", "ctl"))}

    # A pure horizon predicts a plateau. Ours declines; say by how much.
    rep = res["median_by_k"]["rep"]
    ext_ks = sorted(k for k in rep if k >= 48)
    if len(ext_ks) >= 2:
        res["decline"] = dict(
            k_peak=int(ext_ks[0]), c_peak=rep[ext_ks[0]],
            k_last=int(ext_ks[-1]), c_last=rep[ext_ks[-1]],
            frac=float(rep[ext_ks[-1]] / rep[ext_ks[0]]) if rep[ext_ks[0]] else np.nan)

    print(f"\npooled: N*_rep {pooled['rep']['n_star']:.1f} "
          f"[{pooled['rep']['lo']:.1f},{pooled['rep']['hi']:.1f}], "
          f"N*_ctl {pooled['ctl']['n_star']:.1f} "
          f"[{pooled['ctl']['lo']:.1f},{pooled['ctl']['hi']:.1f}], "
          f"ratio {pooled['ratio']:.2f}  ({len(models)} models)")
    if "decline" in res:
        dc = res["decline"]
        print(f"repeated median falls {dc['c_peak']:.0f} -> {dc['c_last']:.0f} "
              f"between k={dc['k_peak']} and k={dc['k_last']} "
              f"({dc['frac']:.2f}x): a plateau is not the whole story.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
