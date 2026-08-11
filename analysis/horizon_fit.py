#!/usr/bin/env python3
"""Test prediction P2: does the measured contraction factor predict the
behavioural collapse point?

Theorem A(iv) gives `N* = log(mu/(2LC)) / log q`. Neither the readout margin
`mu` nor the Lipschitz constant `L` is observable from activations, but they
enter only through one product that is shared across a model's items. The
theory's cross-model content is therefore

    k*  =  beta * 1/log(1/q_hat)  +  alpha,

a two-parameter regression over the whole panel. If contraction is doing the
work, `beta > 0` and the fit is tight; if the collapse point is unrelated to the
state dynamics, it is not.

Also reports the mechanism check: within each model, is the repeated-text
contraction stronger (smaller q_hat) than the length-matched control?

Usage:
  python analysis/horizon_fit.py --behavioural data/results/behavioural.csv \
                                 --state data/results/state.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collapse_point(sub: pd.DataFrame, floor_acc: float = 0.5) -> dict:
    """k* = the largest k at which counting accuracy is still above `floor_acc`.

    Reported as a step-threshold rather than a logistic midpoint: accuracy-vs-k
    is not always monotone (a model can get lucky at one k), and the threshold
    crossing is what "the model can still count this many" means operationally.
    A logistic midpoint is also fitted for comparison.
    """
    acc = sub.groupby("k")["correct"].mean().sort_index()
    if acc.empty:
        return dict(k_star=np.nan, k_star_logistic=np.nan, acc_by_k={})
    ks = acc.index.to_numpy(dtype=float)
    av = acc.to_numpy(dtype=float)
    above = ks[av > floor_acc]
    k_star = float(above.max()) if above.size else float(ks.min())

    # logistic midpoint via a simple grid search (n is small; no optimiser dep)
    best, best_mid = np.inf, np.nan
    for mid in np.linspace(0.5, 40, 400):
        for slope in (0.3, 0.5, 0.8, 1.2, 2.0):
            pred = 1.0 / (1.0 + np.exp(slope * (ks - mid)))
            err = float(((pred - av) ** 2).sum())
            if err < best:
                best, best_mid = err, mid
    return dict(k_star=k_star, k_star_logistic=float(best_mid),
                acc_by_k={int(k): float(v) for k, v in acc.items()})


def model_q(state: pd.DataFrame, model: str, family: str,
            r2_min: float = 0.5, deep_frac: float = 0.6) -> dict:
    """Median q_hat over deep probe layers, on well-determined fits only.

    Deep layers are the ones feeding the readout, and the R^2 filter drops items
    where the boundary distances were too few or too noisy to support a
    geometric fit at all (rather than silently averaging in garbage).
    """
    s = state[(state.model == model) & (state.family == family)
              & (state.layer_frac > deep_frac) & state.q_hat.notna()
              & (state.r2 > r2_min)]
    if s.empty:
        return dict(q=np.nan, n=0, q_iqr=np.nan)
    q = s.q_hat.to_numpy()
    return dict(q=float(np.median(q)), n=int(len(q)),
                q_iqr=float(np.percentile(q, 75) - np.percentile(q, 25)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural.csv")
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--out", default="data/results/summary.json")
    ap.add_argument("--families", nargs="+", default=["word_rep", "sentence_rep"])
    args = ap.parse_args()

    beh = pd.read_csv(args.behavioural)
    state = pd.read_csv(args.state) if Path(args.state).exists() else pd.DataFrame()

    summary: dict = {"models": {}}
    for model in sorted(beh.model.unique()):
        b = beh[(beh.model == model) & (beh.family.isin(args.families))]
        entry = dict(n_items=int(len(b)))
        entry.update(collapse_point(b))

        # the periodicity dissociation: repeated vs length-matched control
        wr = beh[(beh.model == model) & (beh.family == "word_rep") & (beh.k >= 6)]
        ct = beh[(beh.model == model) & (beh.family == "control_word") & (beh.k >= 6)]
        entry["acc_rep_highk"] = float(wr.correct.mean()) if len(wr) else np.nan
        entry["n_rep_highk"] = int(len(wr))
        # controls are scored on rendering all k distinct fillers, which the
        # counting metric expresses as "not truncated and not degenerate"
        if len(ct):
            ok = ct.outcome.isin(["correct", "overcount", "undercount"]) & \
                 (ct.duration_ratio > 0.7) & (ct.spectral_flatness < 0.35)
            entry["acc_ctl_highk"] = float(ok.mean())
        else:
            entry["acc_ctl_highk"] = np.nan
        entry["n_ctl_highk"] = int(len(ct))

        if len(state):
            entry["q_rep"] = model_q(state, model, "word_rep")
            entry["q_ctl"] = model_q(state, model, "control_word")
            q = entry["q_rep"]["q"]
            entry["horizon_scale"] = (float(1.0 / np.log(1.0 / q))
                                      if np.isfinite(q) and 0 < q < 1 else np.nan)
        summary["models"][model] = entry

    # ---- P2 regression across the panel -------------------------------
    pts = [(m, e["horizon_scale"], e["k_star"]) for m, e in summary["models"].items()
           if np.isfinite(e.get("horizon_scale", np.nan)) and np.isfinite(e["k_star"])]
    if len(pts) >= 3:
        x = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts])
        beta, alpha = np.polyfit(x, y, 1)
        pred = beta * x + alpha
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        spear = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else np.nan
        summary["p2"] = dict(
            n=len(pts), beta=float(beta), alpha=float(alpha),
            r2=float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan,
            spearman=spear,
            points=[dict(model=p[0], horizon_scale=p[1], k_star=p[2]) for p in pts],
        )
    else:
        summary["p2"] = dict(n=len(pts), note="too few models with both measures")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}\n")

    print(f"{'model':10s} {'k*':>5s} {'q_rep':>7s} {'q_ctl':>7s} "
          f"{'1/log(1/q)':>10s} {'acc_rep':>8s} {'acc_ctl':>8s}")
    for m, e in summary["models"].items():
        qr = e.get("q_rep", {}).get("q", np.nan)
        qc = e.get("q_ctl", {}).get("q", np.nan)
        print(f"{m:10s} {e['k_star']:5.1f} {qr:7.3f} {qc:7.3f} "
              f"{e.get('horizon_scale', np.nan):10.2f} "
              f"{e['acc_rep_highk']:8.2f} {e['acc_ctl_highk']:8.2f}")
    if summary["p2"].get("r2") is not None:
        p2 = summary["p2"]
        print(f"\nP2: k* = {p2.get('beta', float('nan')):.2f} * 1/log(1/q) "
              f"+ {p2.get('alpha', float('nan')):.2f}   "
              f"R2={p2.get('r2', float('nan')):.3f}  "
              f"rho={p2.get('spearman', float('nan')):.3f}  n={p2['n']}")


if __name__ == "__main__":
    main()
