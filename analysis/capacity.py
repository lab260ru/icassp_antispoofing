#!/usr/bin/env python3
"""Representational capacity growth: the paper's central state measurement.

Theorem A(ii) says the states a decoder visits under periodic conditioning stop
being mutually distinguishable past a horizon. The observable consequence is that
the *number of distinguishable states* a generation visits stops growing when you
ask for more repetitions --- while for the same amount of non-repetitive text it
keeps growing.

We measure the number of distinguishable states as the effective rank of the
generation trajectory,

    N_eff = exp( -sum_i p_i log p_i ),   p_i = sigma_i / sum sigma,

the same observable the companion ASR study used, computed at deep probe layers.
The statistic is the *capacity gain* `dN_eff / d log k`, fitted separately on
repeated items and on their length-matched controls. The controls are what turn
this from a description into a test: they hold word count and syntax fixed and
remove only the periodicity, so a gap between the two gains cannot be explained
by sequence length, by generation duration, or by the model simply having more
text to render.

Confidence intervals bootstrap over stimulus templates rather than over items,
because items sharing a template are not independent.

Usage:
  python analysis/capacity.py --state data/results/state.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = {"repeated": "word_rep", "control": "control_word"}

# Ablation variants are re-runs of a panel member under a changed decoding
# setting, not additional checkpoints. They are reported individually but must
# never enter panel-level aggregates, or the panel size and every pooled
# statistic are inflated by a non-independent copy.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import ABLATIONS, panel as restrict, describe  # noqa: E402


def gain(df: pd.DataFrame, col: str = "n_eff", n_boot: int = 2000,
         seed: int = 0) -> dict:
    """Slope of `col` against log k, with a template-bootstrap CI."""
    d = df[(df.k >= 2) & df[col].notna()]
    if d.k.nunique() < 4 or len(d) < 8:
        return dict(gain=np.nan, lo=np.nan, hi=np.nan, n=len(d), r2=np.nan,
                    sat=np.nan, lo_k=np.nan, hi_k=np.nan)
    x = np.log(d.k.to_numpy(dtype=float))
    y = d[col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    rng = np.random.default_rng(seed)
    tmpls = d.template.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(tmpls, size=len(tmpls), replace=True)
        sub = pd.concat([d[d.template == t] for t in pick])
        if sub.k.nunique() < 3:
            continue
        boots.append(np.polyfit(np.log(sub.k.to_numpy(float)),
                                sub[col].to_numpy(float), 1)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))

    # saturation: median value in the top k-band, as a plateau estimate
    top = d[d.k >= d.k.max() / 2]
    return dict(gain=float(slope), lo=float(lo), hi=float(hi), n=int(len(d)),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan,
                sat=float(top[col].median()) if len(top) else np.nan,
                lo_k=float(d[d.k <= 3][col].median()) if len(d[d.k <= 3]) else np.nan,
                hi_k=float(top[col].median()) if len(top) else np.nan)


def paired_gap(df: pd.DataFrame, col: str = "n_eff", n_boot: int = 2000,
               seed: int = 0) -> dict:
    """Per-(template, k) difference control minus repeated, then bootstrap.

    The stimuli were built as matched pairs, so the paired difference is the
    estimator with the least nuisance variance: template identity and k cancel.
    """
    rep = df[df.family == FAMILIES["repeated"]].groupby(["template", "k"])[col].median()
    ctl = df[df.family == FAMILIES["control"]].groupby(["template", "k"])[col].median()
    j = pd.concat([rep.rename("rep"), ctl.rename("ctl")], axis=1).dropna()
    if len(j) < 6:
        return dict(gap=np.nan, lo=np.nan, hi=np.nan, n=0, frac_pos=np.nan)
    diff = (j.ctl - j.rep).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(diff, size=diff.size, replace=True)))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(gap=float(diff.mean()), lo=float(lo), hi=float(hi),
                n=int(diff.size), frac_pos=float((diff > 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--out", default="data/results/capacity.json")
    ap.add_argument("--deep-frac", type=float, default=0.6)
    ap.add_argument("--metric", default="n_eff")
    args = ap.parse_args()

    st = pd.read_csv(args.state)
    st = st[st.layer_frac > args.deep_frac]
    # Ablation arms stay in (they are printed for the mitigation table and
    # excluded from panel statistics below), but the judge-unmeasurable template
    # goes, so the capacity comparison runs on the same items as the
    # behavioural one.
    st, drop = restrict(st, ablations=True, degenerate=True, cap_hits=False)
    print(describe(drop))

    result: dict = {"metric": args.metric, "models": {}}
    print(f"{'model':10s} {'gain_rep':>18s} {'gain_ctl':>18s} {'ratio':>6s} "
          f"{'paired gap':>18s} {'frac':>5s}")
    for m in sorted(st.model.unique()):
        s = st[st.model == m]
        gr = gain(s[s.family == FAMILIES["repeated"]], args.metric)
        gc = gain(s[s.family == FAMILIES["control"]], args.metric)
        pg = paired_gap(s, args.metric)
        ratio = (gr["gain"] / gc["gain"]
                 if np.isfinite(gr["gain"]) and np.isfinite(gc["gain"]) and gc["gain"] > 1e-9
                 else np.nan)
        # is the repeated gain significantly below the control gain?
        sep = (np.isfinite(gr["hi"]) and np.isfinite(gc["lo"]) and gr["hi"] < gc["lo"])
        result["models"][m] = dict(repeated=gr, control=gc, ratio=float(ratio),
                                   paired=pg, ci_separated=bool(sep))
        print(f"{m:10s} {gr['gain']:7.1f} [{gr['lo']:5.1f},{gr['hi']:5.1f}] "
              f"{gc['gain']:7.1f} [{gc['lo']:5.1f},{gc['hi']:5.1f}] "
              f"{ratio:6.2f} {pg['gap']:7.1f} [{pg['lo']:5.1f},{pg['hi']:5.1f}] "
              f"{pg['frac_pos']:5.2f}{'  *' if sep else ''}")

    panel_models = {k: v for k, v in result["models"].items() if k not in ABLATIONS}
    n_sep = sum(1 for v in panel_models.values() if v["ci_separated"])
    result["n_models"] = len(panel_models)
    result["n_separated"] = n_sep
    result["ablations"] = sorted(set(result["models"]) & ABLATIONS)
    ratios = [v["ratio"] for v in panel_models.values() if np.isfinite(v["ratio"])]
    result["ratio_median"] = float(np.median(ratios)) if ratios else np.nan
    print(f"\npanel ({len(panel_models)} checkpoints; ablations excluded: "
          f"{result['ablations'] or 'none'}): capacity gain on repeated text is a "
          f"factor {result['ratio_median']:.2f} of control; "
          f"{n_sep}/{len(panel_models)} with disjoint 95% CIs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
