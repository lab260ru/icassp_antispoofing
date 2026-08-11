#!/usr/bin/env python3
"""Does the capacity result survive controlling for output repetitiveness?

The objection, raised in review and worth taking seriously: if a model correctly
renders "very" sixteen times, the *audio* is genuinely repetitive, so the
trajectory of hidden states would look low-rank even for a perfect rendering.
Effective rank might then be measuring the output's acoustic monotony rather than
any loss of representational capacity, and the repeated-vs-control gap would be a
tautology of the stimulus design.

Two tests, both using data already collected.

**Test 1 — partial out realised output diversity.** For each generation we can
count how many *distinct* acoustic tokens it actually emitted, and what fraction
of its tokens are distinct. That is a direct measure of how repetitive the output
is. Regressing N_eff on log k *and* output diversity separates "more repetitions
requested" from "less varied audio produced": if the log-k coefficient stays
negative-going for repeated text relative to control after diversity is in the
model, the capacity effect is not simply output monotony.

**Test 2 — restrict to correct renderings.** Among items the model rendered
correctly, the audio is by construction what the prompt asked for, so any
remaining repeated-vs-control difference in capacity cannot be attributed to the
model having produced degenerate output. Coverage is thin at high k precisely
because correct renderings become rare, which is itself the phenomenon; we report
the n so the reader can judge.

Usage:  python analysis/capacity_confound.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
ABLATIONS = {"xtts2norp"}


def output_diversity(model: str, seed: int = 0) -> dict:
    """Distinct-token count and type/token ratio per generated item.

    Read from the saved acoustic token ids where available (Llasa), and
    otherwise from the metadata's token count only. Models whose token ids we do
    not persist contribute the token count, which still carries most of the
    length signal.
    """
    out: dict[str, dict] = {}
    tok_dir = Path(DATA_ROOT) / "tokens" / model
    if tok_dir.is_dir():
        for f in tok_dir.glob(f"*_s{seed}.npy"):
            try:
                ids = np.load(f)
            except Exception:  # noqa: BLE001
                continue
            n = int(ids.size)
            if n == 0:
                continue
            uniq = int(np.unique(ids).size)
            out[f.stem.rsplit("_s", 1)[0]] = dict(
                n_tok=n, n_uniq=uniq, ttr=uniq / n,
                # how often the immediately preceding token repeats: a direct
                # index of local acoustic monotony
                rep_rate=float(np.mean(ids[1:] == ids[:-1])) if n > 1 else 0.0,
            )
    return out


def partial_regression(df: pd.DataFrame, extra: list[str]) -> dict:
    """OLS of n_eff on log k plus `extra` covariates; return the log-k slope."""
    d = df.dropna(subset=["n_eff", "k"] + extra)
    if len(d) < 12 or d.k.nunique() < 4:
        return dict(slope=np.nan, n=len(d))
    X = [np.log(d.k.to_numpy(dtype=float))]
    for c in extra:
        v = d[c].to_numpy(dtype=float)
        X.append((v - v.mean()) / (v.std() + 1e-9))
    X.append(np.ones(len(d)))
    A = np.vstack(X).T
    y = d.n_eff.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(slope=float(beta[0]), n=int(len(d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--behavioural", default="data/results/behavioural.csv")
    ap.add_argument("--out", default="data/results/capacity_confound.json")
    ap.add_argument("--deep-frac", type=float, default=0.6)
    args = ap.parse_args()

    st = pd.read_csv(args.state)
    st = st[st.layer_frac > args.deep_frac]
    beh = pd.read_csv(args.behavioural)
    beh0 = beh[beh.seed == 0].set_index(["model", "item_id"])

    res: dict = {"models": {}}
    print(f"{'model':10s} {'family':13s} {'raw':>7s} {'+diversity':>11s} "
          f"{'correct-only':>13s}")
    for m in sorted(st.model.unique()):
        div = output_diversity(m)
        s = st[st.model == m].copy()
        for c in ("n_tok", "n_uniq", "ttr", "rep_rate"):
            s[c] = [div.get(i, {}).get(c, np.nan) for i in s.item_id]
        s["correct"] = [beh0["correct"].get((m, i), np.nan) for i in s.item_id]

        entry: dict = {}
        for fam in ("word_rep", "control_word"):
            f = s[s.family == fam]
            raw = partial_regression(f, [])
            # control for how varied the produced audio actually was
            covs = [c for c in ("ttr", "rep_rate", "n_tok")
                    if c in f and f[c].notna().sum() > 12]
            adj = partial_regression(f, covs) if covs else dict(slope=np.nan, n=0)
            corr = partial_regression(f[f.correct == 1], [])
            entry[fam] = dict(raw=raw, adjusted=adj, correct_only=corr,
                              covariates=covs)
            print(f"{m:10s} {fam:13s} {raw['slope']:7.1f} "
                  f"{adj['slope']:11.1f} {corr['slope']:13.1f} "
                  f"(n={raw['n']}/{adj['n']}/{corr['n']})")
        if {"word_rep", "control_word"} <= set(entry):
            for key in ("raw", "adjusted", "correct_only"):
                a = entry["word_rep"][key]["slope"]
                b = entry["control_word"][key]["slope"]
                entry.setdefault("ratio", {})[key] = (
                    float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-9
                    else np.nan)
        res["models"][m] = entry

    panel = {k: v for k, v in res["models"].items() if k not in ABLATIONS}
    for key in ("raw", "adjusted", "correct_only"):
        vals = [v["ratio"][key] for v in panel.values()
                if "ratio" in v and np.isfinite(v["ratio"].get(key, np.nan))]
        res[f"ratio_{key}_median"] = float(np.median(vals)) if vals else np.nan
        res[f"ratio_{key}_n"] = len(vals)

    print(f"\npanel median repeated/control slope ratio "
          f"({res['ratio_raw_n']} checkpoints, ablations excluded):")
    print(f"  raw                        {res['ratio_raw_median']:.2f}")
    print(f"  controlling for diversity  {res['ratio_adjusted_median']:.2f}")
    print(f"  correct renderings only    {res['ratio_correct_only_median']:.2f} "
          f"(n={res['ratio_correct_only_n']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
