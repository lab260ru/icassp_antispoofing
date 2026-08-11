#!/usr/bin/env python3
"""Is the horizon counted in repetitions, or in acoustic tokens?

A reviewer's alternative locus for the collapse: perhaps nothing text-side is
happening and the decoder simply falls into an attractor in *acoustic* token
space once it has emitted enough near-identical frames. That account and ours
make opposite predictions about the unit the horizon is measured in.

A sentence-repetition unit takes several times more acoustic tokens to render
than a word-repetition unit. So:

  * if the collapse is driven by accumulated acoustic tokens, sentence repetition
    should reach the limit in far FEWER repetitions than word repetition, and the
    token count at collapse should match across families;
  * if the collapse is driven by the number of times the periodic map has been
    applied --- Theorem A's claim --- the collapse should happen at a comparable
    number of REPETITIONS, and the token count at collapse should differ.

Usage:  python analysis/unit_invariance.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Re-runs of a panel member under a changed decoding setting are reported
# separately and never counted as additional checkpoints.
ABLATIONS = {"xtts2norp"}


def k_star(sub: pd.DataFrame, thr: float = 0.5) -> tuple[float, float]:
    """Largest k with accuracy above `thr`, and the median token count there."""
    if sub.empty:
        return np.nan, np.nan
    acc = sub.groupby("k")["correct"].mean()
    above = acc.index[acc > thr]
    if not len(above):
        return 0.0, np.nan
    k = float(above.max())
    tok = sub[sub.k == k]["n_speech_tokens"]
    return k, float(tok.median()) if len(tok) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural.csv")
    ap.add_argument("--out", default="data/results/unit_invariance.json")
    args = ap.parse_args()

    d = pd.read_csv(args.behavioural)
    rows, res = [], {"models": {}}
    for m in sorted(set(d.model.unique()) - ABLATIONS):
        kw, tw = k_star(d[(d.model == m) & (d.family == "word_rep")])
        ks, ts = k_star(d[(d.model == m) & (d.family == "sentence_rep")])
        if not (np.isfinite(kw) and np.isfinite(ks)):
            continue
        res["models"][m] = dict(k_word=kw, k_sent=ks, tok_word=tw, tok_sent=ts)
        rows.append((m, kw, ks, tw, ts))
        print(f"{m:10s} k*(word)={kw:4.0f} (~{tw:6.0f} tok)   "
              f"k*(sent)={ks:4.0f} (~{ts:6.0f} tok)")

    if rows:
        kw = np.array([r[1] for r in rows], float)
        ks = np.array([r[2] for r in rows], float)
        tw = np.array([r[3] for r in rows], float)
        ts = np.array([r[4] for r in rows], float)
        ok = np.isfinite(tw) & np.isfinite(ts) & (tw > 0)
        res["mean_abs_k_diff"] = float(np.abs(kw - ks).mean())
        res["mean_k_ratio"] = float(np.mean(ks[kw > 0] / kw[kw > 0])) if (kw > 0).any() else np.nan
        res["mean_tok_ratio"] = float(np.mean(ts[ok] / tw[ok])) if ok.any() else np.nan
        res["n_models"] = len(rows)
        print(f"\nacross {len(rows)} models:")
        print(f"  mean |k*(word) - k*(sent)|      = {res['mean_abs_k_diff']:.1f} repetitions")
        print(f"  mean token-count ratio at k*    = {res['mean_tok_ratio']:.2f}x")
        print("  -> the horizon tracks repetitions, not accumulated acoustic tokens"
              if res["mean_abs_k_diff"] <= 2.5 else
              "  -> inconclusive: horizons differ across units")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
