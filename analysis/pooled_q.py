#!/usr/bin/env python3
"""Pooled estimate of the per-repetition contraction factor, with a bootstrap CI.

A single item gives only k-1 boundary distances, and those are one noisy
realisation of a stochastic decoder, so a per-item exponential fit is
underpowered --- its R^2 says more about sampling noise than about whether the
geometric law holds. Pooling is the right estimator: within a (model, family)
cell, normalise each item's distances by its own first distance and regress
`log(d_m / d_0)` on `m` through the origin. Every item then contributes its
*shape* and nothing else, so items of different scales combine without one
dominating.

The confidence interval bootstraps over items, not over points, because points
within an item are not independent.

Requires the per-boundary distances, so it reads the activations directly rather
than the summary CSV.

Usage:
  python analysis/pooled_q.py --models llasa1b xtts2 --out data/results/pooled_q.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402

from common import offset_tok as tokmod  # noqa: E402
from common.boundaries import (  # noqa: E402
    boundary_distances, boundary_steps_from_attention, boundary_steps_uniform,
    unit_columns,
)
from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices  # noqa: E402

MIN_GAPS = 5   # need enough boundaries for a decay to be a decay


def pooled_fit(curves: list[np.ndarray], n_boot: int = 2000,
               rng_seed: int = 0) -> dict:
    """Regress log(d_m/d_0) on m through the origin, pooled over items."""
    curves = [c for c in curves if c.size >= MIN_GAPS and np.all(np.isfinite(c)) and c[0] > 0
              and np.all(c > 0)]
    if len(curves) < 4:
        return dict(q=np.nan, lo=np.nan, hi=np.nan, n_items=len(curves), r2=np.nan)

    def slope(sel: list[np.ndarray]) -> float:
        xs, ys = [], []
        for c in sel:
            m = np.arange(c.size, dtype=np.float64)
            xs.append(m)
            ys.append(np.log(c / c[0]))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        return float((x @ y) / (x @ x)) if (x @ x) > 0 else np.nan

    s = slope(curves)
    # variance explained, pooled
    xs = np.concatenate([np.arange(c.size, dtype=np.float64) for c in curves])
    ys = np.concatenate([np.log(c / c[0]) for c in curves])
    ss_res = float(((ys - s * xs) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())

    rng = np.random.default_rng(rng_seed)
    boots = []
    idx = np.arange(len(curves))
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        v = slope([curves[i] for i in pick])
        if np.isfinite(v):
            boots.append(v)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(q=float(np.exp(s)), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                n_items=len(curves), r2=float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan)


def collect(model: str, seed: int, stim: dict, deep_only: bool = True) -> dict:
    """Boundary-distance curves per family, at the deepest probe layer."""
    spec = BY_KEY[model]
    act_dir = Path(DATA_ROOT) / "activations" / model
    tok = tokmod.load(spec)
    out: dict[str, list] = {}
    methods: dict[str, int] = {}
    for f in sorted(act_dir.glob(f"*_s{seed}.npz")):
        item_id = f.stem.rsplit("_s", 1)[0]
        it = stim.get(item_id)
        if it is None or it["k"] < MIN_GAPS + 1:
            continue
        try:
            z = np.load(f)
        except Exception:  # noqa: BLE001
            continue
        hidden = z["hidden"]
        probes = z["probe_layers"].tolist()
        pi = len(probes) - 1 if deep_only else 0
        akeys = sorted([k for k in z.files if k.startswith("attn_l")],
                       key=lambda s: int(s[6:]))
        attn = z[akeys[-1]] if akeys else None
        units = it.get("boundary_units") or [it["target_unit"]] * it["k"]
        cols = unit_columns(tok, it["text"], units) if tok is not None else []
        steps = (boundary_steps_from_attention(attn, cols)
                 if (attn is not None and cols) else np.array([], dtype=int))
        method = "attention"
        if steps.size < MIN_GAPS + 1:
            steps = boundary_steps_uniform(hidden.shape[0], it["k"])
            method = "uniform"
        if steps.size < MIN_GAPS + 1:
            continue
        d = boundary_distances(hidden, steps)
        if d.shape[0] < MIN_GAPS:
            continue
        out.setdefault(it["family"], []).append(d[:, pi].astype(np.float64))
        methods[method] = methods.get(method, 0) + 1
    return dict(curves=out, methods=methods)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/pooled_q.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stim = {}
    for line in open(args.stimuli):
        it = json.loads(line)
        stim[it["item_id"]] = it

    rows = []
    for model in args.models:
        if not (Path(DATA_ROOT) / "activations" / model).exists():
            print(f"[{model}] no activations")
            continue
        got = collect(model, args.seed, stim)
        print(f"[{model}] boundary method: {got['methods']}")
        for fam, curves in sorted(got["curves"].items()):
            fit = pooled_fit(curves)
            rows.append(dict(model=model, family=fam, **fit,
                             method=max(got["methods"], key=got["methods"].get)
                             if got["methods"] else "none"))
            print(f"   {fam:14s} q={fit['q']:.4f} "
                  f"[{fit['lo']:.4f},{fit['hi']:.4f}] "
                  f"n={fit['n_items']:3d} R2={fit['r2']:.3f}")

    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
