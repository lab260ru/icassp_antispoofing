#!/usr/bin/env python3
"""Can a linear readout recover the repetition count from the decoder's states?

This is the decisive test of Theorem A(iii), and it is also the test of a live
competing hypothesis. Venkatesh (arXiv:2605.09239) reports that in text LMs the
count of repeated tokens *is* linearly decodable from the residual stream even at
the layers where the wrong answer forms --- i.e. the information survives and the
failure lies in the output policy, not in the representation. If the same holds
for TTS decoders, our capacity account is wrong, or at least incomplete: the
states would be separable, so the contraction premise (Assumption 2) would be
false and Theorem A would simply not apply.

A linear probe is exactly the Lipschitz readout class the theorem constrains, so
this is not an analogy --- it is the theorem's own claim, tested.

Design. For each instrumented generation we summarise the trajectory by the mean
state over its final third (where, under contraction, the orbit should already
have collapsed), per probe layer. A ridge regression predicts log2(k) from that
summary, scored by leave-one-template-out cross-validation so no probe is ever
evaluated on a carrier sentence it was fitted on. We also report pairwise
discriminability: for every pair of k values, the cross-validated accuracy of a
linear classifier separating their terminal states. Under contraction that
accuracy should fall towards chance as both k grow.

The matched controls make the result interpretable: the same probe, on the same
model, at the same layer, decoding the same quantity from non-repetitive text of
equal length.

Usage:
  python analysis/probe_count.py --models llasa1b xtts2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402

from common.registry import DATA_ROOT  # noqa: E402

TAIL_FRAC = 1.0 / 3.0


def load_features(model: str, stim: dict, seed: int = 0) -> dict:
    """Per family: early/late window summaries, k, template.

    Two windows, not one. The theory does not predict that the count is *never*
    represented --- at the start of generation the decoder has just read the text
    and the count is plainly available there. It predicts the representation
    *degrades as generation proceeds under periodic conditioning*. Probing the
    first and last third of the same trajectory turns that into a within-item
    contrast, so a difference cannot be explained by probe capacity, item count,
    or how hard the k values are to tell apart in general.
    """
    act_dir = Path(DATA_ROOT) / "activations" / model
    out: dict[str, dict] = {}
    for f in sorted(act_dir.glob(f"*_s{seed}.npz")):
        item_id = f.stem.rsplit("_s", 1)[0]
        it = stim.get(item_id)
        if it is None or it["k"] < 2:
            continue
        try:
            z = np.load(f)
            h = z["hidden"]
        except Exception:  # noqa: BLE001
            continue
        T = h.shape[0]
        if T < 24:
            continue
        w = max(4, int(T * TAIL_FRAC))
        early = h[:w].astype(np.float32).mean(axis=0)
        late = h[-w:].astype(np.float32).mean(axis=0)
        norm = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
        d = out.setdefault(it["family"], dict(Xe=[], Xl=[], k=[], tmpl=[]))
        d["Xe"].append(norm(early))
        d["Xl"].append(norm(late))
        d["k"].append(it["k"])
        d["tmpl"].append(it["template"])
    for fam, d in out.items():
        d["Xe"] = np.stack(d["Xe"])                 # [n, P, d]
        d["Xl"] = np.stack(d["Xl"])
        d["X"] = d["Xl"]                            # default view: terminal state
        d["k"] = np.asarray(d["k"], dtype=float)
        d["tmpl"] = np.asarray(d["tmpl"])
    return out


def ridge_loto(X: np.ndarray, y: np.ndarray, tmpl: np.ndarray,
               alpha: float = 1.0) -> dict:
    """Leave-one-template-out ridge. Returns R^2 and MAE in log2 k units.

    Solved in the dual: there are ~50 items and 1-4k features, so the primal
    normal equations are a d-by-d solve (minutes per layer at d=4096) while the
    equivalent dual is n-by-n (microseconds). Same estimator, same predictions.
    """
    if X.shape[0] < 8 or len(set(tmpl.tolist())) < 3:
        return dict(r2=np.nan, mae=np.nan, n=int(X.shape[0]))
    X = X.astype(np.float64)
    preds = np.full(y.shape, np.nan)
    for t in sorted(set(tmpl.tolist())):
        te = tmpl == t
        tr = ~te
        if tr.sum() < 6 or te.sum() < 1:
            continue
        mu = X[tr].mean(0, keepdims=True)
        A = X[tr] - mu
        ybar = y[tr].mean()
        b = y[tr] - ybar
        # w = A^T (A A^T + alpha I)^-1 b  ==  (A^T A + alpha I)^-1 A^T b
        K = A @ A.T + alpha * np.eye(A.shape[0])
        dual = np.linalg.solve(K, b)
        preds[te] = ((X[te] - mu) @ (A.T @ dual)) + ybar
    ok = np.isfinite(preds)
    if ok.sum() < 4:
        return dict(r2=np.nan, mae=np.nan, n=int(ok.sum()))
    res = float(((y[ok] - preds[ok]) ** 2).sum())
    tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return dict(r2=float(1 - res / tot) if tot > 1e-9 else np.nan,
                mae=float(np.abs(y[ok] - preds[ok]).mean()), n=int(ok.sum()))


def pairwise_sep(X: np.ndarray, k: np.ndarray, tmpl: np.ndarray) -> dict:
    """Leave-one-template-out nearest-centroid accuracy for each pair of k."""
    ks = sorted(set(k.tolist()))
    res: dict[str, float] = {}
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            sel = (k == a) | (k == b)
            Xs, ys, ts = X[sel], (k[sel] == b).astype(int), tmpl[sel]
            if len(set(ts.tolist())) < 3 or Xs.shape[0] < 8:
                continue
            correct, total = 0, 0
            for t in sorted(set(ts.tolist())):
                te, tr = ts == t, ts != t
                if tr.sum() < 4 or te.sum() < 1:
                    continue
                c0 = Xs[tr][ys[tr] == 0].mean(0) if (ys[tr] == 0).any() else None
                c1 = Xs[tr][ys[tr] == 1].mean(0) if (ys[tr] == 1).any() else None
                if c0 is None or c1 is None:
                    continue
                d0 = np.linalg.norm(Xs[te] - c0, axis=1)
                d1 = np.linalg.norm(Xs[te] - c1, axis=1)
                correct += int(((d1 < d0).astype(int) == ys[te]).sum())
                total += int(te.sum())
            if total >= 4:
                res[f"{int(a)}v{int(b)}"] = correct / total
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/probe.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stim = {}
    for line in open(args.stimuli):
        it = json.loads(line)
        stim[it["item_id"]] = it

    result: dict = {"models": {}}
    for model in args.models:
        if not (Path(DATA_ROOT) / "activations" / model).exists():
            continue
        feats = load_features(model, stim, args.seed)
        entry: dict = {}
        for fam in ("word_rep", "control_word"):
            d = feats.get(fam)
            if d is None or d["Xl"].shape[0] < 8:
                continue
            P = d["Xl"].shape[1]
            y = np.log2(d["k"])
            fam_entry: dict = dict(n_items=int(d["Xl"].shape[0]))
            for win, key in (("Xe", "early"), ("Xl", "late")):
                per_layer = [ridge_loto(d[win][:, p, :], y, d["tmpl"]) for p in range(P)]
                r2s = [r["r2"] if np.isfinite(r["r2"]) else -9 for r in per_layer]
                best = int(np.argmax(r2s))
                fam_entry[key] = dict(best_layer_idx=best, best=per_layer[best],
                                      deep=per_layer[-1],
                                      by_layer=[r["r2"] for r in per_layer])
            fam_entry["retention"] = (
                fam_entry["late"]["best"]["r2"] / fam_entry["early"]["best"]["r2"]
                if np.isfinite(fam_entry["early"]["best"]["r2"])
                and fam_entry["early"]["best"]["r2"] > 0.05 else np.nan)
            fam_entry["pairwise"] = pairwise_sep(d["Xl"][:, -1, :], d["k"], d["tmpl"])
            entry[fam] = fam_entry
        result["models"][model] = entry

        print(f"\n[{model}] ridge probe decoding log2(k), leave-one-template-out")
        for fam, e in entry.items():
            eb, lb = e["early"]["best"], e["late"]["best"]
            print(f"  {fam:14s} n={e['n_items']:3d}  "
                  f"early R2={eb['r2']:.3f} (MAE {eb['mae']:.2f})  "
                  f"late R2={lb['r2']:.3f} (MAE {lb['mae']:.2f})  "
                  f"retention={e['retention']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
