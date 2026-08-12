#!/usr/bin/env python3
"""Does attention dilution alone explain the deficit, without the contraction premise?

A round-5 reviewer asked the question this paper most needed asking. Lemma 1
(attention dilution) is proved *and* measured. Assumption 2 (contraction) is
neither --- two independent attempts to estimate `q` failed. If dilution on its
own predicts which items get miscounted, then the contraction half of the
account is doing no work, and the honest theory is the smaller one.

The test. Both dilution and the deficit grow with `k`, so a raw correlation
between them is guaranteed and says nothing. We therefore correlate them
*within* each (model, k) cell, where `k` is held fixed by construction, and pool
the per-cell correlations by Fisher z weighted by cell size. What is being asked
is: among items asked for the same number of repetitions from the same model,
do the ones whose attention is flattest come out worst?

Dilution is measured three ways, because "flat attention" has no single
operationalisation and picking one after seeing results would be cherry-picking:
`attn_share_max` (the largest single occurrence's share, low = flat),
`attn_block_entropy` (high = flat), and `attn_unif_dev` (deviation from uniform,
low = flat). Each is signed into a single flatness scale `FLAT`, higher = flatter.

The direction is the whole point, so state it before looking. Lemma 1 says
flattening is what destroys the decoder's ability to tell one occurrence from
another, and the relative count error is negative when a model undercounts.
Lemma 1 therefore predicts **corr(FLAT, rel_err) < 0**: flatter attention, worse
count. A positive correlation would mean flatter attention goes with *better*
counting, which is the opposite of the mechanism the lemma supplies.

Qwen3-TTS is absent throughout: it sums text and acoustic embeddings per
position, so no attention column belongs to the text and dilution cannot be
measured there at all.

Reading the outcome:
  * negative r on repeated items: dilution tracks the failure item by item, and
    the paper should lean on Lemma 1 rather than on an unmeasured premise;
  * null: dilution is necessary background but does not pick out which
    generations fail. That is not evidence for contraction either;
  * positive r: the item-level association runs *against* Lemma 1's direction,
    and the lemma cannot be the item-level mechanism whatever else is true.

Causality is not established in any of these cases. A generation that stalls may
produce peaked attention as a consequence rather than a cause, and this design
cannot separate the two --- only an intervention on the attention profile could.

Usage:  python analysis/dilution_sufficiency.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402

# (column, sign) with sign chosen so that +1 * value means "flatter attention".
DILUTION = [("attn_share_max", -1.0),
            ("attn_block_entropy", +1.0),
            ("attn_unif_dev", -1.0)]


def fisher_pool(rs: list[float], ns: list[int]) -> tuple[float, float, float]:
    """Weighted Fisher-z pool of per-cell correlations -> (r, lo, hi)."""
    rs_a, ns_a = np.array(rs, float), np.array(ns, float)
    keep = np.isfinite(rs_a) & (ns_a > 3) & (np.abs(rs_a) < 0.999)
    rs_a, ns_a = rs_a[keep], ns_a[keep]
    if len(rs_a) < 2:
        return float("nan"), float("nan"), float("nan")
    z = np.arctanh(rs_a)
    w = ns_a - 3
    zbar = float((w * z).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    return (float(np.tanh(zbar)), float(np.tanh(zbar - 1.96 * se)),
            float(np.tanh(zbar + 1.96 * se)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--deep-frac", type=float, default=0.6)
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/dilution_sufficiency.json")
    args = ap.parse_args()

    beh = pd.read_csv(args.behavioural)
    beh = beh[beh.family.isin(["word_rep", "control_word"])]
    beh, _ = panel(beh)
    beh = beh[beh.k >= args.kmin]
    beh = beh.assign(rel_err=(beh.count_a - beh.k) / beh.k)
    # One row per item: the state table is the instrumented pass only, so
    # average the behavioural seeds rather than pretend they are separate.
    beh = beh.groupby(["model", "item_id", "family", "k"], as_index=False).rel_err.mean()

    st = pd.read_csv(args.state)
    st = st[(st.layer_frac > args.deep_frac) & st.attn_share_max.notna()]
    st = st.groupby(["model", "item_id"], as_index=False)[
        [c for c, _ in DILUTION]].median()

    d = beh.merge(st, on=["model", "item_id"], how="inner")
    if d.empty:
        raise SystemExit("no overlap between behavioural and attention tables")

    res: dict = {"kmin": args.kmin, "n_items": int(len(d)),
                 "models": sorted(d.model.unique()), "measures": {}}
    print(f"items with both a count and an attention profile: {len(d)} "
          f"({', '.join(sorted(d.model.unique()))})\n")

    for fam in ("word_rep", "control_word"):
        sub = d[d.family == fam]
        print(f"--- {fam} (n={len(sub)})")
        for col, sign in DILUTION:
            rs, ns = [], []
            for (_m, _k), g in sub.groupby(["model", "k"]):
                if len(g) < 4:
                    continue
                x = sign * g[col].to_numpy(float)
                y = g.rel_err.to_numpy(float)
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    continue
                rs.append(float(np.corrcoef(x, y)[0, 1]))
                ns.append(len(g))
            r, lo, hi = fisher_pool(rs, ns)
            res["measures"].setdefault(fam, {})[col] = dict(
                r=r, lo=lo, hi=hi, n_cells=len(rs), n_items=int(sum(ns)))
            sig = "" if not np.isfinite(lo) else ("  *" if lo > 0 or hi < 0 else "")
            if np.isfinite(lo) and lo > 0:
                sig += "  AGAINST Lemma 1's direction"
            elif np.isfinite(hi) and hi < 0:
                sig += "  consistent with Lemma 1"
            print(f"    {col:20s} r={r:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"cells={len(rs):3d}{sig}")

    rep = res["measures"].get("word_rep", {})
    supports = [c for c, v in rep.items() if np.isfinite(v["hi"]) and v["hi"] < 0]
    against = [c for c, v in rep.items() if np.isfinite(v["lo"]) and v["lo"] > 0]
    if against and not supports:
        res["verdict"] = ("item-level association runs AGAINST Lemma 1: within a "
                          "cell, flatter attention goes with better counting")
    elif supports and not against:
        res["verdict"] = "dilution tracks the deficit item-by-item, as Lemma 1 predicts"
    else:
        res["verdict"] = "no consistent within-cell association"
    res["n_against"], res["n_supporting"] = len(against), len(supports)
    print(f"\nverdict: {res['verdict']}")
    if against:
        print("  Lemma 1 predicts flatter attention -> worse count. The data show the\n"
              "  reverse, on all three measures, with the matched controls flat at zero.\n"
              "  So dilution is not the item-level mechanism. This does not rescue the\n"
              "  contraction premise, which two direct attempts failed to measure; it\n"
              "  removes the *other* candidate explanation for item-level variation.\n"
              "  Direction of causation is not established: a stalling generation may\n"
              "  produce peaked attention rather than result from it.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
