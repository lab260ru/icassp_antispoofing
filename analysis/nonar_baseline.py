#!/usr/bin/env python3
r"""Is the deficit autoregressive, or is it just what happens to long repeated text?

The paper claims something about autoregressive decoders, and until now it
had never run one that was not autoregressive. Round-6 reviewers said so. VITS
(`facebook/mms-tts-eng`) is the contrast: a text encoder, a duration predictor
and a normalising-flow decoder that emits the whole waveform at once. There is no
generation-step recurrence for a per-repetition state map to act on, so
Assumption 1 has nothing to apply to and Theorem 1 predicts nothing about it.

The comparison is the \emph{within-model} dissociation, not the absolute score.
VITS is smaller and older than the panel, so its overall accuracy is not
comparable and we do not compare it. What is comparable is whether a model treats
repeated text differently from its own length-matched control, which is the
paper's actual claim and is scale-free.

This doubles as the control the judge most needed. If CTC blank-collapse were
manufacturing the dissociation by merging adjacent copies of a word (S9), it
would do so on VITS audio too --- same judge, same stimuli, same repeated words.
A null here is therefore evidence that the AR panel's gap is not an artifact of
the recogniser, which no amount of re-auditing the recogniser alone could show.

Usage:  python analysis/nonar_baseline.py
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


def summarise(d: pd.DataFrame) -> dict:
    rep = d[d.family == "word_rep"]
    ctl = d[d.family == "control_word"]
    out = dict(
        n_rep=int(len(rep)), n_ctl=int(len(ctl)),
        exact_rep=float((rep.err == 0).mean()) if len(rep) else np.nan,
        exact_ctl=float((ctl.err == 0).mean()) if len(ctl) else np.nan,
        median_rep=float(rep.err.median()) if len(rep) else np.nan,
        median_ctl=float(ctl.err.median()) if len(ctl) else np.nan)
    out["exact_gap"] = out["exact_ctl"] - out["exact_rep"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--nonar", default="data/results/behavioural_vits.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/nonar_baseline.json")
    args = ap.parse_args()

    def load(path: str, keep_ablations: bool = False) -> pd.DataFrame:
        d = pd.read_csv(path)
        d = d[d.family.isin(["word_rep", "control_word"])]
        d, _ = panel(d, ablations=keep_ablations)
        d = d[d.k >= args.kmin]
        return d.assign(err=(d.count_a - d.k) / d.k)

    # The baseline lives in ABLATIONS -- it is a contrast, not a panel member --
    # so it has to be asked for explicitly here.
    ar, nonar = load(args.ar), load(args.nonar, keep_ablations=True)

    res: dict = {"kmin": args.kmin}
    res["nonar"] = summarise(nonar)
    res["ar_pooled"] = summarise(ar)
    res["ar_per_model"] = {m: summarise(g) for m, g in ar.groupby("model")}

    print(f"{'model':14s} {'rep exact':>10s} {'ctl exact':>10s} {'gap (pts)':>10s}")
    for m, v in sorted(res["ar_per_model"].items()):
        print(f"{m:14s} {100*v['exact_rep']:9.1f}% {100*v['exact_ctl']:9.1f}% "
              f"{100*v['exact_gap']:+9.1f}")
    v = res["ar_pooled"]
    print(f"{'AR pooled':14s} {100*v['exact_rep']:9.1f}% {100*v['exact_ctl']:9.1f}% "
          f"{100*v['exact_gap']:+9.1f}")
    v = res["nonar"]
    print(f"{'VITS (non-AR)':14s} {100*v['exact_rep']:9.1f}% {100*v['exact_ctl']:9.1f}% "
          f"{100*v['exact_gap']:+9.1f}")

    # Does every AR checkpoint separate from the non-AR baseline?
    gaps = [v["exact_gap"] for v in res["ar_per_model"].values()]
    res["n_ar_above_nonar"] = int(sum(g > res["nonar"]["exact_gap"] for g in gaps))
    res["n_ar"] = len(gaps)
    res["nonar_dissociates"] = bool(res["nonar"]["exact_gap"] > 0.05)

    print(f"\nAR checkpoints with a larger dissociation than the non-AR baseline: "
          f"{res['n_ar_above_nonar']}/{res['n_ar']}")
    if not res["nonar_dissociates"]:
        print("The non-AR synthesiser shows no periodicity-specific deficit: it\n"
              "degrades with length on both families alike. Two things follow.\n"
              "First, the effect is not a property of long repeated *text*, since\n"
              "the same text through a non-recurrent decoder does not produce it.\n"
              "Second, the judge is not manufacturing it: CTC blank-collapse would\n"
              "have depressed VITS's repeated items too, and did not.")

    # by k, for the figure and for the claim that VITS degrades with length
    res["nonar_by_k"] = {
        int(k): dict(exact_rep=float((g[g.family == "word_rep"].err == 0).mean()),
                     exact_ctl=float((g[g.family == "control_word"].err == 0).mean()))
        for k, g in nonar.groupby("k")}

    # Is the null just a floor effect -- VITS too weak to show any structure?
    # A reviewer raised it as the primary objection, and the band where the AR
    # panel's gap is already large settles it: if VITS were pinned at the floor
    # there, both its arms would be low. They are not, and they are ordered the
    # other way.
    def band(d, lo, hi):
        g = d[(d.k >= lo) & (d.k <= hi)]
        r, c = g[g.family == "word_rep"], g[g.family == "control_word"]
        if not len(r) or not len(c):
            return None
        er, ec = float((r.err == 0).mean()), float((c.err == 0).mean())
        return dict(exact_rep=er, exact_ctl=ec, gap=ec - er, n=int(len(g)))

    def load_all(path: str, keep_ablations: bool) -> pd.DataFrame:
        d = pd.read_csv(path)
        d = d[d.family.isin(["word_rep", "control_word"])]
        d, _ = panel(d, ablations=keep_ablations)
        return d.assign(err=(d.count_a - d.k) / d.k)

    ar_all = load_all(args.ar, False)
    nonar_all = load_all(args.nonar, True)
    res["bands"] = {}
    print(f"\n{'k band':10s} {'VITS rep':>9s} {'VITS ctl':>9s} {'gap':>7s} | "
          f"{'AR rep':>7s} {'AR ctl':>7s} {'gap':>7s}")
    for lo, hi in [(2, 4), (6, 8), (12, 16), (24, 32)]:
        nb, ab = band(nonar_all, lo, hi), band(ar_all, lo, hi)
        if not nb or not ab:
            continue
        res["bands"][f"{lo}-{hi}"] = dict(nonar=nb, ar=ab)
        print(f"k={lo}-{hi:<7d} {100*nb['exact_rep']:8.1f}% {100*nb['exact_ctl']:8.1f}% "
              f"{100*nb['gap']:+6.1f} | {100*ab['exact_rep']:6.1f}% "
              f"{100*ab['exact_ctl']:6.1f}% {100*ab['gap']:+6.1f}")
    mid = res["bands"].get("6-8")
    if mid:
        res["floor_effect_ruled_out"] = bool(
            mid["nonar"]["exact_rep"] > 0.5 and mid["nonar"]["exact_ctl"] > 0.5
            and mid["ar"]["gap"] > 0.3)
        if res["floor_effect_ruled_out"]:
            print("\nAt k=6-8 the AR panel already shows a "
                  f"{100*mid['ar']['gap']:.0f}-point gap while VITS renders "
                  f"{100*mid['nonar']['exact_rep']:.1f}% of repeated and "
                  f"{100*mid['nonar']['exact_ctl']:.1f}% of control items exactly.\n"
                  "Both arms are far off the floor and ordered the other way, so\n"
                  "the null is not VITS being too weak to show structure.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
