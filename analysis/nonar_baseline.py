#!/usr/bin/env python3
r"""Is the deficit autoregressive? On two baselines, the answer is no.

The paper claims something about autoregressive decoders, and for a long time it
had never run one that was not. Round-6 reviewers said so; rounds 8-10 then said
one 2021 model could not carry the claim alone. Both were right, and running the
second baseline cost us the claim.

Two non-AR systems, and they disagree:

* **VITS** (`facebook/mms-tts-eng`, 2021): text encoder, per-phoneme duration
  predictor, normalising-flow decoder. Shows *no* dissociation.
* **F5-TTS** (2024): flow matching over a diffusion transformer, a total-duration
  estimate, the whole mel denoised in parallel. Shows the dissociation at roughly
  the panel's size, growing with `k` as the AR models' does.

Neither has a generation-step recurrence, so "autoregressive" cannot be what
separates them. What does differ is where the count has to live. VITS expands
each input token to its own predicted duration, so "how many" is carried
structurally by the input sequence and never has to be represented. F5-TTS
estimates one total duration and denoises globally, so "how many" must be
represented somewhere internal --- the same burden an AR decoder carries in its
state. That is a better hypothesis than AR-versus-not, and these two baselines
are evidence for it, but it was formed after seeing them and is labelled as such
wherever it appears.

The comparison is the \emph{within-model} dissociation, not the absolute score:
whether a model treats repeated text differently from its own length-matched
control. That is scale-free, which matters because VITS is much the smaller
model. F5-TTS removes that objection --- it is contemporary with the panel and
comparably invested, so its result cannot be dismissed as vintage.

The VITS null still does the judge's work. If CTC blank-collapse were
manufacturing the dissociation by merging adjacent copies of a word (S9), it
would do so on VITS audio too --- same judge, same stimuli, same repeated words.
It does not, and no amount of re-auditing the recogniser in isolation could have
shown that.

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
    ap.add_argument("--nonar", nargs="+",
                    default=["data/results/behavioural_vits.csv",
                             "data/results/behavioural_f5.csv"],
                    help="one or more non-autoregressive baselines")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/nonar_baseline.json")
    args = ap.parse_args()

    def load(path, keep_ablations: bool = False) -> pd.DataFrame:
        paths = [path] if isinstance(path, str) else list(path)
        frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
        if not frames:
            raise SystemExit(f"no baseline table found among {paths}")
        d = pd.concat(frames, ignore_index=True)
        d = d[d.family.isin(["word_rep", "control_word"])]
        d, _ = panel(d, ablations=keep_ablations)
        d = d[d.k >= args.kmin]
        return d.assign(err=(d.count_a - d.k) / d.k)

    # The baseline lives in ABLATIONS -- it is a contrast, not a panel member --
    # so it has to be asked for explicitly here.
    ar, nonar = load(args.ar), load(args.nonar, keep_ablations=True)

    res: dict = {"kmin": args.kmin}
    res["nonar"] = summarise(nonar)
    res["nonar_per_model"] = {m: summarise(g) for m, g in nonar.groupby("model")}
    res["ar_pooled"] = summarise(ar)
    res["ar_per_model"] = {m: summarise(g) for m, g in ar.groupby("model")}

    print(f"{'model':14s} {'rep exact':>10s} {'ctl exact':>10s} {'gap (pts)':>10s}")
    for m, v in sorted(res["ar_per_model"].items()):
        print(f"{m:14s} {100*v['exact_rep']:9.1f}% {100*v['exact_ctl']:9.1f}% "
              f"{100*v['exact_gap']:+9.1f}")
    v = res["ar_pooled"]
    print(f"{'AR pooled':14s} {100*v['exact_rep']:9.1f}% {100*v['exact_ctl']:9.1f}% "
          f"{100*v['exact_gap']:+9.1f}")
    for m, v in sorted(res["nonar_per_model"].items()):
        print(f"{m + ' (non-AR)':14s} {100*v['exact_rep']:9.1f}% "
              f"{100*v['exact_ctl']:9.1f}% {100*v['exact_gap']:+9.1f}")

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

    def load_all(path, keep_ablations: bool) -> pd.DataFrame:
        paths = [path] if isinstance(path, str) else list(path)
        frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
        d = pd.concat(frames, ignore_index=True)
        d = d[d.family.isin(["word_rep", "control_word"])]
        d, _ = panel(d, ablations=keep_ablations)
        return d.assign(err=(d.count_a - d.k) / d.k)

    ar_all = load_all(args.ar, False)
    nonar_all = load_all(args.nonar, True)
    # Per baseline, never pooled: the two disagree, and averaging them would
    # hide exactly the fact that matters.
    res["bands"] = {}
    names = sorted(nonar_all.model.unique())
    hdr = " | ".join(f"{n[:9]:>16s}" for n in names + ["AR panel"])
    print(f"\n{'k band':9s} {hdr}")
    for lo, hi in [(2, 4), (6, 8), (12, 16), (24, 32)]:
        cells, row = [], {}
        for n in names:
            b = band(nonar_all[nonar_all.model == n], lo, hi)
            row[n] = b
            cells.append(f"{100*b['exact_rep']:6.1f}/{100*b['exact_ctl']:5.1f}"
                         f"{100*b['gap']:+5.1f}" if b else " " * 16)
        ab = band(ar_all, lo, hi)
        row["ar"] = ab
        cells.append(f"{100*ab['exact_rep']:6.1f}/{100*ab['exact_ctl']:5.1f}"
                     f"{100*ab['gap']:+5.1f}" if ab else " " * 16)
        res["bands"][f"{lo}-{hi}"] = row
        print(f"k={lo}-{hi:<5d} " + " | ".join(cells))
    print("  (repeated exact / control exact, gap in points)")

    # The floor-effect rebuttal belongs to whichever baseline shows the null.
    nulls = [n for n in names
             if res["nonar_per_model"][n]["exact_gap"] < 0.05]
    res["baselines_without_dissociation"] = nulls
    res["baselines_with_dissociation"] = [n for n in names if n not in nulls]
    for n in nulls:
        mid = res["bands"]["6-8"][n]
        if mid and mid["exact_rep"] > 0.5 and mid["exact_ctl"] > 0.5:
            print(f"\n{n} shows no dissociation and it is not a floor effect: at "
                  f"k=6-8 it renders\n{100*mid['exact_rep']:.1f}% of repeated and "
                  f"{100*mid['exact_ctl']:.1f}% of control items exactly, off the "
                  f"floor and ordered\nthe other way, where the AR panel's gap is "
                  f"already {100*res['bands']['6-8']['ar']['gap']:.0f} points.")
    if res["baselines_with_dissociation"]:
        print(f"\nBut {', '.join(res['baselines_with_dissociation'])} does show it, "
              f"so 'autoregressive' is not what\nseparates these systems. See the "
              f"module docstring for what plausibly does.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
