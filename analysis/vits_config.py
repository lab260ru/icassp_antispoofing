#!/usr/bin/env python3
"""Is VITS immune, or is its stock duration-predictor setting immune?

The paper withdrew its architectural claim because two non-autoregressive
baselines disagreed: F5-TTS shows the repetition deficit and VITS does not. A
reviewer pressed the obvious weakness in that: VITS was run once, at defaults,
while the autoregressive panel got a repetition-penalty sweep. If VITS's stock
duration predictor happens to sit in a regime immune to the effect for reasons
unrelated to architecture, the comparison proves nothing.

The duration predictor is exactly the component that would have to carry the
count --- output length is the sum of predicted phoneme durations --- so it is
the right thing to perturb. Two settings, both well away from stock:

    noise_scale_duration  0.8 -> 1.6    duration-predictor temperature
    speaking_rate         1.0 -> 1.35   divides the predicted durations

Both degrade synthesis; that is the point of choosing them. What matters is
whether degrading the duration predictor produces the panel's signature --- a
large *positive* control-minus-repeated gap in exact rate --- or leaves VITS's
sign where it was.

Read the gap column carefully: it is control minus repeated, so the panel's
+76 points means the control is counted right and the repeated item is not.
Every VITS arm is negative, meaning the repeated item is counted *better* than
its own control.

The absolute rates fall on both arms under perturbation, and the control falls
further. That is not a counting effect: controls carry many distinct words and
faster or noisier speech costs the CTC judge more on distinct words than on a
repeated one. It inflates the magnitude of VITS's negative gap and says nothing
about repetition, which is why the sign, not the size, is what this script
reports on.

Usage:  python analysis/vits_config.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEGENERATE = {"empty", "degenerate"}
STOCK = {"noise_scale_duration": 0.8, "speaking_rate": 1.0}
ARMS = {
    "vits": "stock",
    "vitsdur": "noise_scale_duration 1.6",
    "vitsrate": "speaking_rate 1.35",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="data/results/behavioural_vits.csv")
    ap.add_argument("--configs", default="data/results/behavioural_vitscfg.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/vits_config.json")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in (args.stock, args.configs) if Path(p).exists()]
    if not frames:
        raise SystemExit("no VITS results found")
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.k >= args.kmin) & (~d.outcome.isin(DEGENERATE))]
    d = d.assign(err=(d.count_a - d.k) / d.k)

    res: dict = {"kmin": args.kmin, "stock_settings": STOCK, "arms": {}}
    print(f"{'config':10s} {'setting':24s} {'rep exact':>10s} {'ctl exact':>10s} "
          f"{'gap':>7s} {'rep med':>8s}")
    for m, label in ARMS.items():
        s = d[d.model == m]
        if not len(s):
            continue
        r, c = s[s.family == "word_rep"], s[s.family == "control_word"]
        gap = float((c.err == 0).mean() - (r.err == 0).mean())
        res["arms"][m] = dict(
            setting=label, n=int(len(s)),
            rep_exact=float((r.err == 0).mean()),
            ctl_exact=float((c.err == 0).mean()),
            rep_median_err=float(r.err.median()), gap=gap)
        print(f"{m:10s} {label:24s} {100*(r.err==0).mean():9.1f}% "
              f"{100*(c.err==0).mean():9.1f}% {100*gap:+7.1f} "
              f"{r.err.median():+8.3f}")

    gaps = {m: v["gap"] for m, v in res["arms"].items()}
    res["n_arms"] = len(gaps)
    res["all_negative"] = bool(gaps and all(g < 0 for g in gaps.values()))
    res["all_median_zero"] = bool(all(v["rep_median_err"] == 0
                                      for v in res["arms"].values()))
    res["verdict"] = (
        "VITS shows no repeated-side deficit at any duration-predictor setting "
        "tested; its immunity is not an artifact of stock configuration"
        if res["all_negative"] else
        "at least one setting reverses the sign: VITS's immunity is "
        "configuration-dependent and the architectural comparison cannot rest "
        "on the stock run")
    print(f"\n{res['verdict']}.")
    if res["all_median_zero"]:
        print("The repeated-side median relative error is exactly zero in every "
              "arm.")
    print("\nWhat this does not show: two settings of one knob on one non-AR\n"
          "model. It rules out the specific objection that stock defaults were\n"
          "load-bearing, not the broader one that VITS is small and old.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
