#!/usr/bin/env python3
r"""How big is the effect against the instrument's own disagreement?

Round-9 reviewers asked for the number that makes every other number readable and
that we had never computed: the judge's error on genuine model output, and hence
the noise floor under the reported effect sizes. Without it, "a median count
error of -8.3%" floats free -- a reader cannot tell whether that is comfortably
above the recogniser's own uncertainty or inside it.

Two independent estimates of the floor, both on real generated audio:

* **Two recognisers on the same audio.** wav2vec2 against HuBERT --- different
  pretraining objectives, so their errors should not be correlated. Their mean
  absolute disagreement is the resolution of the measurement.
* **Perturbation.** Re-transcribing the same clips under mild noise and a
  resampling round trip.

The comparison is made in counts, not percentages, because that is the unit the
recogniser errs in.

The answer is uncomfortable and belongs in the paper: the median item is missing
about one repetition, and two recognisers disagree by about half a count. The
per-item effect is roughly twice the instrument's resolution, which is thin.

Two things keep the claim standing, and both are checkable here rather than
asserted. The deficit is measured against a matched control through the same
judge, and a recogniser with indiscriminate half-count noise could not leave
controls exact in 94% of generations. And where the two recognisers disagree,
ours reports the *higher* count in the large majority of cases --- so our
instrument under-states the deficit rather than manufacturing it.

Usage:  python analysis/noise_floor.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="data/results/ctc_field_validation.json")
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/noise_floor.json")
    args = ap.parse_args()

    fv = json.loads(Path(args.field).read_text())
    items = fv["check4_second_recogniser"]["items"]
    perturb = fv["check3_perturbation_stability"]["by_group"]["word_rep"]["overall"]

    diffs = [abs(i["count_primary"] - i["count_secondary"]) for i in items]
    dis = [i for i in items if i["count_primary"] != i["count_secondary"]]
    ours_higher = sum(1 for i in dis if i["count_primary"] > i["count_secondary"])
    hi_k = [i for i in items if i["k"] >= 12]
    dis_hi = [i for i in hi_k if i["count_primary"] != i["count_secondary"]]

    floor = dict(
        two_recognisers_mean_abs_diff=float(st.mean(diffs)),
        two_recognisers_n=len(items),
        two_recognisers_mean_abs_diff_high_k=float(
            st.mean(abs(i["count_primary"] - i["count_secondary"]) for i in hi_k)),
        n_disagreements=len(dis), ours_higher=ours_higher,
        n_disagreements_high_k=len(dis_hi),
        ours_higher_high_k=sum(1 for i in dis_hi
                               if i["count_primary"] > i["count_secondary"]),
        noise_mean_abs_delta=float(perturb["mean_abs_diff_noise"]),
        resample_mean_abs_delta=float(perturb["mean_abs_diff_resample"]))

    d = pd.read_csv(args.behavioural)
    d = d[d.family.isin(["word_rep", "control_word"])]
    d, _ = panel(d)
    d = d[d.k >= args.kmin]
    d = d.assign(missing=d.k - d.count_a, err=(d.count_a - d.k) / d.k)
    rep = d[d.family == "word_rep"]
    ctl = d[d.family == "control_word"]

    effect = dict(
        median_missing=float(rep.missing.median()),
        median_missing_by_k={int(k): float(g.missing.median())
                             for k, g in rep.groupby("k")},
        mean_missing=float(rep.missing.mean()),
        control_exact_rate=float((ctl.err == 0).mean()))

    ratio = effect["mean_missing"] / floor["two_recognisers_mean_abs_diff"]
    res = dict(floor=floor, effect=effect, effect_over_floor=float(ratio))

    print("noise floor, in repetitions:")
    print(f"  two recognisers disagree by  {floor['two_recognisers_mean_abs_diff']:.2f}"
          f"  (n={floor['two_recognisers_n']}; "
          f"{floor['two_recognisers_mean_abs_diff_high_k']:.2f} at k>=12)")
    print(f"  30 dB noise moves the count  {floor['noise_mean_abs_delta']:.2f}")
    print(f"  resampling moves the count   {floor['resample_mean_abs_delta']:.2f}")
    print("\neffect, same units:")
    print(f"  median repetitions missing   {effect['median_missing']:.1f}")
    print(f"  mean repetitions missing     {effect['mean_missing']:.2f}"
          f"  ({ratio:.1f}x the two-recogniser floor)")
    print(f"\nwhere the two recognisers disagree, ours reports the HIGHER count in "
          f"{floor['ours_higher']} of {floor['n_disagreements']} cases "
          f"({floor['ours_higher_high_k']} of {floor['n_disagreements_high_k']} "
          f"at k>=12): our judge under-states the deficit.")
    print(f"controls come back exact in {100*effect['control_exact_rate']:.1f}% of "
          f"generations through the same judge, which indiscriminate half-count\n"
          f"noise could not produce.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
