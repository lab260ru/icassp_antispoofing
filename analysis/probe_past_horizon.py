#!/usr/bin/env python3
"""The theorem's own prediction, tested where it applies, on every checkpoint we
could run it on.

Theorem 1(iv) forbids an $L$-Lipschitz readout from separating counts *past* the
horizon. It says nothing below it. Our linear probe ran on the main ladder,
$k\\le32$, at or below the fitted saturation scale --- so it was operating in the
region where the theorem permits a readout to succeed, and its success there
tested nothing. S12 named the missing experiment: probe states past the point
where tracking stops.

This runs it. The extension items were regenerated with hidden-state capture and
the same ridge probe applied, so the only thing that changes is the range.

Reading a low $R^2$ as vindication would be a mistake, and the comparison here is
built to prevent it. $R^2$ falls when the target varies less, and
$k\\in\\{48,\\ldots,128\\}$ spans about a third of the range $k\\in\\{2,\\ldots,32\\}$
does in $\\log_2$. The honest yardstick is a predictor that ignores the states
entirely and always answers the mean: if the trained probe cannot beat that, it
has learned nothing about the count, whatever $R^2$ says.

Two arms exist because either could have been the artifact:

  control, same k range   rules out "the extended range is too narrow to decode
                          from". The theorem says nothing about aperiodic
                          carriers, so the probe should succeed on them --- and
                          it does.
  more checkpoints        rules out "one model, one seed, twelve items". This is
                          the arm that did not come back the way we hoped.

The comparison is diagnostic, which is why the split matters rather than merely
disappointing. The rival account has the count surviving in the states while only
the output policy fails, so it predicts the probe *can* still read the count past
the horizon; Theorem 1(iv) predicts it cannot. A checkpoint that loses the count
is evidence for the theorem and against the rival, and one that keeps it is the
reverse. Every count printed below is therefore a vote, and the script reports
the tally rather than the checkpoints that voted the way we hoped.

Usage:  python analysis/probe_past_horizon.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAST_KS = [48, 64, 96, 128]
PAST_REPS = 3
BELOW_KS = [2, 3, 4, 6, 8, 12, 16, 24, 32]
BELOW_REPS = 6


def constant_predictor_mae(ks: list[int], reps: int) -> float:
    """MAE in log2 units of always predicting the mean of the target."""
    y = np.log2(np.repeat(np.array(ks, dtype=float), reps))
    return float(np.mean(np.abs(y - y.mean())))


def row(mae: float, r2: float, n, ks: list[int], reps: int) -> dict:
    const = constant_predictor_mae(ks, reps)
    return dict(ks=ks, reps=reps, mae=mae, r2=r2, n=n, const_mae=const,
                beats_constant=bool(mae < const), mae_ratio=float(mae / const))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--below", default="data/results/probe.json")
    ap.add_argument("--past", nargs="+",
                    default=["llasa1b=data/results/probe_past_horizon.json",
                             "llasa3b=data/results/probe_past_horizon_3b.json",
                             "llasa8b=data/results/probe_past_horizon_8b.json",
                             "qwen06b=data/results/probe_past_horizon_q06b.json",
                             "qwen17b=data/results/probe_past_horizon_q17b.json"],
                    help="model=path pairs, one per checkpoint probed past N*")
    # One control arm rules the range confound out for one checkpoint. It was a
    # single checkpoint for as long as only one had control activations, and the
    # rebuttal was correspondingly narrow: the probe reads control states at
    # 0.86 *on Llasa-1B*. Every checkpoint that lost the count needs its own
    # control over its own k range, so this takes a list.
    ap.add_argument("--past-control", nargs="+",
                    default=["llasa1b=data/results/probe_past_horizon_ctl.json",
                             "llasa3b=data/results/probe_past_horizon_ctl_3b.json",
                             "llasa8b=data/results/probe_past_horizon_ctl_8b.json",
                             "qwen06b=data/results/probe_past_horizon_ctl_q06b.json",
                             "qwen17b=data/results/probe_past_horizon_ctl_q17b.json"],
                    help="control items over the same k range: the test that "
                         "rules out 'the range is too narrow'")
    ap.add_argument("--out", default="data/results/probe_horizon_compare.json")
    args = ap.parse_args()

    below_all = json.loads(Path(args.below).read_text())["models"]
    rows: dict[str, dict] = {}

    for spec in args.past:
        model, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            continue
        past = json.loads(p.read_text())["models"][model]["word_rep"]
        rows[f"past:{model}"] = row(past["late"]["best"]["mae"],
                                    past["late"]["best"]["r2"],
                                    past.get("n_items"), PAST_KS, PAST_REPS)
        if model in below_all:
            b = below_all[model]["word_rep"]
            rows[f"below:{model}"] = row(b["late"]["best"]["mae"],
                                         b["late"]["best"]["r2"],
                                         b.get("n_items"), BELOW_KS, BELOW_REPS)

    # Control items over the identical k range. The theorem says nothing about
    # them, so if the probe reads the count off *their* states while failing on
    # the repeated ones, that failure cannot be blamed on the range being narrow.
    for spec in args.past_control:
        cmodel, _, cpath = spec.partition("=")
        if not Path(cpath).exists():
            continue
        c = json.loads(Path(cpath).read_text())["models"].get(cmodel, {}).get("control_word")
        if c:
            rows[f"control:{cmodel}"] = row(c["late"]["best"]["mae"],
                                            c["late"]["best"]["r2"],
                                            c.get("n_items"), PAST_KS, PAST_REPS)

    print(f"{'arm':>26s} {'probe MAE':>10s} {'constant MAE':>13s} "
          f"{'ratio':>6s} {'R2':>7s}")
    for key in sorted(rows, key=lambda k: (k.split(":")[0] != "below", k)):
        kind, model = key.split(":")
        lab = {"below": "repeated k=2..32",
               "past": "repeated k=48..128",
               "control": "control  k=48..128"}[kind]
        v = rows[key]
        print(f"{lab + ' ' + model:>26s} {v['mae']:10.3f} {v['const_mae']:13.3f} "
              f"{v['mae_ratio']:6.2f} {v['r2']:+7.3f}")

    past_keys = [k for k in rows if k.startswith("past:")]
    lost = [k.split(":")[1] for k in past_keys if not rows[k]["beats_constant"]]
    kept = [k.split(":")[1] for k in past_keys if rows[k]["beats_constant"]]
    # "Beats the constant predictor" is a threshold, and a threshold decides
    # borderline cases by fiat. A checkpoint whose MAE ratio is 0.99 has beaten
    # it by six thousandths of a log2 unit on twelve items, which is not a
    # margin -- reporting that as "recovers the count" would be as misleading as
    # rounding it the other way. Anything within TIE of 1.0 is called
    # indistinguishable and named as such, whichever side of 1.0 it falls.
    TIE = 0.02
    tied = [k.split(":")[1] for k in past_keys
            if abs(rows[k]["mae_ratio"] - 1.0) <= TIE]
    ctl_models = sorted(k.split(":")[1] for k in rows if k.startswith("control:"))
    ctl_key = next((k for k in rows if k.startswith("control:")), None)

    res = dict(rows=rows, n_checkpoints=len(past_keys),
               lost_past_horizon=sorted(lost), kept_past_horizon=sorted(kept),
               tie_band=TIE, indistinguishable=sorted(tied),
               clearly_kept=sorted(k for k in kept if k not in tied))
    res["replicates"] = bool(past_keys and not kept)
    if ctl_models:
        res["controls"] = ctl_models
        res["control_ratios"] = {m: rows[f"control:{m}"]["mae_ratio"]
                                 for m in ctl_models}
        res["controls_beating_constant"] = [
            m for m in ctl_models if rows[f"control:{m}"]["beats_constant"]]
        # Narrowness is ruled out only for the checkpoints that actually lost
        # the count: on a checkpoint that keeps it, there is nothing to explain.
        # And it is ruled out *per checkpoint*, not once for the panel --- a
        # control arm on one model says nothing about the range on another.
        need = [m for m in lost if m in ctl_models]
        res["range_confound_checked_on"] = need
        res["range_confound_unchecked"] = [m for m in lost if m not in ctl_models]
        res["range_confound_ruled_out"] = bool(
            need and all(rows[f"control:{m}"]["beats_constant"] for m in need)
            and not res["range_confound_unchecked"])
        # The same question under the tie-band reading, where a checkpoint that
        # merely ties the constant predictor also has nothing recoverable and so
        # also needs its range excused.
        null_side = sorted(set(lost) | set(tied))
        res["range_confound_null_side"] = [
            m for m in null_side if m in ctl_models
            and rows[f"control:{m}"]["beats_constant"]]
        res["range_confound_null_side_unchecked"] = [
            m for m in null_side if m not in ctl_models]

    if res["replicates"]:
        res["verdict"] = ("the probe loses the count past the horizon in every "
                          "checkpoint tested")
    elif lost:
        res["verdict"] = (
            f"the prediction holds in {len(lost)} of {len(past_keys)} checkpoints: "
            f"{', '.join(lost)} lose the count past the horizon, "
            f"{', '.join(kept)} do not")
    else:
        res["verdict"] = ("no checkpoint loses the count past the horizon; the "
                          "theorem's conclusion is not supported here")
    print(f"\n{res['verdict']}.")
    if tied:
        names = ", ".join(tied)
        print(f"\nBut {names} sit within {100*TIE:.0f}% of the constant "
              f"predictor, so the\nstrict threshold is deciding them by fiat. "
              f"By effect size {len(tied) + len(lost) - len(set(tied) & set(lost))}"
              f" of {len(past_keys)} checkpoints show no\nrecoverable count past "
              "the horizon and "
              f"{len(res['clearly_kept'])} clearly does. We quote the strict "
              "number,\nwhich is the less favourable one.")

    if lost and kept:
        print(f"\nThe checkpoints disagree, so this is not a panel result. "
              f"Quoting the\n{len(lost)} that lost the count without the "
              f"{len(kept)} that did not would be\ncherry-picking; the paper "
              f"reports {len(lost)} of {len(past_keys)}.")
    if ctl_models:
        ratios = ", ".join(f"{m} {rows[f'control:{m}']['mae_ratio']:.2f}"
                           for m in ctl_models)
        print(f"\nControl arms over the SAME k range ({len(ctl_models)} "
              f"checkpoints): {ratios}.")
        failed_ctl = [m for m in ctl_models
                      if not rows[f"control:{m}"]["beats_constant"]]
        if failed_ctl:
            print(f"{', '.join(failed_ctl)} fail to beat the constant predictor "
                  "on control\nstates too, so on those checkpoints a narrow range "
                  "cannot be excluded.\nDo not quote their repeated-side null as "
                  "clean.")
        if res.get("range_confound_ruled_out"):
            print(f"Every checkpoint that lost the count "
                  f"({', '.join(res['range_confound_checked_on'])}) has its own "
                  f"control\narm recovering it over the identical k, so where the "
                  "count is lost a narrow\nrange is not why.")
        elif res.get("range_confound_unchecked"):
            print(f"No control arm for {', '.join(res['range_confound_unchecked'])}"
                  ", which lost the count, so the\nrange rebuttal does not cover "
                  "the whole null side.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
