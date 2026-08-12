#!/usr/bin/env python3
"""The theorem's own prediction, tested where it applies.

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

Usage:  python analysis/probe_past_horizon.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def constant_predictor_mae(ks: list[int], reps: int) -> float:
    """MAE in log2 units of always predicting the mean of the target."""
    y = np.log2(np.repeat(np.array(ks, dtype=float), reps))
    return float(np.mean(np.abs(y - y.mean())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--below", default="data/results/probe.json")
    ap.add_argument("--past", default="data/results/probe_past_horizon.json")
    ap.add_argument("--past-control",
                    default="data/results/probe_past_horizon_ctl.json",
                    help="control items over the same k range: the test that "
                         "rules out 'the range is too narrow'")
    ap.add_argument("--model", default="llasa1b")
    ap.add_argument("--out", default="data/results/probe_horizon_compare.json")
    args = ap.parse_args()

    below = json.loads(Path(args.below).read_text())["models"][args.model]["word_rep"]
    past = json.loads(Path(args.past).read_text())["models"][args.model]["word_rep"]

    rows = {
        "below": dict(ks=[2, 3, 4, 6, 8, 12, 16, 24, 32], reps=6,
                      r2=below["late"]["best"]["r2"], mae=below["late"]["best"]["mae"],
                      n=below.get("n_items")),
        "past": dict(ks=[48, 64, 96, 128], reps=3,
                     r2=past["late"]["best"]["r2"], mae=past["late"]["best"]["mae"],
                     n=past.get("n_items")),
    }
    # Control items over the identical k range. The theorem says nothing about
    # them, so if the probe reads the count off *their* states while failing on
    # the repeated ones, the failure cannot be blamed on the range being narrow.
    ctl_path = Path(args.past_control)
    if ctl_path.exists():
        c = json.loads(ctl_path.read_text())["models"][args.model].get("control_word")
        if c:
            rows["past_control"] = dict(
                ks=[48, 64, 96, 128], reps=3,
                r2=c["late"]["best"]["r2"], mae=c["late"]["best"]["mae"],
                n=c.get("n_items"))
    for v in rows.values():
        v["const_mae"] = constant_predictor_mae(v["ks"], v["reps"])
        v["beats_constant"] = bool(v["mae"] < v["const_mae"])
        v["mae_ratio"] = float(v["mae"] / v["const_mae"])

    print(f"{'range':>26s} {'probe MAE':>10s} {'constant MAE':>13s} "
          f"{'ratio':>6s} {'R2':>7s}")
    for key, lab in (("below", "repeated, k=2..32  (below N*)"),
                     ("past", "repeated, k=48..128 (past N*)"),
                     ("past_control", "control,  k=48..128 (past N*)")):
        v = rows.get(key)
        if not v:
            continue
        print(f"{lab:>26s} {v['mae']:10.3f} {v['const_mae']:13.3f} "
              f"{v['mae_ratio']:6.2f} {v['r2']:+7.3f}")

    res = dict(model=args.model, rows=rows)
    res["verdict"] = (
        "the probe recovers the count below the horizon and not past it"
        if rows["below"]["beats_constant"] and not rows["past"]["beats_constant"]
        else "the split is not clean; do not quote this as support")
    pc = rows.get("past_control")
    if pc:
        res["range_confound_ruled_out"] = bool(
            pc["beats_constant"] and not rows["past"]["beats_constant"])
    print(f"\n{res['verdict']}.")
    if not rows["past"]["beats_constant"]:
        print("Past the horizon the trained probe matches a predictor that ignores\n"
              "the states and always answers the mean: it has learned nothing.\n"
              "This is Theorem 1(iv)'s prediction tested where it applies.")
    if rows.get("past_control"):
        pc = rows["past_control"]
        if res.get("range_confound_ruled_out"):
            print("\nOver the SAME k range the probe does recover the count from\n"
                  f"control states (MAE {pc['mae']:.2f} against "
                  f"{pc['const_mae']:.2f} for the constant predictor), so the\n"
                  "failure on repeated states is not the range being narrow.")
        else:
            print("\nThe control arm over the same range also fails to beat the\n"
                  "constant predictor, so a narrow range cannot be excluded as the\n"
                  "explanation. Do not quote the repeated-side null as clean.")
    print("\nOne checkpoint, one seed, "
          f"{rows['past']['n']} items past the horizon: a single clean\n"
          "observation, not a panel result.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
