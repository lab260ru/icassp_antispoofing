#!/usr/bin/env python3
"""The theorem's own prediction, tested where it applies --- on two checkpoints.

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
  a second checkpoint     rules out "one model, one seed, twelve items". This is
                          the arm that did not come back the way we hoped.

Llasa-1B loses the count past the horizon. Llasa-8B does not: its repeated-side
probe beats the constant predictor by exactly the margin the control arm does.
So the theorem's conclusion holds in one of the two checkpoints we could test it
on, and the paper says one of two rather than quoting the checkpoint that
worked. A prediction that survives in half the cases it was tested in is not
support; it is an open question with one encouraging instance.

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
                             "llasa8b=data/results/probe_past_horizon_8b.json"],
                    help="model=path pairs, one per checkpoint probed past N*")
    ap.add_argument("--past-control",
                    default="llasa1b=data/results/probe_past_horizon_ctl.json",
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
    cmodel, _, cpath = args.past_control.partition("=")
    if Path(cpath).exists():
        c = json.loads(Path(cpath).read_text())["models"][cmodel].get("control_word")
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
    ctl_key = next((k for k in rows if k.startswith("control:")), None)

    res = dict(rows=rows, n_checkpoints=len(past_keys),
               lost_past_horizon=sorted(lost), kept_past_horizon=sorted(kept))
    res["replicates"] = bool(past_keys and not kept)
    if ctl_key:
        # Narrowness is ruled out only for the checkpoints that actually lost
        # the count: on a checkpoint that keeps it, there is nothing to explain.
        res["range_confound_ruled_out"] = bool(rows[ctl_key]["beats_constant"] and lost)

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

    if lost and kept:
        print("\nThis is a failure to replicate, not a panel result. Quoting the\n"
              "checkpoint that lost the count without the one that did not would\n"
              "be cherry-picking; the paper reports one of two.")
    if ctl_key and res.get("range_confound_ruled_out"):
        v = rows[ctl_key]
        print(f"\nOver the SAME k range the probe does recover the count from\n"
              f"control states (MAE {v['mae']:.2f} against {v['const_mae']:.2f} "
              f"for the constant\npredictor), so where the count is lost, a narrow "
              "range is not why.")
    elif ctl_key:
        print("\nThe control arm over the same range also fails to beat the\n"
              "constant predictor, so a narrow range cannot be excluded as the\n"
              "explanation. Do not quote the repeated-side null as clean.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
