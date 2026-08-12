#!/usr/bin/env python3
"""Does the standard decoding-rule mitigation fix the count deficit?

The paper argues that mitigations acting on the decoding rule treat a symptom:
under Theorem A the horizon is set by the contraction factor `q` and the readout
margin, neither of which a sampling penalty touches. That is an assertion until
somebody sweeps one, so we sweep the most widely deployed instance of it.

XTTS-v2 ships `repetition_penalty = 5.0` on acoustic tokens. We regenerate the
whole benchmark at penalties 1, 2, 3, 5 and 8 and measure the relative count
error at k >= 6 with the CTC judge. Three outcomes are distinguishable and each
means something different:

  * error shrinks monotonically toward zero as the penalty rises -- the lever
    works, and the paper's framing is wrong;
  * error is roughly flat across the usable range -- the lever does not act on
    the quantity that governs the horizon, which is the paper's claim;
  * error is non-monotone, with degradation at both extremes -- the penalty is
    trading one failure for another rather than fixing anything.

The control condition is the check that the sweep is doing anything at all: if
control error also moves with the penalty, the setting is changing general
rendering quality rather than repetition behaviour specifically.

Usage:  python analysis/mitigation_sweep.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# registry key -> repetition_penalty actually used, per architecture.
# Two review rounds objected that a sweep on one model cannot support a claim
# about the field's standard mitigation, so Qwen3-TTS-0.6B is swept too. Its
# shipped penalty is 1.05 against XTTS-v2's 5.0, so the two cover different
# parts of the range rather than repeating one.
SWEEPS = {
    "XTTS-v2": {"xtts2norp": 1.0, "xtts2rp2": 2.0, "xtts2rp3": 3.0,
                "xtts2": 5.0, "xtts2rp8": 8.0},
    "Qwen3-TTS-0.6B": {"qwen06brp10": 1.0, "qwen06b": 1.05,
                       "qwen06brp15": 1.5, "qwen06brp30": 3.0},
}
SWEEP = {k: v for d in SWEEPS.values() for k, v in d.items()}
ARCH_OF = {k: arch for arch, d in SWEEPS.items() for k in d}
DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"


def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = [float(np.median(rng.choice(x, x.size, replace=True))) for _ in range(n_boot)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/mitigation_sweep.json")
    ap.add_argument("--csv", default="data/results/behavioural_sweep.csv")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--no-rescore", action="store_true")
    args = ap.parse_args()

    have = [m for m in SWEEP
            if Path(DATA_ROOT, "asr_ctc", f"{m}.jsonl").exists()]
    missing = sorted(set(SWEEP) - set(have))
    if missing:
        print(f"note: no CTC transcripts yet for {missing}; reporting what exists")
    if not have:
        sys.exit("no sweep arms transcribed yet")

    if not args.no_rescore:
        subprocess.run([sys.executable, "src/common/score_counts.py", "--models",
                        *have, "--judge", "ctc", "--out", args.csv],
                       check=True, capture_output=True)

    d = pd.read_csv(args.csv)
    d = d[d.family.isin(["word_rep", "control_word"])].copy()
    d["rel_err"] = (d.count_a - d.k) / d.k
    ok = ~d.outcome.isin(["empty", "degenerate"])

    res: dict = {"kmin": args.kmin, "arms": {}}
    print(f"{'penalty':>8s} {'repeated rel.err':>26s} {'control rel.err':>26s} {'degen%':>7s}")
    for m in sorted(have, key=lambda k: SWEEP[k]):
        row: dict = {"model": m, "penalty": SWEEP[m], "arch": ARCH_OF[m]}
        for fam, key in (("word_rep", "rep"), ("control_word", "ctl")):
            s = d[(d.model == m) & (d.family == fam) & (d.k >= args.kmin) & ok]
            v = s.rel_err.to_numpy(dtype=float)
            lo, hi = boot_ci(v)
            row[key] = dict(median=float(np.median(v)) if v.size else np.nan,
                            lo=lo, hi=hi, n=int(v.size))
        dg = d[(d.model == m) & (d.family == "word_rep") & (d.k >= args.kmin)]
        row["degen_rate"] = float((~ok[dg.index]).mean()) if len(dg) else np.nan
        res["arms"][m] = row
        print(f"{SWEEP[m]:8.1f} {row['rep']['median']:+8.3f} "
              f"[{row['rep']['lo']:+.3f},{row['rep']['hi']:+.3f}] n={row['rep']['n']:4d} "
              f"{row['ctl']['median']:+8.3f} "
              f"[{row['ctl']['lo']:+.3f},{row['ctl']['hi']:+.3f}] n={row['ctl']['n']:4d} "
              f"{100*row['degen_rate']:6.1f}")

    # Is the deficit explained by the penalty? Restrict to arms whose control is
    # intact, since an arm that cannot render the control at all says nothing
    # about repetition specifically.
    # Fit each architecture separately: the two ship penalties an order of
    # magnitude apart, so a single slope across both would be meaningless.
    res["by_arch"] = {}
    for arch, keys in SWEEPS.items():
        usable = [r for k, r in res["arms"].items()
                  if k in keys and np.isfinite(r["ctl"]["median"])
                  and abs(r["ctl"]["median"]) < 0.05
                  and np.isfinite(r["rep"]["median"])]
        if len(usable) < 3:
            print(f"\n{arch}: {len(usable)} usable arms, too few to fit")
            continue
        x = np.array([r["penalty"] for r in usable])
        y = np.array([r["rep"]["median"] for r in usable])
        slope = float(np.polyfit(x, y, 1)[0])
        res["by_arch"][arch] = dict(
            n_usable=len(usable), slope_per_penalty_unit=slope,
            rep_err_range=[float(y.min()), float(y.max())],
            penalties=[float(v) for v in x],
            removes_deficit=bool(y.max() > -0.02))
        print(f"\n{arch}: {len(usable)} arms with an intact control "
              f"(penalties {list(x)})")
        print(f"  repeated error spans {y.min():+.3f} to {y.max():+.3f}, "
              f"slope {slope:+.4f} per unit")
        print("  -> the penalty does not remove the deficit"
              if y.max() < -0.02 else
              "  -> the penalty removes the deficit here; revisit the claim")
    # Keep the flat keys the paper's macros already read, from XTTS-v2.
    x_arch = res["by_arch"].get("XTTS-v2")
    if x_arch:
        res["n_usable_arms"] = x_arch["n_usable"]
        res["slope_per_penalty_unit"] = x_arch["slope_per_penalty_unit"]
        res["rep_err_range"] = x_arch["rep_err_range"]
        res["usable_penalties"] = x_arch["penalties"]
    res["n_architectures"] = len(res["by_arch"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
