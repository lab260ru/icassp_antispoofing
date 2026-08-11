#!/usr/bin/env python3
"""An ASR-free replication of the behavioural dissociation.

Why this exists. Our instrument audit (`analysis/asr_reliability.py`) found that
Whisper large-v3 systematically undercounts *genuinely correct* repeated
material: on concatenative audio with known ground truth its counted/true ratio
falls to 0.27--0.65 for k >= 4, while matched distinct-word sequences of the same
length transcribe at ratio 1.00. The judge therefore has a bias pointing in the
same direction as our headline effect, and any accuracy figure that depends on a
transcript count is deflated for repeated items by an unknown amount.

That is a threat to the behavioural claim and has to be met with a measurement
that does not use the transcript at all.

Duration does not. If a model renders k repetitions faithfully, the audio lasts
about as long as k repetitions take, and the expected duration is fitted per
model and template from the low-k regime where rendering is reliable. An item is
counted `duration-correct` when its duration falls in the acceptance band and it
is not flagged degenerate by the audio-level detectors (silence, spectral
flatness, clipping) --- none of which consult the ASR either.

This is a weaker criterion than the conjunctive one: a model that babbles for
exactly the right length passes it. That weakness is the point. It cannot be
inflated by ASR bias, so if the repeated-vs-control gap survives here, the gap is
not an artefact of the judge. Reported alongside, never instead of, the primary
metric.

Usage:  python analysis/duration_only.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ABLATIONS = {"xtts2norp"}
DUR_LO, DUR_HI = 0.70, 1.45


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * min(c + h, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural.csv")
    ap.add_argument("--out", default="data/results/duration_only.json")
    args = ap.parse_args()

    d = pd.read_csv(args.behavioural)
    # instrument-independent correctness
    degenerate = (d.outcome.isin(["empty", "degenerate"])
                  | (d.spectral_flatness > 0.35)
                  | (d.rms < 1e-3))
    d["dur_ok"] = ((d.duration_ratio >= DUR_LO) & (d.duration_ratio <= DUR_HI)
                   & ~degenerate).astype(int)

    panel = d[~d.model.isin(ABLATIONS)]
    res: dict = {"criterion": f"duration_ratio in [{DUR_LO},{DUR_HI}] and not degenerate",
                 "models": {}}

    print(f"{'model':11s} {'rep k>=6':>18s} {'ctl k>=6':>18s}  gap")
    for m in sorted(panel.model.unique()):
        row = {}
        for fam, key in (("word_rep", "rep"), ("control_word", "ctl")):
            s = panel[(panel.model == m) & (panel.family == fam) & (panel.k >= 6)]
            n, k = len(s), int(s.dur_ok.sum())
            lo, hi = wilson(k, n)
            row[key] = dict(pct=100 * k / n if n else np.nan, lo=lo, hi=hi, n=n)
        res["models"][m] = row
        print(f"{m:11s} {row['rep']['pct']:6.1f} [{row['rep']['lo']:5.1f},{row['rep']['hi']:5.1f}] "
              f"{row['ctl']['pct']:6.1f} [{row['ctl']['lo']:5.1f},{row['ctl']['hi']:5.1f}]  "
              f"{row['ctl']['pct'] - row['rep']['pct']:+6.1f}")

    hi = panel[panel.k >= 6]
    for fam, key in (("word_rep", "rep"), ("control_word", "ctl")):
        s = hi[hi.family == fam]
        n, k = len(s), int(s.dur_ok.sum())
        lo, up = wilson(k, n)
        res[f"pooled_{key}"] = dict(pct=100 * k / n if n else np.nan, lo=lo, hi=up, n=n)

    r, c = res["pooled_rep"], res["pooled_ctl"]
    res["pooled_gap"] = c["pct"] - r["pct"]
    res["ci_disjoint"] = bool(r["hi"] < c["lo"])
    res["n_models_gap_positive"] = int(sum(
        1 for v in res["models"].values() if v["ctl"]["pct"] > v["rep"]["pct"]))
    res["n_models"] = len(res["models"])

    print(f"\npooled at k>=6, ASR-free criterion:")
    print(f"  repeated {r['pct']:.1f}% [{r['lo']:.1f},{r['hi']:.1f}]  n={r['n']}")
    print(f"  control  {c['pct']:.1f}% [{c['lo']:.1f},{c['hi']:.1f}]  n={c['n']}")
    print(f"  gap {res['pooled_gap']:+.1f} points, CIs disjoint: {res['ci_disjoint']}, "
          f"positive in {res['n_models_gap_positive']}/{res['n_models']} models")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
