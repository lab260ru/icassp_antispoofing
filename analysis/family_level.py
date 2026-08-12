#!/usr/bin/env python3
"""How many independent replicates does this panel really have?

A round-6 reviewer pointed out that treating the six checkpoints as six
independent samples overstates the panel: Llasa contributes three of them and
Qwen3-TTS two, so checkpoints are clustered within three architecture families
and the effective number of architectural replicates is nearer three than six.
Sibling checkpoints share a tokenizer, a training recipe and, for Llasa, a
backbone lineage; their errors are not independent draws.

This recomputes the checkpoint-level contrasts one level up. Each family is
reduced to the mean of its checkpoints' effects, and inference runs across the
three families. With n=3 no rank test can say anything --- the exact two-sided
signed-rank floor is 0.25 --- so we report the family means and the range, and
say plainly that the panel supports a direction and not a p-value.

The point is not that the checkpoint-level numbers were wrong. It is that
"holds in six of six checkpoints" and "holds in three of three families" are
different strengths of claim, and the paper should quote the weaker one.

Usage:  python analysis/family_level.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAMILY = {"llasa1b": "Llasa", "llasa3b": "Llasa", "llasa8b": "Llasa",
          "xtts2": "XTTS", "qwen06b": "Qwen3-TTS", "qwen17b": "Qwen3-TTS"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/results/checkpoint_level.json")
    ap.add_argument("--out", default="data/results/family_level.json")
    args = ap.parse_args()

    ck = json.loads(Path(args.checkpoint).read_text())
    res: dict = {"note": "checkpoints clustered within architecture families; "
                         "n=3 families gives an exact signed-rank floor of 0.25, "
                         "so no rank test is reported"}

    for key in ("exact_rate_gap", "count_error_gap", "capacity_gap"):
        per = ck.get(key, {}).get("per_model")
        if not per:
            continue
        fam: dict[str, list[float]] = {}
        for m, v in per.items():
            fam.setdefault(FAMILY.get(m, m), []).append(float(v))
        means = {f: float(np.mean(vs)) for f, vs in fam.items()}
        vals = np.array(list(means.values()))
        scale = 100 if key != "capacity_gap" else 1
        res[key] = dict(per_family=means,
                        n_families=len(means),
                        n_checkpoints=len(per),
                        mean=float(vals.mean()),
                        lo=float(vals.min()), hi=float(vals.max()),
                        all_positive=bool((vals > 0).all()))
        print(f"\n{key}  ({len(means)} families from {len(per)} checkpoints)")
        for f, v in sorted(means.items()):
            n = len(fam[f])
            print(f"    {f:12s} {scale*v:+8.1f}   (mean of {n} checkpoint"
                  f"{'s' if n > 1 else ''})")
        print(f"  family mean {scale*vals.mean():+.1f}, "
              f"range [{scale*vals.min():+.1f}, {scale*vals.max():+.1f}], "
              f"all positive: {bool((vals > 0).all())}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("With three families the honest statement is a direction that holds in\n"
          "every family and a magnitude that varies across them. A signed-rank\n"
          "test at n=3 cannot go below p=0.25 and is not reported.")


if __name__ == "__main__":
    main()
