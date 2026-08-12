#!/usr/bin/env python3
"""Does the deficit survive greedy decoding? The ablation the theorem invites.

Theorem 1 bounds a readout. No property of the sampling rule enters the proof, so
the bound is stated as covering greedy, sampled and beam decoding alike --- and
that claim sat next to an experimental panel where every generation was sampled.
A reviewer was right that this is the cheapest ablation the paper could run and
the one most directly relevant to its own premise.

The premise is what makes it interesting rather than merely tidy. Assumption 1
posits a single time-invariant map $F$ between repetition boundaries. Stochastic
token choice is exactly the per-step perturbation that would break that
autonomy: what gets sampled at repetition $m$ changes what conditions repetition
$m+1$. Greedy decoding removes the perturbation, so it is the setting in which
Assumption 1 is *most* defensible --- and therefore the setting in which the
deficit has the fewest excuses.

Two readings, and they point opposite ways:

  deficit survives    sampling noise is not the cause. The failure is in the
                      conditioning, not the draw, which is what the theorem
                      needs and what the repetition-penalty sweep could not show.
  deficit disappears  the effect is a sampling artifact and the theorem's frame
                      is the wrong one for it. That would be a bigger finding
                      than the paper's, and we would report it as such.

The comparison is against Qwen3-TTS-0.6B under its shipped sampling settings.
Greedy is deterministic, so one seed is the entire experiment on that arm; the
sampled arm is reported both at seed 0 (one draw, the matched comparison) and
pooled over its three seeds (the panel number), because quoting only whichever
is more favourable would be the obvious way to cheat here.

Usage:  python analysis/greedy_decoding.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import cap_flags  # noqa: E402

DEGENERATE = {"empty", "degenerate"}


def stats(d: pd.DataFrame) -> dict:
    out: dict = {}
    for fam, tag in (("word_rep", "rep"), ("control_word", "ctl")):
        s = d[d.family == fam]
        err = (s.count_a - s.k) / s.k
        out[tag] = dict(n=int(len(s)),
                        exact=float((err == 0).mean()) if len(s) else float("nan"),
                        median_err=float(err.median()) if len(s) else float("nan"))
    out["gap"] = out["ctl"]["exact"] - out["rep"]["exact"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampled", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--greedy", default="data/results/behavioural_greedy.csv")
    ap.add_argument("--model", default="qwen06b")
    ap.add_argument("--greedy-model", default="qwen06bgreedy")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/greedy_decoding.json")
    args = ap.parse_args()

    frames = [pd.read_csv(p) for p in (args.sampled, args.greedy) if Path(p).exists()]
    d = pd.concat(frames, ignore_index=True)
    # Greedy decoding on repetitive text runs into the generation budget far
    # more often than sampling does, and budget-truncated counts are censored
    # downward -- exactly the direction that would manufacture a greedy deficit.
    # The same cap-hit rule the panel uses is applied here, and the rate is
    # reported per arm so the reader can see how unequal it was.
    flags = cap_flags()
    d["cap"] = [flags.get((r.model, r.item_id, r.seed), False) for r in d.itertuples()]
    d = d[(d.k >= args.kmin) & (~d.outcome.isin(DEGENERATE))
          & d.family.isin(["word_rep", "control_word"])]
    cap_rate = {m: float(g.cap.mean()) for m, g in d.groupby("model")}
    d = d[~d.cap]

    samp = d[d.model == args.model]
    arms = {
        "sampled_pooled": samp,
        "sampled_seed0": samp[samp.seed == 0],
        "greedy": d[d.model == args.greedy_model],
    }
    res: dict = {"kmin": args.kmin, "model": args.model, "arms": {},
                 "cap_hit_rate": cap_rate}
    print("cap-hit rate before exclusion: "
          + ", ".join(f"{m} {100*v:.1f}%" for m, v in sorted(cap_rate.items())))
    print(f"{'arm':>16s} {'n':>5s} {'exact rep':>10s} {'exact ctl':>10s} "
          f"{'gap':>7s} {'med rep':>8s}")
    for name, s in arms.items():
        if not len(s):
            continue
        st = stats(s)
        st["n"] = int(len(s))
        res["arms"][name] = st
        print(f"{name:>16s} {len(s):5d} {100*st['rep']['exact']:9.1f}% "
              f"{100*st['ctl']['exact']:9.1f}% {100*st['gap']:6.1f} "
              f"{st['rep']['median_err']:+8.3f}")

    if "greedy" not in res["arms"]:
        raise SystemExit("no greedy results yet")
    g = res["arms"]["greedy"]["gap"]
    s0 = res["arms"].get("sampled_seed0", {}).get("gap")
    sp = res["arms"].get("sampled_pooled", {}).get("gap")
    res["gap_greedy"] = float(g)
    res["gap_sampled_seed0"] = float(s0) if s0 is not None else None
    res["gap_sampled_pooled"] = float(sp) if sp is not None else None
    res["survives"] = bool(g > 0.5 * max(x for x in (s0, sp) if x is not None))

    print(f"\ngreedy gap {100*g:.1f} points against "
          f"{100*s0:.1f} sampled at seed 0 and {100*sp:.1f} pooled")
    res["verdict"] = (
        "the deficit survives greedy decoding: it is not a sampling artifact"
        if res["survives"] else
        "the deficit largely disappears under greedy decoding, which makes it a "
        "property of the sampling rule and not of the conditioning")
    print(f"{res['verdict']}.")
    print("\nGreedy removes the per-step perturbation that would break "
          "Assumption 1's\nautonomy, so this is the setting where that "
          "assumption is most defensible ---\nand where the deficit has the "
          "fewest excuses. One checkpoint, one decoding\nmode: it does not "
          "speak for the panel.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
