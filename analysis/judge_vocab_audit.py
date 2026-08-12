#!/usr/bin/env python3
"""Which stimulus words can the judge actually transcribe?

A control item is scored on whether its k filler words come back. That only
measures the model if the judge can render each filler in the first place. Ours
cannot render all of them: `okay` comes back as "o k", never as "okay", so
template t2's controls score the judge's orthography rather than the decoder.

Separating judge failure from model failure needs a criterion that cannot be
satisfied by any decoder behaviour. Restricting to low k does not give one: at
k <= 4 exact-match is only 60-74%, so low-k misses are not attributable to the
judge by assumption, and the pool words past the fourth never occur at low k at
all. The criterion used instead is *near-total absence*. A word the judge emits
in essentially none of its occurrences is not measuring a decoder: six
independent checkpoints do not fail on one filler in 181 of 181 attempts while
rendering its neighbours in the same breath. Whatever the acoustics were, the
judge's orthography is what the score is reading.

So: for every scored word, the fraction of items containing it whose transcript
contains it, over the whole ladder. Words below `--floor` are unmeasurable, and a
template is unusable if any of its scored vocabulary is. The floor is set low
deliberately --- it is meant to catch words that never appear, not words that
appear less often than one would like --- and the rule is applied to both item
families, so no exclusion can quietly favour the control side of the comparison.

Usage:  python analysis/judge_vocab_audit.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.score_counts import normalise  # noqa: E402

ABLATIONS = {"xtts2norp", "xtts2rp2", "xtts2rp3", "xtts2rp8"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--kmax", type=int, default=0,
                    help="restrict to k<=kmax; 0 uses the whole ladder")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="a word delivered below this rate is unmeasurable")
    ap.add_argument("--out", default="data/results/judge_vocab_audit.json")
    args = ap.parse_args()

    stim = {}
    for line in Path(args.stimuli).open():
        it = json.loads(line)
        stim[it["item_id"]] = it

    d = pd.read_csv(args.behavioural)
    d = d[(~d.model.isin(ABLATIONS)) & (~d.outcome.isin(["empty", "degenerate"]))]
    if args.kmax:
        d = d[d.k <= args.kmax]

    seen: dict[str, list[int]] = defaultdict(list)
    word_tmpl: dict[str, set] = defaultdict(set)
    for r in d.itertuples():
        it = stim.get(r.item_id)
        if not it:
            continue
        units = it.get("boundary_units") or []
        toks = normalise(str(r.transcript) if isinstance(r.transcript, str) else "")
        for u in set(units):
            parts = normalise(u)
            if not parts:
                continue
            n = len(parts)
            hit = any(toks[i:i + n] == parts for i in range(max(0, len(toks) - n + 1)))
            seen[u].append(int(hit))
            word_tmpl[u].add(it["template"])

    rates = {w: sum(v) / len(v) for w, v in seen.items() if len(v) >= 10}
    bad = {w: r for w, r in rates.items() if r < args.floor}

    scope = f"k<={args.kmax}" if args.kmax else "the whole ladder"
    print(f"judge delivery over {scope}\n")
    for w, r in sorted(rates.items(), key=lambda kv: kv[1]):
        mark = "  <-- BELOW FLOOR" if r < args.floor else ""
        print(f"  {w:10s} {r:5.2f}  (n={len(seen[w]):3d}, templates "
              f"{','.join(sorted(word_tmpl[w]))}){mark}")

    bad_tmpl = sorted({t for w in bad for t in word_tmpl[w]})
    print(f"\nwords below floor {args.floor}: {sorted(bad) or 'none'}")
    print(f"=> unusable templates: {bad_tmpl or 'none'}")
    if bad_tmpl:
        print("   Their items measure the judge's orthography, not the decoder,\n"
              "   and are excluded from both families.")

    res = dict(kmax=args.kmax or None, floor=args.floor,
               delivery={w: dict(rate=r, n=len(seen[w]),
                                 templates=sorted(word_tmpl[w]))
                         for w, r in sorted(rates.items())},
               below_floor=sorted(bad), excluded_templates=bad_tmpl)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
