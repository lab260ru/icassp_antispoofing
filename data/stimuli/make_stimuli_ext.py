#!/usr/bin/env python3
"""Extension ladder: k far past the main sweep, to test for a counting horizon.

Why this file exists. Theorem A(iv) predicts that the rendered repetition count
*saturates* at a constant `N*`. Fitting the main ladder (k <= 32) with
`c(k) = N*(1 - exp(-k/N*))` returns `N*` between 30 and 640 depending on the
model --- that is, at or beyond the top of the ladder. So the main sweep cannot
distinguish "a horizon we have not reached yet" from "no horizon at all", and
saying otherwise would be reading a prediction into data that does not test it.
This ladder reaches k = 128 to settle it.

Two departures from `make_stimuli.py`, both deliberate:

1. **Controls draw from a pool of distinct words, never cycled.** The main
   control pools hold eight words each and are cycled for k > 8, so those
   controls carry period-8 repetition rather than none. That is tolerable when
   the contrast is against period-1 repetition, but at k = 128 a cycled pool of
   eight would be sixteen full cycles, and the control would stop being a
   control. `--check` asserts the pool is large enough and fails loudly.

2. **Template t2 is dropped.** Our CTC judge transcribes its filler `okay` as
   "o k" in 181 of 181 occurrences, so t2 control items measure the judge's
   orthography rather than the model. Fixed in scoring (see
   `src/common/score_counts.py`), excluded here as well.

Output: data/stimuli/stimuli_ext.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Far ladder. Log-spaced: the question is the *shape* of c(k) over a wide span,
# not resolution at any one point.
K_LADDER_EXT = [48, 64, 96, 128]

# Same carriers and targets as the main sweep so the two ladders splice, minus
# t2 (see module docstring).
WORD_TEMPLATES = [
    ("t1", "The dog was", "very", "big and it ran across the field."),
    ("t3", "He walked", "far", "into the forest before turning back."),
    ("t4", "The light was", "blue", "before the storm arrived."),
]

# 144 distinct common words, all ASR-robust in the same sense the targets were
# chosen to be: frequent, no near-homophones, no orthographic variants of the
# kind that made `okay` unscoreable. Large enough that k = 128 controls never
# repeat a word, with headroom for the per-template target exclusion.
FILLER_POOL = [
    "quite", "really", "truly", "fairly", "rather", "somewhat", "extremely",
    "notably", "deeply", "clearly", "plainly", "surely", "barely", "hardly",
    "mostly", "nearly", "partly", "simply", "widely", "wholly", "badly",
    "boldly", "briskly", "calmly", "coldly", "coolly", "crudely", "darkly",
    "dearly", "densely", "dimly", "dryly", "dully", "eagerly", "early",
    "easily", "evenly", "exactly", "faintly", "falsely", "famously", "fiercely",
    "finely", "firmly", "flatly", "fondly", "formally", "frankly", "freely",
    "freshly", "gently", "gladly", "gravely", "grimly", "harshly", "heavily",
    "highly", "hotly", "hugely", "humbly", "keenly", "kindly", "largely",
    "lately", "lightly", "likely", "lively", "loosely", "loudly", "lowly",
    "madly", "mainly", "meekly", "merely", "mildly", "neatly", "newly",
    "nicely", "oddly", "openly", "poorly", "primly", "promptly", "proudly",
    "purely", "quickly", "quietly", "rapidly", "rarely", "readily", "richly",
    "rightly", "roughly", "roundly", "rudely", "sadly", "safely", "sharply",
    "shortly", "shyly", "slightly", "slowly", "smoothly", "softly", "solely",
    "solidly", "sorely", "soundly", "sourly", "sparsely", "speedily", "starkly",
    "steadily", "sternly", "stiffly", "stoutly", "strictly", "strongly",
    "sturdily", "subtly", "sweetly", "swiftly", "tamely", "tautly", "tensely",
    "thickly", "thinly", "tightly", "tiredly", "trimly", "vaguely", "vastly",
    "vividly", "warmly", "weakly", "wearily", "weekly", "wetly", "wildly",
    "wisely", "worthily", "wrongly", "yearly", "coarsely", "curtly", "deftly",
]


def build(ks: list[int]) -> list[dict]:
    items: list[dict] = []
    for tid, prefix, target, suffix in WORD_TEMPLATES:
        pool = [w for w in FILLER_POOL if w != target]
        for k in ks:
            if k > len(pool):
                raise SystemExit(
                    f"filler pool holds {len(pool)} usable words but k={k} needs "
                    f"{k} distinct ones; extend FILLER_POOL rather than cycling "
                    f"it, or the control stops being a control")
            rep = " ".join([target] * k)
            text = f"{prefix} {rep} {suffix}"
            items.append(dict(
                item_id=f"wr_{target}_{tid}_k{k:03d}",
                family="word_rep", template=tid, text=text,
                target_unit=target, k=k, expected_count=k,
                expected_words=len(text.split()), control_of=None,
                boundary_units=[target] * k, ladder="ext",
            ))
            used = pool[:k]
            ctext = f"{prefix} {' '.join(used)} {suffix}"
            items.append(dict(
                item_id=f"ct_{target}_{tid}_k{k:03d}",
                family="control_word", template=tid, text=ctext,
                target_unit=target, k=k, expected_count=0,
                expected_words=len(ctext.split()),
                control_of=f"wr_{target}_{tid}_k{k:03d}",
                boundary_units=list(used), control_units=list(used),
                ladder="ext",
            ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "stimuli_ext.jsonl"))
    ap.add_argument("--ks", type=int, nargs="+", default=K_LADDER_EXT)
    args = ap.parse_args()

    if len(set(FILLER_POOL)) != len(FILLER_POOL):
        dupes = sorted({w for w in FILLER_POOL if FILLER_POOL.count(w) > 1})
        raise SystemExit(f"FILLER_POOL has duplicates, so controls would repeat: {dupes}")

    items = build(args.ks)
    out = Path(args.out)
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    n_rep = sum(1 for i in items if i["family"] == "word_rep")
    longest = max(i["expected_words"] for i in items)
    print(f"wrote {len(items)} items to {out} "
          f"({n_rep} repeated + {len(items) - n_rep} controls), "
          f"k in {args.ks}, longest {longest} words, "
          f"pool {len(FILLER_POOL)} distinct")


if __name__ == "__main__":
    main()
