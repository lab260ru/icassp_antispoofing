#!/usr/bin/env python3
"""Minimal disambiguators at period 1: can the decoder be helped to individuate?

WHY THIS FILE EXISTS
--------------------
Four mechanistic accounts of the counting deficit have been tested to a
conclusion and all four failed: contraction (q = 21-348 on both arms),
attention dilution (dose-response sign-reversed), the stop head reading
duration (predictors inseparable), and the count being erased from the state (a
rank-1 patch transplants the count coordinate cleanly and the decoder ignores
it). What survived points one way: the count *is* linearly decodable and stays
decodable late; VITS, which binds each input token to its own output span, is
immune; F5-TTS, which estimates one total duration, is not, and supplying it the
true duration fixes k<12 and not k>=12; CosyVoice 2 makes about the right
*amount* of speech and still loses the count; and a period-8 cycled control
shows no deficit at all.

The account that fits: **the decoder cannot individuate identical spans.** The
count is represented, but the generation policy runs on local context, and
token-identical neighbours give it nothing to advance against.

`make_stimuli_period.py` tests that by varying the period. This file tests it
from the other side, at period 1, by leaving the repetition verbatim and making
successive spans *locally distinguishable* with the smallest edit available:

    bare      The dog was very very very ... very big and it ran ...
    comma     The dog was very, very, very, ... very big and it ran ...
    stop      The dog was very. Very. Very. ... Very. big and it ran ...
    and       The dog was very and very and very ... and very big and it ran ...

The counted unit is the same word in all four. The count is the same k in all
four. `comma` and `stop` hold the word count *exactly* fixed -- punctuation
attaches to a word and `text.split()` is unchanged -- so they are the clean
cells. `and` does not: it inserts k-1 function words, and it is included anyway
because it is the one variant a text front-end cannot normalise away (see the
uninterpretability section).

WHAT WOULD SUPPORT AND WHAT WOULD FALSIFY  (pre-committed; written before any
disambiguator audio existed)
-----------------------------------------------------------------------------
Let E(bare) be the exact rate of the published repeated arm, E(dis) the exact
rate of a disambiguated arm at the same k and carrier, and E(8) the period-8
control's rate -- the "no deficit" ceiling. Define the recovered fraction

    R = (E(dis) - E(bare)) / (E(8) - E(bare)).

  * R >= 0.50  -> INDIVIDUATION SUPPORTED. A local edit that changes neither
                  the counted word nor the count recovers most of the deficit.
                  That is the account's central prediction and nothing else on
                  the table predicts it.
  * R <= 0.15  -> INDIVIDUATION FALSIFIED. Making successive spans locally
                  distinguishable does nothing. The decoder is not failing for
                  want of something to advance against, and the account should
                  be dropped from the paper rather than softened.
  * in between -> PARTIAL. Report as partial; it is not the clean confirmation
                  the account needs, and a mechanism section should not be
                  written on it.

R is computed on `comma` and `stop` first, because those are the length-matched
cells. `and` is reported beside them, never averaged into them.

Ordering prediction, also pre-committed: if the account is right, `stop` (a
sentence boundary, the strongest local break) should recover at least as much
as `comma` (a phrase break), and both should be positive. `stop` < `comma`
< 0 would be a shape no version of the account predicts.

WHAT WOULD MAKE THIS UNINTERPRETABLE
------------------------------------
1. **A front-end that strips punctuation.** If a model's text normaliser drops
   commas, `comma` is *character-identical to bare after normalisation* and its
   null result is an artefact of the pipeline, not evidence about the decoder.
   This is not hypothetical -- TTS front-ends normalise aggressively. The check
   is decisive and mechanical: generation is seeded per item, so if the
   front-end sees the same string, the audio is the same audio. The analysis
   compares the `bare` and `comma` waveforms for each (item, seed) and reports
   any cell where they are identical as UNINTERPRETABLE rather than as a null.
   `and` exists precisely because no normaliser can delete a lexical token.
2. **Length.** `and` is 2k-1 repetition words against bare's k. Length hurts
   counting -- that is the paper's own result -- so the confound pushes `and`
   *down*. A recovery there is therefore conservative, and a null there is not
   interpretable on its own.
3. **The judge.** Inserting material between repetitions changes the acoustics
   the recogniser sees, and the score is occurrences of the target word. If the
   judge's delivery rate for the target drops in a disambiguated arm we are
   measuring the recogniser. `analysis/disambiguation.py` audits target-word
   delivery per arm before reading any rate.
4. **Prosody is not individuation.** `comma` and `stop` also change *duration*
   -- pauses lengthen the audio. If the recovery tracks added duration rather
   than the break itself, "individuation" is the wrong word for it and
   "the model needed more time" is the right one. Duration per arm is reported
   so the two can be told apart, and `and` (which adds tokens, not just pauses)
   discriminates them.

Output: data/stimuli/stimuli_disambig.jsonl
Usage:  python data/stimuli/make_stimuli_disambig.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stimuli_period import K_LADDER, WORD_TEMPLATES  # noqa: E402

# Item ids share the period ladder's `pd_` prefix on purpose: that prefix is
# what `population.NON_PANEL_ITEM_PREFIXES` keys on, so this arm inherits the
# guard that keeps it out of the English panel instead of needing its own.
PREFIX = "pd_"


def variants(target: str, k: int) -> list[tuple[str, str, str]]:
    """(variant id, rendered repetition block, one-line description).

    `bare` is generated here as well as in the period ladder even though the
    two are character-identical, because the recovered fraction R is a ratio of
    differences and the denominator has to come from the same generation run as
    the numerator. Reusing the period ladder's p=1 rows would work today and
    break silently the first time one of the two arms is re-run.
    """
    rep = [target] * k
    return [
        ("bare", " ".join(rep), "the published repeated arm"),
        # Comma after every repetition except the last, so the block joins the
        # suffix exactly as the bare block does.
        ("comma", ", ".join(rep), "phrase break between repetitions"),
        # Sentence break. Capitalised because a full stop followed by a
        # lower-case word is text no front-end was trained on, and the point is
        # to add a boundary, not to add an oddity.
        ("stop", ". ".join(w.capitalize() for w in rep) + ".",
         "sentence break between repetitions"),
        # The lexical disambiguator. Word count is NOT held fixed here; see the
        # module docstring.
        ("and", " and ".join(rep), "conjunction between repetitions"),
    ]


def build(ks: list[int]) -> list[dict]:
    items: list[dict] = []
    for tid, prefix, target, suffix, _pool in WORD_TEMPLATES:
        for k in ks:
            for vid, block, desc in variants(target, k):
                text = f"{prefix} {block} {suffix}"
                items.append(dict(
                    item_id=f"{PREFIX}{target}_{tid}_k{k:02d}_d{vid}",
                    family="word_rep", template=tid, text=text,
                    target_unit=target, k=k, expected_count=k,
                    expected_words=len(text.split()), control_of=None,
                    # k copies of one word, so `score_counts.py` takes the
                    # unbounded-occurrence branch exactly as it does for the
                    # published `wr_*` items. The inserted commas, stops and
                    # conjunctions are never counted; the scored target is the
                    # repeated content word and nothing else, which is what
                    # keeps this a measurement of the model rather than of the
                    # judge's handling of the material we inserted.
                    boundary_units=[target] * k,
                    ladder="disambig", period=1, rotation=0, pool_id="target",
                    n_distinct=1, variant=vid, variant_desc=desc,
                    instrumented=False,
                ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "stimuli_disambig.jsonl"))
    ap.add_argument("--ks", type=int, nargs="+", default=K_LADDER)
    args = ap.parse_args()

    items = build(args.ks)

    seen: set[str] = set()
    for it in items:
        assert it["item_id"] not in seen, f"duplicate id {it['item_id']}"
        seen.add(it["item_id"])

    # The length invariant the clean cells rest on: `bare`, `comma` and `stop`
    # are the same number of words at every (template, k); `and` is 2k-1 and is
    # asserted to be, so that its inequality is a stated design fact rather than
    # an accident nobody checked.
    from collections import defaultdict
    cells: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for it in items:
        cells[(it["template"], it["k"])][it["variant"]] = it
    for (tid, k), g in cells.items():
        w = {v: g[v]["expected_words"] for v in ("bare", "comma", "stop")}
        assert len(set(w.values())) == 1, f"{tid} k={k}: length not held fixed {w}"
        assert g["and"]["expected_words"] == w["bare"] + k - 1, (
            f"{tid} k={k}: `and` arm is not bare+{k-1} words")

    out = Path(args.out)
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    from collections import Counter
    per = Counter(i["variant"] for i in items)
    print(f"wrote {len(items)} items -> {out}")
    print(f"  k         {args.ks}")
    print(f"  templates {[t[0] for t in WORD_TEMPLATES]}")
    for v, n in per.items():
        print(f"  {v:6s} {n:4d} items")
    ex = [i for i in items if i["template"] == "t1" and i["k"] == 16]
    print("\n  t1, k=16:")
    for i in ex:
        print(f"    {i['variant']:6s} ({i['expected_words']:2d} words) {i['text'][:90]}...")


if __name__ == "__main__":
    main()
