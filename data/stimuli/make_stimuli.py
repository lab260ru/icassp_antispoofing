#!/usr/bin/env python3
"""Build the repetition-ladder stimulus set.

The set is a *parametric* probe: every item carries an explicit repetition count
k and a ground-truth number of occurrences of a designated `target_unit`, so a
model's output can be scored by counting rather than by string match. Matched
controls hold word count and syntax fixed while removing the repetition, which
separates "long text is hard" from "repeated text is hard".

Output: data/stimuli/stimuli.jsonl, one item per line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# k-ladder. Dense at the low end (where models still count) and log-spaced above,
# because the collapse point k* is expected in the 4-16 range and we need
# resolution there without paying for 32 separate values.
K_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# Targets chosen for ASR robustness: common, monosyllabic-or-disyllabic, no
# homophone confusions, and unlikely to be swallowed by Whisper's LM prior.
WORD_TEMPLATES = [
    # (template_id, prefix, target, suffix, filler_pool)
    ("t1", "The dog was", "very", "big and it ran across the field.",
     ["quite", "really", "truly", "fairly", "rather", "somewhat", "extremely", "notably"]),
    ("t2", "She said", "no", "to the offer and then left the room.",
     ["yes", "maybe", "sure", "fine", "okay", "well", "right", "hmm"]),
    ("t3", "He walked", "far", "into the forest before turning back.",
     ["deep", "fast", "long", "wide", "high", "low", "near", "past"]),
    ("t4", "The light was", "blue", "before the storm arrived.",
     ["green", "bright", "pale", "dim", "warm", "cold", "sharp", "soft"]),
    ("t5", "The runner moved", "quick", "along the narrow path.",
     ["swift", "smooth", "steady", "light", "sharp", "loose", "tight", "clean"]),
    ("t6", "They were", "really", "tired after the long journey.",
     ["truly", "quite", "very", "rather", "deeply", "clearly", "plainly", "surely"]),
]

# Number phrases. English number words are the natural repetition stress test:
# the same digit-word recurs at several magnitudes and the model must keep an
# internal place-value counter that repetition-attractor dynamics would destroy.
NUMBER_PHRASES = [
    ("n01", "six hundred sixty six thousand six hundred sixty six", "six", 6),
    ("n02", "seven hundred seventy seven thousand seven hundred seventy seven", "seven", 6),
    ("n03", "nine hundred ninety nine thousand nine hundred ninety nine", "nine", 6),
    ("n04", "three hundred thirty three thousand three hundred thirty three", "three", 6),
    ("n05", "six hundred sixty six million six hundred sixty six thousand six hundred sixty six",
     "six", 9),
    ("n06", "eight hundred eighty eight thousand eight hundred eighty eight", "eight", 6),
    ("n07", "five hundred fifty five", "five", 3),
    ("n08", "four hundred forty four", "four", 3),
    ("n09", "two hundred twenty two million two hundred twenty two thousand two hundred twenty two",
     "two", 9),
    ("n10", "one hundred eleven thousand one hundred eleven", "one", 5),
]

SENTENCES = [
    ("s1", "The bell rang twice.", "bell"),
    ("s2", "He counted the stones.", "stones"),
    ("s3", "Rain fell on the roof.", "rain"),
    ("s4", "The door stayed open.", "door"),
]
SENTENCE_K = [1, 2, 3, 4, 6, 8, 12, 16]

TWISTERS = [
    ("w1", "She sells sea shells by the sea shore.", "sea", 2),
    ("w2", "Peter Piper picked a peck of pickled peppers.", "peck", 1),
    ("w3", "How much wood would a woodchuck chuck.", "chuck", 1),
    ("w4", "Red lorry yellow lorry red lorry yellow lorry.", "lorry", 4),
    ("w5", "The sixth sick sheikh's sixth sheep is sick.", "sixth", 2),
    ("w6", "Six slick slim sycamore saplings.", "s", 0),
    ("w7", "Fresh French fried fly fritters.", "fr", 0),
    ("w8", "Truly rural truly rural truly rural.", "rural", 3),
]


def word_rep_items() -> list[dict]:
    items = []
    for tid, prefix, target, suffix, fillers in WORD_TEMPLATES:
        for k in K_LADDER:
            rep = " ".join([target] * k)
            text = f"{prefix} {rep} {suffix}"
            items.append(dict(
                item_id=f"wr_{target}_{tid}_k{k:02d}",
                family="word_rep", template=tid, text=text,
                target_unit=target, k=k, expected_count=k,
                expected_words=len(text.split()), control_of=None,
            ))
            # Matched control: same length, same syntax, distinct adverbs. Only
            # meaningful for k>=2 (k=1 IS its own control).
            if k >= 2:
                distinct = [fillers[i % len(fillers)] for i in range(k)]
                # guarantee the target itself never appears in the control
                distinct = [d for d in distinct if d != target]
                while len(distinct) < k:
                    distinct.append(fillers[(len(distinct) + 3) % len(fillers)])
                ctext = f"{prefix} {' '.join(distinct[:k])} {suffix}"
                items.append(dict(
                    item_id=f"ct_{target}_{tid}_k{k:02d}",
                    family="control_word", template=tid, text=ctext,
                    target_unit=target, k=k, expected_count=0,
                    expected_words=len(ctext.split()),
                    control_of=f"wr_{target}_{tid}_k{k:02d}",
                ))
    return items


def number_items() -> list[dict]:
    items = []
    for nid, text, target, count in NUMBER_PHRASES:
        items.append(dict(
            item_id=f"nm_{nid}", family="numbers", template=nid,
            text=text.capitalize() + ".", target_unit=target, k=count,
            expected_count=count, expected_words=len(text.split()), control_of=None,
        ))
    return items


def sentence_items() -> list[dict]:
    items = []
    for sid, sent, target in SENTENCES:
        for k in SENTENCE_K:
            text = " ".join([sent] * k)
            items.append(dict(
                item_id=f"sr_{sid}_k{k:02d}", family="sentence_rep", template=sid,
                text=text, target_unit=target, k=k, expected_count=k,
                expected_words=len(text.split()), control_of=None,
            ))
    return items


def twister_items() -> list[dict]:
    items = []
    for wid, text, target, per in TWISTERS:
        for k in [1, 2, 4]:
            full = " ".join([text] * k)
            items.append(dict(
                item_id=f"tw_{wid}_k{k}", family="twisters", template=wid,
                text=full, target_unit=target, k=k, expected_count=per * k,
                expected_words=len(full.split()), control_of=None,
            ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "stimuli.jsonl"))
    args = ap.parse_args()

    items = word_rep_items() + number_items() + sentence_items() + twister_items()

    # Instrumented subset: repetition families with a clean boundary structure.
    # Twisters have no single repeating unit at k=1, numbers are short; both are
    # behavioural-only. Cap by k so activation storage stays bounded.
    for it in items:
        it["instrumented"] = bool(
            it["family"] in ("word_rep", "sentence_rep", "control_word")
            and it["k"] >= 2
        )

    seen = set()
    for it in items:
        assert it["item_id"] not in seen, f"duplicate id {it['item_id']}"
        seen.add(it["item_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    from collections import Counter
    fam = Counter(i["family"] for i in items)
    print(f"wrote {len(items)} items -> {out}")
    for k, v in sorted(fam.items()):
        print(f"  {k:14s} {v:4d}")
    print(f"  instrumented   {sum(i['instrumented'] for i in items):4d}")


if __name__ == "__main__":
    main()
