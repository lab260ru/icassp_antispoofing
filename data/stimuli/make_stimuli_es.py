#!/usr/bin/env python3
"""Build the Spanish repetition ladder — the cross-lingual arm.

Why this exists. Three review rounds have made the same objection: the headline
framing ("the limit tracks periodicity, not length") is stated without
linguistic qualification while every measurement behind it is English. The
Limitations section currently concedes the point. This file is the stimulus half
of removing that concession; `analysis/crosslingual_es.py` is the audit and
reporting half.

Design. A verbatim structural mirror of `data/stimuli/make_stimuli.py`'s
`word_rep_items()`, and nothing else. Six carrier templates, each with one
target word repeated k in {1,2,3,4,6,8,12,16,24,32}; for every k>=2 a
length-matched control with the same carrier and the same number of words in the
same slot, the repeated token replaced by fillers drawn from an eight-word pool.
The other English families (numbers, sentence_rep, twisters) are deliberately
NOT ported: the paper's exact-rate gap is defined on word_rep vs control_word,
and porting families that do not enter that number would spend generation budget
without touching the objection. Mirroring includes the parts that are not ideal
--- the filler pool cycles at k>8, so a k=32 control contains four copies of
each of eight fillers rather than 32 genuinely distinct words --- because a
control built to a *different* rule than the English one would make the two arms
incomparable, which is the only thing this arm is for.

Two Spanish-specific constraints on the vocabulary, both about the judge rather
than the model:

  * **Every scored word is pure ASCII.** `src/common/score_counts.normalise`
    keeps only `[a-z0-9]`, so `mas` survives and `mas`-with-an-accent would be
    shredded into two tokens and never match. Carriers may carry accents (they
    are never scored, and unaccented Spanish would degrade the TTS front end);
    targets and fillers may not. Nothing here needs an accent to be correctly
    spelled Spanish.
  * **Target and fillers are short.** XTTS-v2 caps Spanish input at 239
    characters (`VoiceBpeTokenizer.char_limits`, vs 250 for English) and caps
    generation at ~602 mel tokens regardless of language. Long targets would hit
    the *text* limit at high k on top of the mel limit and confound two different
    censoring mechanisms. Targets here are 2--5 characters, so at k=32 the
    repeated block is at most 192 characters and every repeated item stays under
    the text cap; the generator asserts this rather than trusting it.

Template ids are `et1`..`et6`, not `t1`..`t6`. This is load-bearing, not
cosmetic. `population.panel()` drops rows whose template appears in
`data/results/judge_vocab_audit.json`, and the English audit lists `t2` there
(the judge never emits `okay` or `hmm`). Had the Spanish templates reused the
English names, a routine `panel()` call would have silently deleted every
Spanish `t2` row for a reason established about English orthography. The Spanish
audit writes its own exclusion file and `panel(audit=...)` is pointed at it.

Output: data/stimuli/stimuli_es.jsonl, one item per line, same schema as
stimuli.jsonl so `score_counts.py` and `population.panel()` read it unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Identical to the English ladder. Any change here would make the two arms
# incomparable at exactly the k values the collapse point lives in.
K_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# XTTS-v2's Spanish input cap (TTS/tts/layers/xtts/tokenizer.py:608). Checked
# here so a stimulus that would be silently truncated by the front end is a
# build-time failure rather than a mystery in the results.
ES_CHAR_LIMIT = 239

# (template_id, prefix, target, suffix, filler_pool)
#
# Slot choice mirrors English: an intensifier/manner/attribute position where a
# run of identical tokens is degenerate but parsable, and where eight distinct
# fillers are mutually substitutable so the control differs from the repeated
# item in periodicity alone. Targets avoid homophones with the carrier and never
# occur inside their own carrier, so an occurrence count of the target is a
# count of the repeated block and not of the sentence around it.
#
# Filler pools are all short (<=6 characters). That is not cosmetic. A pool of
# long fillers pushes the k=32 *control* past the 239-character input cap while
# the repeated item, built from a 2--5 character target, stays comfortably under
# it --- a censoring mechanism that applies to one family only, which is exactly
# the asymmetry that would make the repeated-vs-control gap uninterpretable. The
# first draft of this file had seven over-cap items, all of them controls; the
# pools were shortened until the count was zero and the generator now asserts it.
WORD_TEMPLATES = [
    ("et1", "El perro era", "muy", "grande y corría por el campo.",
     ["tan", "poco", "algo", "harto", "casi", "medio", "nada", "bien"]),
    ("et2", "Ella dijo", "no", "a la oferta y luego salió.",
     ["claro", "bueno", "vale", "listo", "justo", "fijo", "cierto", "exacto"]),
    ("et3", "El caminó", "lejos", "en el bosque antes de volver.",
     ["hondo", "cerca", "alto", "recto", "abajo", "solo", "raso", "lento"]),
    ("et4", "La luz era", "azul", "antes de la tormenta.",
     ["verde", "blanca", "clara", "tenue", "suave", "gris", "rosa", "opaca"]),
    ("et5", "El corredor iba", "veloz", "por el camino estrecho.",
     ["firme", "suave", "recto", "lento", "raudo", "tieso", "fino", "llano"]),
    ("et6", "Ellos estaban", "tan", "cansados tras el largo viaje.",
     ["muy", "poco", "algo", "medio", "harto", "casi", "nada", "bien"]),
]


def _ascii_ok(w: str) -> bool:
    return all("a" <= c <= "z" for c in w)


def word_rep_items() -> list[dict]:
    items = []
    for tid, prefix, target, suffix, fillers in WORD_TEMPLATES:
        assert _ascii_ok(target), f"scored target {target!r} is not ASCII"
        for f in fillers:
            assert _ascii_ok(f), f"scored filler {f!r} is not ASCII"
        # The target must not appear in its own carrier, or an occurrence count
        # would start at one before the model has repeated anything.
        carrier_toks = [t.strip(".,") for t in (prefix + " " + suffix).lower().split()]
        assert target not in carrier_toks, \
            f"target {target!r} occurs in its own carrier ({tid})"
        # Same requirement for the control side: a filler that also occurs in
        # the carrier would be credited by `count_units` from the carrier alone,
        # inflating the control's score for a word the model never repeated.
        for f in fillers:
            assert f not in carrier_toks, \
                f"filler {f!r} occurs in its own carrier ({tid})"

        for k in K_LADDER:
            rep = " ".join([target] * k)
            text = f"{prefix} {rep} {suffix}"
            items.append(dict(
                item_id=f"wr_{target}_{tid}_k{k:02d}",
                family="word_rep", template=tid, text=text,
                target_unit=target, k=k, expected_count=k,
                expected_words=len(text.split()), control_of=None,
                boundary_units=[target] * k,
            ))
            if k >= 2:
                distinct = [fillers[i % len(fillers)] for i in range(k)]
                distinct = [d for d in distinct if d != target]
                while len(distinct) < k:
                    distinct.append(fillers[(len(distinct) + 3) % len(fillers)])
                used = distinct[:k]
                ctext = f"{prefix} {' '.join(used)} {suffix}"
                items.append(dict(
                    item_id=f"ct_{target}_{tid}_k{k:02d}",
                    family="control_word", template=tid, text=ctext,
                    target_unit=target, k=k, expected_count=0,
                    expected_words=len(ctext.split()),
                    control_of=f"wr_{target}_{tid}_k{k:02d}",
                    boundary_units=list(used),
                    control_units=list(used),
                ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "stimuli_es.jsonl"))
    args = ap.parse_args()

    items = word_rep_items()

    # `instrumented` mirrors stimuli.jsonl's flag. It is False throughout: this
    # arm is behavioural only. No activation is stored, no probe is fit on it,
    # and nothing in the state-space analysis should ever see a Spanish item.
    for it in items:
        it["instrumented"] = False

    seen = set()
    for it in items:
        assert it["item_id"] not in seen, f"duplicate id {it['item_id']}"
        seen.add(it["item_id"])

    over = [it for it in items if len(it["text"]) > ES_CHAR_LIMIT]
    assert not over, (
        "these items exceed XTTS-v2's Spanish 239-char input cap, which would "
        "censor one family and not the other: "
        + ", ".join(f"{it['item_id']}({len(it['text'])})" for it in over[:8])
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    fam = Counter(i["family"] for i in items)
    print(f"wrote {len(items)} items -> {out}")
    for k, v in sorted(fam.items()):
        print(f"  {k:14s} {v:4d}")
    print(f"  max text chars: {max(len(i['text']) for i in items)} "
          f"(XTTS-v2 es cap {ES_CHAR_LIMIT})")
    # Controls are allowed over the text cap only if it never happens; report
    # rather than assert so the number is visible even when it is zero.
    print(f"  items over the es char cap: {len(over)}"
          + (" -> " + ", ".join(f"{i['item_id']}({len(i['text'])})" for i in over)
             if over else ""))


if __name__ == "__main__":
    main()
