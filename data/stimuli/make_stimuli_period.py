#!/usr/bin/env python3
"""Period ladder: vary the period p at fixed length and fixed k.

WHY THIS FILE EXISTS
--------------------
The paper is titled "Periodicity, Not Length" and claims the counting deficit
tracks *periodicity*. Four reviewers in the latest round made the same
objection, and it is correct: the design never varies periodicity on its own.
The repeated arm is one word repeated k times -- period 1 *and* verbatim token
identity. Every control is either aperiodic (`stimuli_ext.jsonl`,
`stimuli_aperiodic.jsonl`, 146-word pool, never cycled) or period-8 (the main
ladder's controls above k=8, which cycle a pool of eight). Nothing between
period 1 and period 8 has ever been rendered, so "periodicity" and "the same
token, again" are two names for the same manipulation in this corpus.

This file builds the missing rungs. At a fixed carrier sentence, a fixed word
count and a fixed k, the only thing that changes is the period p of the
repeating block:

    p = 1     A A A A A A A A A A A A          (the published repeated arm)
    p = 2     A B A B A B A B A B A B
    p = 3     A B C A B C A B C A B C          (added later; see ODD RUNGS below)
    p = 4     A B C D A B C D A B C D
    p = 6     A B C D E F A B C D E F          (added later; see ODD RUNGS below)
    p = 8     A B C D E F G H A B C D          (the published cycled control)
    p = k     twelve distinct words            (the published never-cycled control)

Every arm is generated here, including the three that already exist, so the
whole ladder comes out of one generator, one generation run, one set of seeds
and one scoring pass. Re-rendering the published arms costs a ninth of the
compute and removes the possibility that a difference between rungs is a
difference between two runs made months apart.

WHAT WOULD SUPPORT WHICH READING  (pre-committed; no results existed when this
was written, and no generation had been launched)
---------------------------------------------------------------------------
Let E(p) be the exact rate -- the fraction of generations whose transcript
carries all k requested units, the paper's own statistic -- at fixed k and
carrier. Take the period-8 control as the ceiling anchor, since it is the arm
whose published value (~0.94 exact) says "no deficit", and define the deficit

    D(p) = E(8) - E(p).

D(1) is the published effect. The question is entirely about D(2) and D(4).

* **PERIODICITY.** D(2) >= 0.5 * D(1), averaged over checkpoints. A two-word
  alternation carries no verbatim adjacent repetition, so if it still loses
  most of what the period-1 arm loses, the failure is driven by the periodic
  structure and not by any single token recurring. The title stands, and the
  reviewer's requested control has been supplied and passed.

* **VERBATIM TOKEN IDENTITY.** D(2) <= 0.25 * D(1), i.e. the deficit is gone as
  soon as two distinct tokens alternate, and E(2) ~ E(4) ~ E(8). Then the
  manipulation the paper has been calling "periodicity" is verbatim repetition
  of one token, the title is wrong, and the contribution must be restated as
  isolating verbatim-token-repetition from length.

* **AMBIGUOUS / GRADED.** 0.25 * D(1) < D(2) < 0.5 * D(1): a real but minority
  periodicity residue on top of a verbatim-dominated effect. This still fails
  the reviewer's stated bar ("the deficit still appears at the same period
  length"), so it still forces a retitle; the honest phrase would be
  "verbatim repetition, with a weaker penalty for short periods".

Secondary, pre-committed: D(4) is expected to be no larger than D(2) under
either reading, so a non-monotone curve (D(4) > D(2)) means something other
than period is moving and the ladder should be read as uninterpretable until
that is explained.

WHAT WOULD MAKE THE ANSWER UNINTERPRETABLE
------------------------------------------
1. **Vocabulary.** A period-p arm needs p distinct fillers, so a naive design
   would let the p=2 arm use two words and the p=8 arm use eight *different*
   words, and the CTC judge does not transcribe all eight equally (delivery
   rates inside one pool of eight run 0.86 to 1.00). A p=2 arm built from the
   two worst-delivered words would show a deficit made of orthography.
   Controlled here by *rotation*: for p in {2, 4} every rotation of the pool is
   generated (4 rotations at p=2, 2 at p=4), so pooled over rotations each arm
   uses the pool of eight exactly once per word per (template, k, seed). The
   arms at p = 2, 4, 8 are therefore vocabulary-identical in aggregate, by
   construction, and any residual difference between them is not lexical.
   NOT controlled: p=1 uses the template's target word and nothing else (that
   is what the published repeated arm is), and p=k uses the 146-word extension
   pool, whose tail transcribes at 0.79-0.85. Both are inherited confounds and
   both must be reported as such. The clean, vocabulary-matched comparison is
   p in {2, 4, 8}; the p=1 and p=k rungs are attached to it with a caveat.

2. **Overcount blindness.** `score_counts.py` scores an all-identical unit list
   with an unbounded occurrence count and a mixed unit list with a monotone
   scan that cannot exceed k. So a looping model is visible as an overcount at
   p=1 and invisible at p>1, which would manufacture exactly the collapse the
   verbatim reading predicts. This is not a hypothetical: it is the same
   asymmetry the published repeated-vs-control contrast already carries. The
   analysis must recount every arm with the *same* unbounded rule from the
   stored transcript and report both numbers; if the collapse at p=2 survives
   only under the capped rule, the ladder says nothing.

3. **Length and duration.** Word count is identical across all rungs at a given
   (template, k) by construction -- same prefix, same suffix, k filler slots.
   Syllable count is not: "quite" and "extremely" are not the same length in
   the mouth. Rotation averages this over the pool for p in {2, 4, 8} but not
   for p=1. Report the per-arm mean word and duration.

4. **Censoring.** Cap hits are censored downward and must be excluded via
   `population.panel()`. If the arms differ in cap-hit rate the comparison is
   between differently censored populations; report the rate per arm.

ODD RUNGS: p = 3 AND p = 6 AT k = 24  (second pre-commitment, written and
committed before any p=3 or p=6 audio existed; nothing below was adjusted after
the fact)
---------------------------------------------------------------------------
A reviewer objected, correctly, that the ladder above tests **only powers of
two** -- p in {1, 2, 4, 8} -- and contains no odd period at all. Two rivals to
"periodicity" survive that design untouched:

  * **Power-of-two block structure.** A decoder whose bookkeeping is tied to
    block sizes that are powers of two would produce a monotone ladder over
    exactly these four rungs and nothing else.
  * **Pool cycling / loop attractor.** The p in {2,4,8} arms all cycle a pool
    whose size divides the pool of eight the rotation scheme partitions, so
    "how the filler pool cycles" and "the period" are not separated either.

Under either rival the observed ladder would look identical. The only way to
separate them from period is to put rungs *between* the powers of two, so this
file adds, **at k = 24 only** (the sole k in K_LADDER divisible by 3):

    p = 3   odd, not a power of two, strictly between p=2 and p=4
    p = 6   even, not a power of two, strictly between p=4 and p=8

Everything else is held exactly as above: same five carriers, same word count,
same k, same three seeds, same eight-word filler pools, same `pd_` id scheme,
same four checkpoints already in the ladder (llasa1b, qwen06b, qwen17b, xtts2).
llasa3b is deliberately absent from the ladder: it fails its replication gate on
the p=1 rung, and adding it here would import that failure.

Vocabulary balance is preserved by the same rotation argument, generalised: 3
and 6 do not divide 8, so the stride scheme in `rotations()` cannot be used, and
`window_rotations()` below takes p *consecutive* pool words mod 8 instead. The
number of rotations is the smallest that balances the pool exactly -- eight for
p=3, four for p=6 -- so pooled over rotations every pool word is used the same
number of times in each new arm, exactly as for p in {2,4,8}. This is asserted,
not hoped (see `build()`).

The comparison must be made **within k = 24**, not against the ladder pooled
over k in {16,24,32}, because the deficit depends on k as well as p. The k=24
values recomputed from `data/results/behavioural_period.csv` through
`population.panel()` and the paper's own exact-rate statistic (the ordered scan,
cap hits excluded), pooled as the unweighted mean over the four checkpoints, are

    E(1) = 8.5%   E(2) = 51.2%   E(4) = 75.2%   E(8) = 92.9%   E(24) = 85.0%

and these -- not the all-k pooled 10.4 / 47.9 / 73.6 / 93.1 -- are the numbers
the two tests below are stated against.

The two tests are interpolation tests, pre-registered here:

    TEST A:  E(2) < E(3) < E(4)     i.e.  51.2% < E(3) < 75.2%
    TEST B:  E(4) < E(6) < E(8)     i.e.  75.2% < E(6) < 92.9%

both at k=24, pooled as the unweighted mean over the four checkpoints, under
the paper's exact-rate statistic.

* **PERIODICITY CONFIRMED (interpolation holds).** Both tests pass: E(3) falls
  between E(2) and E(4), and E(6) falls between E(4) and E(8). Then the deficit
  is a smooth monotone function of the period with no special status for powers
  of two, the pool-cycling / loop-attractor rival is refuted -- a decoder
  cycling a pool has no reason to place an odd period neatly between its two
  power-of-two neighbours -- and the title stands.

* **PERIODICITY REFUTED / POWER-OF-TWO ARTIFACT.** E(3) or E(6) lands at or
  above E(8) = 92.9% (the non-power-of-two rungs behave like the no-deficit
  ceiling), or below E(2) = 51.2%. Either way the ordering is not monotone in
  the period and something other than the period is driving the ladder. This
  must be reported loudly and first; it forces a retitle.

* **AMBIGUOUS.** Anything else -- E(3) inside its band but E(6) outside it (or
  the reverse), or pooled interpolation that holds while it inverts within a
  checkpoint. The report must state exactly which of the two tests failed and
  on which checkpoints.

Per-checkpoint rates are reported alongside the pooled ones in every case. A
pooled interpolation that holds while two checkpoints individually invert is a
different result and is not to be hidden behind the mean.

Output: data/stimuli/stimuli_period.jsonl
Usage:  python data/stimuli/make_stimuli_period.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The k values where the deficit is large and where cycling actually bites: at
# k <= 8 a pool of eight covers k without repeating, so there is no period
# ladder to build. Every k here is a whole multiple of every period below,
# which is why k=12 is absent even though the published aperiodic arm covers
# it: 8 does not divide 12, so a period-8 rung at k=12 would be one and a half
# cycles and its first four pool words would appear twice as often as its last
# four. That is precisely the lexical imbalance the rotation scheme exists to
# remove, and buying one extra k with it would be a bad trade. k in {16,24,32}
# is inside the published range and inside the regime the deficit is large in.
K_LADDER = [16, 24, 32]

# Periods. Every p divides 8 so that the rotation scheme below covers the pool
# exactly, and every p divides every k in K_LADDER so that each word of a
# p-periodic item appears exactly k/p times -- a partial final cycle would make
# "period p" a half-truth and would put more of one word than another into the
# item. p=1 (the repeated arm) and p="k" (never cycled) are added separately.
PERIODS = [2, 4, 8]

# The odd/non-power-of-two rungs, added to answer the "only powers of two were
# tested" objection. Keyed by k because they exist at k=24 only: 3 divides
# neither 16 nor 32, so a p=3 rung there would be a partial final cycle and
# "period 3" would be a half-truth. The value is the list of periods and the
# number of consecutive-window rotations that balances the eight-word pool for
# each (see `window_rotations`): eight rotations for p=3, four for p=6.
ODD_PERIODS: dict[int, list[int]] = {24: [3, 6]}

# Carriers, targets and filler pools copied verbatim from `make_stimuli.py`, so
# the rendered text of the p=1 and p=8 rungs is character-for-character the
# published `wr_*` and `ct_*` text. t2 is dropped: its fillers `okay` and `hmm`
# are never emitted by the CTC judge, `judge_vocab_audit.py` excludes the
# template panel-wide, and generating audio we are committed to discarding
# would only buy a bigger n in a column we cannot report.
WORD_TEMPLATES = [
    ("t1", "The dog was", "very", "big and it ran across the field.",
     ["quite", "really", "truly", "fairly", "rather", "somewhat", "extremely", "notably"]),
    ("t3", "He walked", "far", "into the forest before turning back.",
     ["deep", "fast", "long", "wide", "high", "low", "near", "past"]),
    ("t4", "The light was", "blue", "before the storm arrived.",
     ["green", "bright", "pale", "dim", "warm", "cold", "sharp", "soft"]),
    ("t5", "The runner moved", "quick", "along the narrow path.",
     ["swift", "smooth", "steady", "light", "sharp", "loose", "tight", "clean"]),
    ("t6", "They were", "really", "tired after the long journey.",
     ["truly", "quite", "very", "rather", "deeply", "clearly", "plainly", "surely"]),
]

# The extension ladder's never-cycled pool, for the p=k rung. Imported rather
# than re-typed: a second copy of a 144-word list is a second copy to drift.
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stimuli_ext import FILLER_POOL as EXT_POOL  # noqa: E402


def rotations(pool: list[str], p: int) -> list[list[str]]:
    """The 8/p disjoint p-word rotations of an 8-word pool.

    Rotation r takes every (8/p)-th word starting at r, so the rotations
    partition the pool: p=2 gives {0,4},{1,5},{2,6},{3,7} and p=4 gives
    {0,2,4,6},{1,3,5,7}. Generating all of them and pooling is what makes the
    p=2, p=4 and p=8 arms use the same eight words the same number of times.
    Taking one rotation and calling it "the p=2 arm" would confound period with
    whichever two words were picked, which is the mistake this whole file
    exists to stop making.
    """
    step = len(pool) // p
    assert step * p == len(pool), f"period {p} does not divide pool of {len(pool)}"
    return [[pool[r + j * step] for j in range(p)] for r in range(step)]


def window_rotations(pool: list[str], p: int) -> list[list[str]]:
    """Balanced p-word rotations of an 8-word pool when p does not divide 8.

    `rotations()` above takes every (8/p)-th word, which needs p | 8 and so
    covers p in {1,2,4,8} and nothing else. The odd rungs need p = 3 and p = 6,
    for which no stride exists, so this takes p *consecutive* words starting at
    r, wrapping mod 8.

    The number of starts is chosen as the smallest s such that starts
    {0, 8/s, 2*8/s, ...} use every pool word the same number of times, i.e. the
    smallest s dividing 8 with s*p divisible by 8: s = 8 for p=3, s = 4 for
    p=6. Pooled over those rotations each of the eight words appears in exactly
    (s*p)/8 rotations, so the p=3 and p=6 arms are vocabulary-matched to each
    other and to p in {2,4,8} in aggregate, which is the whole point of
    rotating at all. Balance is asserted in `build()` rather than trusted.

    Consecutive rather than strided is a real difference from the arms above:
    it changes *which* words co-occur inside one item, not how often each word
    is used across the arm. Since the pooled comparison is over rotations, and
    every word appears equally often in every arm, that difference cannot move
    the arm-level exact rate lexically -- but it is stated here so nobody has
    to infer it from the code.
    """
    n = len(pool)
    starts = next(s for s in (1, 2, 4, 8) if s <= n and (s * p) % n == 0)
    step = n // starts
    return [[pool[(r * step + j) % n] for j in range(p)] for r in range(starts)]


def build(ks: list[int]) -> list[dict]:
    items: list[dict] = []
    for tid, prefix, target, suffix, pool in WORD_TEMPLATES:
        assert target not in pool, f"{tid}: target {target} is in its own filler pool"
        assert len(set(pool)) == len(pool) == 8, f"{tid}: pool must be 8 distinct words"
        ext = [w for w in EXT_POOL if w != target]
        for k in ks:
            # --- p = 1: the published repeated arm, re-rendered here so the
            # whole ladder shares a generation run. `family` and `boundary_units`
            # match `make_stimuli.py` exactly, which is what makes
            # `score_counts.py` take the unbounded-occurrence branch for it, as
            # it does for `wr_*`.
            text = f"{prefix} {' '.join([target] * k)} {suffix}"
            items.append(dict(
                item_id=f"pd_{target}_{tid}_k{k:02d}_p01_r0",
                family="word_rep", template=tid, text=text,
                target_unit=target, k=k, expected_count=k,
                expected_words=len(text.split()), control_of=None,
                boundary_units=[target] * k,
                ladder="period", period=1, rotation=0, pool_id="target",
                n_distinct=1,
            ))

            # --- p in {2, 4, 8}: the vocabulary-matched interior of the ladder.
            for p in PERIODS:
                assert k % p == 0, f"k={k} is not a whole number of period-{p} cycles"
                for r, words in enumerate(rotations(pool, p)):
                    used = [words[i % p] for i in range(k)]
                    ctext = f"{prefix} {' '.join(used)} {suffix}"
                    items.append(dict(
                        item_id=f"pd_{target}_{tid}_k{k:02d}_p{p:02d}_r{r}",
                        family="control_word", template=tid, text=ctext,
                        target_unit=target, k=k, expected_count=0,
                        expected_words=len(ctext.split()),
                        control_of=f"pd_{target}_{tid}_k{k:02d}_p01_r0",
                        boundary_units=list(used), control_units=list(used),
                        ladder="period", period=p, rotation=r, pool_id="main8",
                        n_distinct=p,
                    ))

            # --- p in {3, 6} at k=24: the odd / non-power-of-two rungs. Same
            # family, same fields and the same id scheme as the interior above,
            # so every downstream reader treats them identically; only the
            # rotation scheme differs, because 3 and 6 do not divide 8.
            for p in ODD_PERIODS.get(k, []):
                assert k % p == 0, f"k={k} is not a whole number of period-{p} cycles"
                assert p not in PERIODS and p != 1, f"p={p} is already in the ladder"
                for r, words in enumerate(window_rotations(pool, p)):
                    assert len(set(words)) == p, f"p={p} r={r}: repeated word in rotation"
                    used = [words[i % p] for i in range(k)]
                    ctext = f"{prefix} {' '.join(used)} {suffix}"
                    items.append(dict(
                        item_id=f"pd_{target}_{tid}_k{k:02d}_p{p:02d}_r{r}",
                        family="control_word", template=tid, text=ctext,
                        target_unit=target, k=k, expected_count=0,
                        expected_words=len(ctext.split()),
                        control_of=f"pd_{target}_{tid}_k{k:02d}_p01_r0",
                        boundary_units=list(used), control_units=list(used),
                        ladder="period", period=p, rotation=r, pool_id="main8",
                        n_distinct=p,
                    ))

            # --- p = k: never cycled, from the 146-word extension pool. The
            # published aperiodic rung, re-rendered for the same reason as p=1.
            if k > len(ext):
                raise SystemExit(f"extension pool too small for k={k}")
            used = ext[:k]
            ctext = f"{prefix} {' '.join(used)} {suffix}"
            items.append(dict(
                item_id=f"pd_{target}_{tid}_k{k:02d}_p{k:02d}_r0",
                family="control_word", template=tid, text=ctext,
                target_unit=target, k=k, expected_count=0,
                expected_words=len(ctext.split()),
                control_of=f"pd_{target}_{tid}_k{k:02d}_p01_r0",
                boundary_units=list(used), control_units=list(used),
                ladder="period", period=k, rotation=0, pool_id="ext146",
                n_distinct=k,
            ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "stimuli_period.jsonl"))
    ap.add_argument("--ks", type=int, nargs="+", default=K_LADDER)
    args = ap.parse_args()

    items = build(args.ks)

    # Behavioural only. Storing activations for 180 more items buys nothing the
    # main ladder's instrumented subset does not already have, and the state
    # dumps are the slow half of a Llasa run.
    for it in items:
        it["instrumented"] = False

    seen: set[str] = set()
    for it in items:
        assert it["item_id"] not in seen, f"duplicate id {it['item_id']}"
        seen.add(it["item_id"])

    # The invariants the whole comparison rests on, asserted rather than hoped:
    # at each (template, k) every rung is the same number of words, and the
    # p in {2,4,8} rungs use each pool word the same number of times in total.
    from collections import Counter, defaultdict
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        by_cell[(it["template"], it["k"])].append(it)
    for (tid, k), group in by_cell.items():
        widths = {i["expected_words"] for i in group}
        assert len(widths) == 1, f"{tid} k={k}: rungs differ in length {widths}"
        use: Counter = Counter()
        for i in group:
            if i["pool_id"] == "main8":
                use.update(i["boundary_units"])
        assert len(set(use.values())) == 1, (
            f"{tid} k={k}: rotations do not balance the pool: {use}")
        # Per *arm*, not only per cell. The cell-level check above can be
        # satisfied by two arms whose imbalances cancel, which would leave each
        # individual rung lexically confounded while the cell looked clean --
        # and the arm is the unit every rate in this experiment is computed
        # over. Every main8 rung must use each of its template's eight pool
        # words the same number of times, on its own.
        by_arm: dict[int, Counter] = defaultdict(Counter)
        for i in group:
            if i["pool_id"] == "main8":
                by_arm[i["period"]].update(i["boundary_units"])
        for p, u in by_arm.items():
            assert len(u) == 8 and len(set(u.values())) == 1, (
                f"{tid} k={k} p={p}: arm does not balance the pool: {dict(u)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    per_p = Counter(i["period"] if i["pool_id"] != "ext146" else "k" for i in items)
    print(f"wrote {len(items)} items -> {out}")
    print(f"  k        {args.ks}")
    print(f"  templates {[t[0] for t in WORD_TEMPLATES]}")
    for p, n in sorted(per_p.items(), key=lambda kv: str(kv[0])):
        print(f"  period {str(p):>2s}  {n:4d} items")
    print(f"  longest  {max(i['expected_words'] for i in items)} words")


if __name__ == "__main__":
    main()
