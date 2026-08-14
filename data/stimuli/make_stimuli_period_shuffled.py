#!/usr/bin/env python3
"""Shuffled twins: hold the word multiset fixed, destroy only the periodicity.

WHY THIS FILE EXISTS
--------------------
The period ladder (`make_stimuli_period.py`) climbs p = 1, 2, 3, 4, 6, 8 at
fixed carrier, fixed word count and fixed k, and the exact rate climbs with it.
The paper reads that as "periodicity, not length". A reviewer has named the one
confound that ladder does not break, and they are right:

    every rung varies the period AND the lexical diversity together. At p=1 the
    item contains one distinct word, at p=2 two, at p=8 eight. Type-token ratio,
    text entropy and vocabulary size move in lockstep with the period. A decoder
    that fails on low lexical diversity rather than on periodicity per se would
    produce exactly the observed ladder.

This is the same criticism, one level up, that produced the period ladder in the
first place. There is exactly one way to break it: hold the **multiset of
words** fixed and vary only the **order**.

    p=4 periodic:  A B C D A B C D A B C D ...
    p=4 shuffled:  A B C D B A D C C D A B ...   same multiset, no period

Vocabulary, type-token ratio, unigram entropy, word count, character count and k
are identical between a periodic item and its shuffled twin *by construction*
-- the twin is a permutation of the same 24 filler tokens in the same carrier.
Only the order differs. If the deficit survives the shuffle, the ladder measured
lexical diversity. If the shuffle removes the deficit, the ladder measured
periodicity.

Scope: p in {2, 4, 8} at k = 24 -- the rung the odd periods were run at, and the
only k where the whole cycled interior of the ladder already exists at matched
vocabulary. The four checkpoints are the four the ladder contains: llasa1b,
qwen06b, qwen17b, xtts2. `data/results/behavioural_period.csv` has no llasa8b
rows and never had any; adding it here would compare a shuffled arm against a
periodic arm that does not exist for that checkpoint. p=1 has no shuffled twin
-- a multiset of one word has exactly one arrangement -- and p=3, p=6 are left
out because the periodic arm's own vocabulary-balance argument there rests on a
different (consecutive-window) rotation scheme; the clean, strided, fully
vocabulary-matched interior is p in {2, 4, 8}.

WHAT WOULD SUPPORT WHICH READING  (pre-committed; written and committed before
any shuffled audio existed, before the generator had been run, and not adjusted
afterwards)
---------------------------------------------------------------------------
Let E_per(p) and E_shuf(p) be the exact rate at k = 24 -- the paper's own
statistic, the ordered scan through `score_counts.py` with cap hits excluded via
`population.panel()` -- pooled as the unweighted mean over the four checkpoints.
The periodic arm's values, recomputed from `data/results/behavioural_period.csv`
before this file was written and identical to the ones `make_stimuli_period.py`
pre-registered for the odd rungs, are

    E_per(2) = 51.2%    E_per(4) = 75.2%    E_per(8) = 92.9%

CHANGED AT PILOT STAGE, BEFORE ANY AUDIO EXISTED  (recorded here, at the point
of change, rather than backfilled)
..........................................................................
The first version of this pre-commitment took E_per(p) from the *published*
ladder rows above and nothing else. That is unsafe for one checkpoint: the
`coqui` env's transformers was bumped 4.x -> 5.x on 2026-08-12, after the
published XTTS-v2 rows were generated, and the shuffled arm has to be generated
through `src/models/xtts_gen_compat.py` on the new one. For XTTS-v2 the
periodic-vs-shuffled contrast would then confound word order with a major
version bump of the stack that samples the audio.

So this file also emits, for every one of the 35 periodic items it twins, a
**re-rendered periodic control**: byte-identical text, a new `_z` item id, the
same three seeds, generated in the *same run and the same environment* as its
shuffled twins. E_per(p) in every test above is taken from that arm. The
published rows are reported beside it as a replication check, and if the
re-render fails to reproduce them (pooled |difference| > 10 points at any p) the
failure is reported as a finding in its own right and every verdict is given
under both.

This is a change of measurement, so its direction has to be stated: it can move
the verdict either way, because it moves the baseline the shuffled arm is
compared against, and nothing is known yet about which way the re-render will
land. It costs about 25% more generation and it is the only way to make the
XTTS-v2 cell mean anything. The re-render is emitted for all four checkpoints,
not only XTTS-v2, because a control that exists on one checkpoint and not the
others cannot be pooled.

so the deficits the ladder attributes to periodicity, against the p=8 ceiling,
are

    D(2) = 41.7 points    D(4) = 17.7 points.

Define the **recovery fraction**

    R(p) = [ E_shuf(p) - E_per(p) ] / D(p),     D(p) = E_per(8) - E_per(p)

for p in {2, 4}. R = 1 means shuffling the multiset removes the whole deficit;
R = 0 means shuffling changes nothing. Let Rbar = mean(R(2), R(4)).

The primary statistic is `exact` (the ordered scan, the paper's rule). The
**co-primary** is `exact_u`, the unbounded recount of
`analysis/period_ladder.py` -- total occurrences of every distinct requested
unit, uncapped and order-insensitive. It is co-primary and not a footnote
because the ordered scan is order-sensitive and the shuffled arm asks the model
to reproduce an irregular order, so the scan is biased *against* the shuffled
arm; `exact_u` is the statistic that cannot be gamed by that bias in either
direction.

* **PERIODICITY CONFIRMED.** Rbar >= 0.50 under `exact`, with
  min(R(2), R(4)) >= 0.25, and E_shuf(p) > E_per(p) at both p = 2 and p = 4
  under `exact_u` as well. Removing periodicity while holding lexical diversity
  exactly fixed removes at least half the deficit: the confound is broken and
  the title stands.

* **LEXICAL DIVERSITY, NOT PERIODICITY.** Rbar <= 0.20 under `exact` AND under
  `exact_u`. Then shuffling the same words into an aperiodic order leaves the
  deficit essentially where it was, what the ladder measures is type-token
  ratio, and the title is wrong. This must be reported first and loudly; it
  forces a retitle.

* **INTERMEDIATE.** Anything else: 0.20 < Rbar < 0.50; or the two statistics
  disagree about which band Rbar is in; or R(2) and R(4) disagree in sign; or
  Rbar clears 0.50 while min(R(2), R(4)) < 0.25. This licenses only the weaker
  claim that periodicity contributes *part* of the ladder alongside lexical
  diversity, names which rung and which statistic dissented, and still requires
  the title to be softened, because the paper's claim is that periodicity is the
  variable, not that it is one of two.

Per-checkpoint values are reported next to every pooled one. A pooled verdict
that holds while a checkpoint individually inverts is a different result and is
not to be hidden behind the mean. Verdicts are also computed under the
leave-one-checkpoint-out means, and under the paired within-(template, rotation,
seed) differencing, and any disagreement between those and the pooled call is
reported as a weakening.

PRE-COMMITTED PLACEBO
---------------------
p = 8 carries no periodicity deficit to remove (E_per(8) is the ceiling anchor),
so Delta(8) = E_shuf(8) - E_per(8) measures what the *shuffle manipulation
itself* costs, independent of periodicity: the residual order-sensitivity of the
scan, and any acoustic cost of an irregular word order. Pre-committed: if
Delta(8) <= -5 points, the manipulation carries a cost, R(2) and R(4) are
downward-biased, and the offset-corrected

    R_adj(p) = [ Delta(p) - Delta(8) ] / D(p)

is reported beside R(p). If the raw and offset-corrected calls fall in different
bands, the outcome is INTERMEDIATE regardless of which band the raw call is in.

WHAT WOULD MAKE THE ANSWER UNINTERPRETABLE, AND WHAT IS DONE ABOUT IT
---------------------------------------------------------------------
1. **A shuffle that is not actually aperiodic.** A uniform random permutation of
   a low-type multiset still contains runs and near-periodic stretches, and at
   p=2 a random shuffle can land close to alternation. So the shuffles here are
   not random: they are annealed to flatten the autocorrelation of the token
   sequence, and every one is accepted against thresholds fixed below. For each
   item both arms carry their realised statistics into the stimulus file, so a
   reader can verify the shuffled arm is aperiodic rather than trust the label.

2. **Adjacent identical tokens.** A shuffle that puts two copies of the same
   word side by side reintroduces locally exactly the p=1 phenomenon the ladder
   is about, and the periodic arms have an adjacent-repeat rate of exactly zero
   at every p. At p in {4, 8} zero adjacent repeats is achievable and is
   enforced as a hard constraint, so the arms are matched on adjacency. **At
   p = 2 it is not achievable**: a multiset of 12 A and 12 B has exactly one
   arrangement with no adjacent repeat, and that arrangement is the perfect
   alternation -- i.e. the periodic item itself. So at p=2 removing periodicity
   *necessarily* introduces verbatim adjacent repetition. This is minimised
   (subject to the aperiodicity threshold) and reported, and its direction is
   stated here in advance: it pushes the shuffled p=2 arm *down*, toward the
   p=1 arm, so it can only suppress R(2), never manufacture it. A positive R(2)
   is therefore conservative; an R(2) near zero is confounded by adjacency and
   must be read together with R(4), where adjacency is matched exactly.

3. **One lucky permutation.** Every periodic item gets `N_SHUFFLE = 3` distinct
   shuffles, each its own stimulus item with its own id, generated under all
   three model seeds. The analysis reports the spread of the exact rate across
   the three shuffles; if the spread across shuffles is comparable to the
   periodic-vs-shuffled gap, the gap is a property of a permutation and not of
   periodicity, and the result must be reported as such.

4. **Order-sensitivity of the scan.** Handled by the co-primary `exact_u` above,
   and additionally by `exact_ms` (the transcript carries exactly the target
   count of every distinct word), which is reported but is not part of the
   pre-committed test.

5. **Censoring.** Cap hits are censored downward and are excluded via
   `population.panel()`; the cap-hit rate is reported per arm per checkpoint,
   and a column of exact zeros everywhere is a bug in the flag rather than a
   result.

6. **Vocabulary.** No rotation scheme is needed here and none is used: a
   shuffled twin uses the *same words the same number of times* as its periodic
   twin, so the arms are vocabulary-matched item by item, not merely in
   aggregate. The rotations of the periodic arm are inherited (each periodic
   item is twinned separately), so pooled over rotations both arms use each
   template's pool of eight equally, exactly as the periodic ladder does.

THE SHUFFLE CONSTRUCTOR AND ITS THRESHOLDS
------------------------------------------
These thresholds are properties of the *text* and were fixed by a feasibility
probe over sequences alone, before any audio existed and before any outcome
could be seen. They are recorded here as constants; if one had to be repaired,
the repair is recorded at the constant, not backfilled.

For a token sequence x of length n = 24, the match autocorrelation at lag l is

    rho(l) = #{ i : x_i == x_{i+l} } / (n - l)

and its value under a uniform random permutation of a multiset with p types of
n/p copies each is  rho_base(p) = (n/p - 1) / (n - 1)  --  0.478 at p=2, 0.217
at p=4, 0.087 at p=8. Define the **periodicity index**

    PI = max_{l = 2..12} | rho(l) - rho_base |

which is 1 - rho_base for a p-periodic sequence (0.522, 0.783, 0.913 at p = 2,
4, 8) and ~0 for an aperiodic one. Lag 1 is excluded from PI and reported
separately as the adjacent-repeat rate, because adjacency is deliberately
*matched to the periodic arm* (driven to zero) rather than left at its random
value, so its deviation from rho_base is a design choice and not evidence of
periodicity.

Acceptance, checked for every emitted item and asserted rather than hoped:

    PI <= TAU_PI = 0.10                 (vs 0.522 / 0.783 / 0.913 periodic)
    adjacent repeats == 0               for p in {4, 8}
    longest p-periodic window <= 2p     tokens, i.e. never more than two
                                        consecutive cycles anywhere
    pairwise Hamming distance between the 3 shuffles of one item >= 6

Also carried per item, for the reader rather than for a threshold: normalised
LZ76 complexity of the token sequence (low for periodic, high for shuffled),
type-token ratio and unigram entropy (identical between twins by construction --
that is the point), and rho at the twin's own period p.

Output: data/stimuli/stimuli_period_shuffled.jsonl -- a SEPARATE file from
`stimuli_period.jsonl`, deliberately. `analysis/period_ladder.py` and
`analysis/period_odd.py` load the period file and inner-join on it, so keeping
these items out of it means the landed ladder results cannot move. The item ids
still begin with `pd_`, which is what keeps them out of the English panel via
`population.NON_PANEL_ITEM_PREFIXES` and what makes the existing `pd_*.wav` /
`pd_*.npy` globs in the generation and ASR steps pick them up unchanged.

Usage:  python data/stimuli/make_stimuli_period_shuffled.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stimuli_period import WORD_TEMPLATES, rotations  # noqa: E402

K = 24
PERIODS = [2, 4, 8]
N_SHUFFLE = 3
TAU_PI = 0.10          # see docstring; fixed by a text-only feasibility probe
MIN_HAMMING = 6
MAX_LAG = K // 2
SEED = 20260813


# --------------------------------------------------------------------------
# sequence statistics
# --------------------------------------------------------------------------
def rho(seq: list, lag: int) -> float:
    """Match autocorrelation at `lag`: fraction of pairs (i, i+lag) that agree."""
    n = len(seq)
    return sum(1 for i in range(n - lag) if seq[i] == seq[i + lag]) / (n - lag)


def rho_base(seq: list) -> float:
    """Expected match rate under a uniform random permutation of this multiset.

    Written from the realised counts rather than from p, so it is correct for
    any multiset and does not silently assume a balanced one.
    """
    n = len(seq)
    c = Counter(seq)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def periodicity_index(seq: list) -> float:
    b = rho_base(seq)
    return max(abs(rho(seq, l) - b) for l in range(2, MAX_LAG + 1))


def adjacent_repeats(seq: list) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i] == seq[i + 1])


def longest_periodic_window(seq: list, p: int) -> int:
    """Length in tokens of the longest stretch that is exactly p-periodic.

    A run of m consecutive positions with x_i == x_{i+p} spans m + p tokens.
    The periodic arm returns len(seq); an aperiodic one should return a small
    number of tokens, and never more than two cycles under the threshold above.
    """
    n = len(seq)
    if p >= n:
        return p
    best = run = 0
    for i in range(n - p):
        run = run + 1 if seq[i] == seq[i + p] else 0
        best = max(best, run)
    return best + p if best else min(p, n)


def lz76(seq: list) -> int:
    """Lempel-Ziv 1976 complexity: number of distinct phrases in the parse.

    A structural measure that separates the arms while type-token ratio and
    unigram entropy are held identical: a p-periodic sequence parses into ~p+2
    phrases, an aperiodic one into many more.
    """
    s = [str(x) for x in seq]
    n = len(s)
    # Exhaustive-history parse, written out rather than as the terse index
    # dance: walk left to right, and at each position take the shortest phrase
    # that has not already appeared anywhere in the history including its own
    # first character.
    c, parsed = 1, 0
    while parsed < n:
        L = 1
        while parsed + L <= n:
            phrase = s[parsed:parsed + L]
            hist = s[:parsed + L - 1]
            found = any(hist[j:j + L] == phrase for j in range(len(hist) - L + 1))
            if not found:
                break
            L += 1
        parsed += L
        c += 1
    return c - 1


def lz_norm(seq: list) -> float:
    """LZ76 complexity divided by its n / log_b(n) asymptote for b types."""
    n, b = len(seq), max(2, len(set(seq)))
    return lz76(seq) / (n / math.log(n, b))


def unigram_entropy(seq: list) -> float:
    n = len(seq)
    return -sum((v / n) * math.log2(v / n) for v in Counter(seq).values())


def bigram_entropy(seq: list) -> float:
    """Conditional entropy H(x_{i+1} | x_i). Identical unigram statistics, and
    this is where the two arms are *supposed* to differ: a p-periodic sequence
    has H = 0, a shuffled one has H ~ the unigram entropy."""
    pairs = Counter(zip(seq, seq[1:]))
    firsts = Counter(seq[:-1])
    tot = sum(pairs.values())
    h = 0.0
    for (a, _b), v in pairs.items():
        h -= (v / tot) * math.log2(v / firsts[a])
    return h


def seq_stats(seq: list, p: int) -> dict:
    n = len(seq)
    return dict(
        rho_base=round(rho_base(seq), 4),
        rho_at_p=round(rho(seq, p), 4) if p < n else None,
        periodicity_index=round(periodicity_index(seq), 4),
        rho_by_lag={l: round(rho(seq, l), 4) for l in range(1, MAX_LAG + 1)},
        adjacent_repeats=adjacent_repeats(seq),
        adjacent_repeat_rate=round(adjacent_repeats(seq) / (n - 1), 4),
        longest_periodic_window=longest_periodic_window(seq, p),
        lz76=lz76(seq),
        lz_norm=round(lz_norm(seq), 4),
        ttr=round(len(set(seq)) / n, 4),
        unigram_entropy=round(unigram_entropy(seq), 4),
        bigram_entropy=round(bigram_entropy(seq), 4),
    )


# --------------------------------------------------------------------------
# the shuffle search
# --------------------------------------------------------------------------
def anneal(counts: list[int], p: int, rng: random.Random,
           *, hard_adjacency: bool, iters: int = 40000) -> list[int]:
    """One aperiodic arrangement of the multiset given by `counts`.

    Simulated annealing over transpositions. The objective is a hard barrier on
    the periodicity index (PI <= TAU_PI) and on the longest p-periodic window,
    with the adjacent-repeat count as the thing minimised inside the feasible
    set -- so the search buys the flattest achievable autocorrelation first and
    then matches the periodic arm's zero adjacency as closely as the multiset
    allows. At p in {4, 8} that is exactly zero; at p = 2 it is not, for the
    reason in the docstring.
    """
    seq = [w for w, c in enumerate(counts) for _ in range(c)]
    rng.shuffle(seq)
    lim = 2 * p

    def cost(s: list[int]) -> float:
        pen = 500.0 * max(0.0, periodicity_index(s) - TAU_PI)
        pen += 50.0 * max(0, longest_periodic_window(s, p) - lim)
        a = adjacent_repeats(s)
        pen += (100.0 if hard_adjacency else 1.0) * a
        return pen

    cur = cost(seq)
    best, best_seq = cur, list(seq)
    t0, t1 = 2.0, 0.01
    n = len(seq)
    for t in range(iters):
        temp = t0 * (t1 / t0) ** (t / iters)
        i, j = rng.randrange(n), rng.randrange(n)
        if seq[i] == seq[j]:
            continue
        seq[i], seq[j] = seq[j], seq[i]
        c = cost(seq)
        if c <= cur or rng.random() < math.exp(-(c - cur) / temp):
            cur = c
            if c < best:
                best, best_seq = c, list(seq)
        else:
            seq[i], seq[j] = seq[j], seq[i]
    return best_seq


def hamming(a: list, b: list) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def make_shuffles(words: list[str], p: int, rng: random.Random) -> list[list[str]]:
    """`N_SHUFFLE` accepted, mutually distinct aperiodic arrangements."""
    counts = [K // p] * p
    hard = p >= 4
    out: list[list[int]] = []
    for attempt in range(400):
        if len(out) == N_SHUFFLE:
            break
        cand = anneal(counts, p, rng, hard_adjacency=hard)
        if periodicity_index(cand) > TAU_PI + 1e-9:
            continue
        if longest_periodic_window(cand, p) > 2 * p:
            continue
        if hard and adjacent_repeats(cand) != 0:
            continue
        if any(hamming(cand, o) < MIN_HAMMING for o in out):
            continue
        out.append(cand)
    if len(out) != N_SHUFFLE:
        raise SystemExit(f"could not find {N_SHUFFLE} accepted shuffles for p={p}")
    return [[words[i] for i in s] for s in out]


# --------------------------------------------------------------------------
def build() -> list[dict]:
    rng = random.Random(SEED)
    items: list[dict] = []
    for tid, prefix, target, suffix, pool in WORD_TEMPLATES:
        for p in PERIODS:
            for r, words in enumerate(rotations(pool, p)):
                periodic = [words[i % p] for i in range(K)]
                twin_id = f"pd_{target}_{tid}_k{K:02d}_p{p:02d}_r{r}"
                per_stats = seq_stats(periodic, p)
                ptext = f"{prefix} {' '.join(periodic)} {suffix}"
                # The within-run periodic control. Text byte-identical to the
                # published item it re-renders, so any difference between this
                # arm and the published rows is the pipeline drifting, not the
                # stimulus changing.
                items.append(dict(
                    item_id=f"{twin_id}_z",
                    family="control_word", template=tid, text=ptext,
                    target_unit=target, k=K, expected_count=0,
                    expected_words=len(ptext.split()),
                    control_of=f"pd_{target}_{tid}_k{K:02d}_p01_r0",
                    boundary_units=list(periodic), control_units=list(periodic),
                    ladder="period_shuffled", arm="periodic",
                    period=p, source_period=p, rotation=r, shuffle=-1,
                    twin_id=twin_id, pool_id="main8", n_distinct=p,
                    instrumented=False,
                    seq_stats=per_stats, twin_seq_stats=per_stats,
                ))
                for s, seq in enumerate(make_shuffles(words, p, rng)):
                    assert Counter(seq) == Counter(periodic), "multiset not preserved"
                    text = f"{prefix} {' '.join(seq)} {suffix}"
                    st = seq_stats(seq, p)
                    items.append(dict(
                        item_id=f"{twin_id}_x{s}",
                        family="control_word", template=tid, text=text,
                        target_unit=target, k=K, expected_count=0,
                        expected_words=len(text.split()),
                        control_of=f"pd_{target}_{tid}_k{K:02d}_p01_r0",
                        boundary_units=list(seq), control_units=list(seq),
                        ladder="period_shuffled", arm="shuffled",
                        period=p, source_period=p, rotation=r, shuffle=s,
                        twin_id=twin_id, pool_id="main8", n_distinct=p,
                        instrumented=False,
                        seq_stats=st, twin_seq_stats=per_stats,
                    ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "stimuli_period_shuffled.jsonl"))
    args = ap.parse_args()
    items = build()

    seen: set[str] = set()
    for it in items:
        assert it["item_id"] not in seen, f"duplicate id {it['item_id']}"
        seen.add(it["item_id"])
        assert it["item_id"].startswith("pd_"), "id must stay off the English panel"

    # The invariants the whole comparison rests on, asserted rather than hoped.
    for it in items:
        st, tw = it["seq_stats"], it["twin_seq_stats"]
        assert st["ttr"] == tw["ttr"], "type-token ratio moved"
        assert st["unigram_entropy"] == tw["unigram_entropy"], "unigram entropy moved"
        assert it["expected_words"] == len(it["text"].split())
        if it["arm"] == "shuffled":
            assert st["periodicity_index"] <= TAU_PI + 1e-9
            assert st["longest_periodic_window"] <= 2 * it["period"]
            if it["period"] >= 4:
                assert st["adjacent_repeats"] == 0
        else:
            # the re-rendered periodic control: maximally periodic by definition
            assert st["longest_periodic_window"] == K
            assert st["adjacent_repeats"] == 0

    # The re-rendered periodic control must be byte-identical to the published
    # item it re-renders. If it is not, it is not a control.
    pub = {}
    pp = Path(__file__).parent / "stimuli_period.jsonl"
    if pp.exists():
        pub = {json.loads(l)["item_id"]: json.loads(l) for l in pp.open()}
        for it in items:
            if it["arm"] == "periodic":
                src = pub[it["twin_id"]]
                assert it["text"] == src["text"], f"{it['item_id']}: text drifted"
                assert it["boundary_units"] == src["boundary_units"]

    # Character count, which is what XTTS-v2 truncates on, is identical between
    # twins because the multiset is: a permutation cannot change total length.
    by_twin: dict[str, list[dict]] = {}
    for it in items:
        by_twin.setdefault(it["twin_id"], []).append(it)
    for tw, g in by_twin.items():
        assert len({len(i["text"]) for i in g}) == 1, f"{tw}: arms differ in chars"
        assert len(g) == N_SHUFFLE + 1

    out = Path(args.out)
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    n_sh = sum(1 for i in items if i["arm"] == "shuffled")
    print(f"wrote {len(items)} items -> {out}  "
          f"({n_sh} shuffled, {len(items) - n_sh} re-rendered periodic controls)")
    for p in PERIODS:
        g = [i for i in items if i["period"] == p and i["arm"] == "shuffled"]
        pi = [i["seq_stats"]["periodicity_index"] for i in g]
        ar = [i["seq_stats"]["adjacent_repeat_rate"] for i in g]
        lz = [i["seq_stats"]["lz_norm"] for i in g]
        lzt = [i["twin_seq_stats"]["lz_norm"] for i in g]
        rp = [i["seq_stats"]["rho_at_p"] for i in g]
        print(f"  p={p}: {len(g):3d} items  PI {min(pi):.3f}-{max(pi):.3f} "
              f"(periodic {g[0]['twin_seq_stats']['periodicity_index']:.3f})  "
              f"rho(p) {min(rp):.3f}-{max(rp):.3f} (periodic 1.000)  "
              f"adj-rate {min(ar):.3f}-{max(ar):.3f} (periodic 0.000)  "
              f"LZnorm {min(lz):.2f}-{max(lz):.2f} (periodic {lzt[0]:.2f})")


if __name__ == "__main__":
    main()
