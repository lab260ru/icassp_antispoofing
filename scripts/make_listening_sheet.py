#!/usr/bin/env python3
"""A blinded, listen-and-type sampling sheet for the human counting audit.

Five review rounds have made the same point in different words: the CTC judge
(`src/common/score_counts.py`) is validated against *concatenative* audio with a
known ground-truth count (`analysis/ctc_validation.py`), never against a human
listening to *real generated* audio. The paper ships a 165-clip stratified
sample (`data/audio_sample/`, built by `scripts/make_audio_sample.py`) precisely
so a reader could check the judge by ear -- but nobody has done it, and a
reviewer said the honest thing to do about that is say so, not gesture at the
sample's existence. This script builds the instrument that makes running the
check take about an hour instead of a research project.

WHY THESE DESIGN CHOICES, EACH FORCED BY SOMETHING THAT WOULD OTHERWISE GO WRONG.

**Sampled from the released 165, not the private 3.4 GB corpus.** A reader who
wants to reproduce the audit needs to be able to fetch the same audio; the
released sample is the only audio anyone outside this machine can reach.

**Stratified by arm (repeated vs. control) and by k band, and the two arms are
size-balanced.** The question this audit exists to answer is whether the
judge's error is *arm-asymmetric* -- whether it specifically undercounts
repeated material relative to a human, while agreeing with a human on controls
(`analysis/independent_judge.py` runs exactly this comparison against a second
recogniser; this is the same comparison against a human). An 80/20 split
between controls and repeats cannot see an arm asymmetry no matter how many
clips it contains, so the two arms get equal allocation, each split as evenly
as possible across the three k bands (`low` k<=4, `mid` 4<k<=12, `high` k>12,
the same bins `make_audio_sample.py` uses).

**Degenerate and empty clips are excluded by default (`--include-degenerate` to
override).** Those outcomes are audio-level failures (near-silence, noise) that
`classify()` in `score_counts.py` catches from RMS and spectral flatness before
the transcript is even consulted -- there is no coherent target word for a human
to count, so including them would not test the thing under audit (transcript-based
counting) and would inflate the `unclear` rate for reasons that have nothing to
do with the judge's counting logic. This removes 28 of the 165 released clips;
see PROTOCOL.md and the script's printed strata sizes for exactly which are left.

**The sheet a listener opens carries none of: model name, arm, CTC count, or
transcript.** A listener who can see the machine's answer is not an independent
measurement of it -- that is not a stylistic nicety, it is the whole point of
running this audit instead of re-reading the manifest. The blinded key that
carries that information lives in a *separate* file
(`data/listening/listening_key.csv`) that only the scorer opens.

**The target word/list shown is what the judge actually scores, not a proxy for
it.** For a repeated item (`word_rep`) the CTC's count is occurrences of one
word, so the sheet shows that one word. For its length-matched control
(`control_word`) the model was asked to say k *distinct* filler words instead of
repeating one, and the CTC's count for that item is how many of those k fillers
came out (`count_units` in `score_counts.py`) -- a different quantity, not
"occurrences of the missing word". So the control rows show the filler list (in
the order and with whatever repeats the stimulus itself has), and the listener
is asked the matched question: how many of these were said. Showing the filler
words is not showing the answer -- they are the stimulus, already public in
`data/stimuli/stimuli.jsonl` -- it is showing what to listen for, which a
repeated-item row does too by naming the target word.

**Row order is randomised, independent of arm/k/model.** Otherwise thirty
listens of a mounting k, or thirty repeats-then-thirty-controls, would let a
listener's expectation drift with the block instead of judging each clip cold.

Usage:
  python scripts/make_listening_sheet.py                  # default n=60
  python scripts/make_listening_sheet.py --n 40 --seed 1
  python scripts/make_listening_sheet.py --include-degenerate
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

K_BINS = [0, 4, 12, 64]
K_LABELS = ["low", "mid", "high"]
ARM_NAME = {"word_rep": "repeated", "control_word": "control"}
SECONDS_PER_CLIP = 15  # audio only; the ~1 min/clip estimate below adds thinking
DEGENERATE = {"empty", "degenerate"}


def load_stimuli(path: Path) -> dict:
    import json
    stim = {}
    for line in path.open():
        it = json.loads(line)
        stim[it["item_id"]] = it
    return stim


def target_word_field(row: pd.Series, stim: dict) -> str:
    """What the listener is told to listen for.

    `word_rep`: the single word the item repeats -- what the CTC judge counts
    occurrences of. `control_word`: the ordered list of distinct filler words
    the model was asked to say instead -- what the CTC judge counts delivery
    of. Both are drawn from the stimulus text itself, not from any scoring
    output, so showing them is not showing the judge's answer.
    """
    it = stim.get(row.item_id)
    if it is None:
        raise SystemExit(f"item_id {row.item_id!r} (clip {row.clip}) not found in "
                          f"data/stimuli/stimuli.jsonl -- the released sample and "
                          f"the stimuli file have drifted apart; regenerate one "
                          f"of them before building the sheet.")
    if row.family == "word_rep":
        return str(it["target_unit"])
    units = it.get("control_units") or it.get("boundary_units") or []
    return " / ".join(str(u) for u in units)


def allocate(pool: pd.DataFrame, n: int) -> tuple[dict, list[str]]:
    """How many clips to draw from each (arm, k_band) cell.

    Arms split as evenly as the total allows (n//2 and the remainder), and each
    arm's share splits as evenly as possible across the three k bands. If a
    cell does not have enough clips to fill its share, the shortfall is handed
    to the other bands *within the same arm* first, so the arm balance --
    the thing this audit exists to protect -- is the last thing to give.
    Returns the per-cell target counts and a list of notes about any
    departures from the ideal, so a shortfall is printed, not silently eaten.
    """
    notes: list[str] = []
    avail = {(arm, kb): int(((pool.arm == arm) & (pool.k_band == kb)).sum())
             for arm in ARM_NAME.values() for kb in K_LABELS}

    arms = list(ARM_NAME.values())
    base = n // len(arms)
    arm_targets = {a: base for a in arms}
    for a in arms[: n - base * len(arms)]:
        arm_targets[a] += 1

    alloc: dict[tuple[str, str], int] = {}
    for arm in arms:
        want = arm_targets[arm]
        kb_base = want // len(K_LABELS)
        kb_targets = {kb: kb_base for kb in K_LABELS}
        for kb in K_LABELS[: want - kb_base * len(K_LABELS)]:
            kb_targets[kb] += 1
        # Cap by availability, track the shortfall.
        short = 0
        for kb in K_LABELS:
            got = min(kb_targets[kb], avail[(arm, kb)])
            if got < kb_targets[kb]:
                short += kb_targets[kb] - got
                notes.append(f"{arm}/{kb}: wanted {kb_targets[kb]}, only "
                             f"{avail[(arm, kb)]} available after excluding "
                             f"degenerate/empty clips")
            alloc[(arm, kb)] = got
        # Redistribute the shortfall to whichever bands in this arm still have
        # spare clips, largest spare first.
        while short > 0:
            spare = {kb: avail[(arm, kb)] - alloc[(arm, kb)] for kb in K_LABELS}
            kb = max(spare, key=spare.get)
            if spare[kb] <= 0:
                notes.append(f"{arm}: {short} clip(s) short of its target and "
                             f"no spare capacity left in any k band")
                break
            alloc[(arm, kb)] += 1
            short -= 1
    return alloc, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/audio_sample/manifest.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--sheet-out", default="data/listening/listening_sheet.csv")
    ap.add_argument("--key-out", default="data/listening/listening_key.csv")
    ap.add_argument("--n", type=int, default=60,
                    help="clips to sample (default 60: ~15 min of audio at "
                         "15s/clip plus thinking time, see printed estimate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-degenerate", action="store_true",
                    help="also draw from empty/degenerate outcome clips "
                         "(there is no coherent target word to count in most "
                         "of these; off by default)")
    args = ap.parse_args()

    d = pd.read_csv(REPO / args.manifest)
    if not args.include_degenerate:
        n0 = len(d)
        d = d[~d.outcome.isin(DEGENERATE)].copy()
        print(f"excluded {n0 - len(d)} empty/degenerate clips "
              f"({len(d)} of {n0} remain eligible); pass --include-degenerate "
              f"to keep them")
    if d.family.isin(ARM_NAME).sum() < len(d):
        drop = sorted(set(d.family) - set(ARM_NAME))
        raise SystemExit(f"manifest has families outside {{word_rep, "
                          f"control_word}}: {drop}. This script assumes the "
                          f"released sample is the repeated/control pair; "
                          f"update ARM_NAME before sampling from a different "
                          f"family set.")
    d = d.assign(arm=d.family.map(ARM_NAME),
                 k_band=pd.cut(d.k, K_BINS, labels=K_LABELS))
    if d.k_band.isna().any():
        bad = d[d.k_band.isna()][["clip", "k"]]
        raise SystemExit(f"{len(bad)} clip(s) have k outside the {K_BINS} "
                          f"banding, e.g.\n{bad.head()}")

    if args.n > len(d):
        raise SystemExit(f"--n {args.n} exceeds the {len(d)} eligible clips; "
                          f"pass --include-degenerate or a smaller --n")

    alloc, notes = allocate(d, args.n)
    for note in notes:
        print(f"[allocation] {note}")

    rng = np.random.default_rng(args.seed)
    picked_parts = []
    for (arm, kb), k in alloc.items():
        if k <= 0:
            continue
        cell = d[(d.arm == arm) & (d.k_band == kb)]
        idx = rng.choice(cell.index.to_numpy(), size=k, replace=False)
        picked_parts.append(d.loc[idx])
    picked = pd.concat(picked_parts) if picked_parts else d.iloc[0:0]

    n_got = len(picked)
    if n_got != args.n:
        print(f"[allocation] built {n_got} clips against a target of {args.n} "
              f"-- see notes above; this is the actual eligible pool talking, "
              f"not a bug")

    stim = load_stimuli(REPO / args.stimuli)
    picked = picked.assign(
        target_word=[target_word_field(r, stim) for _, r in picked.iterrows()])

    # Randomise row order so arm/k/model do not correlate with listening
    # position, then assign blinded row ids.
    order = rng.permutation(len(picked))
    picked = picked.iloc[order].reset_index(drop=True)
    width = max(3, len(str(len(picked))))
    picked = picked.assign(row_id=[f"L{i+1:0{width}d}" for i in range(len(picked))])

    sheet_cols = ["row_id", "clip", "target_word", "human_count", "unclear"]
    sheet = picked[["row_id", "clip", "target_word"]].copy()
    sheet["human_count"] = ""
    sheet["unclear"] = ""

    key_cols = ["row_id", "clip", "model", "arm", "family", "k", "k_band",
                "item_id", "seed", "outcome", "target_word", "ctc_count",
                "requested"]
    key = picked.rename(columns={"counted": "ctc_count"})[
        ["row_id", "clip", "model", "arm", "family", "k", "k_band", "item_id",
         "seed", "outcome", "target_word", "ctc_count", "requested"]]

    sheet_path, key_path = REPO / args.sheet_out, REPO / args.key_out
    sheet_path.parent.mkdir(parents=True, exist_ok=True)

    est_min = len(sheet) * (SECONDS_PER_CLIP + 45) / 60.0  # 15s audio + ~45s to
    # find/replay/type -- see PROTOCOL.md for the basis of that figure
    header = (
        f"# Human listening audit sheet -- {len(sheet)} clips, generated by "
        f"scripts/make_listening_sheet.py --n {args.n} --seed {args.seed}\n"
        f"# Roughly {est_min:.0f} minutes at ~15s of audio per clip plus "
        f"listen/replay/type overhead. See data/listening/PROTOCOL.md.\n"
        f"# Do not open data/listening/listening_key.csv before finishing this "
        f"sheet -- it contains the answer.\n"
    )
    with sheet_path.open("w", newline="") as fh:
        fh.write(header)
        w = csv.DictWriter(fh, fieldnames=sheet_cols)
        w.writeheader()
        for row in sheet.to_dict("records"):
            w.writerow(row)

    with key_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_cols)
        w.writeheader()
        for row in key.to_dict("records"):
            w.writerow(row)

    print(f"\nwrote {len(sheet)} rows -> {sheet_path}")
    print(f"wrote blinded key -> {key_path}")
    print(f"estimated time: ~{est_min:.0f} min "
          f"({len(sheet)} clips x ~{SECONDS_PER_CLIP}s audio + overhead)")

    print(f"\nstrata (arm x k band), n={len(sheet)}:")
    tab = key.groupby(["arm", "k_band"], observed=True).size().unstack(fill_value=0)
    print(tab.to_string())
    by_arm = key.arm.value_counts()
    print(f"\narm balance: {dict(by_arm)}")
    if by_arm.max() - by_arm.min() > 1:
        print(f"[warning] arms differ by more than one clip; the allocator's "
              f"shortfall notes above explain why")

    # Self-check: every row in the sheet has a matching row in the key and
    # nothing else, so a reader trusts the join the scorer will do.
    assert set(sheet.row_id) == set(key.row_id), "sheet/key row_id mismatch"
    assert len(sheet) == len(set(sheet.row_id)), "duplicate row_id in sheet"
    assert not (set(sheet.columns) & {"model", "arm", "ctc_count", "transcript"}), (
        "the blinded sheet must not carry model, arm, ctc_count or transcript")
    print("\nself-check passed: sheet/key rows align 1:1, sheet carries no "
          "model/arm/ctc_count/transcript column")


if __name__ == "__main__":
    main()
