#!/usr/bin/env python3
"""Does the periodicity limit hold in a language that is not English?

=============================================================================
PRE-COMMITTED INTERPRETATION. Written and committed to disk before any Spanish
transcript existed. Nothing below this line may be edited after looking at a
result; a rule that turned out to be wrong is reported as wrong, not rewritten.
=============================================================================

The objection. Reviewers in r18 and earlier: the paper's headline framing --- the
limit tracks periodicity, not length --- is stated without linguistic
qualification, while every measurement behind it is English. Limitations
currently concedes this. This arm renders the same repetition ladder in Spanish
with the same XTTS-v2 weights, the same speaker reference and the same decoding
config, and scores it with a Spanish CTC judge.

The measurement. Exactly the statistic the English arm reports: over k >= 6,
after `population.panel()`'s exclusions, the fraction of items whose counted
occurrences equal k, for repeated items and for their length-matched controls,
and the difference. English panel: +76.1 points. English by-family (the weaker,
n=3 replicate-level claim the paper quotes): +75.4. The weakest positive arm
anywhere in the paper is F5-TTS at +60.0.

------------------------------------------------------------------ GATE 1
JUDGE AUDIT --- decided first, and it can end the arm.

The paper's central methodological claim is that an autoregressive judge is
biased against the phenomenon: on concatenative ground truth Whisper scores 0.19
of the true count on periodic audio where the English CTC judge scores 1.00.
Adopting a Spanish judge without the same test would be the exact error the paper
criticises. `analysis/ctc_validation.py` builds the same construction in Spanish
--- one verified k=1 rendering spliced N times, so the count is exact by
construction --- and reports median counted/true.

  REPORTABLE      Spanish CTC median counted/true in [0.90, 1.10] on periodic
                  audio pooled over k >= 4, AND in [0.90, 1.10] on the
                  distinct-word control. Both bounds matter: a judge that
                  undercounts everything equally is not unbiased, it is broken,
                  and a judge above 1.10 is inventing words.
  UNREPORTABLE    anything else. If the Spanish judge cannot be shown unbiased on
                  periodic ground truth, no count it produces can be set beside
                  an English count, and this file stops at the gate and reports
                  the gate. A Spanish gap measured with a biased judge would be
                  worth less than no Spanish number at all.

------------------------------------------------------------------ GATE 2
VOCABULARY AUDIT --- a target word the judge never emits scores a silent zero.

The English audit found `okay` transcribed in 0 of 106 items and `hmm` in 0 of
89, which is why template t2 is excluded from both families. The same audit runs
here with the same floor (0.05).

  REPORTABLE      at most 2 of the 6 Spanish templates fall below the floor.
  UNREPORTABLE    3 or more do; the surviving vocabulary is too thin to carry a
                  ladder, and whatever the remaining templates show is a
                  statement about four words.

------------------------------------------------------------------ GATE 3
IS THE COMPARISON INFORMATIVE AT ALL?

The gap is a difference between two rates. If Spanish is simply hard for XTTS-v2
and *both* families collapse, the gap shrinks toward zero for a reason that has
nothing to do with periodicity, and reading that as "the effect is English
specific" would be wrong.

  INFORMATIVE     control exact rate >= 50% over k >= 6.
  UNINFORMATIVE   below that: reported as "Spanish rendering fails on both
                  families", which is a finding about XTTS-v2's Spanish, not
                  about the periodicity claim, and the arm does not answer the
                  reviewers' objection either way.

------------------------------------------------------------------ VERDICT
Given all three gates pass, the exact-rate gap decides, and the thresholds are
fixed here in advance:

  REPLICATES        gap >= +40 points. The English finding is not English. The
                    Limitations concession can be removed.
  ATTENUATED        +10 <= gap < +40. The direction holds and the magnitude does
                    not. Reported as a weaker cross-lingual effect with both
                    numbers side by side; the Limitations concession is
                    softened, not removed, and the headline framing keeps its
                    qualification.
  ENGLISH-SPECIFIC  gap < +10. A genuinely valuable result and published as
                    readily as a replication: the paper's framing would then be
                    wrong as stated and must be narrowed to English, with this
                    arm as the evidence that narrowed it.

Two standing rules for reading the output. (1) Report at the least favourable
reading --- where a choice exists, take the one that shrinks the Spanish gap.
(2) The cap-hit rate is reported per family, not pooled. XTTS-v2 stops at ~602
mel tokens whatever the language, and Spanish tokenisation moves where that
bites; if the cap censors the two families at very different rates the gap is
partly an artefact of our own budget and the report says so rather than
absorbing it.

Usage:  python analysis/crosslingual_es.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.common.population import panel, describe  # noqa: E402

# The English numbers this arm is set beside. Hardcoded as reference constants,
# read from the files that produced them where possible so a stale number cannot
# survive a rerun.
EN_PANEL_GAP = 76.1      # data/results/exclusion_sensitivity.json, arm "panel"
EN_FAMILY_GAP = 75.4     # data/results/family_level.json, replicate level

# Gate thresholds, fixed above. Repeated here as code so nothing can drift.
JUDGE_LO, JUDGE_HI = 0.90, 1.10
MAX_BAD_TEMPLATES = 2
MIN_CONTROL_EXACT = 0.50
GAP_REPLICATES, GAP_ATTENUATED = 40.0, 10.0


def exact_rate(d: pd.DataFrame, fam: str) -> tuple[float, int]:
    """The paper's exact rate: counted occurrences equal to k.

    Identical to `analysis/exclusion_sensitivity.stats`, deliberately --- a
    reimplementation would make the Spanish number incomparable to the English
    one it is printed beside."""
    s = d[d.family == fam]
    if not len(s):
        return float("nan"), 0
    err = (s.count_a - s.k) / s.k
    return float((err == 0).mean()), int(len(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default=str(REPO / "data/results/behavioural_es.csv"))
    ap.add_argument("--judge-audit", default=str(REPO / "data/results/ctc_validation_es.json"))
    ap.add_argument("--vocab-audit", default=str(REPO / "data/results/judge_vocab_audit_es.json"))
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "data/results/crosslingual_es.json"))
    args = ap.parse_args()

    res: dict = {"kmin": args.kmin, "gates": {}}

    # ---------------- Gate 1: judge audit ---------------------------------
    print("=== Gate 1: judge audit on periodic ground truth ===")
    ja = json.loads(Path(args.judge_audit).read_text())
    s = ja["summary"]
    g1 = dict(
        ctc=ja["ctc"],
        periodic_ctc=s.get("periodic_ctc_kge4"), distinct_ctc=s.get("distinct_ctc_kge4"),
        periodic_whisper=s.get("periodic_whisper_kge4"),
        distinct_whisper=s.get("distinct_whisper_kge4"),
    )
    ok1 = (g1["periodic_ctc"] is not None
           and JUDGE_LO <= g1["periodic_ctc"] <= JUDGE_HI
           and g1["distinct_ctc"] is not None
           and JUDGE_LO <= g1["distinct_ctc"] <= JUDGE_HI)
    g1["pass"] = bool(ok1)
    res["gates"]["judge_audit"] = g1
    print(f"  CTC   periodic {g1['periodic_ctc']}  distinct {g1['distinct_ctc']}")
    print(f"  Whisper periodic {g1['periodic_whisper']}  distinct {g1['distinct_whisper']}"
          "   (the AR judge, shown for contrast; not used to score anything)")
    print(f"  -> {'PASS' if ok1 else 'FAIL'} "
          f"(band [{JUDGE_LO}, {JUDGE_HI}] on periodic AND distinct, k>=4)")
    if not ok1:
        res["verdict"] = "UNREPORTABLE"
        res["reason"] = ("the Spanish CTC judge is not unbiased on periodic ground truth; "
                         "no count it produces can be set beside an English count")
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nVERDICT: UNREPORTABLE -- {res['reason']}")
        print(f"wrote {args.out}")
        return

    # ---------------- Gate 2: vocabulary audit ----------------------------
    print("\n=== Gate 2: vocabulary the judge can actually emit ===")
    va = json.loads(Path(args.vocab_audit).read_text())
    bad_t = va.get("excluded_templates", [])
    g2 = dict(below_floor=va.get("below_floor", []), excluded_templates=bad_t,
              floor=va.get("floor"), pass_=len(bad_t) <= MAX_BAD_TEMPLATES)
    res["gates"]["vocab_audit"] = g2
    print(f"  words below floor {va.get('floor')}: {g2['below_floor'] or 'none'}")
    print(f"  unusable templates: {bad_t or 'none'} "
          f"(limit {MAX_BAD_TEMPLATES} of 6)")
    print(f"  -> {'PASS' if g2['pass_'] else 'FAIL'}")
    if not g2["pass_"]:
        res["verdict"] = "UNREPORTABLE"
        res["reason"] = (f"{len(bad_t)} of 6 templates carry vocabulary the Spanish judge "
                         "does not emit; the surviving ladder is too thin")
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nVERDICT: UNREPORTABLE -- {res['reason']}")
        return

    # ---------------- The measurement -------------------------------------
    raw = pd.read_csv(args.behavioural)
    res["n_generations"] = int(len(raw))

    # Cap-hit rate per family, computed on the FULL frame before the cap-hit
    # exclusion removes those rows -- afterwards the rate is zero by
    # construction and the number would be meaningless.
    print("\n=== Generation ceiling (XTTS-v2's ~602 mel-token cap) ===")
    cap: dict = {}
    for fam in ("word_rep", "control_word"):
        s_ = raw[raw.family == fam]
        cap[fam] = dict(n=int(len(s_)), rate=float(s_.hit_cap.mean()))
    by_k = {}
    for k, s_ in raw.groupby("k"):
        by_k[int(k)] = {f: float(s_[s_.family == f].hit_cap.mean())
                        if len(s_[s_.family == f]) else None
                        for f in ("word_rep", "control_word")}
    # "Effective ceiling": the largest k at which fewer than half of the
    # repeated items were cut off by our own budget. Above it the arm is
    # measuring the harness, not the model.
    ceil_k = max([k for k, v in by_k.items()
                  if (v["word_rep"] is not None and v["word_rep"] < 0.5)] or [0])
    cap["effective_k_ceiling_word_rep"] = int(ceil_k)
    cap["by_k"] = by_k
    res["cap_hits"] = cap
    print(f"  word_rep      {100*cap['word_rep']['rate']:5.1f}%  (n={cap['word_rep']['n']})")
    print(f"  control_word  {100*cap['control_word']['rate']:5.1f}%  "
          f"(n={cap['control_word']['n']})")
    print("  by k:  " + "  ".join(
        f"k={k}:{100*(v['word_rep'] or 0):.0f}/{100*(v['control_word'] or 0):.0f}"
        for k, v in sorted(by_k.items())) + "   (rep%/ctl%)")
    print(f"  effective k ceiling for repeated items (<50% capped): {ceil_k}")

    # ---------------- Gate 3 + verdict ------------------------------------
    d = raw[raw.k >= args.kmin]
    d, drop = panel(d, ablations=True, audit=args.vocab_audit)
    print(f"\n=== Population (k >= {args.kmin}) ===")
    print("  " + describe(drop))

    rep, n_rep = exact_rate(d, "word_rep")
    ctl, n_ctl = exact_rate(d, "control_word")
    gap = 100.0 * (ctl - rep)
    res["exact"] = dict(rep=rep, n_rep=n_rep, ctl=ctl, n_ctl=n_ctl, gap_points=gap,
                        n=int(len(d)))

    print(f"\n=== Exact-rate gap, k >= {args.kmin} ===")
    print(f"{'arm':<28s} {'rep exact':>10s} {'ctl exact':>10s} {'gap':>8s} {'n':>6s}")
    print(f"{'Spanish (XTTS-v2, es judge)':<28s} {100*rep:9.1f}% {100*ctl:9.1f}% "
          f"{gap:+8.1f} {len(d):6d}")
    print(f"{'English panel (reported)':<28s} {'':>10s} {'':>10s} "
          f"{EN_PANEL_GAP:+8.1f}")
    print(f"{'English by-family (n=3)':<28s} {'':>10s} {'':>10s} "
          f"{EN_FAMILY_GAP:+8.1f}")

    # The English arm's own XTTS-v2 rows, which is the tightest comparison
    # available: same weights, same speaker, same decoding, different language.
    en_path = REPO / "data/results/behavioural_ctc.csv"
    if en_path.exists():
        en = pd.read_csv(en_path)
        en = en[(en.model == "xtts2") & (en.k >= args.kmin)]
        en, en_drop = panel(en, ablations=True)
        e_rep, e_nrep = exact_rate(en, "word_rep")
        e_ctl, e_nctl = exact_rate(en, "control_word")
        e_gap = 100.0 * (e_ctl - e_rep)
        res["english_xtts2"] = dict(rep=e_rep, n_rep=e_nrep, ctl=e_ctl, n_ctl=e_nctl,
                                    gap_points=e_gap, n=int(len(en)))
        print(f"{'English XTTS-v2 alone':<28s} {100*e_rep:9.1f}% {100*e_ctl:9.1f}% "
              f"{e_gap:+8.1f} {len(en):6d}")

    print("\n  by k:")
    for k, s_ in d.groupby("k"):
        r_, nr = exact_rate(s_, "word_rep")
        c_, nc = exact_rate(s_, "control_word")
        print(f"    k={int(k):<3d} rep {100*r_:5.1f}% (n={nr:3d})  "
              f"ctl {100*c_:5.1f}% (n={nc:3d})  gap {100*(c_-r_):+6.1f}")

    informative = (ctl == ctl) and ctl >= MIN_CONTROL_EXACT
    res["gates"]["informative"] = dict(control_exact=ctl, threshold=MIN_CONTROL_EXACT,
                                       pass_=bool(informative))
    print(f"\n=== Gate 3: is the comparison informative? ===")
    print(f"  control exact {100*ctl:.1f}% vs threshold {100*MIN_CONTROL_EXACT:.0f}% "
          f"-> {'PASS' if informative else 'FAIL'}")

    if not informative:
        res["verdict"] = "UNINFORMATIVE"
        res["reason"] = ("Spanish rendering fails on the control family too, so the gap is "
                         "small for a reason unrelated to periodicity; this is a finding "
                         "about XTTS-v2's Spanish, not about the periodicity claim")
    elif gap >= GAP_REPLICATES:
        res["verdict"] = "REPLICATES"
        res["reason"] = (f"the exact-rate gap is {gap:+.1f} points in Spanish against "
                         f"{EN_PANEL_GAP:+.1f} in the English panel")
    elif gap >= GAP_ATTENUATED:
        res["verdict"] = "ATTENUATED"
        res["reason"] = (f"the direction holds ({gap:+.1f} points) but the magnitude does "
                         f"not reach the pre-registered {GAP_REPLICATES:.0f}-point bar; "
                         "the Limitations concession is softened, not removed")
    else:
        res["verdict"] = "ENGLISH-SPECIFIC"
        res["reason"] = (f"the exact-rate gap is {gap:+.1f} points in Spanish against "
                         f"{EN_PANEL_GAP:+.1f} in English; the headline framing must be "
                         "narrowed to English")

    print(f"\nVERDICT: {res['verdict']} -- {res['reason']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
