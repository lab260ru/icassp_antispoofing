#!/usr/bin/env python3
"""How much audio is behind 0.19, 0.27--0.65 and 1.00, and how far each moves.

The paper's central methodological choice --- score with a CTC recogniser, not
with Whisper --- is justified by one paragraph of Section 2 and one sentence of
the abstract, and both rest on three ratios: Whisper recovers \\WhisperValPer{}
of the true count on periodic ground truth, 0.27--0.65 across $k\\ge4$, while
matched distinct-word audio scores 1.00 and CTC scores \\CtcValPer{}. A reviewer
asked the obvious question and the paper does not answer it: how many trials, and
how wide is the interval?

Neither is in the macro layer. `make_numbers.py` emits four scalars from
`ctc_validation.json`'s `summary` block, and that block stores no $n$ at all.
This file computes both, from the per-trial rows that both audits do persist.

Pre-committed reading, fixed before any interval was computed:

* **The audit is small by construction and that is not automatically a
  problem.** It is a *ground-truth* audit: the audio is concatenated from
  verified-correct $k{=}1$ renderings, so the true count is known exactly and
  each trial is a clean measurement rather than a noisy one. What a small $n$
  costs here is coverage --- of voices, of words, of $k$ --- not precision about
  any one cell.
* **The unit of evidence is the donor clip, not the trial.** Six words, one
  voice, one checkpoint, one seed, each concatenated at five $k$; the five
  trials from one donor are the same speech repeated more times, not five
  independent observations. Every interval is therefore reported twice: over
  trials (what a reader reconstructing the published number would do) and over
  donors (what the design actually supports). Where the two disagree, the ratio
  is fragile and the paper must say so.
* **A ratio whose bootstrap interval is a single point is not precise, it is
  degenerate.** Three of these cells are medians of values that take two or
  three distinct levels, and one is a run of identical successes. For those the
  bootstrap says nothing useful and an exact bound is reported instead: for a
  run of $m$ successes with no failure, the 95% upper bound on the failure rate
  is $1-0.05^{1/m}$ --- the rule of three. That is the honest statement of what
  "scores 1.00" is worth.
* **Verdict rule.** A ratio is called *fragile* if the donor-level 95% interval
  spans more than 0.20, or if it is a point estimate resting on fewer than ten
  independent donors, or if more than a quarter of the trials that should have
  entered it are missing. Fragile ratios must be quoted with their $n$ and
  interval in the main text, not only in the supplement.

Usage:  python analysis/judge_audit_ci.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

N_BOOT = 20000
BOOT_SEED = 0
FRAGILE_WIDTH = 0.20        # donor-level 95% interval wider than this
FRAGILE_DONORS = 10         # fewer independent donors than this
FRAGILE_MISSING = 0.25      # more than this fraction of trials unmeasured


def boot_ci(values: np.ndarray, groups: np.ndarray | None, stat: str,
            n: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap, over rows if `groups` is None else over clusters."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    f = np.median if stat == "median" else np.mean
    if groups is None:
        idx = rng.integers(0, values.size, size=(n, values.size))
        boot = f(values[idx], axis=1)
    else:
        keys = sorted(set(np.asarray(groups).tolist()))
        members = [np.flatnonzero(np.asarray(groups) == k) for k in keys]
        draws = rng.integers(0, len(keys), size=(n, len(keys)))
        boot = np.array([f(values[np.concatenate([members[j] for j in d])])
                         for d in draws])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return dict(point=float(f(values)), ci95=[lo, hi], width=hi - lo,
                degenerate=bool(hi - lo < 1e-12))


def rule_of_three(m: int) -> float:
    """95% upper bound on the failure rate after `m` consecutive successes."""
    return float(1.0 - 0.05 ** (1.0 / m)) if m > 0 else 1.0


def summarise(label: str, values: np.ndarray, donors: np.ndarray, stat: str,
              n_expected: int | None = None) -> dict:
    """One audit cell: n, donors, both intervals, and the fragility verdict."""
    trial = boot_ci(values, None, stat)
    donor = boot_ci(values, donors, stat)
    n_donors = int(len(set(donors.tolist())))
    missing = (0.0 if n_expected is None
               else float(1.0 - values.size / n_expected))
    all_one = bool(values.size and np.all(values >= 1.0 - 1e-12))
    out = dict(label=label, statistic=stat, n_trials=int(values.size),
               n_donors=n_donors, n_expected=n_expected,
               fraction_missing=missing, point=trial["point"],
               ci95_trial=trial["ci95"], ci95_donor=donor["ci95"],
               donor_ci_width=donor["width"],
               bootstrap_degenerate=bool(trial["degenerate"]
                                         and donor["degenerate"]),
               distinct_values=sorted(set(np.round(values, 6).tolist()))[:8])
    if stat == "median":
        # A median over a design this balanced can be constant under every
        # resample --- \WhisperValPer's twelve trials are six 0.25s and six
        # 0.125s, one of each per donor, so the median is 0.1875 for any draw.
        # That is a property of the design, not precision, and the mean is the
        # statistic with something left to say. Reported alongside so the
        # paper has a quotable interval where the median has none.
        alt = boot_ci(values, donors, "mean")
        out["mean_ratio"] = alt["point"]
        out["mean_ci95_donor"] = alt["ci95"]
    out["fraction_exact"] = float(np.mean(values >= 1.0 - 1e-12))
    if all_one:
        # A run of successes has no bootstrap spread and a real upper bound on
        # how often it would fail. Quote the bound, not the empty interval.
        out["all_trials_exact"] = True
        out["failure_rate_upper_95_by_trial"] = rule_of_three(int(values.size))
        out["failure_rate_upper_95_by_donor"] = rule_of_three(n_donors)
    reasons = []
    if donor["width"] > FRAGILE_WIDTH:
        reasons.append(f"donor-level 95% interval spans {donor['width']:.2f}")
    if n_donors < FRAGILE_DONORS:
        reasons.append(f"only {n_donors} independent donors")
    if missing > FRAGILE_MISSING:
        reasons.append(f"{100 * missing:.0f}% of the trials never returned")
    out["fragile"] = bool(reasons)
    out["fragile_because"] = reasons
    return out


# ---------------------------------------------------------------- audit 1


def ctc_validation(out: dict) -> None:
    """`\\WhisperValPer` and `\\CtcValPer`, and their distinct-word twins."""
    src = REPO / "data/results/ctc_validation.json"
    d = json.loads(src.read_text())
    trials, summary = d["trials"], d["summary"]

    res: dict = {"source": str(src.relative_to(REPO)),
                 "macro_source": "analysis/make_numbers.py, from summary.*_kge4",
                 "unit_of_analysis": "one concatenated utterance (atom x k x judge)",
                 "cells": {}}
    print(f"\n[A] {src.name}: the audit behind \\WhisperValPer and \\CtcValPer")
    print(f"{'cell':>22s} {'macro':>14s} {'n':>3s} {'don':>4s} {'ratio':>6s} "
          f"{'95% CI (donor)':>16s}  note")
    for kind in ("periodic", "distinct"):
        for judge in ("ctc", "whisper"):
            sel = [t for t in trials
                   if t["kind"] == kind and t["judge"] == judge
                   and t["true"] >= 4 and t["counted"] >= 0]
            expect = [t for t in trials
                      if t["kind"] == kind and t["judge"] == judge and t["true"] >= 4]
            v = np.array([t["counted"] / t["true"] for t in sel])
            donors = np.array([t["unit"] for t in sel])
            key = f"{kind}_{judge}_kge4"
            # Reproduce the published scalar before reporting anything about it.
            assert abs(float(np.median(v)) - summary[key]) < 1e-12, \
                f"{key} no longer reproduces from trials"
            cell = summarise(key, v, donors, "median", n_expected=len(expect))
            cell["published_value"] = summary[key]
            cell["ks_present"] = sorted({t["true"] for t in sel})
            cell["ks_missing"] = sorted({t["true"] for t in expect}
                                        - {t["true"] for t in sel})
            macro = {"periodic_whisper_kge4": "\\WhisperValPer",
                     "periodic_ctc_kge4": "\\CtcValPer",
                     "distinct_whisper_kge4": "\\WhisperValDis",
                     "distinct_ctc_kge4": "\\CtcValDis"}[key]
            cell["macro"] = macro
            res["cells"][key] = cell
            note = ("; ".join(cell["fragile_because"]) if cell["fragile"]
                    else "stable")
            if cell["bootstrap_degenerate"]:
                note = "median constant under resampling; " + note
            print(f"{key:>22s} {macro:>14s} {cell['n_trials']:3d} "
                  f"{cell['n_donors']:4d} {cell['point']:6.3f} "
                  f"[{cell['ci95_donor'][0]:6.3f},{cell['ci95_donor'][1]:6.3f}]  {note}")
            print(f"{'':>22s} {'(mean)':>14s} {'':3s} {'':4s} "
                  f"{cell['mean_ratio']:6.3f} "
                  f"[{cell['mean_ci95_donor'][0]:6.3f},"
                  f"{cell['mean_ci95_donor'][1]:6.3f}]  "
                  f"{100 * cell['fraction_exact']:.0f}% of trials exact")

    # The two facts a reader of the abstract would want and cannot get from it.
    wp = res["cells"]["periodic_whisper_kge4"]
    res["whisper_periodic_missing_ks"] = wp["ks_missing"]
    res["whisper_periodic_is_median_of_two_levels"] = bool(
        len(wp["distinct_values"]) == 2)
    # Why that cell's interval is a point, which is the opposite of the reason a
    # reviewer would assume. Whisper did not vary: it transcribed the repeated
    # word exactly once in every trial that returned, at both k. The ratio is
    # 1/k by construction, so there is no spread to put an interval around --
    # the failure is uniform, not imprecisely measured. Worth saying in the
    # paper, because "0.19 with no interval" reads as weakness and this is not.
    counted = sorted({t["counted"] for t in trials
                      if t["kind"] == "periodic" and t["judge"] == "whisper"
                      and t["true"] >= 4 and t["counted"] >= 0})
    res["whisper_periodic_counted_values"] = counted
    res["whisper_periodic_uniform_failure"] = bool(counted == [1])
    res["whisper_periodic_note"] = (
        f"all {wp['n_trials']} returning trials transcribed the repeated word "
        f"exactly {counted[0] if len(counted) == 1 else '?'} time(s), so the "
        f"ratio is 1/k by construction and has no sampling spread; the "
        f"{len(wp['ks_missing'])} higher k values "
        f"({', '.join(str(k) for k in wp['ks_missing'])}) returned no transcript "
        f"at all")
    print(f"    {res['whisper_periodic_note']}")
    out["ctc_validation"] = res


# ---------------------------------------------------------------- audit 2


def asr_reliability(out: dict) -> None:
    """The 0.27--0.65 range and the distinct-word 1.00, from the other audit."""
    src = REPO / "data/results/asr_reliability.json"
    d = json.loads(src.read_text())
    rep, ctl = d["synthetic_ground_truth"], d["synthetic_ground_truth_control"]

    res: dict = {"source": str(src.relative_to(REPO)),
                 "judge": d["asr_model"],
                 "note": "quoted in main.tex as literals, not macros",
                 "repeated_by_k": {}, "control_by_k": {}}
    print(f"\n[B] {src.name}: the audit behind the literal 0.27--0.65 and 1.00")
    print(f"{'arm':>10s} {'k':>4s} {'n':>3s} {'don':>4s} {'mean ratio':>10s} "
          f"{'95% CI (donor)':>16s}  note")
    for arm, block, store, gkey in (("repeated", rep, "repeated_by_k", "target_word"),
                                    ("distinct", ctl, "control_by_k", "offset")):
        for k in sorted({r["k_true"] for r in block["rows"]}):
            sub = [r for r in block["rows"] if r["k_true"] == k]
            v = np.array([r["ratio"] for r in sub])
            donors = np.array([str(r[gkey]) for r in sub])
            pub = block["by_k"][str(k)]
            assert abs(float(v.mean()) - pub["mean_ratio"]) < 1e-12, \
                f"{arm} k={k} mean ratio no longer reproduces"
            assert len(sub) == pub["n"], f"{arm} k={k} n moved"
            cell = summarise(f"{arm}_k{k}", v, donors, "mean")
            cell["published_value"] = pub["mean_ratio"]
            res[store][str(k)] = cell
            note = ("; ".join(cell["fragile_because"]) if cell["fragile"]
                    else "stable")
            print(f"{arm:>10s} {k:4d} {cell['n_trials']:3d} {cell['n_donors']:4d} "
                  f"{cell['point']:10.3f} "
                  f"[{cell['ci95_donor'][0]:6.3f},{cell['ci95_donor'][1]:6.3f}]  {note}")

    # The range the paper quotes is a min--max over four point estimates, each
    # with its own interval. State it as such, and state the interval that
    # actually covers the whole range.
    hi_k = [c for k, c in res["repeated_by_k"].items() if int(k) >= 4]
    res["repeated_kge4_range_of_points"] = [min(c["point"] for c in hi_k),
                                            max(c["point"] for c in hi_k)]
    res["repeated_kge4_envelope_of_cis"] = [min(c["ci95_donor"][0] for c in hi_k),
                                            max(c["ci95_donor"][1] for c in hi_k)]
    rows = [r for r in rep["rows"] if r["k_true"] >= 4]
    res["repeated_kge4_pooled"] = summarise(
        "repeated_kge4_pooled",
        np.array([r["ratio"] for r in rows]),
        np.array([r["target_word"] for r in rows]), "mean")
    crows = ctl["rows"]
    res["distinct_kge4_pooled"] = summarise(
        "distinct_kge4_pooled",
        np.array([r["ratio"] for r in crows if r["k_true"] >= 4]),
        np.array([str(r["offset"]) for r in crows if r["k_true"] >= 4]), "mean")
    pts = ", ".join("{:.2f}".format(c["point"]) for c in hi_k)
    print(f"\n    the quoted range 0.27--0.65 is the min and max of four point "
          f"estimates\n    ({pts}), "
          f"each n=6 on 6 donors; the envelope of their\n    own 95% intervals is "
          f"[{res['repeated_kge4_envelope_of_cis'][0]:.2f}, "
          f"{res['repeated_kge4_envelope_of_cis'][1]:.2f}]")
    p = res["repeated_kge4_pooled"]
    print(f"    pooled over k>=4: {p['point']:.2f} "
          f"[{p['ci95_donor'][0]:.2f}, {p['ci95_donor'][1]:.2f}] "
          f"(n={p['n_trials']} trials on {p['n_donors']} donors)")
    q = res["distinct_kge4_pooled"]
    print(f"    distinct-word control at k>=4: {q['point']:.2f}, "
          f"{q['n_trials']} trials on {q['n_donors']} rotations -- no interval "
          f"(all exact);\n    the 95% upper bound on how often it would miscount "
          f"is {100 * q['failure_rate_upper_95_by_trial']:.0f}% by trial and "
          f"{100 * q['failure_rate_upper_95_by_donor']:.0f}% by rotation")
    out["asr_reliability"] = res


# ---------------------------------------------------------------- conflict


def cross_check(out: dict) -> None:
    """Which arm actually carries "a bias specific to periodicity", and on what.

    One paragraph of main.tex chains four numbers as though they came from one
    audit --- "the same ground truth", and the supplement's "the identical
    audio". They come from two, built differently:

      `ctc_validation.py`   0.12 s gaps. Both of its arms concatenate ONE atom
                            $N$ times, so both are periodic at the utterance
                            level; the arm named `distinct` differs only in
                            using a control *sentence* as the atom. It is not a
                            distinct-word control and \\WhisperValDis (0.25)
                            must not be read as one. This audit is the one that
                            compares the two judges on identical material, so
                            \\WhisperValPer vs \\CtcValPer is a fair comparison.
      `asr_reliability.py`  0.25 s gaps, Whisper only. Its control cycles
                            through six different carrier sentences, so it is a
                            genuine distinct-word sequence of matched length.
                            This is the arm --- and the only arm --- that
                            supports "specific to periodicity". CTC was never
                            run on this material at all.

    So the paragraph's logic holds, but every clause is sourced from a different
    experiment than the reader would assume, and the load-bearing distinct-word
    claim rests on the smallest arm in either file.
    """
    cv = out["ctc_validation"]["cells"]
    ar = out["asr_reliability"]
    q = ar["distinct_kge4_pooled"]
    notes = dict(
        distinct_word_claim_supported_by="asr_reliability.py cyclic-rotation "
                                         "control (the only genuine "
                                         "distinct-word arm)",
        distinct_word_n_trials=q["n_trials"],
        distinct_word_n_rotations=q["n_donors"],
        distinct_word_ratio=q["point"],
        distinct_word_failure_rate_upper_95_by_rotation=q[
            "failure_rate_upper_95_by_donor"],
        ctc_validation_distinct_arm_is_not_distinct_words=True,
        ctc_validation_distinct_whisper_kge4=cv["distinct_whisper_kge4"]["point"],
        ctc_validation_distinct_note=(
            "0.25, but that arm repeats one control sentence N times, so it is "
            "periodic audio too; it is consistent with the periodicity account, "
            "not a counterexample to it, and \\WhisperValDis should stay unused "
            "or be renamed"),
        whisper_vs_ctc_on_identical_material=dict(
            whisper=cv["periodic_whisper_kge4"]["point"],
            ctc=cv["periodic_ctc_kge4"]["point"],
            n_trials_whisper=cv["periodic_whisper_kge4"]["n_trials"],
            n_trials_ctc=cv["periodic_ctc_kge4"]["n_trials"],
            same_audit=True))
    out["cross_experiment_notes"] = notes
    print(f"\n[C] which arm carries which claim")
    print(f"    Whisper {notes['whisper_vs_ctc_on_identical_material']['whisper']:.2f} "
          f"vs CTC {notes['whisper_vs_ctc_on_identical_material']['ctc']:.2f}: "
          f"same audit, same audio -- a fair comparison "
          f"(n={cv['periodic_whisper_kge4']['n_trials']} and "
          f"{cv['periodic_ctc_kge4']['n_trials']} trials on 6 donors)")
    print(f"    'distinct-word sequences score 1.00' rests on "
          f"{notes['distinct_word_n_trials']} trials over "
          f"{notes['distinct_word_n_rotations']} rotations in a DIFFERENT audit "
          f"(0.25 s gaps, Whisper only);\n    CTC was never run on that material")
    print(f"    {notes['ctc_validation_distinct_note']}")

    frag = {k: v["fragile_because"] for k, v in cv.items() if v["fragile"]}
    frag.update({f"asr_reliability repeated k={k}": v["fragile_because"]
                 for k, v in ar["repeated_by_k"].items() if v["fragile"]})
    out["fragile_ratios"] = frag
    print(f"\nFragile by the pre-committed rule ({len(frag)}):")
    for k, why in frag.items():
        print(f"  {k:34s} {'; '.join(why)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/results/judge_audit_ci.json"))
    args = ap.parse_args()
    out: dict = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                 "fragility_rule": {"donor_ci_width_above": FRAGILE_WIDTH,
                                    "fewer_donors_than": FRAGILE_DONORS,
                                    "missing_trial_fraction_above": FRAGILE_MISSING}}
    ctc_validation(out)
    asr_reliability(out)
    cross_check(out)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
