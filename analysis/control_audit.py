#!/usr/bin/env python3
"""Adversarial audit of the control-word count: is 0.0% real or an artifact?

A reviewer flagged the paper's headline dissociation as too perfect: repeated
items undercount by a median -8.3% at k>=6, while their length-matched
controls (k copies of one word replaced by k *distinct* fillers) score
*exactly* 0.0% -- median 0.0, bootstrap CI [0.0, 0.0] -- for every one of six
checkpoints, at every k up to 32. The specific worry: "the count-error metric
only tallies how many units a transcript contains, not which words they are,
so the reported perfect control result may reflect counting the wrong
things." This script tries to break that result four independent ways. It is
read-only with respect to `src/`, the paper, and every existing result file:
it imports `count_units` / `count_occurrences` / `normalise` from
`src/common/score_counts.py` unchanged and only ever *calls* them.

  1. IDENTITY CHECK -- feed `count_units` transcripts with the wrong words,
     one filler repeated instead of k distinct ones, and reordered fillers.
     If any of these score as a full match, the reviewer's worry is confirmed
     and the paper's control result is void.

  2. SATURATION CHECK -- `count_units` iterates over `units` (len == k), so
     it can never emit more than k matches: overcounting is *structurally*
     impossible for the control direction by construction, not because
     models never over-render fillers. That asymmetry is disclosed here
     explicitly, and separately, a synthetic partial-match transcript proves
     the function CAN still register a non-zero (negative) error -- so "0.0%"
     is not simply unreachable-otherwise.

  3. REAL-DATA CHECK -- pull actual CTC transcripts at k>=16 controls from
     the raw ASR jsonl files (not just the CSV, so nothing is laundered
     through an intermediate join) and print stimulus vs. transcript side by
     side for manual audit, mixing exact matches and non-zero cases.

  4. DISTRIBUTION CHECK -- the paper reports only the median. This computes
     mean, IQR, min, max, and the zero/negative/positive fractions per model
     at k>=6, because a median of 0.0 is compatible with a very unclean
     underlying distribution, and a bootstrap CI of a median is [0.0, 0.0]
     whenever a bare majority of the sample is exactly 0 -- regardless of how
     bad the rest of the sample is.

Along the way this also checks the stimulus generator itself
(`data/stimuli/make_stimuli.py`), because the manual audit surfaces something
neither the reviewer nor the four checks above were looking for: the "k
distinct fillers" description is only true up to k=8. The filler pool per
carrier template has exactly 8 words (`WORD_TEMPLATES` in make_stimuli.py),
so every control item with k>8 (k in {12,16,24,32} -- most of the k>=6 mass)
is built by *cycling* that 8-word pool, i.e. it contains repeats. That is
reported here as a stimulus-construction fact, verified directly against
`data/stimuli/stimuli.jsonl`, independent of the scoring-code question.

Usage:
  python analysis/control_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))
# Imported, never modified: these are the exact functions that produced the
# paper's numbers. Testing copies would not test anything.
from score_counts import count_units, count_occurrences, normalise  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
ABLATIONS = {"xtts2norp"}  # re-run of a panel member, never pooled (see other analyses)


# --------------------------------------------------------------------------
# 1. IDENTITY CHECK
# --------------------------------------------------------------------------
def identity_check() -> dict:
    """Can a transcript containing k of the *wrong* words score as correct?

    Each case below sets up a transcript engineered to be maximally
    favourable to the "counting isn't checking identity" hypothesis, then
    asks `count_units` what it scores. If the metric were merely tallying
    token count (or word count, or any identity-blind measure), all of these
    would come back k/k. They must not.
    """
    units = ["apple", "banana", "cherry", "damson"]  # k=4, all distinct
    k = len(units)
    cases = []

    def run(name: str, tokens: list[str], expect_full_credit: bool, note: str):
        c = count_units(tokens, units)
        cases.append(dict(
            name=name, tokens=tokens, units=units, k=k, count_units_result=c,
            rel_err=(c - k) / k, expect_full_credit=expect_full_credit,
            scored_full_credit=(c == k), note=note,
        ))

    # (a) Entirely wrong, same cardinality: k tokens, none of them the
    #     requested fillers. This is the reviewer's exact scenario.
    run("wrong_words_same_count",
        ["kumquat", "elderberry", "fig", "guava"],
        expect_full_credit=False,
        note="k=4 tokens present, zero overlap with the requested fillers")

    # (b) One filler repeated k times instead of k distinct ones -- the
    #     literal alternative hypothesis named in the task.
    run("one_filler_repeated_k_times",
        ["banana", "banana", "banana", "banana"],
        expect_full_credit=False,
        note="4 tokens, but only 1 distinct word -- must not score 4/4")

    # (c) Same k distinct words, correct multiset, wrong order (full
    #     reversal). Order matters to the task ("say these in order"), and to
    #     the forward-only matcher; this should not fetch full credit either.
    run("correct_words_reversed_order",
        ["damson", "cherry", "banana", "apple"],
        expect_full_credit=False,
        note="right words, wrong order -- forward-only match should not "
             "recover all 4")

    # (d) A single adjacent transposition (near-miss ordering), still not the
    #     literal expected order.
    run("adjacent_swap",
        ["banana", "apple", "cherry", "damson"],
        expect_full_credit=False,
        note="first two fillers swapped")

    # (e) Positive control: the actually-correct rendering. If this does NOT
    #     score k/k, the function is broken in a way that would show up as
    #     spuriously *negative* control error everywhere, not spuriously
    #     positive -- included so the other cases are interpretable against a
    #     working baseline, not against a function that never returns k.
    run("correct_rendering",
        ["apple", "banana", "cherry", "damson"],
        expect_full_credit=True,
        note="sanity check: exact match must score full credit")

    # (f) Correct words, correct order, embedded in carrier text (the
    #     realistic case) -- also must score full credit.
    run("correct_rendering_with_carrier",
        ["the", "dog", "was", "apple", "banana", "cherry", "damson", "big"],
        expect_full_credit=True,
        note="realistic case: fillers embedded in a carrier sentence")

    all_pass = all(c["scored_full_credit"] == c["expect_full_credit"] for c in cases)
    return dict(cases=cases, all_behave_as_expected=all_pass)


# --------------------------------------------------------------------------
# 2. SATURATION / CLIPPING CHECK
# --------------------------------------------------------------------------
def saturation_check() -> dict:
    """Is 0.0% the only value the metric can produce -- by construction?

    `count_units` loops `for u in units`, so the running count `c` can never
    exceed `len(units)`. For a control item, `units` holds exactly the k
    requested fillers, so `count_a <= k` always, and `rel_err = (count_a-k)/k`
    can never be positive. That is a real, disclosed asymmetry (word_rep
    scores overcounting via `count_occurrences`, deliberately unbounded; see
    `src/common/score_counts.py` around the `count_units` call site) -- but it
    only forecloses the *overcount* direction. It says nothing about whether
    the *undercount* direction can register a non-zero value, which is the
    direction that would actually falsify "exactly 0.0%".
    """
    units = ["apple", "banana", "cherry", "damson"]
    k = len(units)

    # (a) Feed MORE correct material than requested -- extra genuine copies
    #     of the fillers after the k-th one. If the matcher could overcount,
    #     this would return > k.
    tokens_extra = units + units  # the whole distinct set said twice
    c_extra = count_units(tokens_extra, units)

    # (b) A transcript that is correct for the first half and garbage for the
    #     second -- this must NOT come back as k (full credit), proving a
    #     genuine partial failure is representable, i.e. the function is not
    #     a disguised constant-k stub.
    tokens_partial = ["apple", "banana", "xxxxx", "yyyyy"]
    c_partial = count_units(tokens_partial, units)

    # (c) Complete miss.
    tokens_miss = ["nothing", "here", "matches", "at", "all"]
    c_miss = count_units(tokens_miss, units)

    return dict(
        overcount_structurally_possible=bool(c_extra > k),
        overcount_probe=dict(tokens=tokens_extra, units=units, k=k,
                              result=c_extra,
                              interpretation="result is capped at k=%d even "
                                             "though %d correct fillers were "
                                             "present in the transcript"
                                             % (k, len(tokens_extra))),
        undercount_representable=bool(c_partial < k and c_partial > 0),
        partial_probe=dict(tokens=tokens_partial, units=units, k=k,
                            result=c_partial, rel_err=(c_partial - k) / k),
        complete_miss_probe=dict(tokens=tokens_miss, units=units, k=k,
                                  result=c_miss, rel_err=(c_miss - k) / k),
        conclusion=(
            "count_units is capped above at k by construction (control error "
            "can never be positive: overcounting is impossible for controls "
            "regardless of what the model does), but the undercount "
            "direction is fully representable -- a partial or total miss "
            "returns a value strictly below k. So a non-zero control error "
            "is not merely possible in principle; see section 4 for whether "
            "it happens in the actual data."
        ),
    )


# --------------------------------------------------------------------------
# 3 & 4: REAL DATA -- manual transcript audit and full distribution
# --------------------------------------------------------------------------
def load_stimuli(path: Path) -> dict:
    stim = {}
    for line in open(path):
        it = json.loads(line)
        stim[it["item_id"]] = it
    return stim


def stimuli_pool_cycling_check(stim: dict) -> dict:
    """Are control fillers really 'k distinct words, no repetition' at all k?

    `make_stimuli.py` draws each control's filler list as
    `[fillers[i % len(fillers)] for i in range(k)]` from an 8-word pool per
    carrier template. That is only actually k *distinct* words when k<=8.
    This matters for interpreting section 4: if a chunk of the k>=6 mass is
    built from a filler list that itself repeats (period 8), the "controls
    have no periodicity" framing is not quite true of the stimulus at high k,
    independent of anything the scoring code does with it.
    """
    rows = []
    for it in stim.values():
        if it["family"] != "control_word":
            continue
        units = it.get("control_units") or it.get("boundary_units") or []
        n_distinct = len(set(units))
        rows.append(dict(item_id=it["item_id"], template=it["template"],
                          k=it["k"], n_units=len(units), n_distinct=n_distinct,
                          has_internal_repeat=bool(n_distinct < len(units))))
    df = pd.DataFrame(rows)
    by_k = (df.groupby("k")["has_internal_repeat"].mean().to_dict())
    return dict(
        pool_size_per_template=8,
        fraction_of_control_items_with_internal_repeat_by_k={int(k): float(v) for k, v in by_k.items()},
        n_items_k_gt_8_with_repeat=int(df[(df.k > 8)].has_internal_repeat.sum()),
        n_items_k_gt_8_total=int(len(df[df.k > 8])),
        conclusion=(
            "Every control item with k>8 (k in {12,16,24,32}, i.e. the "
            "majority of the k>=6 range examined) is built by cycling an "
            "8-word filler pool, so its filler list itself contains repeats "
            "(period 8). Table 1 of the paper (paper/main.tex) describes "
            "control_word items as '$k$ distinct filler words (no "
            "repetition)'; that is only true for k<=8. The supplementary "
            "(paper/supplementary/supp.tex) does disclose 'cycled through "
            "the pool', so the fact is not hidden from the codebase, but "
            "the main-paper table's parenthetical is not accurate at the "
            "k values that dominate the k>=6 headline statistic."
        ),
    )


def manual_transcript_audit(behavioural: pd.DataFrame, stim: dict, kmin: int,
                             n_sample: int, seed: int) -> list[dict]:
    """Pull real CTC transcripts for control items at high k and print them
    next to the stimulus text, so a human (not this script) can judge whether
    the model actually rendered what `count_units` says it rendered.

    Deliberately oversamples non-zero cases: at k>=16 the large majority of
    rows are exact zero (see section 4), so a uniform random sample would
    mostly just show boring perfect matches. The interesting audit question
    is what the *non-zero* rows actually contain -- genuine model failure, or
    an ASR artifact -- so this pulls a mix of both and lets both be judged.
    """
    c = behavioural[(behavioural.family == "control_word") & (behavioural.k >= kmin)].copy()
    c["rel_err"] = (c.count_a - c.k) / c.k

    # Raw jsonl re-fetch (not the CSV's cached `transcript` column) per the
    # task's instruction to pull straight from asr_ctc/<model>.jsonl.
    raw_cache: dict[str, dict] = {}

    def raw_text(model: str, stem: str) -> str | None:
        if model not in raw_cache:
            d = {}
            p = Path(DATA_ROOT) / "asr_ctc" / f"{model}.jsonl"
            if p.exists():
                for line in open(p):
                    try:
                        r = json.loads(line)
                        d[r["stem"]] = r.get("text", "")
                    except Exception:  # noqa: BLE001
                        pass
            raw_cache[model] = d
        return raw_cache[model].get(stem)

    rng = np.random.default_rng(seed)
    nonzero = c[c.rel_err != 0]
    zero = c[c.rel_err == 0]
    n_nz = min(n_sample - 5, len(nonzero)) if len(nonzero) else 0
    n_z = min(n_sample - n_nz, len(zero)) if len(zero) else 0
    picks = pd.concat([
        nonzero.sample(n=n_nz, random_state=seed) if n_nz else nonzero.iloc[:0],
        zero.sample(n=n_z, random_state=seed) if n_z else zero.iloc[:0],
    ]).sort_values(["model", "template", "k"])

    out = []
    for _, r in picks.iterrows():
        stem = f"{r.item_id}_s{int(r.seed)}"
        raw = raw_text(r.model, stem)
        it = stim.get(r.item_id, {})
        out.append(dict(
            model=r.model, item_id=r.item_id, k=int(r.k), seed=int(r.seed),
            template=r.template, count_a=int(r.count_a), rel_err=float(r.rel_err),
            outcome=r.outcome,
            stimulus_text=it.get("text", ""),
            boundary_units=it.get("boundary_units", []),
            raw_ctc_transcript=raw,
            csv_transcript=r.transcript,
            transcript_matches_csv=(raw == r.transcript) if raw is not None else None,
        ))
    return out


def distribution_check(behavioural: pd.DataFrame, kmin: int) -> dict:
    """Full distribution of control relative error at k>=kmin, per model.

    The paper reports only the bootstrap CI of the *median*. A median of 0.0
    with a [0.0, 0.0] CI is exactly what you get whenever a bare majority of
    a sample sits at 0, no matter how negative the rest of it is -- so the
    median alone cannot distinguish "genuinely perfect" from "usually
    perfect, sometimes badly wrong". This reports mean, IQR, extremes, and the
    zero/negative/positive split so that distinction is visible.
    """
    d = behavioural[behavioural.family.isin(["word_rep", "control_word"]) & (behavioural.k >= 2)].copy()
    degen = d.outcome.isin(["empty", "degenerate"])
    d["rel_err"] = (d.count_a - d.k) / d.k

    def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
        if x.size < 3:
            return float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        meds = [float(np.median(rng.choice(x, x.size, replace=True))) for _ in range(n_boot)]
        return tuple(np.percentile(meds, [2.5, 97.5]))

    per_model = {}
    for m in sorted(d.model.unique()):
        s = d[(d.model == m) & (d.family == "control_word") & (d.k >= kmin) & ~degen]
        v = s.rel_err.to_numpy(dtype=float)
        if v.size == 0:
            continue
        lo, hi = boot_ci(v)
        per_model[m] = dict(
            n=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)),
            median_ci=[lo, hi],
            iqr_25=float(np.percentile(v, 25)), iqr_75=float(np.percentile(v, 75)),
            min=float(v.min()), max=float(v.max()),
            frac_exact_zero=float(np.mean(v == 0)),
            frac_negative=float(np.mean(v < 0)),
            frac_positive=float(np.mean(v > 0)),
            is_ablation=(m in ABLATIONS),
            # a mean much more negative than the median is the tell for a
            # misleadingly clean headline number
            mean_median_gap=float(np.mean(v) - np.median(v)),
        )

    # Per-template breakdown at k>=kmin, panel only: does the non-zero mass
    # concentrate in one carrier template? (It does -- see report.) This is
    # the check that surfaces the ASR-vocabulary confound: some fillers
    # ("okay", "hmm", "right") are words a CTC recognizer without a language
    # model is prone to mis-spell ("o k", "um"/"m", "write"), which would
    # make count_units correctly report a miss for a word the model may well
    # have said correctly.
    panel = d[~d.model.isin(ABLATIONS)]
    per_template = {}
    for t, g in panel[(panel.family == "control_word") & (panel.k >= kmin) & ~degen.loc[panel.index]].groupby("template"):
        v = g.rel_err.to_numpy(dtype=float)
        per_template[t] = dict(n=int(v.size), mean=float(v.mean()),
                                median=float(np.median(v)),
                                frac_nonzero=float(np.mean(v != 0)))

    # Pooled, panel only (mirrors analysis/count_error.py's ABLATIONS handling)
    pooled = panel[(panel.family == "control_word") & (panel.k >= kmin) & ~degen.loc[panel.index]]
    pv = pooled.rel_err.to_numpy(dtype=float)
    lo, hi = boot_ci(pv)

    return dict(
        kmin=kmin,
        per_model=per_model,
        per_template_panel=per_template,
        pooled_panel=dict(
            n=int(pv.size), median=float(np.median(pv)), mean=float(np.mean(pv)),
            median_ci=[lo, hi], frac_exact_zero=float(np.mean(pv == 0)),
            frac_negative=float(np.mean(pv < 0)), frac_positive=float(np.mean(pv > 0)),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--out", default="data/results/control_audit.json")
    ap.add_argument("--kmin", type=int, default=6,
                     help="matches the kmin used in analysis/count_error.py")
    ap.add_argument("--audit-kmin", type=int, default=16,
                     help="k threshold for the manual transcript audit sample")
    ap.add_argument("--n-audit-sample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    behavioural = pd.read_csv(args.behavioural)
    stim = load_stimuli(Path(args.stimuli))

    print("=" * 78)
    print("1. IDENTITY CHECK -- does count_units check word identity?")
    print("=" * 78)
    ident = identity_check()
    for c in ident["cases"]:
        status = "OK " if c["scored_full_credit"] == c["expect_full_credit"] else "FAIL"
        print(f"[{status}] {c['name']:32s} tokens={c['tokens']}")
        print(f"        -> count_units = {c['count_units_result']}/{c['k']}  "
              f"rel_err={c['rel_err']:+.3f}  ({c['note']})")
    print(f"\nall cases behave as expected: {ident['all_behave_as_expected']}")

    print("\n" + "=" * 78)
    print("2. SATURATION / CLIPPING CHECK")
    print("=" * 78)
    sat = saturation_check()
    print(f"overcount structurally possible for controls: "
          f"{sat['overcount_structurally_possible']}")
    print(f"  probe: {sat['overcount_probe']}")
    print(f"undercount representable (partial match != 0 and != k): "
          f"{sat['undercount_representable']}")
    print(f"  probe: {sat['partial_probe']}")
    print(f"  complete miss probe: {sat['complete_miss_probe']}")
    print(f"\n{sat['conclusion']}")

    print("\n" + "=" * 78)
    print("STIMULI CHECK -- are control fillers really k distinct words at all k?")
    print("=" * 78)
    pool = stimuli_pool_cycling_check(stim)
    print(f"fraction of control items with an internal repeat, by k:")
    for k, v in sorted(pool["fraction_of_control_items_with_internal_repeat_by_k"].items()):
        print(f"  k={k:3d}: {v:.2f}")
    print(f"\n{pool['conclusion']}")

    print("\n" + "=" * 78)
    print(f"3. MANUAL TRANSCRIPT AUDIT (k>={args.audit_kmin}, raw asr_ctc jsonl)")
    print("=" * 78)
    audit = manual_transcript_audit(behavioural, stim, args.audit_kmin,
                                     args.n_audit_sample, args.seed)
    for a in audit:
        print("-" * 78)
        print(f"model={a['model']}  item_id={a['item_id']}  k={a['k']}  seed={a['seed']}  "
              f"count_a={a['count_a']}  rel_err={a['rel_err']:+.3f}  outcome={a['outcome']}")
        print(f"  STIMULUS  : {a['stimulus_text']}")
        print(f"  TRANSCRIPT: {a['raw_ctc_transcript']}")

    print("\n" + "=" * 78)
    print(f"4. DISTRIBUTION CHECK (k>={args.kmin})")
    print("=" * 78)
    dist = distribution_check(behavioural, args.kmin)
    print(f"{'model':11s} {'n':>4s} {'median':>8s} {'mean':>8s} {'IQR':>16s} "
          f"{'min':>7s} {'max':>6s} {'%zero':>6s} {'%neg':>6s} {'%pos':>6s}")
    for m, v in dist["per_model"].items():
        tag = " [ABLATION]" if v["is_ablation"] else ""
        print(f"{m:11s} {v['n']:4d} {v['median']:8.4f} {v['mean']:8.4f} "
              f"[{v['iqr_25']:+.3f},{v['iqr_75']:+.3f}] {v['min']:7.3f} {v['max']:6.3f} "
              f"{100*v['frac_exact_zero']:6.1f} {100*v['frac_negative']:6.1f} "
              f"{100*v['frac_positive']:6.1f}{tag}")
    print(f"\nper-template breakdown (panel only, k>={args.kmin}):")
    for t, v in sorted(dist["per_template_panel"].items()):
        print(f"  {t}: n={v['n']:4d} mean={v['mean']:+.4f} median={v['median']:+.4f} "
              f"frac_nonzero={v['frac_nonzero']:.3f}")
    p = dist["pooled_panel"]
    print(f"\npooled (panel, k>={args.kmin}): median={p['median']:+.4f} "
          f"CI=[{p['median_ci'][0]:+.4f},{p['median_ci'][1]:+.4f}]  mean={p['mean']:+.4f}  "
          f"n={p['n']}  %zero={100*p['frac_exact_zero']:.1f} "
          f"%neg={100*p['frac_negative']:.1f} %pos={100*p['frac_positive']:.1f}")

    result = dict(
        identity_check=ident,
        saturation_check=sat,
        stimuli_pool_cycling_check=pool,
        manual_transcript_audit=audit,
        distribution_check=dist,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
