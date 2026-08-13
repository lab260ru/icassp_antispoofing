#!/usr/bin/env python3
"""Is the period ladder a *period* ladder, or a power-of-two ladder?

THE OBJECTION
-------------
`analysis/period_ladder.py` reports p in {1, 2, 4, 8, k}. Every interior rung is
a power of two, and there is no odd period anywhere in the design. Two rivals to
"periodicity" survive that untouched:

  * **power-of-two block structure** -- a decoder whose bookkeeping is tied to
    block sizes that are powers of two would produce the same monotone ladder;
  * **pool cycling / loop attractor** -- the p in {2,4,8} arms all cycle a pool
    of a size that divides eight, so "how the filler pool cycles" is not
    separated from "the period" either.

Under either rival the published ladder looks exactly as it does. The only way
to tell them apart is to put rungs *between* the powers of two.
`data/stimuli/make_stimuli_period.py` adds two, at k = 24 only (the sole k in
the ladder divisible by 3):

    p = 3   odd, not a power of two, strictly between p=2 and p=4
    p = 6   even, not a power of two, strictly between p=4 and p=8

vocabulary-balanced by the same rotation argument (eight consecutive-window
rotations for p=3, four for p=6, so each arm uses each of its template's eight
pool words the same number of times), same carriers, same word count, same k,
same three seeds, same four checkpoints.

THE PRE-COMMITMENT  (restated from the generator docstring, which was written
and committed before any p=3 or p=6 audio existed)
---------------------------------------------------------------------------
The comparison is made **within k = 24**, because the deficit depends on k as
well as on p. The k=24 baseline, recomputed through `population.panel()` under
the paper's own exact-rate statistic (the ordered scan, cap hits excluded) and
pooled as the unweighted mean over the four checkpoints, is

    E(1) = 8.5%   E(2) = 51.2%   E(4) = 75.2%   E(8) = 92.9%   E(24) = 85.0%

-- not the all-k pooled 10.4 / 47.9 / 73.6 / 93.1 that the paper quotes, which
average over k=16 and k=32 as well. The two tests are stated against the k=24
numbers.

    TEST A:  E(2) < E(3) < E(4)
    TEST B:  E(4) < E(6) < E(8)

  * **PERIODICITY CONFIRMED** -- both tests pass. The deficit is a smooth
    monotone function of the period, powers of two have no special status, the
    pool-cycling / loop-attractor rival is refuted, and the title stands.
  * **PERIODICITY REFUTED / POWER-OF-TWO ARTIFACT** -- E(3) or E(6) lands at or
    above E(8) (the non-power-of-two rungs behave like the no-deficit ceiling),
    or below E(2). Either way the ordering is not monotone in the period.
    Reported first and loudly; it forces a retitle.
  * **AMBIGUOUS** -- anything else. The report must name which test failed and
    on which checkpoints.

Per-checkpoint rates are reported beside the pooled ones throughout, because a
pooled interpolation that holds while two checkpoints individually invert is a
different result.

WHAT COULD STILL MAKE THIS A LIE, AND WHAT IS DONE ABOUT IT
-----------------------------------------------------------
Everything `analysis/period_ladder.py` documents applies unchanged, and the
counting/censoring machinery is imported from it rather than re-implemented:
the unbounded recount (`count_unbounded`), the strict conjunction, the
cap-hits-kept bound, and `population.panel()` for exclusions. Two things are
specific to this file:

1. **The new rungs are the only ones generated today.** The p in {1,2,4,8,k}
   rungs are the *same rows* the published ladder used -- the generators resume
   on (item_id, seed) -- so the baseline the interpolation is measured against
   was not re-rolled. This is a strength for comparability and a weakness for
   drift detection, so `replication` in `period_ladder.py` remains the check on
   whether the pipeline moved.
2. **n is not equal across rungs.** The rotation counts that balance the pool
   differ by period (8 rotations at p=3, 4 at p=2 and p=6, 2 at p=4, 1 at p=8),
   so the ceiling rungs have the fewest generations and the widest intervals.
   A cluster bootstrap over templates is reported for every adjacent gap so
   that the interpolation is not read off point estimates alone.

Usage:  python analysis/period_odd.py
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
from src.common.population import cap_flags, describe, panel  # noqa: E402
from analysis.period_ladder import load_stimuli, prepare  # noqa: E402

K = 24
# The full ladder at k=24, in period order. "24" is the never-cycled rung (the
# 146-word extension pool), which `prepare()` labels "k".
ARMS = ["1", "2", "3", "4", "6", "8", "k"]
ARM_LABEL = {"k": "24"}
RULES = [("exact", "ordered scan (the published rule)"),
         ("exact_u", "unbounded recount (same rule at every p)"),
         ("exact_both", "strict: in order AND no extras")]


def tables(d: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = d.pivot_table(index="model", columns="arm", values=col, aggfunc="mean")
    n = d.pivot_table(index="model", columns="arm", values=col, aggfunc="size")
    cols = [a for a in ARMS if a in t.columns]
    return t[cols], n[cols]


def show(t: pd.DataFrame, n: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("  " + f"{'checkpoint':11s} " +
          "  ".join(f"p={ARM_LABEL.get(a, a):>2s}" for a in t.columns) + "      (n per cell)")
    for m in t.index:
        cells = "  ".join(f"{100*t.loc[m, a]:5.1f}" for a in t.columns)
        ns = ",".join(str(int(n.loc[m, a])) for a in t.columns)
        print(f"  {m:11s} {cells}      ({ns})")
    print(f"  {'POOLED':11s} " + "  ".join(f"{100*t[a].mean():5.1f}" for a in t.columns)
          + "      (unweighted mean over checkpoints)")


def interpolation(t: pd.DataFrame, label: str) -> dict:
    """The two pre-committed tests, pooled and per checkpoint."""
    have = set(t.columns)
    if not {"2", "3", "4", "6", "8"} <= have:
        return {}
    E = {a: float(t[a].mean()) for a in t.columns}

    def band(p: str, lo: str, hi: str, series: pd.Series | None = None) -> dict:
        v = E if series is None else series
        return dict(lo_arm=lo, hi_arm=hi, lo=v[lo], mid=v[p], hi=v[hi],
                    passes=bool(v[lo] < v[p] < v[hi]))

    a = band("3", "2", "4")
    b = band("6", "4", "8")
    per = {}
    for m in t.index:
        s = t.loc[m]
        per[m] = dict(A=band("3", "2", "4", s), B=band("6", "4", "8", s))

    # The refutation clause, exactly as pre-committed: a non-power-of-two rung
    # at or above the p=8 ceiling, or below the p=2 rung.
    refute = {p: dict(at_or_above_E8=bool(E[p] >= E["8"]), below_E2=bool(E[p] < E["2"]))
              for p in ("3", "6")}
    refuted = any(v["at_or_above_E8"] or v["below_E2"] for v in refute.values())
    if a["passes"] and b["passes"]:
        call = "PERIODICITY CONFIRMED"
    elif refuted:
        call = "PERIODICITY REFUTED / POWER-OF-TWO ARTIFACT"
    else:
        call = "AMBIGUOUS"

    # Monotonicity of the whole cycled ladder, which is the claim the
    # interpolation is a test of. The never-cycled rung is excluded: it draws on
    # a different, harder pool and is an inherited confound, not a rung.
    ladder = [x for x in ("1", "2", "3", "4", "6", "8") if x in have]
    mono = all(E[ladder[i]] < E[ladder[i + 1]] for i in range(len(ladder) - 1))
    inversions = [f"E({ladder[i]})={100*E[ladder[i]]:.1f} >= E({ladder[i+1]})="
                  f"{100*E[ladder[i+1]]:.1f}"
                  for i in range(len(ladder) - 1) if E[ladder[i]] >= E[ladder[i + 1]]]

    failing = [m for m, v in per.items() if not (v["A"]["passes"] and v["B"]["passes"])]
    print(f"\n  [{label}] pooled: " +
          "  ".join(f"E({ARM_LABEL.get(x, x)})={100*E[x]:.1f}" for x in ladder))
    print(f"    TEST A  E(2) < E(3) < E(4):  {100*a['lo']:.1f} < {100*a['mid']:.1f} "
          f"< {100*a['hi']:.1f}   {'PASS' if a['passes'] else 'FAIL'}")
    print(f"    TEST B  E(4) < E(6) < E(8):  {100*b['lo']:.1f} < {100*b['mid']:.1f} "
          f"< {100*b['hi']:.1f}   {'PASS' if b['passes'] else 'FAIL'}")
    print(f"    monotone over p in {{1,2,3,4,6,8}}: {mono}"
          + (f"   inversions: {inversions}" if inversions else ""))
    if failing:
        print(f"    per-checkpoint failures: {failing}")
        for m in failing:
            for name in ("A", "B"):
                v = per[m][name]
                if not v["passes"]:
                    print(f"      {m:9s} TEST {name}: {100*v['lo']:.1f} "
                          f"< {100*v['mid']:.1f} < {100*v['hi']:.1f}  FAIL")
    else:
        print("    both tests hold in every checkpoint individually")
    print(f"    -> {call}")
    return dict(label=label, E_pooled=E, test_A=a, test_B=b, per_checkpoint=per,
                refutation_clause=refute, monotone=bool(mono), inversions=inversions,
                checkpoints_failing=failing, outcome=call)


def leave_one_out(t: pd.DataFrame) -> dict:
    """Both tests recomputed with each checkpoint dropped in turn.

    Four checkpoints is few enough that one of them can carry a pooled mean, and
    one of them has a specific reason to be doubted: XTTS-v2's p=3 and p=6 rows
    were generated after its conda env's transformers was upgraded to 5.x, so
    they are not strictly commensurable with its own published p in {2,4,8}
    rows (see `src/models/xtts_gen_compat.py`). The drop-XTTS-v2 row is
    therefore the one to read, and the rest are there so it cannot be accused
    of being the only leave-one-out that was looked at.
    """
    out = {}
    print("\n--- leave one checkpoint out (pooled mean recomputed each time) ---")
    print(f"  {'dropped':11s} " + "  ".join(f"E({a:>2s})" for a in t.columns)
          + "   TEST A  TEST B")
    for drop in [None, *t.index]:
        s = t.drop(index=drop) if drop is not None else t
        E = {a: float(s[a].mean()) for a in s.columns}
        a_ok = E["2"] < E["3"] < E["4"]
        b_ok = E["4"] < E["6"] < E["8"]
        key = drop or "(none)"
        out[key] = dict(E={k: v for k, v in E.items()}, test_A=bool(a_ok),
                        test_B=bool(b_ok),
                        outcome="PERIODICITY CONFIRMED" if a_ok and b_ok else "not both")
        print(f"  {key:11s} " + "  ".join(f"{100*E[a]:5.1f}" for a in s.columns)
              + f"    {'PASS' if a_ok else 'FAIL':4s}    {'PASS' if b_ok else 'FAIL'}")
    return out


def bootstrap_gaps(d: pd.DataFrame, col: str, *, n_boot: int = 4000,
                   seed: int = 0) -> dict:
    """Cluster bootstrap over templates for every adjacent gap in the ladder.

    The unit of resampling is the carrier template: the rotations, the seeds and
    the periods inside one template share a sentence and are not independent
    draws. Four checkpoints are too few to bootstrap over, so the checkpoint
    mean is recomputed inside each replicate -- the interval is a statement
    about generalising across carriers, not across models. Five templates is a
    small cluster count and the intervals are correspondingly wide; they are
    here to stop a 2-point gap being read as an ordering.
    """
    rng = np.random.default_rng(seed)
    tpl = sorted(d.template.unique())
    idx = {t: d[d.template == t] for t in tpl}
    pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "6"), ("6", "8")]
    acc: dict[str, list[float]] = {f"{a}->{b}": [] for a, b in pairs}
    for _ in range(n_boot):
        s = pd.concat([idx[tpl[i]] for i in rng.integers(0, len(tpl), len(tpl))],
                      ignore_index=True)
        t = s.pivot_table(index="model", columns="arm", values=col, aggfunc="mean")
        for a, b in pairs:
            if a in t.columns and b in t.columns:
                acc[f"{a}->{b}"].append(float(t[b].mean() - t[a].mean()))
    out = {}
    print("\n  adjacent gaps, cluster bootstrap over the five carrier templates "
          "(gap > 0 means the ladder rises with p)")
    for k, v in acc.items():
        if not v:
            continue
        lo, hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        out[k] = dict(mean=float(np.mean(v)), lo=lo, hi=hi, n_boot=len(v),
                      excludes_zero=bool(lo > 0 or hi < 0))
        print(f"    E({k.split('->')[1]}) - E({k.split('->')[0]}) = "
              f"{100*np.mean(v):+6.1f} pts  [{100*lo:+6.1f}, {100*hi:+6.1f}]"
              f"{'  (excludes 0)' if lo > 0 or hi < 0 else ''}")
    return out


def paired(d: pd.DataFrame, col: str) -> dict:
    """The ladder differenced inside each (model, template, seed) cell.

    The tables above average over cells before differencing. This averages the
    rotations of each arm into one number per cell first, then compares only
    cells where every arm is present, so nothing about which cells an arm
    happens to occupy can enter the ordering.
    """
    piv = d.pivot_table(index=["model", "template", "seed"], columns="arm",
                        values=col, aggfunc="mean")
    need = [a for a in ARMS if a in piv.columns and a != "k"]
    piv = piv.dropna(subset=need)
    if not len(piv):
        return {}
    print("\n--- paired within (model, template, seed), rotations averaged first ---")
    print(f"  {'checkpoint':11s} {'cells':>6s} " +
          "  ".join(f"p={a:>2s}" for a in need))
    rows = []
    for m, g in piv.groupby(level=0):
        print(f"  {m:11s} {len(g):6d} " + "  ".join(f"{100*g[a].mean():5.1f}" for a in need))
        rows.append(dict(model=m, n_cells=int(len(g)),
                         **{f"p{a}": float(g[a].mean()) for a in need}))
    pooled = {f"p{a}": float(np.mean([r[f"p{a}"] for r in rows])) for a in need}
    print(f"  {'POOLED':11s} {'':6s} " + "  ".join(f"{100*pooled[f'p{a}']:5.1f}" for a in need))
    a_ok = pooled["p2"] < pooled["p3"] < pooled["p4"]
    b_ok = pooled["p4"] < pooled["p6"] < pooled["p8"]
    print(f"  paired TEST A {'PASS' if a_ok else 'FAIL'}, "
          f"TEST B {'PASS' if b_ok else 'FAIL'}")
    return dict(per_checkpoint=rows, pooled=pooled, test_A=bool(a_ok), test_B=bool(b_ok))


def hygiene(d: pd.DataFrame, d_caps: pd.DataFrame) -> list[dict]:
    """Per-arm cap rate, length, duration and degeneracy.

    The cap rate is the load-bearing one twice over. It is the censoring check
    -- if the arms differ in how often our own token budget truncated them, the
    table compares differently censored populations -- and it is the sanity
    check on `qwen_gen.py`, whose `decode_step_count` off-by-one used to make
    `hit_cap` never fire. A column of exact zeros everywhere is that bug, not a
    clean run.
    """
    flags = cap_flags()
    g = d_caps.assign(cap=[flags.get((r.model, r.item_id, r.seed), False)
                           for r in d_caps.itertuples()])
    rows = []
    print("\n--- per-arm hygiene at k=24 (cap hits excluded from the tables above) ---")
    print(f"  {'arm':>4s} {'n_kept':>7s} {'cap%':>6s} {'words':>7s} {'chars':>7s} "
          f"{'dur_s':>7s} {'degen%':>7s}")
    for a in ARMS:
        gg, gk = g[g.arm == a], d[d.arm == a]
        if not len(gg):
            continue
        r = dict(arm=ARM_LABEL.get(a, a), n_kept=int(len(gk)),
                 cap_rate=float(gg.cap.mean()),
                 words=float(gk.expected_words.mean()), chars=float(gk.n_chars.mean()),
                 duration_s=float(gk.duration_s.mean()),
                 degenerate_rate=float(gg.outcome.isin(["empty", "degenerate"]).mean()))
        rows.append(r)
        print(f"  {r['arm']:>4s} {r['n_kept']:7d} {100*r['cap_rate']:5.1f}% "
              f"{r['words']:7.1f} {r['chars']:7.1f} {r['duration_s']:7.2f} "
              f"{100*r['degenerate_rate']:6.1f}%")
    by_model = g.groupby("model").cap.mean()
    print("  cap-hit rate by checkpoint: " +
          ", ".join(f"{m} {100*v:.1f}%" for m, v in by_model.items()))
    if float(g.cap.mean()) == 0.0:
        print("  !! cap-hit rate is identically zero across every row -- that is the "
              "qwen_gen.py decode_step_count off-by-one, not a clean run. STOP.")
    return rows


def rotation_spread(d: pd.DataFrame) -> dict:
    """How much the new arms' answers move with which pool words were drawn.

    If p=3 swings widely across its eight rotations, the arm-level rate is an
    average over a large lexical confound and the interpolation is being read
    off a noisy point. The published p=2 arm's spread is the yardstick.
    """
    out = {}
    print("\n--- exact rate by rotation (spread = the lexical confound rotation removes) ---")
    for a in ("2", "3", "4", "6"):
        g = d[d.arm == a]
        if not len(g):
            continue
        r = g.groupby("rotation").exact.mean()
        out[a] = {int(k): float(v) for k, v in r.items()}
        print(f"  p={a:>2s}  " + "  ".join(f"r{int(k)}={100*v:.0f}" for k, v in r.items())
              + f"   spread {100*(r.max()-r.min()):.0f} pts")
    return out


def collapse_onto_m(d: pd.DataFrame, pub: pd.DataFrame) -> dict:
    """The new rungs against the published bare-repetition ladder at m = k/p.

    `period_ladder.py` found that the ladder collapses onto m, the number of
    times each distinct token has to repeat. The odd rungs are a free
    out-of-sample test of that: p=3 at k=24 gives m=8 and p=6 gives m=4, both
    already measured on the published `wr_*` ladder, and neither was used to fit
    anything. A residual here of the same size as the published rungs' is a
    second, independent confirmation that the driver is m and not a block size.
    """
    bare = pub[pub.item_id.astype(str).str.startswith("wr_")]
    if not len(bare):
        return {}
    bare = bare.assign(exact=(bare.count_a == bare.k))
    ref = bare.groupby(["model", "k"], observed=True).exact.agg(["mean", "size"])
    rows = []
    print("\n--- the new rungs against the published bare ladder at m = k/p ---")
    print(f"  {'checkpoint':11s} {'p':>3s} {'m':>3s} {'E(p,24)':>8s} {'E_pub(1,m)':>11s} "
          f"{'residual':>9s} {'n':>4s} {'n_pub':>6s}")
    for (mo, p), g in d[d.pool_id == "main8"].groupby(["model", "period"], observed=True):
        m = int(K // int(p))
        if (mo, m) not in ref.index:
            continue
        e, epub = float(g.exact.mean()), float(ref.loc[(mo, m), "mean"])
        rows.append(dict(model=mo, period=int(p), m=m, E=e, E_pub=epub,
                         residual=e - epub, n=int(len(g)),
                         n_pub=int(ref.loc[(mo, m), "size"]), new_rung=int(p) in (3, 6)))
        print(f"  {mo:11s} {int(p):3d} {m:3d} {100*e:7.1f}% {100*epub:10.1f}% "
              f"{100*(e-epub):+8.1f}  {len(g):4d} {int(ref.loc[(mo, m), 'size']):6d}")
    if not rows:
        return {}
    r = pd.DataFrame(rows)
    new, old = r[r.new_rung], r[~r.new_rung]
    out = dict(cells=rows,
               mae_new=float(new.residual.abs().mean()) if len(new) else float("nan"),
               mae_published=float(old.residual.abs().mean()) if len(old) else float("nan"),
               bias_new=float(new.residual.mean()) if len(new) else float("nan"))
    print(f"  mean |residual|: new rungs (p=3,6) {100*out['mae_new']:.1f} pts, "
          f"published rungs {100*out['mae_published']:.1f} pts")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_period.csv")
    ap.add_argument("--published", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli_period.jsonl")
    ap.add_argument("--out", default="data/results/period_odd.json")
    args = ap.parse_args()

    stim = load_stimuli(REPO / args.stimuli)
    d, drop = prepare(REPO / args.behavioural, stim, keep_caps=False)
    d_caps, _ = prepare(REPO / args.behavioural, stim, keep_caps=True)
    d, d_caps = d[d.k == K], d_caps[d_caps.k == K]
    print(describe(drop))
    print(f"k = {K} only.  checkpoints: {', '.join(sorted(d.model.unique()))}")
    print(f"templates: {sorted(d.template.unique())}   "
          f"arms: {sorted(d.arm.unique(), key=lambda a: (a == 'k', a))}")

    missing = [a for a in ("3", "6") if a not in set(d.arm)]
    if missing:
        raise SystemExit(f"no scored rows for p={missing} -- generate and score them first")

    res: dict = {"k": K, "checkpoints": sorted(d.model.unique()),
                 "kept": int(len(d)), "drop": {k: v for k, v in drop.items()
                                               if k != "excluded_templates"},
                 "rates": {}, "n": {}, "tests": {}}

    verdicts = {}
    for col, title in RULES:
        t, n = tables(d, col)
        show(t, n, f"exact rate by period at k={K} -- {title}")
        res["rates"][col] = json.loads(t.to_json())
        res["n"][col] = json.loads(n.to_json())
        verdicts[col] = interpolation(t, title)
        res["tests"][col] = verdicts[col]

    # cap hits kept and counted as the failures they are: the bound in the
    # other direction, since excluding them removes each arm's worst items.
    t_cap, n_cap = tables(d_caps, "exact")
    show(t_cap, n_cap, f"exact rate by period at k={K} -- cap hits KEPT (censored bound)")
    res["rates"]["exact_caps_kept"] = json.loads(t_cap.to_json())
    res["n"]["exact_caps_kept"] = json.loads(n_cap.to_json())
    res["tests"]["exact_caps_kept"] = interpolation(t_cap, "cap hits kept")

    res["leave_one_out"] = leave_one_out(tables(d, "exact")[0])
    res["bootstrap_gaps"] = bootstrap_gaps(d, "exact")
    res["bootstrap_gaps_strict"] = bootstrap_gaps(d, "exact_both")
    res["paired"] = paired(d, "exact")
    res["rotation_spread"] = rotation_spread(d)
    res["hygiene"] = hygiene(d, d_caps)

    if (REPO / args.published).exists():
        pub = pd.read_csv(REPO / args.published)
        pub = pub[pub.model.isin(sorted(d.model.unique()))]
        pub, _ = panel(pub)
        res["collapse_onto_m"] = collapse_onto_m(d, pub)

    # The headline is the published rule -- the statistic the E(p) the tests
    # were pre-committed against were computed under. The other rules are
    # reported beside it, and any disagreement between them is part of the
    # answer rather than a footnote.
    head = verdicts["exact"]["outcome"]
    by_rule = {c: v["outcome"] for c, v in verdicts.items() if v}
    by_rule["exact_caps_kept"] = res["tests"]["exact_caps_kept"]["outcome"]
    res["headline_outcome"] = head
    res["outcome_by_rule"] = by_rule
    res["rules_agree"] = bool(len(set(by_rule.values())) == 1)
    print(f"\n=== {head}   (pre-committed label, ordered-scan rule at k={K})")
    print("    by counting rule: " + ", ".join(f"{k}: {v}" for k, v in by_rule.items()))
    if not res["rules_agree"]:
        print("    !! the counting rules do not agree -- report all of them")

    out = REPO / args.out
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
