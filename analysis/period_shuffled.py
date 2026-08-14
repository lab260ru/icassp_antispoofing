#!/usr/bin/env python3
"""Is the period ladder a *period* ladder, or a lexical-diversity ladder?

The ladder in `analysis/period_ladder.py` varies p at fixed carrier, fixed word
count and fixed k, and the exact rate climbs with p. But every rung varies the
period AND the lexical diversity together -- one distinct word at p=1, two at
p=2, eight at p=8 -- so type-token ratio, unigram entropy and vocabulary size
move in lockstep with the period, and a decoder that fails on low lexical
diversity rather than on periodicity would produce the identical ladder.

`data/stimuli/make_stimuli_period_shuffled.py` breaks that confound the only way
it can be broken: it holds the **multiset of words** fixed and varies only the
**order**. For every periodic item at k=24 with p in {2, 4, 8} it emits three
shuffled twins -- same carrier, same word count, same character count, exactly
the same words the same number of times, reordered to flatten the sequence
autocorrelation -- plus a re-rendered periodic control with byte-identical text,
generated in the same run and the same environment as the twins.

The pre-commitment, the recovery-fraction test, the three outcome labels, the
placebo and the pilot-stage change that added the re-rendered control are all in
that file's docstring, written before any of this audio existed. This file only
reads them out. In summary:

    R(p) = [E_shuf(p) - E_per(p)] / [E_per(8) - E_per(p)],   Rbar = mean(R(2), R(4))

    PERIODICITY CONFIRMED         Rbar >= 0.50, min(R(2), R(4)) >= 0.25,
                                  and E_shuf > E_per at p=2 and p=4 under
                                  the order-insensitive statistic too
    LEXICAL DIVERSITY, NOT
      PERIODICITY                 Rbar <= 0.20 under BOTH statistics
    INTERMEDIATE                  anything else

with E_per taken from the **re-rendered** periodic control (primary) and from
the published ladder rows (reported beside it), the placebo Delta(8) reported
and its offset correction applied when it exceeds the pre-committed -5 points,
and every pooled number given per checkpoint and under leave-one-checkpoint-out.

Three counting rules are carried through everything, exactly as
`analysis/period_odd.py` does:

  exact       the ordered scan, the paper's own rule. Order-SENSITIVE, and the
              shuffled arm asks a model to reproduce an irregular order, so this
              rule is biased AGAINST the shuffled arm. Primary because it is the
              statistic the paper's claim is stated in.
  exact_u     the unbounded recount: total occurrences of every distinct
              requested unit, uncapped and order-insensitive. Co-primary,
              because it is the rule the shuffle cannot be penalised by.
  exact_both  the strict conjunction of the two.

and one more that only this analysis needs:

  exact_ms    the transcript carries exactly the target count of every distinct
              word. Order-insensitive and stricter than `exact_u`, which a
              transcript can satisfy by over-delivering one word and
              under-delivering another. Reported, not part of the test.

Inputs:  data/results/behavioural_period_shuffled.csv  (the side CSV; the
         published `behavioural_period.csv` is never written by this arm)
         data/stimuli/stimuli_period.jsonl
         data/stimuli/stimuli_period_shuffled.jsonl
Output:  data/results/period_shuffled.json
Usage:   python analysis/period_shuffled.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.common.population import describe, panel  # noqa: E402
from src.common.score_counts import count_occurrences, normalise  # noqa: E402
from analysis.period_ladder import count_unbounded  # noqa: E402

K = 24
PERIODS = [2, 4, 8]
TEST_PERIODS = [2, 4]        # p=8 is the placebo, not a test rung
R_CONFIRM, R_LEXICAL, R_MIN = 0.50, 0.20, 0.25
PLACEBO_TRIGGER = -0.05      # Delta(8) below this means the shuffle itself costs
REPLICATION_TOL = 0.10       # re-render vs published, pooled, per period


# --------------------------------------------------------------------------
def load_stimuli() -> pd.DataFrame:
    """The two arms plus the published periodic rows, in one frame.

    `arm` is the only thing that distinguishes them downstream:
      periodic_pub   the published ladder item (behavioural_period.csv rows)
      periodic       the re-rendered periodic control, `_z` ids, this run
      shuffled       the shuffled twins, `_x{s}` ids, this run
    """
    rows = []
    for line in (REPO / "data/stimuli/stimuli_period.jsonl").open():
        r = json.loads(line)
        if r["k"] != K:
            continue
        # The p=1 and p=k rungs are carried as CONTEXT only. They have no
        # shuffled twin -- a multiset of one word has exactly one arrangement,
        # and the never-cycled rung is already aperiodic -- so they take no part
        # in any test. They are here because the two-factor table below is
        # unreadable without the two ends of the ladder.
        if r["pool_id"] != "main8" or r["period"] not in PERIODS:
            rows.append(dict(item_id=r["item_id"], arm="context", period=r["period"],
                             rotation=r["rotation"], shuffle=-1, twin_id=r["item_id"],
                             n_chars=len(r["text"]),
                             units=json.dumps(r["boundary_units"]),
                             **{f"s_{k}": v for k, v in _flat(r).items()}))
            continue
        rows.append(dict(item_id=r["item_id"], arm="periodic_pub", period=r["period"],
                         rotation=r["rotation"], shuffle=-1, twin_id=r["item_id"],
                         n_chars=len(r["text"]), units=json.dumps(r["boundary_units"]),
                         **{f"s_{k}": v for k, v in _flat(r).items()}))
    for line in (REPO / "data/stimuli/stimuli_period_shuffled.jsonl").open():
        r = json.loads(line)
        rows.append(dict(item_id=r["item_id"], arm=r["arm"], period=r["period"],
                         rotation=r["rotation"], shuffle=r["shuffle"],
                         twin_id=r["twin_id"], n_chars=len(r["text"]),
                         units=json.dumps(r["boundary_units"]),
                         **{f"s_{k}": v for k, v in _flat(r).items()}))
    return pd.DataFrame(rows)


def _flat(r: dict) -> dict:
    """The realised sequence statistics, for the arm-level report.

    Recomputed here for the published rows, which predate the statistics file
    and carry none, so that both arms are described by the same code rather than
    by a stored number on one side and a recomputation on the other.
    """
    st = r.get("seq_stats")
    if st is None:
        seq = r["boundary_units"]
        st = _seq_stats(seq, r["period"])
    return {k: v for k, v in st.items() if k != "rho_by_lag"}


def _seq_stats(seq: list, p: int) -> dict:
    """A local copy of the generator's statistics, for stimuli that predate it."""
    sys.path.insert(0, str(REPO / "data/stimuli"))
    from make_stimuli_period_shuffled import seq_stats  # noqa: PLC0415
    st = seq_stats(seq, p)
    return st


def exact_multiset(transcript: str, units_json: str) -> bool:
    """Every distinct requested word delivered exactly its requested number of
    times. Order-insensitive and stricter than the unbounded recount, which a
    transcript can satisfy by over-delivering one word and under-delivering
    another by the same amount."""
    toks = normalise(transcript if isinstance(transcript, str) else "")
    want = Counter(json.loads(units_json))
    return all(count_occurrences(toks, w) == c for w, c in want.items())


def prepare(csv: Path, stim: pd.DataFrame, *, keep_caps: bool) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(csv)
    d = d[d.item_id.astype(str).str.startswith("pd_")]
    d, drop = panel(d, period_arm=True, cap_hits=not keep_caps)
    d = d.merge(stim, on="item_id", how="inner", validate="many_to_one")
    d = d.assign(
        exact=(d.count_a == d.k),
        count_u=[count_unbounded(t, u) for t, u in zip(d.transcript, d.units)],
        exact_ms=[exact_multiset(t, u) for t, u in zip(d.transcript, d.units)],
    )
    return d.assign(exact_u=(d.count_u == d.k),
                    exact_both=(d.count_a == d.k) & (d.count_u == d.k)), drop


# --------------------------------------------------------------------------
def rate(d: pd.DataFrame, col: str, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = d[d.arm == arm]
    t = g.pivot_table(index="model", columns="period", values=col, aggfunc="mean")
    n = g.pivot_table(index="model", columns="period", values=col, aggfunc="size")
    return t.reindex(columns=PERIODS), n.reindex(columns=PERIODS)


def show(tables: dict[str, pd.DataFrame], ns: dict[str, pd.DataFrame], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'checkpoint':11s} " + "  ".join(
        f"{a:>22s}" for a in tables) + "      (n per cell)")
    print(f"  {'':11s} " + "  ".join(
        "  ".join(f"p={p:<2d}" for p in PERIODS).rjust(22) for a in tables))
    models = sorted(set().union(*[set(t.index) for t in tables.values()]))
    for m in models:
        cells = []
        for a, t in tables.items():
            if m in t.index:
                cells.append("  ".join(f"{100*t.loc[m, p]:4.0f}" for p in PERIODS).rjust(22))
            else:
                cells.append(" " * 22)
        nn = ",".join(str(int(ns[a].loc[m, p])) for a in tables for p in PERIODS
                      if m in ns[a].index)
        print(f"  {m:11s} " + "  ".join(cells) + f"      ({nn})")
    print(f"  {'MEAN':11s} " + "  ".join(
        "  ".join(f"{100*t[p].mean():4.1f}" for p in PERIODS).rjust(22)
        for t in tables.values()))


def recovery(t_shuf: pd.DataFrame, t_per: pd.DataFrame, label: str) -> dict:
    """R(p) per checkpoint and pooled, plus the placebo and its correction."""
    E_s, E_p = t_shuf.mean(), t_per.mean()
    D = {p: float(E_p[8] - E_p[p]) for p in PERIODS}
    Delta = {p: float(E_s[p] - E_p[p]) for p in PERIODS}
    R = {p: (float("nan") if abs(D[p]) < 1e-9 else Delta[p] / D[p]) for p in TEST_PERIODS}
    Rbar = float(np.nanmean([R[p] for p in TEST_PERIODS]))
    placebo = Delta[8]
    R_adj = {p: (float("nan") if abs(D[p]) < 1e-9 else (Delta[p] - placebo) / D[p])
             for p in TEST_PERIODS}
    Rbar_adj = float(np.nanmean([R_adj[p] for p in TEST_PERIODS]))

    per_ck = {}
    for m in t_shuf.index:
        if m not in t_per.index:
            continue
        dm = {p: float(t_per.loc[m, 8] - t_per.loc[m, p]) for p in PERIODS}
        per_ck[m] = {
            "E_per": {p: float(t_per.loc[m, p]) for p in PERIODS},
            "E_shuf": {p: float(t_shuf.loc[m, p]) for p in PERIODS},
            "delta": {p: float(t_shuf.loc[m, p] - t_per.loc[m, p]) for p in PERIODS},
            "R": {p: (None if abs(dm[p]) < 1e-9 else
                      round(float(t_shuf.loc[m, p] - t_per.loc[m, p]) / dm[p], 4))
                  for p in TEST_PERIODS},
        }
    print(f"\n  [{label}]")
    for p in PERIODS:
        tag = ("PLACEBO" if p == 8 else "")
        rv = "" if p == 8 else f"   R({p}) = {R[p]:+.3f}"
        print(f"    p={p}:  E_per={100*E_p[p]:5.1f}  E_shuf={100*E_s[p]:5.1f}  "
              f"Delta={100*Delta[p]:+6.1f} pts  D={100*D[p]:5.1f} pts{rv}  {tag}")
    print(f"    Rbar = {Rbar:+.3f}   (offset-corrected {Rbar_adj:+.3f}, "
          f"placebo Delta(8) = {100*placebo:+.1f} pts)")
    return dict(label=label, E_per={p: float(E_p[p]) for p in PERIODS},
                E_shuf={p: float(E_s[p]) for p in PERIODS},
                D=D, delta=Delta, R=R, Rbar=Rbar,
                placebo=placebo, R_adj=R_adj, Rbar_adj=Rbar_adj,
                per_checkpoint=per_ck)


def band(rbar: float, rmin: float) -> str:
    if rbar >= R_CONFIRM and rmin >= R_MIN:
        return "PERIODICITY CONFIRMED"
    if rbar <= R_LEXICAL:
        return "LEXICAL DIVERSITY, NOT PERIODICITY"
    return "INTERMEDIATE"


def verdict(rec_exact: dict, rec_u: dict) -> dict:
    """The pre-committed call, applied without adjustment."""
    rmin = min(rec_exact["R"][p] for p in TEST_PERIODS)
    call = band(rec_exact["Rbar"], rmin)
    # co-primary gate: the CONFIRMED label additionally requires the
    # order-insensitive statistic to move in the same direction at both rungs
    up_u = all(rec_u["delta"][p] > 0 for p in TEST_PERIODS)
    if call == "PERIODICITY CONFIRMED" and not up_u:
        call = "INTERMEDIATE"
        reason = ("Rbar cleared 0.50 under the ordered scan but the "
                  "order-insensitive recount does not agree in sign at both rungs")
    elif call == "LEXICAL DIVERSITY, NOT PERIODICITY" and rec_u["Rbar"] > R_LEXICAL:
        call = "INTERMEDIATE"
        reason = ("Rbar <= 0.20 under the ordered scan but not under the "
                  "order-insensitive recount; the two statistics disagree")
    else:
        reason = "pre-committed bands applied to Rbar and min R(p)"
    # placebo clause
    placebo_applied = rec_exact["placebo"] <= PLACEBO_TRIGGER
    if placebo_applied:
        adj_call = band(rec_exact["Rbar_adj"],
                        min(rec_exact["R_adj"][p] for p in TEST_PERIODS))
        if adj_call != call:
            reason += (f"; placebo Delta(8) = {100*rec_exact['placebo']:+.1f} pts "
                       f"triggered the offset correction, which lands in "
                       f"'{adj_call}' -- pre-committed to INTERMEDIATE on disagreement")
            call = "INTERMEDIATE"
    sign_split = (np.sign(rec_exact["R"][2]) != np.sign(rec_exact["R"][4]))
    if sign_split and call != "INTERMEDIATE":
        call = "INTERMEDIATE"
        reason += "; R(2) and R(4) disagree in sign"
    # A description, not a threshold. The pre-committed LEXICAL band is
    # "Rbar <= 0.20", which a large NEGATIVE Rbar also satisfies -- but
    # "shuffling the same words made the model materially worse" is a third
    # thing, not the reviewer's lexical-diversity reading, and calling it
    # LEXICAL without saying so would be misleading. The label is left exactly
    # as pre-committed; this flag rides alongside it.
    reversal = bool(rec_exact["Rbar"] < -R_LEXICAL)
    if reversal:
        reason += ("; NOTE Rbar is strongly NEGATIVE -- the shuffled arm is worse "
                   "than its periodic twin, which the pre-committed LEXICAL band "
                   "formally covers but does not describe. Read the placebo and "
                   "the adjacent-repeat rates before interpreting")
    return dict(outcome=call, reason=reason, Rbar_exact=rec_exact["Rbar"],
                Rbar_exact_u=rec_u["Rbar"], min_R=rmin,
                placebo_applied=bool(placebo_applied), reversal=reversal,
                Rbar_adj=rec_exact["Rbar_adj"])


# --------------------------------------------------------------------------
def leave_one_out(d: pd.DataFrame, col: str) -> dict:
    out = {}
    models = sorted(d.model.unique())
    for drop_m in models:
        g = d[d.model != drop_m]
        ts, _ = rate(g, col, "shuffled")
        tp, _ = rate(g, col, "periodic")
        E_s, E_p = ts.mean(), tp.mean()
        D = {p: float(E_p[8] - E_p[p]) for p in PERIODS}
        R = {p: (None if abs(D[p]) < 1e-9 else float(E_s[p] - E_p[p]) / D[p])
             for p in TEST_PERIODS}
        rb = float(np.nanmean([v for v in R.values() if v is not None]))
        out[f"without_{drop_m}"] = dict(R=R, Rbar=rb,
                                        band=band(rb, min(v for v in R.values())))
    print("\n--- leave one checkpoint out (does the call rest on one model?) ---")
    for k, v in out.items():
        print(f"  {k:20s} Rbar={v['Rbar']:+.3f}  R(2)={v['R'][2]:+.3f} "
              f"R(4)={v['R'][4]:+.3f}   {v['band']}")
    return out


def paired(d: pd.DataFrame, col: str) -> dict:
    """The gap differenced inside each (model, template, rotation, seed) cell.

    The pooled means above treat every generation as exchangeable, but a
    shuffled twin and its periodic control share a carrier, a rotation, a word
    multiset and a seed, so the paired difference removes all of that. The three
    shuffles of one twin are averaged first, so a twin contributes one number
    however many shuffles it has.
    """
    keys = ["model", "template", "rotation", "period", "seed"]
    s = d[d.arm == "shuffled"].groupby(keys, observed=True)[col].mean()
    p = d[d.arm == "periodic"].groupby(keys, observed=True)[col].mean()
    j = pd.concat({"shuf": s, "per": p}, axis=1).dropna()
    out = {}
    print("\n--- paired within (model, template, rotation, seed) ---")
    for per, g in j.groupby(level="period"):
        diff = g["shuf"] - g["per"]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        out[int(per)] = dict(n_pairs=int(len(diff)), mean_diff=float(diff.mean()),
                             se=se, frac_positive=float((diff > 0).mean()),
                             frac_zero=float((diff == 0).mean()))
        print(f"  p={per}: n={len(diff):3d} pairs  mean(shuf - per) = "
              f"{100*diff.mean():+6.1f} pts (SE {100*se:.1f})  "
              f"{100*(diff > 0).mean():.0f}% of pairs positive, "
              f"{100*(diff == 0).mean():.0f}% tied")
    return out


def shuffle_spread(d: pd.DataFrame, col: str) -> dict:
    """Is the gap a property of periodicity or of one lucky permutation?"""
    g = d[d.arm == "shuffled"]
    out = {}
    print("\n--- spread across the three shuffles (a large spread means the "
          "result is a permutation, not a period) ---")
    for per, gp in g.groupby("period"):
        by_s = gp.groupby(["model", "shuffle"])[col].mean().unstack()
        pooled = by_s.mean()
        # within-twin spread: sd of the item-level rate across the 3 shuffles
        item = gp.groupby(["model", "twin_id", "shuffle"])[col].mean().unstack()
        within = item.std(axis=1, ddof=1).mean()
        out[int(per)] = dict(
            pooled_by_shuffle={int(k): float(v) for k, v in pooled.items()},
            range_pooled=float(pooled.max() - pooled.min()),
            mean_within_twin_sd=float(within),
            by_model_shuffle=json.loads(by_s.to_json()))
        print(f"  p={per}: pooled per shuffle "
              f"{'  '.join(f'{100*v:.1f}' for v in pooled)}  "
              f"(range {100*(pooled.max()-pooled.min()):.1f} pts)   "
              f"mean within-twin sd across shuffles {100*within:.1f} pts")
    return out


def p2_is_uninformative(stim: pd.DataFrame) -> dict:
    """Why the p=2 rung cannot test periodicity, as a counting argument.

    A binary string of 12 A and 12 B with no two adjacent symbols equal is
    determined by its first symbol -- once you forbid a repeat, every position
    is forced -- so there are exactly TWO such strings, ABAB...  and BABA...,
    and both ARE the period-2 periodic item (one is a rotation of the other).

    Therefore at p=2, "aperiodic" and "no adjacent repeats" are mutually
    exclusive by construction. Any shuffled twin of a p=2 item necessarily
    reintroduces adjacent verbatim repetition -- the local period-1 structure
    the ladder identifies as the most damaging condition of all (E(1) = 8.9% at
    k=24). R(2) therefore measures adjacency and periodicity confounded
    together, in that order of magnitude, and cannot be read as evidence about
    periodicity either way.

    The direction of the confound was pre-committed in
    `make_stimuli_period_shuffled.py` before any audio existed ("it pushes the
    shuffled p=2 arm down, toward the p=1 arm, so it can only suppress R(2),
    never manufacture it"). What was NOT anticipated is how large the forced
    adjacency is: the realised rate is reported here against the floor.

    p=4 and p=8 are unaffected -- there the multiset admits arrangements with
    exactly zero adjacent repeats, which is what the constructor enforces, so
    those rungs match their periodic twins on adjacency exactly and are the
    only rungs at which the experiment's question can be asked.
    """
    out = {}
    for p in PERIODS:
        g = stim[(stim.arm == "shuffled") & (stim.period == p)]
        gp = stim[(stim.arm == "periodic") & (stim.period == p)]
        zero_possible = p >= 4
        out[p] = dict(
            zero_adjacency_possible=zero_possible,
            shuffled_adjacent_repeat_rate=float(g.s_adjacent_repeat_rate.mean()),
            periodic_adjacent_repeat_rate=float(gp.s_adjacent_repeat_rate.mean()),
            informative=bool(zero_possible))
    print("\n--- can the shuffle match the periodic arm's zero adjacency? ---")
    for p, v in out.items():
        print(f"  p={p}: zero adjacency achievable = {str(v['zero_adjacency_possible']):5s}   "
              f"realised adj-rate  periodic {v['periodic_adjacent_repeat_rate']:.3f} "
              f"vs shuffled {v['shuffled_adjacent_repeat_rate']:.3f}   -> rung is "
              f"{'INFORMATIVE' if v['informative'] else 'UNINFORMATIVE BY CONSTRUCTION'}")
    print("  At p=2 the only zero-adjacency arrangement of 12+12 IS the alternation,")
    print("  i.e. the periodic item itself, so aperiodicity forces adjacent repeats and")
    print("  R(2) confounds them. The clean rungs are p=4 and p=8.")
    return out


def delta_ci(d: pd.DataFrame, col: str, *, n_boot: int = 4000, seed: int = 0) -> dict:
    """Template-cluster bootstrap CI on Delta(p) per checkpoint and pooled.

    The pooled p=4 figure is a mean over four checkpoints that do not agree, and
    a mean with no interval on its parts cannot be told apart from noise. One
    interval per (checkpoint, period) is what decides whether the split is real.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    print(f"\n--- Delta(p) per checkpoint with 95% template-cluster CI ({col}) ---")
    print(f"  {'checkpoint':10s} {'p':>2s} {'n_per':>6s} {'n_shuf':>7s} {'E_per':>7s} "
          f"{'E_shuf':>7s} {'Delta':>7s}  95% CI")
    for m in sorted(d.model.unique()):
        for p in PERIODS:
            g = d[(d.model == m) & (d.period == p)]
            gp, gs = g[g.arm == "periodic"], g[g.arm == "shuffled"]
            if not len(gp) or not len(gs):
                continue
            tm = sorted(g.template.unique())
            boot = []
            for _ in range(n_boot):
                pick = rng.choice(len(tm), len(tm), True)
                sub = pd.concat([g[g.template == tm[i]] for i in pick])
                a = sub[sub.arm == "shuffled"][col].mean()
                b = sub[sub.arm == "periodic"][col].mean()
                if np.isfinite(a) and np.isfinite(b):
                    boot.append(a - b)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            dl = float(gs[col].mean() - gp[col].mean())
            sig = "" if lo <= 0 <= hi else "  <- excludes 0"
            out[f"{m}_p{p}"] = dict(n_per=int(len(gp)), n_shuf=int(len(gs)),
                                    E_per=float(gp[col].mean()), E_shuf=float(gs[col].mean()),
                                    delta=dl, ci_lo=float(lo), ci_hi=float(hi),
                                    excludes_zero=bool(not (lo <= 0 <= hi)))
            print(f"  {m:10s} {p:2d} {len(gp):6d} {len(gs):7d} {100*gp[col].mean():7.1f} "
                  f"{100*gs[col].mean():7.1f} {100*dl:+7.1f}  "
                  f"[{100*lo:+6.1f},{100*hi:+6.1f}]{sig}")
    return out


def audio_health(csv: Path, stim: pd.DataFrame) -> dict:
    """Empty / degenerate / cap-hit rates on the RAW rows, before any filtering.

    `population.panel()` removes degenerate audio and cap hits before any rate
    is computed, which is correct but also hides the question "is this
    checkpoint broken on this arm?". A checkpoint whose shuffled exact rate
    collapses while its audio is clean, the right length and the right number of
    words is failing to *count*, which is data; one whose audio is empty or
    degenerate is failing to *speak*, which is not.
    """
    d = pd.read_csv(csv)
    d = d[d.item_id.astype(str).str.startswith("pd_")].merge(stim, on="item_id", how="inner")
    d = d[(d.k == K) & (d.arm.isin(["periodic", "shuffled"]))]
    out = {}
    print("\n--- audio health on RAW rows (is a collapsed cell broken, or just wrong?) ---")
    print(f"  {'checkpoint':10s} {'p':>2s} {'arm':9s} {'n':>4s} {'empty':>6s} {'degen':>6s} "
          f"{'cap':>6s} {'dur_s':>7s} {'n_words':>8s} {'count_a':>8s}")
    for m in sorted(d.model.unique()):
        for p in PERIODS:
            for a in ("periodic", "shuffled"):
                g = d[(d.model == m) & (d.period == p) & (d.arm == a)]
                if not len(g):
                    continue
                rec = dict(n=int(len(g)),
                           empty=float((g.outcome == "empty").mean()),
                           degenerate=float((g.outcome == "degenerate").mean()),
                           cap=float(g.hit_cap.mean()),
                           duration_s=float(g.duration_s.mean()),
                           n_words=float(g.n_words.mean()),
                           count_a=float(g.count_a.mean()))
                out[f"{m}_p{p}_{a}"] = rec
                print(f"  {m:10s} {p:2d} {a:9s} {len(g):4d} {100*rec['empty']:5.1f}% "
                      f"{100*rec['degenerate']:5.1f}% {100*rec['cap']:5.1f}% "
                      f"{rec['duration_s']:7.2f} {rec['n_words']:8.1f} {rec['count_a']:8.2f}")
    return out


def two_factor(d: pd.DataFrame, col: str) -> dict:
    """Every arm at k=24 laid out against the two text variables that survive.

    The experiment was built to separate *periodicity* from *lexical diversity*
    at fixed multiset. The answer it returns is easiest to read as a table of
    every arm in the study against the two properties of its text that are not
    periodicity: how many distinct filler types it contains, and what fraction
    of adjacent filler pairs are the same word. Periodicity is the third column
    and is what the shuffle sets to ~0 while holding the other two.

    This table is descriptive. It is not the pre-committed test and it was
    written after the result was seen -- which is stated here rather than left
    for a reader to work out, because a two-factor reading assembled after the
    fact is a hypothesis, not a finding.
    """
    rows = []
    for (arm, per), g in d.groupby(["arm", "period"], observed=True):
        if arm == "periodic_pub":
            continue          # identical to `periodic`; would double every row
        rows.append(dict(
            arm=("periodic" if arm in ("periodic", "context") else arm),
            period=int(per), n_distinct=int(round(g.s_ttr.mean() * K)),
            adj_rate=float(g.s_adjacent_repeat_rate.mean()),
            rho_at_p=float(g.s_rho_at_p.mean()) if g.s_rho_at_p.notna().all() else None,
            E=float(g[col].mean()), n=int(len(g)),
            by_model={m: round(float(gg[col].mean()), 4)
                      for m, gg in g.groupby("model", observed=True)}))
    rows.sort(key=lambda r: (r["n_distinct"], r["adj_rate"]))
    print(f"\n--- every arm at k={K} against the two non-periodicity variables "
          f"({col}; descriptive, written after the result) ---")
    print(f"  {'arm':10s} {'p':>3s} {'types':>6s} {'adj-rate':>9s} {'rho(p)':>7s} "
          f"{'E':>6s} {'n':>5s}   per checkpoint")
    for r in rows:
        rp = "  n/a " if r["rho_at_p"] is None else f"{r['rho_at_p']:6.3f}"
        bm = " ".join(f"{m}={100*v:.0f}" for m, v in sorted(r["by_model"].items()))
        print(f"  {r['arm']:10s} {r['period']:3d} {r['n_distinct']:6d} "
              f"{r['adj_rate']:9.3f} {rp} {100*r['E']:6.1f} {r['n']:5d}   {bm}")
    return rows


def per_checkpoint_recovery(d: pd.DataFrame, col: str) -> dict:
    """R(p) computed inside each checkpoint, where the pooled mean hides a split.

    Each checkpoint has its own ceiling E(8), so its own D(p). A checkpoint
    whose D(p) is small has an unstable R and is flagged rather than quoted.
    """
    ts, _ = rate(d, col, "shuffled")
    tp, _ = rate(d, col, "periodic")
    out = {}
    print(f"\n--- recovery fraction inside each checkpoint ({col}) ---")
    print(f"  {'checkpoint':11s} " + "  ".join(
        f"p={p}: E_per E_shuf  Delta   D    R" for p in TEST_PERIODS))
    for m in ts.index:
        bits, rec = [], {}
        for p in TEST_PERIODS:
            ep, es = float(tp.loc[m, p]), float(ts.loc[m, p])
            dd, dl = float(tp.loc[m, 8] - ep), es - ep
            r = None if abs(dd) < 1e-9 else dl / dd
            thin = abs(dd) < 0.10
            rec[p] = dict(E_per=ep, E_shuf=es, delta=dl, D=dd,
                          R=(None if r is None else round(r, 3)), thin_denominator=thin)
            bits.append(f"      {100*ep:5.1f} {100*es:5.1f} {100*dl:+6.1f} "
                        f"{100*dd:5.1f} {('   n/a' if r is None else f'{r:+6.2f}')}"
                        f"{'*' if thin else ' '}")
        out[m] = rec
        print(f"  {m:11s} " + "".join(bits))
    print("  * denominator D(p) < 10 pts: that checkpoint has almost no deficit "
          "to recover at this rung, so its R is unstable and should not be quoted.")
    return out


def replication(d: pd.DataFrame, col: str) -> dict:
    """Does the re-rendered periodic control reproduce the published rows?

    If it does not, the published ladder and this run are not measuring the same
    pipeline, which is the specific risk the `coqui` transformers bump created.
    """
    tz, nz = rate(d, col, "periodic")
    tp, np_ = rate(d, col, "periodic_pub")
    print("\n--- re-rendered periodic control vs the published ladder rows ---")
    print(f"  {'checkpoint':11s} " + "  ".join(f"p={p:<2d} pub  rr   diff" for p in PERIODS))
    rows = {}
    for m in tz.index:
        bits, rec = [], {}
        for p in PERIODS:
            a, b = float(tp.loc[m, p]), float(tz.loc[m, p])
            bits.append(f"     {100*a:4.0f} {100*b:4.0f} {100*(b-a):+5.0f}")
            rec[p] = dict(published=a, rerendered=b, diff=b - a)
        rows[m] = rec
        print(f"  {m:11s} " + "".join(bits))
    pooled = {p: dict(published=float(tp[p].mean()), rerendered=float(tz[p].mean()),
                      diff=float(tz[p].mean() - tp[p].mean())) for p in PERIODS}
    worst = max(abs(v["diff"]) for v in pooled.values())
    ok = worst <= REPLICATION_TOL
    print(f"  pooled diffs " + "  ".join(f"p={p}: {100*pooled[p]['diff']:+.1f}"
                                         for p in PERIODS)
          + f"   -> {'REPLICATES' if ok else 'DOES NOT REPLICATE'} "
            f"(tolerance {100*REPLICATION_TOL:.0f} pts)")
    return dict(per_checkpoint=rows, pooled=pooled, worst_abs_diff=worst,
                replicates=bool(ok), tolerance=REPLICATION_TOL)


def resume_check(side: Path, published: Path) -> dict:
    """Did the generators resume, or did they re-roll the published rows?

    The generators key on (item_id, seed), so passing an extended stimulus file
    should leave every pre-existing row byte-identical. That is load-bearing for
    comparability and is checked rather than assumed.
    """
    a = pd.read_csv(side)
    b = pd.read_csv(published)
    key = ["model", "item_id", "seed"]
    a = a[a.item_id.astype(str).str.startswith("pd_")][key + ["count_a", "duration_s"]]
    b = b[b.item_id.astype(str).str.startswith("pd_")][key + ["count_a", "duration_s"]]
    j = a.merge(b, on=key, how="inner", suffixes=("_new", "_pub"))
    same_a = int((j.count_a_new == j.count_a_pub).sum())
    same_d = int(np.isclose(j.duration_s_new, j.duration_s_pub, atol=1e-6).sum())
    print(f"\n--- resume check: {len(j)} pre-existing pd_ rows shared with "
          f"behavioural_period.csv")
    print(f"  count_a identical on {same_a}/{len(j)}   "
          f"duration identical on {same_d}/{len(j)}")
    bad = j[j.count_a_new != j.count_a_pub]
    if len(bad):
        print("  CHANGED ROWS (the resume did not hold):")
        print(bad.head(10).to_string(index=False))
    return dict(n_shared=int(len(j)), count_a_identical=same_a,
                duration_identical=same_d, held=bool(same_a == len(j)))


def hygiene(d_nocaps: pd.DataFrame, d_caps: pd.DataFrame) -> dict:
    """Cap-hit and degeneracy rates per arm, and the acoustic covariates.

    A cap-hit column of exact zeros everywhere is a broken flag, not a result,
    and two arms with different censoring are not comparable however matched
    their text is.
    """
    out = {}
    cap = d_caps.assign(cap=~d_caps.set_index(["model", "item_id", "seed"]).index.isin(
        d_nocaps.set_index(["model", "item_id", "seed"]).index))
    t = cap.pivot_table(index=["model", "arm"], columns="period", values="cap",
                        aggfunc="mean")
    print("\n--- cap-hit rate by (checkpoint, arm, period) ---")
    print(t.round(3).to_string())
    out["cap_hit_rate"] = json.loads(t.to_json())
    out["cap_hit_all_zero"] = bool(np.nansum(t.values) == 0)
    for col in ("duration_s", "n_words", "count_a", "count_u"):
        if col not in d_nocaps.columns:
            continue
        tt = d_nocaps.pivot_table(index=["model", "arm"], columns="period", values=col,
                                  aggfunc="mean")
        print(f"\n--- mean {col} by (checkpoint, arm, period) ---")
        print(tt.round(2).to_string())
        out[col] = json.loads(tt.to_json())
    # Failure *direction*, not only failure rate. If the shuffled arm fails
    # where the periodic arm fails but by overcounting rather than
    # undercounting, "the deficit moved" means something different from "the
    # deficit shrank", and the exact rate alone cannot tell them apart.
    if "outcome" in d_nocaps.columns:
        oc = (d_nocaps.groupby(["arm", "period"]).outcome
              .value_counts(normalize=True).unstack().fillna(0))
        print("\n--- outcome mix by (arm, period) ---")
        print(oc.round(3).to_string())
        out["outcome_mix"] = json.loads(oc.to_json())
    return out


def sequence_report(stim: pd.DataFrame) -> dict:
    """The realised periodicity of the two arms, so the label can be checked."""
    cols = [c for c in stim.columns if c.startswith("s_")]
    g = stim.groupby(["arm", "period"])[cols].agg(["mean", "min", "max"])
    print("\n--- realised sequence statistics (the shuffled arm's aperiodicity, "
          "verifiable rather than asserted) ---")
    keep = ["s_rho_at_p", "s_periodicity_index", "s_adjacent_repeat_rate",
            "s_longest_periodic_window", "s_lz_norm", "s_ttr",
            "s_unigram_entropy", "s_bigram_entropy"]
    hdr = {"s_rho_at_p": "rho(p)", "s_periodicity_index": "PI",
           "s_adjacent_repeat_rate": "adj-rate", "s_longest_periodic_window": "maxper",
           "s_lz_norm": "LZnorm", "s_ttr": "TTR",
           "s_unigram_entropy": "H1", "s_bigram_entropy": "H2|1"}
    print(f"  {'arm':14s} {'p':>2s} " + " ".join(f"{hdr[c]:>22s}" for c in keep))
    for (arm, per), row in g.iterrows():
        cells = " ".join(
            f"{row[(c, 'mean')]:8.3f} [{row[(c, 'min')]:.2f},{row[(c, 'max')]:.2f}]".rjust(22)
            for c in keep)
        print(f"  {arm:14s} {per:2d} {cells}")
    return json.loads(g.to_json())


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural",
                    default="data/results/behavioural_period_shuffled.csv")
    ap.add_argument("--published", default="data/results/behavioural_period.csv")
    ap.add_argument("--out", default="data/results/period_shuffled.json")
    args = ap.parse_args()

    stim = load_stimuli()
    csv = REPO / args.behavioural
    d, drop = prepare(csv, stim, keep_caps=False)
    d_caps, _ = prepare(csv, stim, keep_caps=True)
    d = d[d.k == K]
    d_caps = d_caps[d_caps.k == K]
    print(describe(drop))
    print(f"rows at k={K}: " + ", ".join(
        f"{a}={int((d.arm == a).sum())}" for a in ("periodic_pub", "periodic", "shuffled")))

    res: dict = {"k": K, "periods": PERIODS,
                 "precommitment": {
                     "statistic": "exact rate (ordered scan) primary; unbounded "
                                  "order-insensitive recount co-primary",
                     "R_confirm": R_CONFIRM, "R_lexical": R_LEXICAL,
                     "R_min_per_rung": R_MIN,
                     "placebo_trigger_points": PLACEBO_TRIGGER,
                     "baseline": "re-rendered periodic control (primary); "
                                 "published ladder rows (reported beside it)",
                     "source": "data/stimuli/make_stimuli_period_shuffled.py docstring"},
                 "population": drop}

    res["resume_check"] = resume_check(csv, REPO / args.published)
    res["sequences"] = sequence_report(stim)

    res["rates"], res["n"], res["recovery"], res["verdict_by_rule"] = {}, {}, {}, {}
    for col, title in [("exact", "ordered scan (the published rule) -- PRIMARY"),
                       ("exact_u", "unbounded recount, order-insensitive -- CO-PRIMARY"),
                       ("exact_both", "strict: in order AND no extras"),
                       ("exact_ms", "exact multiset: every word its exact count")]:
        tables, ns = {}, {}
        for a in ("periodic_pub", "periodic", "shuffled"):
            tables[a], ns[a] = rate(d, col, a)
        show(tables, ns, f"{col} rate at k={K} -- {title}")
        res["rates"][col] = {a: json.loads(t.to_json()) for a, t in tables.items()}
        res["n"][col] = {a: json.loads(n.to_json()) for a, n in ns.items()}
        res["recovery"][col] = {
            "vs_rerendered": recovery(tables["shuffled"], tables["periodic"],
                                      f"{col} vs re-rendered periodic (PRIMARY)"),
            "vs_published": recovery(tables["shuffled"], tables["periodic_pub"],
                                     f"{col} vs published periodic"),
        }

    # Cap hits are excluded by `population.panel()` because they are censored
    # downward by our own token budget. But if the two arms hit the cap at
    # different rates -- and the shuffled arm's generations run visibly longer --
    # then excluding them removes a different slice of each arm, in the
    # direction that flatters the arm which loops more. So the whole comparison
    # is repeated with cap hits KEPT, which is the censored lower bound on the
    # shuffled arm. This is a robustness read, not a change to the pre-committed
    # test; if it lands in a different band that is reported as a weakening.
    tk = {a: rate(d_caps, "exact", a)[0] for a in ("periodic", "shuffled")}
    res["recovery"]["exact_caps_kept"] = {"vs_rerendered": recovery(
        tk["shuffled"], tk["periodic"], "exact vs re-rendered, CAP HITS KEPT")}
    tku = {a: rate(d_caps, "exact_u", a)[0] for a in ("periodic", "shuffled")}
    res["recovery"]["exact_u_caps_kept"] = {"vs_rerendered": recovery(
        tku["shuffled"], tku["periodic"], "exact_u vs re-rendered, CAP HITS KEPT")}
    res["verdict_caps_kept"] = verdict(
        res["recovery"]["exact_caps_kept"]["vs_rerendered"],
        res["recovery"]["exact_u_caps_kept"]["vs_rerendered"])

    res["replication"] = replication(d, "exact")
    res["p2_uninformative"] = p2_is_uninformative(stim)
    res["audio_health"] = audio_health(csv, stim)
    res["delta_ci"] = {c: delta_ci(d, c) for c in ("exact", "exact_u")}
    res["two_factor"] = {c: two_factor(d, c) for c in ("exact", "exact_u")}
    # The same pooled test restricted to the rungs where the manipulation is
    # clean -- p=4 and p=8 match their twins on adjacency exactly, p=2 cannot.
    # Reported, not substituted for the pre-committed test, which named p=2 and
    # p=4 as the test rungs before any of this was known.
    ts4, tp4 = rate(d, "exact", "shuffled"), rate(d, "exact", "periodic")
    res["clean_rung_only"] = dict(
        rung=4, note="p=4: adjacency 0 in both arms, multiset identical item by item",
        E_per=float(tp4[0][4].mean()), E_shuf=float(ts4[0][4].mean()),
        delta=float(ts4[0][4].mean() - tp4[0][4].mean()),
        D=float(tp4[0][8].mean() - tp4[0][4].mean()),
        R=float((ts4[0][4].mean() - tp4[0][4].mean())
                / (tp4[0][8].mean() - tp4[0][4].mean())))
    res["per_checkpoint_recovery"] = {c: per_checkpoint_recovery(d, c)
                                      for c in ("exact", "exact_u")}
    res["verdict"] = verdict(res["recovery"]["exact"]["vs_rerendered"],
                             res["recovery"]["exact_u"]["vs_rerendered"])
    res["verdict_vs_published"] = verdict(res["recovery"]["exact"]["vs_published"],
                                          res["recovery"]["exact_u"]["vs_published"])
    res["leave_one_out"] = leave_one_out(d, "exact")
    res["paired"] = {c: paired(d, c) for c in ("exact", "exact_u")}
    res["shuffle_spread"] = {c: shuffle_spread(d, c) for c in ("exact", "exact_u")}
    res["bootstrap"] = bootstrap(d, "exact")
    res["hygiene"] = hygiene(d, d_caps)

    v = res["verdict"]
    print("\n" + "=" * 78)
    print(f"=== {v['outcome']}")
    print(f"    Rbar (ordered scan) = {v['Rbar_exact']:+.3f}   "
          f"Rbar (order-insensitive) = {v['Rbar_exact_u']:+.3f}   "
          f"min R(p) = {v['min_R']:+.3f}")
    print(f"    {v['reason']}")
    print(f"    against the published periodic rows instead: "
          f"{res['verdict_vs_published']['outcome']} "
          f"(Rbar {res['verdict_vs_published']['Rbar_exact']:+.3f})")
    print(f"    with cap hits kept (censored bound): "
          f"{res['verdict_caps_kept']['outcome']} "
          f"(Rbar {res['verdict_caps_kept']['Rbar_exact']:+.3f})")
    print("=" * 78)

    out = REPO / args.out
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}")


def bootstrap(d: pd.DataFrame, col: str, *, n_boot: int = 4000,
              seed: int = 0) -> dict:
    """Cluster bootstrap over templates for Rbar.

    Templates, not rows: the periods and the arms inside one template share a
    carrier sentence and are not independent. Five clusters is few, so the
    interval is wide and is reported as such rather than quoted as a test.
    """
    rng = np.random.default_rng(seed)
    tmpls = sorted(d.template.unique())
    idx = {t: d[d.template == t] for t in tmpls}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(tmpls), size=len(tmpls), replace=True)
        g = pd.concat([idx[tmpls[i]] for i in pick])
        ts, tp = rate(g, col, "shuffled")[0], rate(g, col, "periodic")[0]
        E_s, E_p = ts.mean(), tp.mean()
        rs = []
        for p in TEST_PERIODS:
            den = E_p[8] - E_p[p]
            if abs(den) > 1e-9:
                rs.append((E_s[p] - E_p[p]) / den)
        if rs and np.all(np.isfinite(rs)):
            vals.append(float(np.mean(rs)))
    a = np.array(vals)
    lo, hi = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (
        float("nan"), float("nan"))
    print(f"\n--- cluster bootstrap over the 5 templates: Rbar 95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]  (median {np.median(a):+.3f}, "
          f"P[Rbar>=0.50] = {float((a >= R_CONFIRM).mean()):.3f}, "
          f"P[Rbar<=0.20] = {float((a <= R_LEXICAL).mean()):.3f})")
    return dict(n_boot=len(a), ci_lo=lo, ci_hi=hi, median=float(np.median(a)),
                p_confirm=float((a >= R_CONFIRM).mean()),
                p_lexical=float((a <= R_LEXICAL).mean()))


if __name__ == "__main__":
    main()
