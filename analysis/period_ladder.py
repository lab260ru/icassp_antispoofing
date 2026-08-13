#!/usr/bin/env python3
"""Is it the period, or is it the same token again?

The paper is titled "Periodicity, Not Length". Its design never separated
periodicity from verbatim token identity: the repeated arm is one word k times
(period 1 *and* verbatim), the controls are either period-8 or aperiodic, and
nothing between was ever rendered. `data/stimuli/make_stimuli_period.py` builds
the missing rungs -- p in {1, 2, 4, 8, k} at k in {16, 24, 32}, same carrier,
same word count, same seeds, vocabulary balanced by rotation across p in
{2,4,8} -- and this file reads them.

WHAT SHAPE MEANS WHAT  (pre-committed, restated from the generator; the
generator's docstring was written and committed before any audio existed, and
this file was written while the first generation was still running)
--------------------------------------------------------------------------
With E(p) the exact rate (all k requested units present in the transcript --
the paper's own statistic) and D(p) = E(8) - E(p) the deficit against the
period-8 anchor:

  * D(2) >= 0.50 * D(1)   -> PERIODICITY. Alternating two distinct tokens still
                             loses most of what repeating one loses, so the
                             period is doing the work. Title stands.
  * D(2) <= 0.25 * D(1)   -> VERBATIM TOKEN IDENTITY. The deficit is gone the
                             moment no token repeats adjacently-in-cycle;
                             "periodicity" is the wrong word and the paper must
                             be retitled and its claim rescoped.
  * in between            -> GRADED. A minority periodicity residue on a
                             verbatim-dominated effect. This still fails the
                             reviewer's bar ("the deficit still appears at the
                             same period length"), so it still forces a retitle,
                             with "verbatim repetition, with a weaker penalty at
                             short periods" as the honest phrase.

Monotonicity is the sanity condition: D(4) > D(2) means something other than
period is moving and the ladder should not be read until that is explained.

THE FOUR WAYS THIS COULD BE A LIE, AND WHAT IS DONE ABOUT EACH
--------------------------------------------------------------
1. **Overcount blindness.** `score_counts.py` counts an all-identical unit list
   with an unbounded occurrence count and a mixed one with a monotone scan that
   cannot exceed k. A model that loops is therefore an overcount at p=1 and
   invisible at p>1 -- which would manufacture exactly the collapse the
   verbatim reading predicts, out of nothing but the scoring rule. So every arm
   is *recounted* here from the stored transcript under one unbounded rule
   (`count_unbounded`: total occurrences of every distinct unit, summed), which
   reduces to `count_occurrences` at p=1 and can exceed k at every p. Both
   columns are reported. If the collapse exists only under the capped rule, the
   ladder is uninterpretable and this file says so.
2. **Vocabulary.** p in {2,4,8} are pooled over all rotations of one 8-word
   pool, so they use the same words the same number of times and cannot differ
   lexically. p=1 (the target word alone) and p=k (146-word pool, tail delivery
   0.79-0.85) are inherited confounds; both are reported, and the verdict is
   computed from the vocabulary-matched interior only.
3. **Censoring.** Cap hits are censored downward and are excluded by
   `population.panel()`. They are not uniform across arms -- the p=1 arm loops
   and so hits the budget far more often -- and excluding them removes the p=1
   arm's *worst* items, which flatters p=1 and so works against the verbatim
   reading. Per-arm cap rates are reported, and the verdict is recomputed with
   cap hits kept (counted as failures) as a bound in the other direction.
4. **Pipeline drift.** The p=1, p=8 and p=k rungs are character-identical
   re-renders of published `wr_*`, `ct_*` and `ap_*` items on the same seeds.
   `replication()` puts today's numbers beside the published ones. If they
   disagree, the ladder is measuring a changed pipeline and not a period.

Usage:  python analysis/period_ladder.py
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
from src.common.population import panel, describe, cap_flags  # noqa: E402
from src.common.score_counts import normalise, count_occurrences  # noqa: E402

FAMILY = {"llasa1b": "Llasa", "llasa3b": "Llasa", "llasa8b": "Llasa",
          "xtts2": "XTTS", "qwen06b": "Qwen3-TTS", "qwen17b": "Qwen3-TTS"}


def load_stimuli(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.open()]
    return pd.DataFrame([
        dict(item_id=r["item_id"], period=r["period"], rotation=r["rotation"],
             pool_id=r["pool_id"], n_distinct=r["n_distinct"],
             # Word count is held fixed across the ladder by construction;
             # *character* count is not, because "very" is four letters and
             # "extremely" is nine. That is not cosmetic for XTTS-v2, which
             # warns above 250 characters and may truncate: the high-p arms are
             # the long ones, so any character-length effect pushes them
             # *down*, toward the p=1 arm, and so works against a finding that
             # the deficit collapses at p>1. Carried here so the direction can
             # be checked rather than assumed.
             n_chars=len(r["text"]),
             units=json.dumps(r["boundary_units"]))
        for r in rows])


def count_unbounded(transcript: str, units_json: str) -> int:
    """Total occurrences of every distinct requested unit, uncapped.

    The one counting rule that means the same thing at every period. At p=1 it
    is `count_occurrences` of the single word, which is exactly what
    `score_counts.py` already does for the repeated arm; at p>1 it is the sum
    over the p distinct words, which `score_counts.py` does *not* do -- it runs
    a monotone scan capped at k. A model that renders eighteen units when
    twelve were asked for reads as 18 here and as 12 there, so this column is
    the one in which "exact" means the same thing across the ladder.

    It is a looser criterion than the scan in one respect: it does not require
    the units to arrive in the requested order. That direction of looseness
    favours the high-p arms, so using it cannot manufacture a collapse at p=2;
    it can only fail to show one that the ordered scan would.
    """
    toks = normalise(transcript if isinstance(transcript, str) else "")
    units = json.loads(units_json)
    return sum(count_occurrences(toks, u) for u in dict.fromkeys(units))


def prepare(csv: Path, stim: pd.DataFrame, *, keep_caps: bool) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(csv)
    d = d[d.item_id.astype(str).str.startswith("pd_")]
    d, drop = panel(d, period_arm=True, cap_hits=not keep_caps)
    # Inner, not left. The disambiguator arm (`make_stimuli_disambig.py`) shares
    # the `pd_` prefix deliberately -- that is what puts it behind
    # `population.NON_PANEL_ITEM_PREFIXES` -- so the prefix filter above catches
    # its rows too, and only the period ladder's own stimulus file decides which
    # of them belong here. An outer/left join plus an assertion would turn the
    # arrival of the disambiguator audio into a crash in an unrelated analysis.
    n_pd = len(d)
    d = d.merge(stim, on="item_id", how="inner", validate="many_to_one")
    if n_pd != len(d):
        print(f"  ({n_pd - len(d)} `pd_` rows are not period-ladder items -- "
              "disambiguator arm, read by analysis/disambiguation.py)")
    d = d.assign(
        exact=(d.count_a == d.k),
        count_u=[count_unbounded(t, u) for t, u in zip(d.transcript, d.units)],
    )
    d = d.assign(exact_u=(d.count_u == d.k),
                 # The strict conjunction, and the one to quote: all k units
                 # delivered in the requested order (the scan) AND no extras
                 # (the unbounded recount). At p=1 the two rules coincide, so
                 # this column changes nothing about the repeated arm and can
                 # only take credit away from the arms above it -- which is the
                 # direction that could rescue the title, so it is the reading
                 # least favourable to a verdict of "verbatim".
                 exact_both=(d.count_a == d.k) & (d.count_u == d.k),
                 arm=np.where(d.pool_id == "ext146", "k", d.period.astype(int).astype(str)))
    return d, drop


ARM_ORDER = ["1", "2", "4", "8", "k"]


def rate_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    t = d.pivot_table(index="model", columns="arm", values=col, aggfunc="mean")
    n = d.pivot_table(index="model", columns="arm", values=col, aggfunc="size")
    t = t.reindex(columns=[a for a in ARM_ORDER if a in t.columns])
    n = n.reindex(columns=t.columns)
    return t, n


def show(t: pd.DataFrame, n: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    head = "  ".join(f"p={a:>2s}" for a in t.columns)
    print(f"  {'checkpoint':11s} {head}      (n per cell)")
    for m in t.index:
        cells = "  ".join(f"{100*t.loc[m, a]:5.1f}" for a in t.columns)
        ns = ",".join(str(int(n.loc[m, a])) for a in t.columns)
        print(f"  {m:11s} {cells}      ({ns})")
    means = "  ".join(f"{100*t[a].mean():5.1f}" for a in t.columns)
    print(f"  {'MEAN':11s} {means}")


def verdict(t: pd.DataFrame, label: str, res: dict) -> dict:
    """D(2)/D(1) against the pre-committed thresholds, per checkpoint and mean."""
    if not {"1", "2", "8"} <= set(t.columns):
        return {}
    d1 = t["8"] - t["1"]
    d2 = t["8"] - t["2"]
    d4 = (t["8"] - t["4"]) if "4" in t.columns else pd.Series(np.nan, index=t.index)
    ratio = d2 / d1.replace(0, np.nan)
    mean_ratio = float((d2.mean()) / d1.mean()) if d1.mean() else float("nan")
    call = ("PERIODICITY" if mean_ratio >= 0.50 else
            "VERBATIM TOKEN IDENTITY" if mean_ratio <= 0.25 else "GRADED")
    out = dict(
        label=label, D1=dict(zip(t.index, d1.round(4))), D2=dict(zip(t.index, d2.round(4))),
        D4=dict(zip(t.index, d4.round(4))),
        ratio_per_checkpoint={m: (None if not np.isfinite(v) else round(float(v), 4))
                              for m, v in ratio.items()},
        mean_D1=float(d1.mean()), mean_D2=float(d2.mean()), mean_D4=float(d4.mean()),
        mean_ratio=mean_ratio, monotone=bool(np.nanmean(d4) <= np.nanmean(d2) + 1e-9),
        verdict=call)
    print(f"\n  [{label}] mean D(1)={100*d1.mean():.1f} pts, D(2)={100*d2.mean():.1f} pts, "
          f"D(4)={100*np.nanmean(d4):.1f} pts   ->  D(2)/D(1) = {mean_ratio:.3f}   {call}")
    res.setdefault("verdicts", []).append(out)
    return out


def bootstrap_ratio(d: pd.DataFrame, col: str, res: dict, *, n_boot: int = 4000,
                    seed: int = 0) -> dict:
    """Cluster bootstrap CI for D(2)/D(1), resampling (template, k) cells.

    The unit of resampling is the *cell*, not the generation: the four p=2
    rotations, the three seeds and the two arms inside one (template, k) share
    a carrier sentence and are not independent draws. Checkpoints are too few
    to bootstrap over (three or four), so the checkpoint mean is recomputed
    inside each replicate and the interval is a statement about generalising
    across carriers and k, not across models.
    """
    rng = np.random.default_rng(seed)
    cells = sorted({(t, k) for t, k in zip(d.template, d.k)})
    idx = {c: d[(d.template == c[0]) & (d.k == c[1])] for c in cells}
    ratios, d1s, d2s = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(cells), len(cells))
        s = pd.concat([idx[cells[i]] for i in pick], ignore_index=True)
        t = s.pivot_table(index="model", columns="arm", values=col, aggfunc="mean")
        if not {"1", "2", "8"} <= set(t.columns):
            continue
        a, b = float((t["8"] - t["1"]).mean()), float((t["8"] - t["2"]).mean())
        d1s.append(a)
        d2s.append(b)
        if abs(a) > 1e-9:
            ratios.append(b / a)
    q = lambda v, p: float(np.quantile(v, p)) if v else float("nan")  # noqa: E731
    out = dict(n_boot=len(ratios),
               ratio_lo=q(ratios, 0.025), ratio_hi=q(ratios, 0.975),
               D1_lo=q(d1s, 0.025), D1_hi=q(d1s, 0.975),
               D2_lo=q(d2s, 0.025), D2_hi=q(d2s, 0.975))
    print(f"  cluster bootstrap over (template,k) cells: "
          f"D(1) [{100*out['D1_lo']:.1f}, {100*out['D1_hi']:.1f}] pts, "
          f"D(2) [{100*out['D2_lo']:.1f}, {100*out['D2_hi']:.1f}] pts, "
          f"D(2)/D(1) [{out['ratio_lo']:.3f}, {out['ratio_hi']:.3f}]")
    res["bootstrap"] = out
    return out


def paired(d: pd.DataFrame, res: dict) -> None:
    """The within-cell contrast: same carrier, same k, same seed, p=1 vs p=2.

    The tables average over cells before differencing, which is right for the
    headline but leaves the reader to trust that the arms cover the same cells.
    This differences *inside* each (model, template, k, seed) cell -- averaging
    the four p=2 rotations into one number for that cell first -- so nothing
    about which cells each arm happens to occupy can enter the answer.
    """
    piv = d.pivot_table(index=["model", "template", "k", "seed"], columns="arm",
                        values="exact", aggfunc="mean")
    need = [a for a in ("1", "2", "4", "8") if a in piv.columns]
    piv = piv.dropna(subset=need)
    if not {"1", "2", "8"} <= set(piv.columns):
        return
    rows = []
    print("\n--- paired within (model, template, k, seed): mean exact rate ---")
    print(f"  {'checkpoint':11s} {'cells':>6s} " + "  ".join(f"p={a:>2s}" for a in need))
    for m, g in piv.groupby(level=0):
        cells = "  ".join(f"{100*g[a].mean():5.1f}" for a in need)
        print(f"  {m:11s} {len(g):6d} {cells}")
        rows.append(dict(model=m, n_cells=int(len(g)),
                         **{f"p{a}": float(g[a].mean()) for a in need}))
    res["paired_cells"] = rows


def collapse_onto_m(d: pd.DataFrame, pub: pd.DataFrame, res: dict) -> dict:
    """The reading that survives both halves of this experiment: is it m = k/p?

    Neither of the two hypotheses this experiment was designed to separate
    survives contact with the disambiguator arm. The period ladder is graded --
    the deficit shrinks steadily as p grows, which is not the step function
    "verbatim identity" predicts -- but `analysis/disambiguation.py` finds that
    commas, full stops and an interleaved conjunction recover *nothing*, which
    is not what "periodicity" predicts either: `very and very and very` has
    period 2 in tokens and fails exactly as `very very very` does.

    One variable orders all of it. Let

        m = k / p

    be the number of times *each distinct token* has to be repeated. It is k at
    p=1, k/2 at p=2, and 1 in a never-cycled control. The disambiguated arms all
    have m = k, because inserting punctuation or a conjunction does not reduce
    how often the counted word recurs -- and they behave like the bare arm. The
    p=2 arm has m = k/2 -- and it behaves like a shorter repetition.

    The test is a prediction, not a curve fit. The published main ladder already
    measured the exact rate of a bare repeated item at every k, so for each
    period-ladder cell it supplies a *pre-existing* number to compare against:
    E_period(p, k) against E_published(1, m). Every m needed here -- 1, 2, 3, 4,
    6, 8, 12, 16 -- is on the published ladder, so nothing is interpolated.

    The comparison also separates m from length, which is the paper's own
    question. A period-ladder item at (p, k) is k words long; the published item
    at k'=m is m words long. So if total length costs anything on top of m, the
    period-ladder cell must come in *below* its published twin, and the residual
    E_period - E_published is exactly what length is worth at fixed m.

    Pre-committed reading (written before the residuals were computed; the p=2
    and disambiguator rates above were known, the published-ladder comparison
    was not):

      * |residual| small and unsystematic  -> m governs. The right variable is
        how many times one token repeats, and neither "periodicity" nor "length"
        names it.
      * residual reliably negative and growing with k at fixed m -> m governs
        the bulk and total length adds a real, separable penalty on top.
      * residual reliably positive, or unordered in m -> m does not govern, and
        this section should be dropped rather than argued for.
    """
    bare = pub[pub.item_id.astype(str).str.startswith("wr_")]
    bare = bare.assign(exact=(bare.count_a == bare.k))
    ref = bare.groupby(["model", "k"], observed=True).exact.agg(["mean", "size"])

    dm = d.assign(m=(d.k // d.period).astype(int))
    rows = []
    print("\n--- does the ladder collapse onto m = k/p, the per-token "
          "repetition count? ---")
    print(f"  {'checkpoint':11s} {'p':>3s} {'k':>3s} {'m':>3s} {'E(p,k)':>8s} "
          f"{'E_pub(1,m)':>11s} {'residual':>9s}  {'n':>4s} {'n_pub':>6s}")
    for (mo, p, k), g in dm.groupby(["model", "period", "k"], observed=True):
        m = int(k // p)
        if (mo, m) not in ref.index:
            continue
        e, epub = float(g.exact.mean()), float(ref.loc[(mo, m), "mean"])
        rows.append(dict(model=mo, period=int(p), k=int(k), m=m, E=e, E_pub=epub,
                         residual=e - epub, n=int(len(g)),
                         n_pub=int(ref.loc[(mo, m), "size"])))
        print(f"  {mo:11s} {int(p):3d} {int(k):3d} {m:3d} {100*e:7.1f}% "
              f"{100*epub:10.1f}% {100*(e-epub):+8.1f}  {len(g):4d} "
              f"{int(ref.loc[(mo, m), 'size']):6d}")
    if not rows:
        return {}
    r = pd.DataFrame(rows)
    mae = float(r.residual.abs().mean())
    bias = float(r.residual.mean())
    # Spearman rank correlation between E and m, to check m orders the ladder.
    rho = float(r.E.rank().corr(r.m.rank()))
    print(f"\n  mean |residual| {100*mae:.1f} pts, mean signed {100*bias:+.1f} pts, "
          f"rank corr(E, m) = {rho:+.3f}")
    # Length at fixed m: the residual against total length k, within each m.
    within = r.groupby("m").filter(lambda g: g.k.nunique() > 1)
    slope = float("nan")
    if len(within) > 2:
        slope = float(np.polyfit(within.k, within.residual, 1)[0])
        print(f"  at fixed m, residual vs total length k: {100*slope:+.2f} "
              f"pts per unit k (n={len(within)} cells)")
    out = dict(mae=mae, bias=bias, rank_corr_E_m=rho, length_slope_at_fixed_m=slope,
               cells=rows)
    res["collapse_onto_m"] = out
    return out


def replication(d: pd.DataFrame, pub: pd.DataFrame, res: dict) -> None:
    """Today's re-render of the published arms, against the published numbers.

    The p=1, p=8 and p=k rungs are the same text on the same seeds as `wr_*`,
    `ct_*` and `ap_*`. Their exact rates should agree to sampling noise. They
    are not expected to be identical -- generation is sampled, and the Qwen env
    has been upgraded since the panel was run -- but a large disagreement means
    the ladder's rungs are not commensurable with the published arms and the
    comparison to the paper's numbers has to be dropped.
    """
    pub = pub.assign(exact=(pub.count_a == pub.k))
    pub = pub[pub.k.isin(sorted(d.k.unique())) & pub.template.isin(sorted(d.template.unique()))]
    kind = np.where(pub.item_id.astype(str).str.startswith("wr_"), "1",
                    np.where(pub.item_id.astype(str).str.startswith("ap_"), "k", "8"))
    pub = pub.assign(arm=kind)
    pub = pub[pub.family.isin(["word_rep", "control_word"])]
    rows = []
    print("\n--- rung replication: today's re-render vs the published arm "
          "(same text, same seeds) ---")
    print(f"  {'checkpoint':11s} {'arm':>4s} {'new':>7s} {'published':>10s} {'delta':>7s}"
          f"  {'n_new':>6s} {'n_pub':>6s}")
    for (m, arm), g in d.groupby(["model", "arm"], observed=True):
        if arm not in {"1", "8", "k"}:
            continue
        h = pub[(pub.model == m) & (pub.arm == arm)]
        if not len(h):
            continue
        new, old = float(g.exact.mean()), float(h.exact.mean())
        rows.append(dict(model=m, arm=arm, new=new, published=old, delta=new - old,
                         n_new=len(g), n_pub=len(h)))
        print(f"  {m:11s} {arm:>4s} {100*new:6.1f}% {100*old:9.1f}% {100*(new-old):+6.1f}"
              f"  {len(g):6d} {len(h):6d}")
    res["replication"] = rows
    if rows:
        worst = max(rows, key=lambda r: abs(r["delta"]))
        print(f"  worst disagreement: {worst['model']} p={worst['arm']} "
              f"{100*worst['delta']:+.1f} pts")


def hygiene(d_nocap: pd.DataFrame, d_caps: pd.DataFrame, res: dict) -> None:
    """Per-arm cap rate, length and duration: the columns that make the table a
    comparison of like with like, or reveal that it is not."""
    flags = cap_flags()
    d = d_caps.assign(cap=[flags.get((r.model, r.item_id, r.seed), False)
                           for r in d_caps.itertuples()])
    print("\n--- per-arm hygiene (cap hits are excluded from the tables above) ---")
    print(f"  {'arm':>4s} {'n_kept':>7s} {'cap%':>6s} {'words':>7s} {'chars':>7s} "
          f"{'dur_s':>7s} {'degen%':>7s}")
    rows = []
    for arm in ARM_ORDER:
        g, gk = d[d.arm == arm], d_nocap[d_nocap.arm == arm]
        if not len(g):
            continue
        r = dict(arm=arm, n_kept=int(len(gk)), cap_rate=float(g.cap.mean()),
                 words=float(gk.expected_words.mean()),
                 chars=float(gk.n_chars.mean()),
                 duration_s=float(gk.duration_s.mean()),
                 degenerate_rate=float(g.outcome.isin(["empty", "degenerate"]).mean())
                 if "outcome" in g.columns else float("nan"))
        rows.append(r)
        print(f"  {arm:>4s} {r['n_kept']:7d} {100*r['cap_rate']:5.1f}% {r['words']:7.1f} "
              f"{r['chars']:7.1f} {r['duration_s']:7.2f} {100*r['degenerate_rate']:6.1f}%")
    res["hygiene"] = rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_period.csv")
    ap.add_argument("--published", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--published-aperiodic", default="data/results/behavioural_aperiodic.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli_period.jsonl")
    ap.add_argument("--out", default="data/results/period_ladder.json")
    args = ap.parse_args()

    stim = load_stimuli(REPO / args.stimuli)
    d, drop = prepare(REPO / args.behavioural, stim, keep_caps=False)
    d_caps, _ = prepare(REPO / args.behavioural, stim, keep_caps=True)
    print(describe(drop))
    print(f"checkpoints: {', '.join(sorted(d.model.unique()))}")
    print(f"k: {sorted(d.k.unique())}   templates: {sorted(d.template.unique())}")

    res: dict = {"kept": int(len(d)), "drop": {k: v for k, v in drop.items()
                                              if k != "excluded_templates"}}

    # --- the deliverable table, both counting rules.
    t_scan, n_scan = rate_table(d, "exact")
    show(t_scan, n_scan, "exact rate by period -- ordered scan (the published rule)")
    t_unb, n_unb = rate_table(d, "exact_u")
    show(t_unb, n_unb, "exact rate by period -- unbounded recount (same rule at every p)")
    res["exact_by_period_scan"] = json.loads(t_scan.to_json())
    res["exact_by_period_unbounded"] = json.loads(t_unb.to_json())
    res["n_by_period"] = json.loads(n_scan.to_json())

    t_both, n_both = rate_table(d, "exact_both")
    show(t_both, n_both, "exact rate by period -- STRICT: in order AND no extras")
    res["exact_by_period_strict"] = json.loads(t_both.to_json())

    v_scan = verdict(t_scan, "ordered scan", res)
    v_unb = verdict(t_unb, "unbounded recount", res)
    v_both = verdict(t_both, "strict conjunction", res)
    bootstrap_ratio(d, "exact_both", res)
    paired(d, res)

    # --- the same, with cap hits kept and counted as the failures they are.
    t_cap, n_cap = rate_table(d_caps, "exact")
    show(t_cap, n_cap, "exact rate by period -- cap hits KEPT (censored, bound only)")
    verdict(t_cap, "cap hits kept", res)

    # --- by k, to see whether the shape is stable across the ladder.
    print("\nexact rate by period and k (pooled over checkpoints, ordered scan)")
    bk = d.pivot_table(index="k", columns="arm", values="exact", aggfunc="mean")
    bk = bk.reindex(columns=[a for a in ARM_ORDER if a in bk.columns])
    print(bk.round(3).to_string())
    res["exact_by_period_and_k"] = json.loads(bk.to_json())

    # --- by family, since the paper quotes family means.
    fam = d.assign(fam=d.model.map(FAMILY).fillna(d.model))
    tf = fam.pivot_table(index="fam", columns="arm", values="exact", aggfunc="mean")
    tf = tf.reindex(columns=[a for a in ARM_ORDER if a in tf.columns])
    nf = fam.pivot_table(index="fam", columns="arm", values="exact", aggfunc="size")
    show(tf, nf.reindex(columns=tf.columns), "exact rate by period -- by family")
    verdict(tf, "by family", res)

    # --- does rotation matter? if the p=2 arm's answer depends on which two
    # words were picked, the vocabulary control was necessary and the spread is
    # the size of the confound it removed.
    print("\np=2 exact rate by rotation (spread = size of the lexical confound "
          "a single-rotation design would have carried)")
    rot = d[d.arm == "2"].pivot_table(index="model", columns="rotation",
                                      values="exact", aggfunc="mean")
    print((100 * rot).round(1).to_string())
    res["p2_by_rotation"] = json.loads(rot.to_json())

    pub_frames = [pd.read_csv(REPO / p) for p in (args.published, args.published_aperiodic)
                  if (REPO / p).exists()]
    if pub_frames:
        pub = pd.concat(pub_frames, ignore_index=True)
        pub = pub[pub.model.isin(sorted(d.model.unique()))]
        pub, _ = panel(pub)
        replication(d, pub, res)
        collapse_onto_m(d, pub, res)

    hygiene(d, d_caps, res)

    # The strict conjunction is the criterion to quote: it is the only one whose
    # definition of "exact" is the same at every rung. The other two are
    # reported beside it as the bounds they are.
    calls = {v.get("verdict") for v in (v_scan, v_unb, v_both) if v}
    call = v_both.get("verdict") or v_scan.get("verdict")
    agree = len(calls) == 1
    res["headline_verdict"] = call
    res["rules_agree"] = bool(agree)
    res["verdict_by_rule"] = {k: v.get("verdict") for k, v in
                              (("scan", v_scan), ("unbounded", v_unb), ("strict", v_both)) if v}
    print(f"\n=== {call} "
          f"({'all three counting rules agree' if agree else 'RULES DISAGREE: ' + str(res['verdict_by_rule'])})")
    print("The word for this effect is "
          + ("'periodicity'." if call == "PERIODICITY" else
             "'verbatim token repetition', not 'periodicity'."))

    out = REPO / args.out
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
