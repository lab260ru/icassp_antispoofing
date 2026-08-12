#!/usr/bin/env python3
"""The dissociation, restricted to controls that really are aperiodic.

Two round-6 reviewers made the same fair objection: the paper calls the
repeated-vs-control contrast its sharpest test of "periodicity, not length", but
the main ladder's control fillers are drawn from a pool of eight and cycled, so
a k=32 control repeats each filler four times. That contrast is period-1 against
period-8, not against aperiodic, and the Discussion's claim about what
collapse-by-length would predict does not sit comfortably with it.

Three regions have controls that genuinely carry no repetition:

  * k <= 8 on the main ladder, where the eight-word pool covers k without
    cycling;
  * the whole extension ladder (k = 48..128), whose controls come from a
    146-word pool that is never cycled and whose generator asserts it; and
  * **k = 12..32, re-generated for this check** with the same 146-word pool
    (`data/stimuli/stimuli_aperiodic.jsonl`, `scripts/run_aperiodic.sh`). This
    is the range the objection was actually about, so working around it with
    sub-analyses at the two ends was the cheap answer; running it is the right
    one. These items are paired against the *same* repeated items the main
    ladder already scored, so only the control side changes.

If the effect is about periodicity it must survive in all three, and its absence
in any would mean the headline was partly an artifact of comparing two periodic
conditions.

**And the headline itself is recomputed with the never-cycled controls in it**
(`headline()`, added after a reviewer pointed out that the correction was being
reported as a sub-analysis while the number the paper leads with kept the cycled
arm). The re-generated pool now covers all six checkpoints, so the substitution
is panel-wide rather than a statement about four of them: the exact-rate gap at
k>=6 is reported with cycled and with never-cycled controls side by side, and
the difference is how much of the published gap the residual period-8 structure
in the control arm was worth.

Usage:  python analysis/aperiodic_controls.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel, cap_flags  # noqa: E402

DEGENERATE = {"empty", "degenerate"}


def contrast(d: pd.DataFrame, label: str, res: dict) -> None:
    rep = d[d.family == "word_rep"]
    ctl = d[d.family == "control_word"]
    if not len(rep) or not len(ctl):
        return
    row = dict(
        n_rep=int(len(rep)), n_ctl=int(len(ctl)),
        exact_rep=float((rep.err == 0).mean()), exact_ctl=float((ctl.err == 0).mean()),
        median_rep=float(rep.err.median()), median_ctl=float(ctl.err.median()))
    row["exact_gap"] = row["exact_ctl"] - row["exact_rep"]
    res[label] = row
    print(f"{label:34s} rep n={row['n_rep']:4d} exact={100*row['exact_rep']:5.1f}% "
          f"med={row['median_rep']:+.3f} | ctl n={row['n_ctl']:4d} "
          f"exact={100*row['exact_ctl']:5.1f}% med={row['median_ctl']:+.3f} | "
          f"gap={100*row['exact_gap']:+5.1f} pts")


FAMILY = {"llasa1b": "Llasa", "llasa3b": "Llasa", "llasa8b": "Llasa",
          "xtts2": "XTTS", "qwen06b": "Qwen3-TTS", "qwen17b": "Qwen3-TTS"}


def _cells(d: pd.DataFrame) -> pd.Series:
    """The cell a control row occupies, independent of which pool filled it.

    A main-ladder control (`ct_very_t1_k12`) and its re-generated aperiodic twin
    (`ap_very_t1_k12`) are the same carrier, the same k and the same seed and
    differ only in whether the fillers cycle, so the item id is exactly the
    thing that must not be used to line them up.
    """
    return (d.model.astype(str) + "|" + d.template.astype(str) + "|"
            + d.target_unit.astype(str) + "|" + d.k.astype(str) + "|"
            + d.seed.astype(str))


def headline(main_d: pd.DataFrame, apf: pd.DataFrame, args, res: dict) -> None:
    """The paper's headline statistic, recomputed with never-cycled controls.

    The de-confounded number was previously a robustness check on the k=12..32
    subset, which left the reader to guess what it does to the figure the paper
    leads with. This substitutes the never-cycled control arm into the headline
    itself: same repeated arm, same k>=kmin range, only the control side of the
    cells above the pool size changes.

    Three numbers, because two of them answer different questions:

    * `cycled_full` is the published statistic, recomputed here so that any
      disagreement with `checkpoint_level.py` shows up as a population bug
      rather than as a result. It is asserted against that file.
    * `cycled_matched` restricts the cycled control arm to the cells the
      re-generated arm actually covers. Comparing `never_cycled` against
      `cycled_full` would mix the substitution with whatever the two arms lose
      to different cap hits, so the honest attribution uses this one.
    * `never_cycled` swaps in the 146-word-pool controls on those cells.

    Checkpoints with no re-generated arm keep their cycled controls and are
    named in the output, because a panel-wide claim made from a subset is the
    objection this analysis exists to answer.
    """
    kmin, pool = args.kmin, args.pool_size
    apf, _ = panel(apf)
    apf = apf.assign(err=(apf.count_a - apf.k) / apf.k)
    apf = apf[apf.family == "control_word"]
    ap_cells = dict(zip(_cells(apf), apf.err == 0))
    # Panel (a) of the figure is a *median relative error* curve, not an exact
    # rate, and the objection was aimed at the figure as much as at the number.
    # Carry the errors as well as the hit/miss so both statistics can be
    # recomputed on the same substituted arm.
    ap_err = dict(zip(_cells(apf), apf.err))

    d = main_d[main_d.k >= kmin]
    rows, missing = {}, []
    # Pooled over generations as well as averaged over checkpoints. The two
    # answer different questions -- "is the effect in this corpus" against "does
    # it happen to models of this kind" -- and the paper quotes the second, but
    # the reviewer's objection was phrased about the figure, which is pooled.
    pool_acc: dict[str, list] = {"rep": [], "full": [], "matched": [], "never": []}
    for m, g in d.groupby("model", observed=True):
        rep = g[g.family == "word_rep"]
        ctl = g[g.family == "control_word"]
        if not len(rep) or not len(ctl):
            continue
        lo = ctl[ctl.k <= pool]
        hi = ctl[ctl.k > pool]
        cov = pd.Series([c in ap_cells for c in _cells(hi)], index=hi.index)
        if not cov.any():
            # No re-generated arm for this checkpoint: it keeps its cycled
            # controls in all three columns, so its contribution to the
            # never-cycled panel mean is its published value and the shortfall
            # is visible as the checkpoint being named in `missing`. Zeroing
            # `hi` instead would silently drop k>8 from its control arm and
            # make the substitution look free.
            missing.append(m)
            cov = pd.Series(True, index=hi.index)
        hi_m = hi[cov]          # the cells both control arms cover
        e_rep = float((rep.err == 0).mean())

        def rate(lo_ok, hi_ok):
            v = list(lo_ok) + list(hi_ok)
            return float(np.mean(v)) if v else float("nan")

        lo_ok = list(lo.err == 0)
        full = rate(lo_ok, list(hi.err == 0))
        matched = rate(lo_ok, list(hi_m.err == 0))
        never = rate(lo_ok, [ap_cells.get(c, ok) for c, ok
                             in zip(_cells(hi_m), hi_m.err == 0)])
        pool_acc["rep"] += list(rep.err == 0)
        pool_acc["full"] += lo_ok + list(hi.err == 0)
        pool_acc["matched"] += lo_ok + list(hi_m.err == 0)
        pool_acc["never"] += lo_ok + [ap_cells.get(c, ok) for c, ok
                                      in zip(_cells(hi_m), hi_m.err == 0)]

        # The median-error contrast of `checkpoint_level.py`'s `count_error_gap`,
        # on the same three control arms.
        m_rep = float(rep.err.median())
        med = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
        lo_e = list(lo.err)
        med_full = med(lo_e + list(hi.err))
        med_matched = med(lo_e + list(hi_m.err))
        med_never = med(lo_e + [ap_err.get(c, e) for c, e
                                in zip(_cells(hi_m), hi_m.err)])
        rows[m] = dict(
            n_rep=int(len(rep)), n_ctl_full=int(len(lo) + len(hi)),
            n_ctl_matched=int(len(lo) + len(hi_m)), n_hi_matched=int(len(hi_m)),
            n_hi_full=int(len(hi)), exact_rep=e_rep,
            exact_ctl_cycled_full=full, exact_ctl_cycled_matched=matched,
            exact_ctl_never_cycled=never,
            gap_cycled_full=full - e_rep, gap_cycled_matched=matched - e_rep,
            gap_never_cycled=never - e_rep,
            median_rep=m_rep,
            err_gap_cycled_full=med_full - m_rep,
            err_gap_cycled_matched=med_matched - m_rep,
            err_gap_never_cycled=med_never - m_rep)

    e_rep_p = float(np.mean(pool_acc["rep"]))
    pooled = {"n_rep": len(pool_acc["rep"]), "n_ctl_full": len(pool_acc["full"]),
              "n_ctl_matched": len(pool_acc["matched"]), "exact_rep": e_rep_p}
    for tag in ("full", "matched", "never"):
        pooled[f"exact_ctl_{tag}"] = float(np.mean(pool_acc[tag]))
        pooled[f"gap_{tag}"] = pooled[f"exact_ctl_{tag}"] - e_rep_p
    pooled["periodicity_points"] = 100 * (pooled["gap_matched"] - pooled["gap_never"])

    out: dict = {"kmin": kmin, "pool_size": pool, "per_checkpoint": rows,
                 "pooled_over_generations": pooled,
                 "checkpoints_without_regenerated_arm": missing}
    print(f"\n--- the headline, both ways (k>={kmin}, exact-rate gap) ---")
    print(f"{'checkpoint':11s} {'n_rep':>6s} {'n_ctl':>6s} {'cycled':>9s} "
          f"{'cyc(match)':>11s} {'never':>9s} {'delta':>8s}")
    for m in sorted(rows):
        r = rows[m]
        print(f"{m:11s} {r['n_rep']:6d} {r['n_ctl_matched']:6d} "
              f"{100*r['gap_cycled_full']:+9.1f} {100*r['gap_cycled_matched']:+11.1f} "
              f"{100*r['gap_never_cycled']:+9.1f} "
              f"{100*(r['gap_never_cycled']-r['gap_cycled_matched']):+8.1f}")

    KEYS = ("gap_cycled_full", "gap_cycled_matched", "gap_never_cycled",
            "err_gap_cycled_full", "err_gap_cycled_matched", "err_gap_never_cycled")
    for unit in ("checkpoint", "family"):
        agg = {}
        for key in KEYS:
            vals = {m: rows[m][key] for m in rows}
            if unit == "family":
                fam: dict[str, list[float]] = {}
                for m, v in vals.items():
                    fam.setdefault(FAMILY.get(m, m), []).append(v)
                vals = {f: float(np.mean(vs)) for f, vs in fam.items()}
            agg[key] = dict(per_unit=vals, mean=float(np.mean(list(vals.values()))),
                            n=len(vals))
        agg["periodicity_points"] = 100 * (agg["gap_cycled_matched"]["mean"]
                                           - agg["gap_never_cycled"]["mean"])
        agg["headline_points"] = 100 * (agg["gap_cycled_full"]["mean"]
                                        - agg["gap_never_cycled"]["mean"])
        agg["err_periodicity_points"] = 100 * (agg["err_gap_cycled_matched"]["mean"]
                                               - agg["err_gap_never_cycled"]["mean"])
        out[f"by_{unit}"] = agg
        print(f"  by {unit:10s} (n={agg['gap_cycled_full']['n']}): "
              f"exact-rate gap {100*agg['gap_cycled_full']['mean']:.1f} -> "
              f"{100*agg['gap_never_cycled']['mean']:.1f} pts "
              f"({agg['headline_points']:+.1f}; "
              f"{agg['periodicity_points']:+.1f} on matched cells)   "
              f"median-error gap {100*agg['err_gap_cycled_full']['mean']:.1f} -> "
              f"{100*agg['err_gap_never_cycled']['mean']:.1f} pts")

    # The trap this repo has hit twice: a new script and an old one disagreeing
    # on an estimator because they were fitting different rows, not because the
    # arithmetic moved. The cycled arm here is the *published* statistic and has
    # to look like it. Re-running `checkpoint_level.py` on today's data
    # reproduces this column exactly; the stored file is a hair off on qwen06b
    # (0.9222 against 0.9213) because one repeated item, `wr_really_t6_k24` seed
    # 2, has since been flagged `hit_cap` in the ledger and is now excluded by
    # rule 4. That is 0.015 pt on the panel mean, so the tolerance is set where
    # a real population difference would still fail loudly.
    print(f"  pooled over generations (rep n={pooled['n_rep']}, "
          f"ctl n={pooled['n_ctl_matched']}): "
          f"exact-rate gap {100*pooled['gap_full']:.1f} -> "
          f"{100*pooled['gap_never']:.1f} pts "
          f"({100*(pooled['gap_never']-pooled['gap_full']):+.1f})")

    ck = Path(args.checkpoint)
    if ck.exists():
        pub = json.loads(ck.read_text()).get("exact_rate_gap", {}).get("per_model", {})
        dev = {m: rows[m]["gap_cycled_full"] - pub[m] for m in rows if m in pub}
        worst = max(dev.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
        out["published_deviation"] = dev
        assert abs(worst[1]) < args.tol, (
            f"cycled arm does not reproduce {ck}: {dev}. Suspect the population "
            "before the arithmetic.")
        print(f"  cycled arm reproduces {ck} on {len(dev)} checkpoints "
              f"(worst {worst[0]}: {100*worst[1]:+.2f} pt)")
    if missing:
        print(f"  NOT re-generated, cycled controls kept: {', '.join(missing)}")
    res["headline"] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--checkpoint", default="data/results/checkpoint_level.json",
                    help="published per-checkpoint gaps; the cycled arm must match")
    ap.add_argument("--kmin", type=int, default=6,
                    help="the headline is quoted at k>=6")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="allowed drift of the cycled arm from the stored "
                         "checkpoint-level file before the run fails")
    ap.add_argument("--ext", nargs="+",
                    default=["data/results/behavioural_ext.csv",
                             "data/results/behavioural_ext_llasa.csv"])
    ap.add_argument("--aperiodic", default="data/results/behavioural_aperiodic.csv",
                    help="re-generated aperiodic controls for k=12..32")
    ap.add_argument("--pool-size", type=int, default=8,
                    help="main-ladder filler pool; controls are aperiodic at k<=this")
    ap.add_argument("--out", default="data/results/aperiodic_controls.json")
    args = ap.parse_args()

    res: dict = {"pool_size": args.pool_size}

    main_d = pd.read_csv(args.behavioural)
    main_d = main_d[main_d.family.isin(["word_rep", "control_word"])]
    main_d, _ = panel(main_d)
    main_d = main_d.assign(err=(main_d.count_a - main_d.k) / main_d.k)

    print("Rows marked aperiodic have controls with no repeated filler at all.\n"
          "The period-8 row is the main ladder above k=8, kept for comparison.\n")
    contrast(main_d[(main_d.k >= 2) & (main_d.k <= args.pool_size)],
             f"main ladder, k<={args.pool_size} (aperiodic)", res)
    contrast(main_d[main_d.k > args.pool_size],
             f"main ladder, k>{args.pool_size} (period-8 ctl)", res)

    # The re-generated controls for the range the objection was about. Pair them
    # with the repeated items already scored on the main ladder at the same k,
    # so the only thing that differs between the two arms is whether the control
    # repeats its own fillers.
    apc = Path(args.aperiodic)
    if apc.exists():
        a = pd.read_csv(apc)
        a, _ = panel(a)
        a = a.assign(err=(a.count_a - a.k) / a.k)
        ks = sorted(a.k.unique())
        rep_same = main_d[(main_d.family == "word_rep") & main_d.k.isin(ks)
                          & main_d.model.isin(a.model.unique())]
        contrast(pd.concat([rep_same, a[a.family == "control_word"]]),
                 f"k={min(ks)}-{max(ks)} re-generated (aperiodic)", res)

    frames = [pd.read_csv(p) for p in args.ext if Path(p).exists()]
    if frames:
        e = pd.concat(frames, ignore_index=True)
        e = e[e.family.isin(["word_rep", "control_word"])]
        flags = cap_flags()
        e = e[[not flags.get((r.model, r.item_id, r.seed), False) for r in e.itertuples()]]
        e = e[~e.outcome.isin(DEGENERATE)]
        e = e.assign(err=(e.count_a - e.k) / e.k)
        contrast(e, "extension k>=48 (146-word pool)", res)

    if apc.exists():
        headline(main_d, pd.read_csv(apc), args, res)

    keys = [k for k in res if k not in {"pool_size", "headline"}]
    ap_keys = [k for k in keys if "aperiodic" in k or "146-word" in k]
    gaps = [res[k]["exact_gap"] for k in ap_keys]
    res["aperiodic_regions"] = ap_keys
    res["aperiodic_gaps"] = gaps
    res["holds_in_all_aperiodic"] = bool(all(g > 0 for g in gaps)) if gaps else False
    print(f"\nexact-rate gap in the aperiodic regions: "
          f"{', '.join(f'{100*g:+.1f} pts' for g in gaps)}")
    print("The dissociation is not an artifact of comparing two periodic "
          "conditions."
          if res["holds_in_all_aperiodic"] else
          "The dissociation does NOT survive in a genuinely aperiodic control.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
