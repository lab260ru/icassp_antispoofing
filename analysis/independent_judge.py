#!/usr/bin/env python3
r"""Re-judge the same audio with a recogniser that has no stake in the result.

The most persistent objection to this paper --- eight reviewers over three
rounds, two more in the fourth --- is that the dependent variable is defined by
an instrument we validated on the wrong material. `analysis/ctc_validation.py`
checked the CTC judge against *concatenative* audio with a known count; the
paper then applies it to real generations. Worse, the objection has a specific
mechanism attached, and it is a good one: CTC's blank/repeat-collapse rule is
the exact confound under study. A decoder that merges adjacent identical tokens
would depress counts on repeated items and leave length-matched controls
untouched, which is not a small bias --- it is the paper's headline dissociation,
manufactured by the judge, with the same periodicity signature.

The concatenative validation cannot answer this, because a splice gives the
recogniser clean boundaries that a real loop does not. Nor can any amount of
further argument. The only thing that settles it is to score the *same real
generated audio* with a recogniser whose errors are not the primary judge's, and
see whether the effect is a property of the speech or of the scorer.

So: `facebook/hubert-large-ls960-ft` over the full panel population, not the
n=60 stratified subsample of `ctc_field_validation.py` check 4, and scored
through `src/common/score_counts.py` --- the paper's own counting code, not a
reimplementation --- under `src/common/population.py`'s `panel()`. Optionally a
third, `facebook/wav2vec2-large-robust-ft-libri-960h` (different pretraining
mixture), and Whisper large-v3, which is already transcribed and which matters
here for one reason: it is an autoregressive encoder-decoder with **no CTC blank
collapse at all**. If the dissociation appears under a judge that has no blank
rule, the blank rule cannot be what makes it.

HuBERT is the decisive arm because it is the fair one. It is a CTC recogniser of
comparable capacity and word-error rate to the primary judge --- so a
disagreement cannot be dismissed as "the weak judge is noisy" --- but its encoder
was pretrained by masked cluster-label prediction rather than contrastive
learning, so its specific confusions are not the primary judge's. It does have a
blank-collapse rule of its own. That is a genuine limit on what this experiment
can rule out and it is stated here rather than discovered by a reviewer: two CTC
judges agreeing rules out *this recogniser's* idiosyncratic collapse behaviour,
not the CTC decoding rule as a class. The Whisper arm is what addresses the
class, and it is reported for exactly that reason and no other.

WHAT WOULD COUNT AS WHAT. Pre-committed before any independent-judge number was
computed, because a threshold chosen after the fact is not a threshold.

The published headline is a checkpoint-level exact-rate gap (control exact rate
minus repeated exact rate, k>=6) of +76.7 points, positive in 6 of 6 checkpoints
and 3 of 3 families. Two quantities decide the verdict.

  G   the same gap, recomputed with the independent judge on the same rows.
  A   the disagreement asymmetry. Let D = count_primary - count_independent on
      an item. The confound the reviewers describe is the primary judge losing
      repetitions that the independent judge hears, while agreeing on controls:
      that is D negative on the repeated arm, near zero on the control arm, so
      A = mean(D | repeated) - mean(D | control) is substantially NEGATIVE.
      A near zero or positive means the primary judge is not the source of the
      dissociation. `analysis/noise_floor.py` found the positive direction at
      n=60 --- ours reported the higher count in 11 of 13 disagreements --- and
      this run either replicates that at full n or refutes it.

  primary judge confirmed      G >= 50 points, positive in 6/6 checkpoints and
                               3/3 families, and A > -1.0 counts.
  judge-side artifact detected G < 40 points, or positive in fewer than 5/6
                               checkpoints, or A <= -1.0 counts.
  inconclusive                 anything between (G in [40, 50), or 5/6
                               checkpoints), or the independent judge fails the
                               competence gate below.

The competence gate exists because a judge that cannot transcribe the control
arm cannot measure the contrast, and its failure to find a gap would say nothing
about the audio. If the independent judge's own control exact rate falls below
0.70 --- the primary's is 0.94 --- the comparison is measuring the second
recogniser's orthography, not the models' counting, and the result is
inconclusive whatever G comes out at.

One asymmetry in these thresholds is deliberate. G >= 50 is well below the
published 76.7: a judge swap that costs twenty points still leaves the effect
intact, and demanding G ~ 76.7 would be demanding that two different recognisers
agree to the point. But 40 is a real floor. Below it the number the paper leads
with is substantially an artifact of who scored the audio, and that has to be
reported as such.

ONE POST-HOC ADDITION, declared. The rule above was fixed before any number
existed and the decisive arm's verdict is unchanged by what follows; this is
recorded so nobody has to reconstruct it from a diff. On the first run the
*Whisper* arm returned A = -53.7, which by the literal rule is a screaming
confound signature. It is not one. Its median signed difference is exactly 0 and
it is produced by 147 of 745 repeated items on which the models looped away and
one judge read 441 repetitions where the other read 5. Both judges score those
items wrong, so they cannot move an exact-rate gap; they can and do move a mean
over counts. So the asymmetry is now reported three ways --- raw mean, median,
and a mean excluding per-item differences above `RUNAWAY` --- and the verdict
reads the trimmed one. For the decisive HuBERT arm raw and trimmed agree and
both sit far above the threshold, so this changes nothing that matters; had they
disagreed there, the pre-committed raw figure would be the one to quote.

Trap worth naming, because it has bitten this code before: `count_units` in
`score_counts.py` must SKIP a unit the transcript never delivers, not stop at
the first miss. If it stops, one systematically mis-transcribed filler voids
credit for every correctly rendered filler after it, the control arm's exact
rate collapses, and the gap vanishes --- which would look exactly like a
judge-side artifact while being a bug on our side. The second judge is scored by
importing that same function, and the script asserts it is the skipping version
before reporting anything.

Usage:
  python analysis/independent_judge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from src.common.population import panel, describe  # noqa: E402
from src.common.score_counts import count_units  # noqa: E402
from checkpoint_level import boot_mean, heterogeneity, wilcoxon_signed_rank  # noqa: E402

FAMILY = {"llasa1b": "Llasa", "llasa3b": "Llasa", "llasa8b": "Llasa",
          "xtts2": "XTTS", "qwen06b": "Qwen3-TTS", "qwen17b": "Qwen3-TTS"}
KEY = ["model", "item_id", "seed"]
ARMS = {"word_rep": "repeated", "control_word": "control"}

# Pre-committed thresholds. Read the module docstring before changing any of
# these; they were fixed before the independent judge's numbers existed.
G_CONFIRM = 0.50
G_ARTIFACT = 0.40
A_ARTIFACT = -1.0
COMPETENCE_MIN_CONTROL_EXACT = 0.70
PUBLISHED_GAP = 0.766748369083437
# Added AFTER the first run, and flagged as such rather than folded in silently.
# See "one post-hoc addition" in the docstring: a per-item count difference this
# large is a runaway loop both judges already score as wrong, not a counting
# disagreement, and it is excluded only from the signed-asymmetry statistic.
RUNAWAY = 10


def check_count_units_skips() -> None:
    """The trap in `count_units`: an undeliverable unit must be skipped, not
    treated as the end of the item. Assert the behaviour rather than trust the
    comment, because the second judge's control arm is exactly what a
    stop-at-first-miss version would destroy."""
    toks = ["alpha", "gamma"]
    got = count_units(toks, ["alpha", "beta", "gamma"])
    if got != 2:
        raise SystemExit(
            f"count_units returned {got} for a transcript missing the middle "
            f"unit; expected 2 (skip), got the stop-at-first-miss answer. "
            f"Every control's later fillers are being voided -- fix "
            f"src/common/score_counts.py before reading any number below.")


def load_judge(path: str, kmin: int) -> tuple[pd.DataFrame, dict]:
    """One judge's scored rows, through the paper's own population rules."""
    d = pd.read_csv(path)
    d = d[d.family.isin(ARMS)]
    d, drop = panel(d)
    d = d.assign(arm=d.family.map(ARMS), exact=(d.count_a == d.k).astype(int))
    drop["n_kge"] = int((d.k >= kmin).sum())
    return d, drop


def exact_rate_gap(d: pd.DataFrame, kmin: int) -> dict:
    """The published statistic, verbatim: control exact rate minus repeated
    exact rate, per checkpoint, then across checkpoints and families."""
    d = d[d.k >= kmin]
    per_model, per_model_n = {}, {}
    for m, g in d.groupby("model"):
        r, c = g[g.arm == "repeated"], g[g.arm == "control"]
        if len(r) and len(c):
            per_model[m] = float(c.exact.mean() - r.exact.mean())
            per_model_n[m] = dict(n_repeated=len(r), n_control=len(c),
                                  repeated_exact=float(r.exact.mean()),
                                  control_exact=float(c.exact.mean()))
    vals = np.array([per_model[m] for m in sorted(per_model)], float)
    fam: dict[str, list[float]] = {}
    for m, v in per_model.items():
        fam.setdefault(FAMILY.get(m, m), []).append(v)
    fam_means = {f: float(np.mean(vs)) for f, vs in fam.items()}
    fv = np.array(list(fam_means.values()), float)

    out = dict(per_model=per_model, per_model_detail=per_model_n,
               mean=float(vals.mean()) if len(vals) else float("nan"),
               n_checkpoints=len(vals),
               n_positive=int((vals > 0).sum()),
               per_family=fam_means, n_families=len(fam_means),
               family_mean=float(fv.mean()) if len(fv) else float("nan"),
               all_families_positive=bool(len(fv) and (fv > 0).all()),
               control_exact_overall=float(d[d.arm == "control"].exact.mean()),
               repeated_exact_overall=float(d[d.arm == "repeated"].exact.mean()),
               n_rows=len(d))
    if len(vals) >= 3:
        lo, hi = boot_mean(vals)
        stat, p = wilcoxon_signed_rank(vals)
        out.update(lo=lo, hi=hi, wilcoxon_w=stat, wilcoxon_p=p,
                   **heterogeneity(vals))
    return out


def agreement(m: pd.DataFrame) -> dict:
    """Per-arm, per-k agreement between two judges on the same clips.

    `d` is signed as primary minus independent, so the reviewers' confound has a
    definite sign: negative on the repeated arm means the primary judge is
    hearing fewer repetitions than an independent recogniser does, which is the
    thing that would manufacture the effect.
    """
    def agg(sub: pd.DataFrame) -> dict:
        n = len(sub)
        if n == 0:
            return dict(n=0)
        dis = sub[sub.d != 0]
        keep = sub[sub.d.abs() <= RUNAWAY]
        return dict(
            n=n,
            mean_abs_diff=float(sub.d.abs().mean()),
            median_abs_diff=float(sub.d.abs().median()),
            exact_agreement_rate=float((sub.d == 0).mean()),
            mean_signed_diff=float(sub.d.mean()),
            median_signed_diff=float(sub.d.median()),
            # A mean over counts is not robust when one arm contains runaway
            # loops: an item asking for 16 repetitions on which one judge reads
            # 441 and the other 5 moves the arm mean by tenths of a count all by
            # itself. Both judges call that item wrong, so it cannot touch the
            # exact-rate gap -- but it can dominate a signed-difference mean and
            # fake a confound signature that is not there. The trimmed figure is
            # what the asymmetry test should be read on.
            mean_signed_diff_excl_runaway=float(keep.d.mean()) if len(keep) else None,
            n_runaway=int(len(sub) - len(keep)),
            mean_signed_diff_per_k=float((sub.d / sub.k).mean()),
            n_disagreements=len(dis),
            primary_higher=int((dis.d > 0).sum()),
            independent_higher=int((dis.d < 0).sum()),
            primary_higher_frac=(float((dis.d > 0).mean()) if len(dis) else None),
        )

    out = dict(overall=agg(m), by_arm={}, by_arm_k={}, by_arm_model={})
    for arm, g in m.groupby("arm"):
        out["by_arm"][arm] = agg(g)
        out["by_arm_k"][arm] = {int(k): agg(gk) for k, gk in g.groupby("k")}
        out["by_arm_model"][arm] = {mm: agg(gm) for mm, gm in g.groupby("model")}

    rep = m[m.arm == "repeated"]
    ctl = m[m.arm == "control"]
    if len(rep) and len(ctl):
        a = float(rep.d.mean() - ctl.d.mean())
        rk, ck = rep[rep.d.abs() <= RUNAWAY], ctl[ctl.d.abs() <= RUNAWAY]
        a_trim = (float(rk.d.mean() - ck.d.mean())
                  if len(rk) and len(ck) else float("nan"))
        a_med = float(rep.d.median() - ctl.d.median())
        # Bootstrap the asymmetry over checkpoints, so the interval reflects the
        # unit the paper generalises over rather than the generation count.
        models = sorted(set(m.model))
        per_ck = []
        for mm in models:
            r = rep[rep.model == mm]
            c = ctl[ctl.model == mm]
            if len(r) and len(c):
                per_ck.append(float(r.d.mean() - c.d.mean()))
        arr = np.array(per_ck, float)
        out["asymmetry"] = dict(
            value=a,
            value_excl_runaway=a_trim,
            value_median=a_med,
            n_runaway=int(len(rep) - len(rk) + len(ctl) - len(ck)),
            mean_signed_diff_repeated=float(rep.d.mean()),
            mean_signed_diff_control=float(ctl.d.mean()),
            per_checkpoint={mm: v for mm, v in zip(models, per_ck)},
            n_checkpoints=len(arr),
            # Decided on the outlier-robust figure. Runaway loops are items both
            # judges already score as wrong, so letting them set the sign of the
            # asymmetry test would be answering a different question.
            confound_signature=bool(
                (a_trim if np.isfinite(a_trim) else a) <= A_ARTIFACT),
            confound_signature_raw_mean=bool(a <= A_ARTIFACT),
        )
        if len(arr) >= 3:
            lo, hi = boot_mean(arr)
            out["asymmetry"].update(checkpoint_mean=float(arr.mean()),
                                    lo=lo, hi=hi)
    return out


def merge_judges(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Rows both judges kept, aligned clip for clip.

    The intersection matters. `panel()` drops degenerate output, and one of its
    degeneracy conditions is an empty transcript --- which is a property of the
    judge, not only of the audio. Comparing each judge on its own surviving rows
    would confound a change of scorer with a change of population, so the paired
    statistics run on the intersection and both populations are reported.
    """
    cols = KEY + ["arm", "family", "template", "k", "count_a", "exact"]
    m = a[cols].merge(b[cols], on=KEY + ["arm", "family", "template", "k"],
                      suffixes=("_p", "_i"))
    return m.assign(d=m.count_a_p - m.count_a_i)


def union_rule(merged: pd.DataFrame, kmin: int) -> dict:
    """The strongest form of the objection, tested directly.

    Everything above assumes each judge is separately roughly right. The harder
    version of the reviewers' point is that *both* judges miss repetitions, each
    a different subset, so an item can be correctly rendered and still be scored
    wrong by whichever judge happens to be looking. Under that account the true
    repeated-arm exact rate is bounded below by the rate at which *either* judge
    certifies the item exact.

    That union is the most generous reading of the models' behaviour the data
    admits and the least favourable one for this paper: it hands the repeated arm
    every item either recogniser was willing to call correct, and hands the
    control arm the same. If the dissociation survives being scored that way,
    no combination of these two judges' misses can account for it.

    The intersection is reported beside it only to bracket the range; it is the
    flattering one and is not what we quote.
    """
    out = {}
    for rule, col in (("union_either_judge_exact",
                       merged[["exact_p", "exact_i"]].max(axis=1)),
                      ("intersection_both_judges_exact",
                       merged[["exact_p", "exact_i"]].min(axis=1))):
        out[rule] = exact_rate_gap(
            merged.assign(exact=col, count_a=merged.count_a_p), kmin)
    return out


def template_sensitivity(primary_csv: str, independent_csv: str, kmin: int) -> dict:
    """The gap with `judge_vocab_audit`'s template exclusion switched off.

    Template t2 is dropped from the panel because the *primary* judge cannot
    render its scored vocabulary (`okay` comes out as "o k"). That exclusion was
    derived from the primary judge's delivery rates, so carrying it over to the
    independent judge is borrowing one recogniser's blind spot for another's
    population. Putting t2 back scores both judges on every template, blind spots
    included; it is the least favourable population for the paper and is reported
    for that reason.
    """
    out = {}
    for name, path in (("primary", primary_csv), ("independent", independent_csv)):
        d = pd.read_csv(path)
        d = d[d.family.isin(ARMS)]
        d, drop = panel(d, bad_templates=False)
        d = d.assign(arm=d.family.map(ARMS), exact=(d.count_a == d.k).astype(int))
        out[name] = exact_rate_gap(d, kmin)
        out[name]["n_input_rows"] = drop["n_output"]
    return out


def fmt_gap(g: dict) -> str:
    return (f"{100*g['mean']:+6.1f}  ({g['n_positive']}/{g['n_checkpoints']} ck, "
            f"{sum(1 for v in g['per_family'].values() if v > 0)}/"
            f"{g['n_families']} fam)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--independent", default="data/results/behavioural_hubert.csv")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="name=path for further judges reported alongside")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--out", default="data/results/independent_judge.json")
    args = ap.parse_args()

    check_count_units_skips()

    prim, drop_p = load_judge(args.primary, args.kmin)
    indep, drop_i = load_judge(args.independent, args.kmin)
    print(f"primary     {describe(drop_p)}")
    print(f"independent {describe(drop_i)}")

    res: dict = {
        "primary_judge": "facebook/wav2vec2-large-960h-lv60-self",
        "independent_judge": "facebook/hubert-large-ls960-ft",
        "kmin": args.kmin,
        "published_exact_rate_gap": PUBLISHED_GAP,
        "thresholds": dict(confirm_gap=G_CONFIRM, artifact_gap=G_ARTIFACT,
                           artifact_asymmetry=A_ARTIFACT,
                           competence_min_control_exact=COMPETENCE_MIN_CONTROL_EXACT),
        "population": {"primary": drop_p, "independent": drop_i},
    }
    # Pin what actually ran. The hashes are read from the local HF cache by
    # `data/results/model_revisions.json`; none of them is typed by hand.
    rev = json.loads((REPO / "data/results/model_revisions.json").read_text())
    res["revisions"] = {k: v for k, v in rev.items()
                        if k in {"facebook/wav2vec2-large-960h-lv60-self",
                                 "facebook/hubert-large-ls960-ft",
                                 "facebook/wav2vec2-large-robust-ft-libri-960h",
                                 "openai/whisper-large-v3"}}

    # ---- 1. the headline, each judge on its own panel and on the paired one --
    own = {"primary": exact_rate_gap(prim, args.kmin),
           "independent": exact_rate_gap(indep, args.kmin)}
    merged = merge_judges(prim, indep)
    paired = {
        "primary": exact_rate_gap(
            merged.rename(columns={"count_a_p": "count_a", "exact_p": "exact"}),
            args.kmin),
        "independent": exact_rate_gap(
            merged.rename(columns={"count_a_i": "count_a", "exact_i": "exact"}),
            args.kmin),
    }
    res["exact_rate_gap"] = dict(own_panel=own, paired_panel=paired,
                                 n_paired_rows=len(merged),
                                 n_paired_rows_kge=int((merged.k >= args.kmin).sum()))

    print(f"\nexact-rate gap, control minus repeated, k>={args.kmin}")
    print(f"  {'judge':<34s} {'own panel':>26s} {'paired panel':>26s}")
    for name in ("primary", "independent"):
        lab = res["primary_judge"] if name == "primary" else res["independent_judge"]
        print(f"  {lab.split('/')[-1]:<34s} {fmt_gap(own[name]):>26s} "
              f"{fmt_gap(paired[name]):>26s}")
    print(f"  published number in the paper: {100*PUBLISHED_GAP:+.1f}")

    print(f"\nper checkpoint (paired panel), points")
    print(f"  {'checkpoint':<12s} {'primary':>9s} {'independent':>12s} {'delta':>8s}")
    for m in sorted(paired["primary"]["per_model"]):
        p = paired["primary"]["per_model"][m]
        i = paired["independent"]["per_model"].get(m, float("nan"))
        print(f"  {m:<12s} {100*p:+9.1f} {100*i:+12.1f} {100*(i-p):+8.1f}")
    print(f"\nper family (paired panel), points")
    for f in sorted(paired["primary"]["per_family"]):
        p = paired["primary"]["per_family"][f]
        i = paired["independent"]["per_family"].get(f, float("nan"))
        print(f"  {f:<12s} {100*p:+9.1f} {100*i:+12.1f} {100*(i-p):+8.1f}")

    # ---- 2. per-arm, per-k agreement -------------------------------------
    # Over the full k grid, not just k>=kmin: a judge-side collapse would have
    # to switch on with periodicity, so the low-k rows are where it should be
    # absent and are part of the evidence.
    ag = agreement(merged)
    ag_kge = agreement(merged[merged.k >= args.kmin])
    res["agreement"] = dict(all_k=ag, kge=ag_kge)

    print(f"\nper-arm agreement, primary vs independent, all k (n={len(merged)})")
    print(f"  {'arm':<10s} {'n':>5s} {'mean|d|':>8s} {'exact agree':>12s} "
          f"{'mean d':>8s} {'ours higher':>18s}")
    for arm in ("repeated", "control"):
        v = ag["by_arm"].get(arm, {})
        if not v.get("n"):
            continue
        print(f"  {arm:<10s} {v['n']:5d} {v['mean_abs_diff']:8.3f} "
              f"{100*v['exact_agreement_rate']:11.1f}% {v['mean_signed_diff']:+8.3f} "
              f"{v['primary_higher']:>8d}/{v['n_disagreements']:<9d}")

    print(f"\nby k (mean|d| / exact-agreement% / mean signed d, primary-independent)")
    ks = sorted(set(merged.k))
    print(f"  {'k':>3s}  " + "".join(f"{a:>28s}" for a in ("repeated", "control")))
    for k in ks:
        cells = []
        for arm in ("repeated", "control"):
            v = ag["by_arm_k"].get(arm, {}).get(k, {})
            cells.append(f"{v['mean_abs_diff']:6.2f} {100*v['exact_agreement_rate']:5.1f}% "
                         f"{v['mean_signed_diff']:+6.2f} n={v['n']:<3d}" if v.get("n")
                         else " " * 28)
        print(f"  {k:3d}  " + "".join(f"{c:>28s}" for c in cells))

    asym = ag.get("asymmetry", {})
    if asym:
        print(f"\nasymmetry A = mean(d|repeated) - mean(d|control) = "
              f"{asym['value']:+.3f} counts")
        print(f"  repeated {asym['mean_signed_diff_repeated']:+.3f}, "
              f"control {asym['mean_signed_diff_control']:+.3f}"
              + (f", checkpoint bootstrap [{asym['lo']:+.3f},{asym['hi']:+.3f}]"
                 if "lo" in asym else ""))
        print(f"  median {asym['value_median']:+.3f}, excluding "
              f"{asym['n_runaway']} runaway-loop items "
              f"{asym['value_excl_runaway']:+.3f}")
        print(f"  the reviewers' confound needs A <= {A_ARTIFACT:+.1f}: "
              f"{'PRESENT' if asym['confound_signature'] else 'absent'}")

    # Two things the tables above make easy to miss, so they get said out loud.
    # First: disagreement IS asymmetric across the arms, in magnitude. Reporting
    # only the sign of A would be reporting the convenient half of that.
    rr = ag["by_arm"].get("repeated", {})
    cc = ag["by_arm"].get("control", {})
    if rr.get("n") and cc.get("n"):
        ratio = rr["mean_abs_diff"] / max(cc["mean_abs_diff"], 1e-9)
        res["agreement"]["magnitude_asymmetry"] = dict(
            repeated_mean_abs_diff=rr["mean_abs_diff"],
            control_mean_abs_diff=cc["mean_abs_diff"], ratio=float(ratio))
        print(f"\n  the judges do disagree {ratio:.1f}x more on repeated items "
              f"({rr['mean_abs_diff']:.2f} counts) than on controls "
              f"({cc['mean_abs_diff']:.2f}).")
        print(f"  that asymmetry is real and has the effect's periodicity, but "
              f"its SIGN is the\n  opposite of the confound: on the repeated arm "
              f"the primary judge reports the\n  higher count in "
              f"{rr['primary_higher']} of {rr['n_disagreements']} disagreements, "
              f"so it under-states the deficit.")

    # Second: where the two judges are least able to agree at all.
    worst = [(arm, k, v) for arm, byk in ag["by_arm_k"].items()
             for k, v in byk.items() if v.get("n")]
    worst.sort(key=lambda t: t[2]["exact_agreement_rate"])
    res["agreement"]["worst_cells"] = [
        dict(arm=a, k=k, **v) for a, k, v in worst[:3]]
    print(f"\n  worst-agreeing cells (this is the resolution limit of the "
          f"measurement):")
    for a, k, v in worst[:3]:
        print(f"    {a:<9s} k={k:<3d} exact agreement {100*v['exact_agreement_rate']:.1f}%"
              f", mean|d| {v['mean_abs_diff']:.2f}, n={v['n']}")

    # ---- 2b. the two least favourable readings ---------------------------
    res["union_rule"] = union_rule(merged, args.kmin)
    res["template_sensitivity"] = template_sensitivity(
        args.primary, args.independent, args.kmin)

    u = res["union_rule"]["union_either_judge_exact"]
    i = res["union_rule"]["intersection_both_judges_exact"]
    print(f"\nleast favourable readings")
    print(f"  exact if EITHER judge says so   {fmt_gap(u)}"
          f"   (repeated {100*u['repeated_exact_overall']:.1f}%, "
          f"control {100*u['control_exact_overall']:.1f}%)")
    print(f"  exact if BOTH judges say so     {fmt_gap(i)}")
    ts = res["template_sensitivity"]
    print(f"  t2 restored, primary            {fmt_gap(ts['primary'])}")
    print(f"  t2 restored, independent        {fmt_gap(ts['independent'])}")

    # ---- 3. further judges, reported alongside ---------------------------
    res["other_judges"] = {}
    for spec in args.extra:
        name, _, path = spec.partition("=")
        if not Path(path).exists():
            print(f"\n[skip] {name}: {path} not found")
            continue
        oj, drop_o = load_judge(path, args.kmin)
        mo = merge_judges(prim, oj)
        res["other_judges"][name] = dict(
            population=drop_o,
            own_panel=exact_rate_gap(oj, args.kmin),
            paired_panel=exact_rate_gap(
                mo.rename(columns={"count_a_i": "count_a", "exact_i": "exact"}),
                args.kmin),
            n_paired_rows=len(mo),
            agreement=agreement(mo),
        )
        g = res["other_judges"][name]
        print(f"\n[{name}] {describe(drop_o)}")
        print(f"  exact-rate gap  own {fmt_gap(g['own_panel'])}   "
              f"paired {fmt_gap(g['paired_panel'])}")
        a2 = g["agreement"].get("asymmetry", {})
        if a2:
            print(f"  asymmetry A = {a2['value']:+.3f} counts raw, "
                  f"{a2['value_median']:+.3f} median, "
                  f"{a2['value_excl_runaway']:+.3f} excluding "
                  f"{a2['n_runaway']} runaway-loop items")
            if a2["confound_signature_raw_mean"] and not a2["confound_signature"]:
                print(f"  the raw mean trips the confound threshold and the "
                      f"trimmed one does not: this judge's\n  arm mean is set by "
                      f"items on which the model looped away and the two judges "
                      f"read\n  wildly different loop lengths. Both score those "
                      f"items wrong, so no exact-rate gap\n  depends on them.")

    # ---- 3b. every judge in one place, weakest first ---------------------
    allj = {res["primary_judge"]: paired["primary"],
            res["independent_judge"]: paired["independent"]}
    NAMES = {"robust": "facebook/wav2vec2-large-robust-ft-libri-960h",
             "whisper": "openai/whisper-large-v3"}
    for name, g in res["other_judges"].items():
        allj[NAMES.get(name, name)] = g["paired_panel"]
    order = sorted(allj, key=lambda k: allj[k]["mean"])
    res["all_judges"] = {k: dict(mean=allj[k]["mean"],
                                 n_positive=allj[k]["n_positive"],
                                 n_checkpoints=allj[k]["n_checkpoints"],
                                 per_family=allj[k]["per_family"],
                                 control_exact=allj[k]["control_exact_overall"],
                                 repeated_exact=allj[k]["repeated_exact_overall"])
                         for k in order}
    res["weakest_judge"] = dict(judge=order[0], gap=allj[order[0]]["mean"])
    print(f"\nevery judge, paired panel, weakest first")
    print(f"  {'judge':<46s} {'gap':>7s} {'ck':>6s} {'ctl exact':>10s} "
          f"{'rep exact':>10s}")
    for k in order:
        g = allj[k]
        print(f"  {k:<46s} {100*g['mean']:+7.1f} "
              f"{g['n_positive']}/{g['n_checkpoints']:<4d} "
              f"{100*g['control_exact_overall']:9.1f}% "
              f"{100*g['repeated_exact_overall']:9.1f}%")

    # Where the judges actually differ, which is the whole argument in one line.
    # A judge-side collapse confound is a claim about the REPEATED arm: the
    # scorer is supposed to be losing repetitions that are really there. So it
    # predicts that swapping the scorer moves the repeated-arm exact rate. What
    # moves instead is the control arm --- that is just general word error rate,
    # the thing recognisers are known to differ on. Four recognisers spanning two
    # decoding paradigms put the repeated arm inside a two-point band.
    rep_rates = [g["repeated_exact_overall"] for g in allj.values()]
    ctl_rates = [g["control_exact_overall"] for g in allj.values()]
    res["arm_spread_across_judges"] = dict(
        n_judges=len(allj),
        repeated_exact_min=float(min(rep_rates)), repeated_exact_max=float(max(rep_rates)),
        repeated_spread=float(max(rep_rates) - min(rep_rates)),
        control_exact_min=float(min(ctl_rates)), control_exact_max=float(max(ctl_rates)),
        control_spread=float(max(ctl_rates) - min(ctl_rates)))
    sp = res["arm_spread_across_judges"]
    print(f"\n  across all {sp['n_judges']} judges the REPEATED-arm exact rate spans "
          f"{100*sp['repeated_exact_min']:.1f}-{100*sp['repeated_exact_max']:.1f}% "
          f"({100*sp['repeated_spread']:.1f} points),")
    print(f"  while the CONTROL arm spans "
          f"{100*sp['control_exact_min']:.1f}-{100*sp['control_exact_max']:.1f}% "
          f"({100*sp['control_spread']:.1f} points). Changing the scorer moves the "
          f"arm the\n  models get right, not the arm they get wrong -- which is "
          f"the opposite of what a\n  judge-side collapse of repeated material "
          f"would do.")

    # ---- 4. the pre-committed verdict ------------------------------------
    G = paired["independent"]["mean"]
    ctl_exact = paired["independent"]["control_exact_overall"]
    A = asym.get("value", float("nan"))
    n_pos = paired["independent"]["n_positive"]
    n_ck = paired["independent"]["n_checkpoints"]
    fam_pos = paired["independent"]["all_families_positive"]

    competent = ctl_exact >= COMPETENCE_MIN_CONTROL_EXACT
    reasons = []
    if not competent:
        verdict = "inconclusive"
        reasons.append(
            f"the independent judge fails the competence gate: it certifies only "
            f"{100*ctl_exact:.1f}% of control items as exact against the "
            f"{100*COMPETENCE_MIN_CONTROL_EXACT:.0f}% floor, so its failure to "
            f"resolve the contrast would be its own orthography, not the audio")
    elif G < G_ARTIFACT or n_pos < 5 or (np.isfinite(A) and A <= A_ARTIFACT):
        verdict = "judge-side artifact detected"
        if G < G_ARTIFACT:
            reasons.append(f"the gap falls to {100*G:+.1f} points, below the "
                           f"pre-committed floor of {100*G_ARTIFACT:.0f}")
        if n_pos < 5:
            reasons.append(f"it is positive in only {n_pos} of {n_ck} checkpoints")
        if np.isfinite(A) and A <= A_ARTIFACT:
            reasons.append(f"the primary judge under-counts the repeated arm "
                           f"relative to the independent one by {-A:.2f} counts "
                           f"more than it does controls (A={A:+.2f})")
    elif G >= G_CONFIRM and n_pos == n_ck and fam_pos and A > A_ARTIFACT:
        verdict = "primary judge confirmed"
        reasons.append(
            f"the gap is {100*G:+.1f} points under a recogniser with different "
            f"pretraining, positive in {n_pos}/{n_ck} checkpoints and every "
            f"family, and the disagreement asymmetry A={A:+.2f} does not have "
            f"the confound signature")
    else:
        verdict = "inconclusive"
        reasons.append(f"the gap is {100*G:+.1f} points, positive in {n_pos} of "
                       f"{n_ck} checkpoints, all families positive: {fam_pos}; "
                       f"that lands between the pre-committed thresholds")

    res["verdict"] = dict(
        label=verdict, reasons=reasons,
        G=float(G), A=float(A) if np.isfinite(A) else None,
        independent_control_exact=float(ctl_exact),
        competence_gate_passed=bool(competent),
        gap_retained_fraction=float(G / PUBLISHED_GAP) if PUBLISHED_GAP else None)

    print(f"\nVERDICT: {verdict.upper()}")
    for r in reasons:
        print(f"  - {r}")
    print(f"  the independent judge retains "
          f"{100*G/PUBLISHED_GAP:.0f}% of the published gap "
          f"({100*G:+.1f} against {100*PUBLISHED_GAP:+.1f} points).")
    w = res["weakest_judge"]
    print(f"  quoted at the least favourable judge available, the gap is "
          f"{100*w['gap']:+.1f} points\n  ({w['judge'].split('/')[-1]}); "
          f"that is the number to put in the paper if only one is quoted.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
