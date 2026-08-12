#!/usr/bin/env python3
"""What each of the paper's three load-bearing nulls can actually exclude.

A reviewer read Sections 4--5 and objected to the same move three times: the
rank-1 patch "moves the count not at all", supplying F5-TTS the duration gives
"the gain is 0.0" at $k\\ge12$, and the probe past the horizon is
"indistinguishable" from a predictor that always answers the mean. Each is a
point estimate with an interval that touches or contains zero, and each is then
used as evidence of *absence*. A confidence interval containing zero is not that.
It is compatible with zero and with everything else inside it, and nothing in the
paper says how wide "everything else" is.

This script says how wide. For each null it runs the two one-sided tests against
a margin fixed below, and --- more usefully, because the margin is arguable and
the bound is not --- reports the **smallest effect the data can reject**: the
larger endpoint of the 90% bootstrap interval, which is the tightest margin at
which TOST rejects at $\\alpha=0.05$. If that bound is larger than the effect the
paper's argument needs to rule out, the null is underpowered and the paper is not
entitled to the word "no", whatever its $p$.

Pre-committed margins, and why each. All three are fixed from quantities that
were already published in `data/results/` before this file existed; none is read
off the equivalence bounds it is used to judge.

  1. **Rank-1 patch**, cell `crossk|L7|P128`. The manipulation was built to make
     the receiver render the *donor's* count. A full transfer is therefore a
     shift equal to the donor/receiver baseline gap, already reported in that
     cell as `median_gap` = 0.99 log-count. Margin: half a transfer. Anything
     smaller than half would not license "the decoder renders the donor's count"
     even if it were significant, so half is the largest margin the paper's own
     claim can be asked to survive. Also reported: the bound as a *fraction of a
     full transfer*, which is the scale-free version and the one to quote.
  2. **F5-TTS duration supply**, $k\\ge12$. The claim is a contrast, not a bare
     null: the intervention lifts exact renderings by 33.3 points below $k=12$
     and is said to do nothing at or above it. Margin: half the low-$k$ gain,
     16.7 points. If the high-$k$ gain could be half the low-$k$ one, the split
     that the paragraph rests on is not established.
  3. **Probe vs the mean-answering predictor**, past the extension horizon. Two
     margins, because two different claims are made. (a) The repo's own tie band,
     0.02 in MAE ratio, already used by `probe_past_horizon.py` to call llasa1b
     and llasa3b indistinguishable. (b) The effect the rival account predicts: a
     probe that reads the count off past-horizon states as well as it reads it
     off *control* states over the identical $k$ range, published as ratio 0.86
     on llasa1b --- so a 0.14 improvement over the constant predictor.

What each outcome licenses, fixed before the bounds were computed:

  * bound < margin  --- the effect the paper needs to exclude is excluded. "No
    effect of the size that would matter here" is earned, with the bound quoted.
  * bound > margin  --- the null is underpowered for its job. The paper must say
    "we could not detect", give the bound, and drop "not at all" / "no".
  * bound > the effect seen in the *comparison* arm (a full transfer, the low-$k$
    gain, the below-horizon ratio) --- the null carries no information at all and
    should be cut rather than hedged.

Every reported number is re-derived here from the same inputs the original
analyses used, and the published value is asserted before anything new is
printed: if `causal_count_score.py`, `duration_intervention.py` or
`probe_count.py` move, this file fails loudly rather than reporting an
equivalence bound for a quantity that no longer exists.

Usage:  python analysis/equivalence.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.common.population import panel  # noqa: E402

ALPHA = 0.05
N_BOOT = 20000
BOOT_SEED = 0

# ---- pre-committed margins (see docstring; all from already-published values)
FULL_TRANSFER_FRACTION = 0.5      # rank-1: half a donor/receiver gap
DUR_MARGIN_FRACTION = 0.5         # duration: half the published low-k gain
PROBE_TIE_BAND = 0.02             # probe: the repo's own tie band
PROBE_RIVAL_MARGIN = 0.14         # probe: 1 - 0.86, the published control ratio

# The three cells this file is about.
RANK1_CELL = "crossk|L7|P128"
PROBE_PAST = {"llasa1b": "data/results/probe_past_horizon.json",
              "llasa3b": "data/results/probe_past_horizon_3b.json",
              "llasa8b": "data/results/probe_past_horizon_8b.json",
              "qwen06b": "data/results/probe_past_horizon_q06b.json",
              "qwen17b": "data/results/probe_past_horizon_q17b.json"}
PROBE_CTL = {"llasa1b": "data/results/probe_past_horizon_ctl.json",
             "llasa3b": "data/results/probe_past_horizon_ctl_3b.json",
             "llasa8b": "data/results/probe_past_horizon_ctl_8b.json",
             "qwen06b": "data/results/probe_past_horizon_ctl_q06b.json",
             "qwen17b": "data/results/probe_past_horizon_ctl_q17b.json"}
PROBE_STIM = ["data/stimuli/stimuli_ext_instr.jsonl",
              "data/stimuli/stimuli_ext_instr_ctl.jsonl"]


# ---------------------------------------------------------------- machinery


def boot_dist(x: np.ndarray, stat: str, seed: int = BOOT_SEED,
              n: int = N_BOOT) -> np.ndarray:
    """Bootstrap distribution of `stat` over rows of `x` (columns move together).

    `x` is [n_units] or [n_units, n_cols]; resampling is over units, so a paired
    or ratio statistic sees the same resample in every column.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = x.shape[0]
    idx = rng.integers(0, m, size=(n, m))
    take = x[idx]                                   # [n, m] or [n, m, cols]
    if stat == "median":
        return np.median(take, axis=1)
    if stat == "mean":
        return np.mean(take, axis=1)
    if stat == "mae_ratio":
        # column 0: |probe error|, column 1: |constant-predictor error|
        return take[:, :, 0].mean(axis=1) / take[:, :, 1].mean(axis=1)
    if stat == "median_diff":
        # column 0 minus column 1, medians taken within the same resample
        return np.median(take[:, :, 0], axis=1) - np.median(take[:, :, 1], axis=1)
    raise ValueError(stat)


def boot_dist_cluster(x: np.ndarray, groups: np.ndarray, stat: str,
                      seed: int = BOOT_SEED, n: int = N_BOOT) -> np.ndarray:
    """As `boot_dist`, but resampling whole clusters rather than single rows.

    None of these three designs has independent rows. The rank-1 patch pairs
    share a receiver item, the F5 pairs share a carrier template, and the probe's
    twelve items are four $k$ values inside three templates *which are also its
    cross-validation folds*. Resampling rows treats each as its own unit of
    evidence and buys precision the design never paid for. This is the honest
    version, and it is reported alongside rather than instead: the row bootstrap
    is what a reader reconstructing the published CI would compute, and the gap
    between the two is itself the answer to "how fragile is this null".

    It does not always widen, and where it narrows that is informative rather
    than broken. Resampling rows treats the $k$ grid as sampled when it is a
    fixed design factor; clustering holds each cluster's $k$ composition fixed
    and varies only the unit that really was sampled. Where between-cluster
    spread is small the clustered interval is legitimately the tighter one ---
    but with three or four clusters a percentile bootstrap is very coarse, so
    neither reading is ever quoted on its own.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    keys = sorted(set(np.asarray(groups).tolist()))
    members = [np.flatnonzero(np.asarray(groups) == k) for k in keys]
    g = len(keys)
    draws = rng.integers(0, g, size=(n, g))
    out = np.empty(n)
    for i in range(n):
        idx = np.concatenate([members[j] for j in draws[i]])
        take = x[idx]
        if stat == "median":
            out[i] = np.median(take)
        elif stat == "mean":
            out[i] = take.mean()
        elif stat == "mae_ratio":
            out[i] = take[:, 0].mean() / take[:, 1].mean()
        else:
            raise ValueError(stat)
    return out


def equivalence(boot: np.ndarray, point: float, margin: float,
                direction: int) -> dict:
    """TOST from a bootstrap distribution, plus the tightest excludable bound.

    Rejecting $H_0: |\\theta| \\ge \\Delta$ at $\\alpha$ each side means both
    $P^*(\\theta^* \\ge \\Delta) < \\alpha$ and $P^*(\\theta^* \\le -\\Delta) <
    \\alpha$, which happens exactly when $\\Delta$ exceeds both the upper
    $1-\\alpha$ and the lower $\\alpha$ percentile in magnitude --- i.e. when
    $\\Delta$ exceeds the larger endpoint of the $100(1-2\\alpha)\\%$ interval.
    That endpoint is therefore the smallest margin the data can reject, and it
    is reported as `bound`.

    `direction` names the sign the paper's rival hypothesis predicts (+1 the
    patch pushes the count toward the donor, -1 the probe beats the constant
    predictor), so the one-sided bound in that direction can be quoted on its
    own --- which is usually the honest number, since nobody feared an effect
    the other way.
    """
    lo90, hi90 = (float(np.percentile(boot, 100 * ALPHA)),
                  float(np.percentile(boot, 100 * (1 - ALPHA))))
    lo95, hi95 = (float(np.percentile(boot, 2.5)),
                  float(np.percentile(boot, 97.5)))
    p_upper = float(np.mean(boot >= margin))        # H0: theta >= +margin
    p_lower = float(np.mean(boot <= -margin))       # H0: theta <= -margin
    tost_p = max(p_upper, p_lower)
    bound = max(abs(hi90), abs(lo90))
    one_sided = hi90 if direction > 0 else lo90
    out = dict(point=float(point), ci90=[lo90, hi90], ci95=[lo95, hi95],
               margin=float(margin), tost_p=tost_p,
               equivalent_at_margin=bool(tost_p < ALPHA),
               bound=float(bound), one_sided_bound=float(one_sided),
               # How much room the exclusion has. A margin cleared by a hair is
               # a different claim from one cleared by a factor of five, and a
               # verdict table that prints only True/False hides which is which.
               margin_of_safety=float(margin - bound),
               margin_of_safety_ratio=float(margin / bound) if bound > 0 else None,
               direction=int(direction))
    # The reader would check this by hand; check it here instead. TOST at the
    # margin must agree with the interval it is equivalent to, or one of the
    # two is wrong.
    assert out["equivalent_at_margin"] == (bound < margin), \
        "TOST and the 90% interval disagree; one of them is miscomputed"
    assert lo95 <= lo90 <= hi90 <= hi95, "percentiles out of order"
    return out


def reps_at(delta_log: float, count: float) -> float:
    """A shift of `delta_log` in log(1+count), read as repetitions at `count`."""
    return (1.0 + count) * math.exp(delta_log) - (1.0 + count)


# ---------------------------------------------------------------- null 1


def rank1_patch(out: dict) -> None:
    """The rank-1 causal patch: how much count transfer is excluded?"""
    from analysis.causal_count_score import cell_summary, load_arm  # noqa: PLC0415

    published = json.loads(
        (REPO / "data/results/causal_count_followup.json").read_text()
    )["rank1_patch"]["cells"][RANK1_CELL]

    stim = {json.loads(l)["item_id"]: json.loads(l)
            for l in (REPO / "data/stimuli/stimuli.jsonl").open()}
    rows = load_arm("P1", stim)
    ref = {(r["recv_item"], r["seed"], r["patch_pos"]): r
           for r in rows if r["kind"] == "resume"}
    base = defaultdict(list)
    for r in rows:
        if r["kind"] == "baseline":
            base[(r["template"], r["recv_k"])].append(r["y"])

    signed, gaps, recv_counts, recv_items = [], [], [], []
    for r in rows:
        if r["kind"] != "patch" or r["donor_kind"] == "self":
            continue
        rr = ref.get((r["recv_item"], r["seed"], r["patch_pos"]))
        if rr is None:
            continue
        if f"{r['donor_kind']}|L{r['layer']}|P{r['patch_pos']}" != RANK1_CELL:
            continue
        # Same signing as causal_count_score.analyse_a: a cross-k donor carries
        # a direction, so the quantity tested is the shift *toward the donor*.
        direction = int(np.sign((r["donor_k"] or 0) - r["recv_k"]))
        signed.append((r["y"] - rr["y"]) * direction)
        g_hi = base.get((r["template"], r["donor_k"]), [])
        g_lo = base.get((r["template"], r["recv_k"]), [])
        gaps.append(abs(float(np.median(g_hi)) - float(np.median(g_lo)))
                    if g_hi and g_lo else float("nan"))
        recv_counts.append(float(rr["count"]))
        recv_items.append(r["recv_item"])
    signed = np.asarray(signed)

    # Re-derivation check: this file duplicates the pairing, so it must land on
    # the published cell exactly before any bound computed from it is believed.
    mine = cell_summary(signed)
    for key in ("n", "median", "ci_lo", "ci_hi", "n_pos", "n_neg"):
        assert abs(mine[key] - published[key]) < 1e-12, \
            f"rank-1 cell no longer reproduces: {key} {mine[key]} vs {published[key]}"

    gap = float(np.nanmedian(gaps))
    assert abs(gap - published["median_gap"]) < 1e-12, "donor/receiver gap moved"
    margin = FULL_TRANSFER_FRACTION * gap

    boot = boot_dist(signed, "median")
    eq = equivalence(boot, float(np.median(signed)), margin, direction=+1)
    eq_mean = equivalence(boot_dist(signed, "mean"), float(signed.mean()),
                          margin, direction=+1)
    eq_clu = equivalence(
        boot_dist_cluster(signed, np.array(recv_items), "median"),
        float(np.median(signed)), margin, direction=+1)
    eq_clu["n_clusters"] = len(set(recv_items))

    med_count = float(np.median(recv_counts))
    eq.update(
        cell=RANK1_CELL, n=int(signed.size), unit="log(1+count), signed toward donor",
        full_transfer_logcount=gap,
        bound_as_fraction_of_transfer=eq["bound"] / gap,
        one_sided_bound_as_fraction_of_transfer=eq["one_sided_bound"] / gap,
        median_reference_count=med_count,
        bound_repetitions_at_median_count=reps_at(eq["bound"], med_count),
        bound_repetitions_at_k={str(k): reps_at(eq["bound"], float(k))
                                for k in (16, 24, 32)},
        one_sided_bound_repetitions_at_k={
            str(k): reps_at(eq["one_sided_bound"], float(k)) for k in (16, 24, 32)},
        sensitivity_mean=eq_mean, sensitivity_cluster_by_receiver=eq_clu)
    eq["entitled_to_no"] = bool(eq["equivalent_at_margin"]
                                and eq_clu["equivalent_at_margin"])
    eq["verdict"] = (
        f"a shift larger than {eq['bound']:.3f} log-count "
        f"({100 * eq['bound_as_fraction_of_transfer']:.0f}% of a full transfer, "
        f"{eq['bound_repetitions_at_k']['24']:+.1f} repetitions at k=24) is "
        f"excluded at 95%"
        + ("; the paper may say the patch does not transfer the count"
           if eq["entitled_to_no"] else
           "; too wide to license 'not at all'"))

    print(f"\n[1] rank-1 patch, {RANK1_CELL}, n={eq['n']} paired shifts")
    print(f"    median {eq['point']:+.3f} log-count  "
          f"90% CI [{eq['ci90'][0]:+.3f}, {eq['ci90'][1]:+.3f}]  "
          f"95% CI [{eq['ci95'][0]:+.3f}, {eq['ci95'][1]:+.3f}]")
    print(f"    a full transfer would be {gap:+.3f}; margin (half) {margin:+.3f}; "
          f"TOST p={eq['tost_p']:.4f}")
    print(f"    excluded: any effect above {eq['bound']:.3f} log-count = "
          f"{100 * eq['bound_as_fraction_of_transfer']:.0f}% of a transfer = "
          f"{eq['bound_repetitions_at_k']['24']:.1f} repetitions at k=24 "
          f"({eq['bound_repetitions_at_median_count']:.1f} at the arm's own "
          f"median count of {med_count:.0f})")
    print(f"    one-sided (toward the donor, the only direction predicted): "
          f"{eq['one_sided_bound']:+.3f} log-count = "
          f"{eq['one_sided_bound_repetitions_at_k']['24']:+.1f} at k=24")
    print(f"    clustering by receiver item ({eq_clu['n_clusters']} clusters): "
          f"bound {eq_clu['bound']:.3f} log-count "
          f"({100 * eq_clu['bound'] / gap:.0f}% of a transfer), "
          f"equivalent at margin: {eq_clu['equivalent_at_margin']}")
    print(f"    -> {eq['verdict']}")
    out["rank1_patch"] = eq


# ---------------------------------------------------------------- null 2


def duration_supply(out: dict) -> None:
    """F5-TTS handed the right duration at k>=12: how much gain is excluded?"""
    published = json.loads(
        (REPO / "data/results/duration_intervention.json").read_text())
    split, kmin = published["split"], published["kmin"]

    def load(path: str) -> pd.DataFrame:
        d = pd.read_csv(REPO / path)
        d = d[d.family == "word_rep"]
        # Same population as duration_intervention.py. `ablations=True` because
        # f5tts and f5fix are both listed as ablation arms in population.py and
        # are the whole experiment here; the two arms are never pooled --- they
        # are paired below, item by item.
        d, _ = panel(d, ablations=True)
        d = d[d.k >= kmin]
        return d.assign(err=(d.count_a - d.k) / d.k)

    free = load("data/results/behavioural_f5.csv")
    fixed = load("data/results/behavioural_f5fix.csv")
    hi_f = free[free.k >= split].set_index(["item_id", "seed"]).sort_index()
    hi_x = fixed[fixed.k >= split].set_index(["item_id", "seed"]).sort_index()

    assert hi_f.index.equals(hi_x.index), \
        "the two F5 arms are not item-for-item paired; a paired test is invalid"
    assert len(hi_f) == published["high_k"]["free"]["n"] == 60, "high-k n moved"
    ex_f = (hi_f.err.values == 0).astype(float)
    ex_x = (hi_x.err.values == 0).astype(float)
    assert abs(ex_f.mean() - published["high_k"]["free"]["exact"]) < 1e-12
    assert abs(ex_x.mean() - published["high_k"]["fixed"]["exact"]) < 1e-12
    gain = float(ex_x.mean() - ex_f.mean())
    assert abs(gain - published["high_k_gain"]) < 1e-12, "high-k gain moved"

    low_gain = float(published["low_k_gain"])
    margin = DUR_MARGIN_FRACTION * low_gain
    paired = ex_x - ex_f
    boot = boot_dist(paired, "mean")
    eq = equivalence(boot, gain, margin, direction=+1)
    tmpl = hi_f.template.values
    eq_clu = equivalence(boot_dist_cluster(paired, tmpl, "mean"), gain, margin,
                         direction=+1)
    eq_clu["n_clusters"] = int(len(set(tmpl.tolist())))

    # The paired disagreement counts, which are the whole information content of
    # a McNemar-style comparison: 60 pairs, but only the discordant ones speak.
    b = int(((ex_x == 1) & (ex_f == 0)).sum())
    c = int(((ex_f == 1) & (ex_x == 0)).sum())
    # A continuous reading of the same pairs, in case exact-match is too coarse
    # to see a real but partial improvement: median |relative count error|,
    # which falls if the intervention helps without making the count exact.
    abs_err = np.stack([np.abs(hi_x.err.values), np.abs(hi_f.err.values)], axis=1)
    eq_err = equivalence(boot_dist(abs_err, "median_diff"),
                         float(np.median(abs_err[:, 0]) - np.median(abs_err[:, 1])),
                         margin=0.10, direction=-1)

    eq.update(n_pairs=int(len(hi_f)), unit="exact-rendering rate, difference",
              discordant_fixed_only=b, discordant_free_only=c,
              low_k_gain=low_gain,
              bound_points=100 * eq["bound"],
              one_sided_bound_points=100 * eq["one_sided_bound"],
              low_k_gain_excluded=bool(eq["bound"] < low_gain),
              sensitivity_abs_rel_error=eq_err,
              sensitivity_cluster_by_template=eq_clu,
              median_abs_rel_error_free=float(np.median(abs_err[:, 1])),
              median_abs_rel_error_fixed=float(np.median(abs_err[:, 0])),
              # The same bound as a ceiling a reader can picture: the best the
              # intervention could be doing, in the units of the table it sits
              # next to, and in whole items out of sixty.
              bound_ceiling_exact_pct=100 * (float(ex_f.mean())
                                             + eq["one_sided_bound"]),
              bound_extra_items=eq["one_sided_bound"] * len(hi_f))
    eq["entitled_to_no"] = bool(eq["equivalent_at_margin"]
                                and eq_clu["equivalent_at_margin"])
    eq["verdict"] = (
        f"an improvement larger than {100 * eq['one_sided_bound']:.1f} points is "
        f"excluded at 95% (one-sided); the low-k gain of "
        f"{100 * low_gain:.1f} points is "
        + ("excluded" if eq["low_k_gain_excluded"] else "NOT excluded")
        + (", so the low/high split is established but a smaller genuine benefit "
           "is not ruled out" if eq["low_k_gain_excluded"] else ""))

    print(f"\n[2] F5-TTS duration supplied, k>={split}, n={eq['n_pairs']} pairs")
    print(f"    exact renderings {100 * ex_f.mean():.1f}% -> {100 * ex_x.mean():.1f}%"
          f"  (gain {100 * gain:+.1f} points; discordant pairs {b} up, {c} down)")
    print(f"    90% CI [{100 * eq['ci90'][0]:+.1f}, {100 * eq['ci90'][1]:+.1f}] pts  "
          f"95% CI [{100 * eq['ci95'][0]:+.1f}, {100 * eq['ci95'][1]:+.1f}] pts  "
          f"TOST p={eq['tost_p']:.4f} at margin {100 * margin:.1f} pts")
    print(f"    excluded: any gain above {100 * eq['bound']:.1f} points "
          f"(one-sided {100 * eq['one_sided_bound']:+.1f}); the low-k gain of "
          f"{100 * low_gain:.1f} points is "
          f"{'excluded' if eq['low_k_gain_excluded'] else 'NOT excluded'}")
    print(f"    clustering by carrier template ({eq_clu['n_clusters']} clusters): "
          f"bound {100 * eq_clu['bound']:.1f} pts, equivalent at margin: "
          f"{eq_clu['equivalent_at_margin']}")
    print(f"    -> {eq['verdict']}")
    out["duration_supply"] = eq


# ---------------------------------------------------------------- null 3


def probe_vs_constant(out: dict) -> None:
    """The probe past the horizon against a predictor that answers the mean."""
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "probe_count", REPO / "analysis/probe_count.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    def preds_loto(X: np.ndarray, y: np.ndarray, tmpl: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
        """`probe_count.ridge_loto`, returning the held-out predictions.

        Identical estimator and identical folds; the original returns only the
        two summary scalars, and per-item errors are what a bootstrap needs.
        The scalars are recomputed from these predictions and asserted equal to
        the published ones below, which is the check that this is the same probe.
        """
        X = X.astype(np.float64)
        preds = np.full(y.shape, np.nan)
        for t in sorted(set(tmpl.tolist())):
            te = tmpl == t
            tr = ~te
            if tr.sum() < 6 or te.sum() < 1:
                continue
            mu = X[tr].mean(0, keepdims=True)
            A = X[tr] - mu
            ybar = y[tr].mean()
            K = A @ A.T + alpha * np.eye(A.shape[0])
            dual = np.linalg.solve(K, y[tr] - ybar)
            preds[te] = ((X[te] - mu) @ (A.T @ dual)) + ybar
        return preds

    compare = json.loads(
        (REPO / "data/results/probe_horizon_compare.json").read_text())["rows"]
    stim = {}
    for f in PROBE_STIM:
        for line in (REPO / f).open():
            it = json.loads(line)
            stim[it["item_id"]] = it

    res: dict = {"margins": {"tie_band": PROBE_TIE_BAND,
                             "rival_control_ratio": PROBE_RIVAL_MARGIN},
                 "checkpoints": {}}
    print(f"\n[3] probe vs the mean-answering predictor, past the horizon")
    print(f"{'checkpoint':>12s} {'arm':>8s} {'n':>3s} {'ratio':>6s} "
          f"{'90% CI':>16s} {'bound':>7s} {'clust.':>7s}  verdict")
    for model, path in PROBE_PAST.items():
        feats = pc.load_features(model, stim, 0)
        for fam, src, tag in (("word_rep", path, "past"),
                              ("control_word", PROBE_CTL[model], "control")):
            pub = json.loads((REPO / src).read_text())["models"][model][fam]["late"]
            d = feats[fam]
            y = np.log2(d["k"])
            p = preds_loto(d["Xl"][:, pub["best_layer_idx"], :], y, d["tmpl"])
            mae = float(np.abs(y - p).mean())
            r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            assert abs(mae - pub["best"]["mae"]) < 1e-12, \
                f"{model}/{fam} probe MAE no longer reproduces"
            assert abs(r2 - pub["best"]["r2"]) < 1e-12, \
                f"{model}/{fam} probe R2 no longer reproduces"

            # The rival is the constant predictor of probe_past_horizon.py: it
            # answers the mean of the target and never looks at a state. Its
            # per-item error is therefore |y_i - mean(y)|, and the comparison is
            # paired item by item.
            e_probe = np.abs(y - p)
            e_const = np.abs(y - y.mean())
            ratio = float(e_probe.mean() / e_const.mean())
            key = f"{tag}:{model}"
            assert abs(ratio - compare[key]["mae_ratio"]) < 1e-12, \
                f"{key} MAE ratio no longer reproduces"
            assert abs(e_const.mean() - compare[key]["const_mae"]) < 1e-12

            paired = np.stack([e_probe, e_const], axis=1)
            boot = boot_dist(paired, "mae_ratio")
            boot_clu = boot_dist_cluster(paired, d["tmpl"], "mae_ratio")
            # The estimand is the *improvement* 1 - ratio, so that zero is the
            # null and the equivalence machinery reads the same way as above.
            improvement = 1.0 - ratio
            eq = equivalence(1.0 - boot, improvement, PROBE_RIVAL_MARGIN,
                             direction=+1)
            eq_tie = equivalence(1.0 - boot, improvement, PROBE_TIE_BAND,
                                 direction=+1)
            eq_clu = equivalence(1.0 - boot_clu, improvement,
                                 PROBE_RIVAL_MARGIN, direction=+1)
            eq_clu["n_clusters"] = int(len(set(d["tmpl"].tolist())))
            eq.update(model=model, arm=tag, n_items=int(y.size),
                      sensitivity_cluster_by_template=eq_clu,
                      layer=int(pub["best_layer_idx"]),
                      mae=mae, const_mae=float(e_const.mean()), mae_ratio=ratio,
                      ratio_ci90=[1 - eq["ci90"][1], 1 - eq["ci90"][0]],
                      ratio_ci95=[1 - eq["ci95"][1], 1 - eq["ci95"][0]],
                      unit="improvement in MAE over the constant predictor, "
                           "as a fraction of the constant predictor's MAE",
                      equivalent_at_tie_band=eq_tie["equivalent_at_margin"],
                      tost_p_tie_band=eq_tie["tost_p"],
                      # The probe layer was chosen to maximise R^2 over all
                      # layers. Resampling items at the *chosen* layer holds that
                      # choice fixed, so this interval is if anything too narrow
                      # -- which matters, because a too-narrow interval makes an
                      # equivalence claim look stronger than it is.
                      layer_selection_not_bootstrapped=True,
                      # The ratio in units a reader can picture. The constant
                      # predictor is off by 0.5 log2 units on this k grid, so an
                      # excluded improvement of `bound` is an excluded reduction
                      # of bound*0.5 log2 units -- i.e. the probe cannot be
                      # locating k to better than this factor.
                      bound_log2_units=eq["bound"] * float(e_const.mean()),
                      bound_factor_in_k=float(
                          2 ** (eq["bound"] * float(e_const.mean()))))
            eq["verdict"] = (
                f"improvements above {100 * eq['one_sided_bound']:.0f}% of the "
                f"constant predictor's MAE are excluded")
            res["checkpoints"][key] = eq
            print(f"{model:>12s} {tag:>8s} {y.size:3d} {ratio:6.2f} "
                  f"[{eq['ratio_ci90'][0]:5.2f},{eq['ratio_ci90'][1]:5.2f}] "
                  f"{eq['bound']:7.2f} {eq_clu['bound']:7.2f}  {eq['verdict']}")

    null_side = ["past:llasa1b", "past:llasa3b"]     # the checkpoints called
    # indistinguishable by probe_past_horizon.py's tie band, i.e. the ones whose
    # null the paper leans on.
    res["null_side"] = null_side
    res["worst_bound"] = max(res["checkpoints"][k]["bound"] for k in null_side)
    res["worst_bound_log2_units"] = max(
        res["checkpoints"][k]["bound_log2_units"] for k in null_side)
    res["worst_bound_factor_in_k"] = max(
        res["checkpoints"][k]["bound_factor_in_k"] for k in null_side)
    res["worst_bound_clustered"] = max(
        res["checkpoints"][k]["sensitivity_cluster_by_template"]["bound"]
        for k in null_side)
    res["rival_effect_excluded"] = bool(
        all(res["checkpoints"][k]["equivalent_at_margin"] for k in null_side))
    res["rival_effect_excluded_clustered"] = bool(
        all(res["checkpoints"][k]["sensitivity_cluster_by_template"]
            ["equivalent_at_margin"] for k in null_side))
    res["tie_band_excluded"] = bool(
        all(res["checkpoints"][k]["equivalent_at_tie_band"] for k in null_side))
    # Both readings must agree before the paper is allowed the word. The row
    # bootstrap is the generous one and the cluster bootstrap is the design's
    # own unit of evidence; where they disagree the null is fragile, not clean.
    res["entitled_to_no"] = bool(res["rival_effect_excluded"]
                                 and res["rival_effect_excluded_clustered"])
    # Three separate reasons this one is the shakiest of the three nulls, all
    # worth saying out loud rather than hiding behind a True.
    res["caveats"] = []
    if res["entitled_to_no"]:
        slack = PROBE_RIVAL_MARGIN - res["worst_bound"]
        res["margin_of_safety"] = float(slack)
        if slack < 0.05:
            res["caveats"].append(
                f"the rival effect is excluded by only {slack:.02f} in MAE-ratio "
                f"units ({PROBE_RIVAL_MARGIN:.2f} margin vs "
                f"{res['worst_bound']:.2f} bound)")
    res["caveats"].append(
        "the probe layer was chosen to maximise R^2 and that choice is held "
        "fixed under resampling, so every interval here is narrower than the "
        "procedure's true sampling variability")
    res["caveats"].append(
        "three carrier templates are both the sampled unit and the "
        "cross-validation folds, so the clustered bootstrap has three clusters "
        "and is coarse")
    res["fragile"] = bool(
        res["rival_effect_excluded"] != res["rival_effect_excluded_clustered"]
        or res.get("margin_of_safety", 1.0) < 0.05)
    res["verdict"] = (
        f"at n=12 per checkpoint the tightest excludable improvement on the null "
        f"side is {100 * res['worst_bound']:.0f}% of the constant predictor's MAE "
        f"({100 * res['worst_bound_clustered']:.0f}% clustering by template, the "
        f"design's own unit); the rival's predicted effect "
        f"({100 * PROBE_RIVAL_MARGIN:.0f}%) is "
        + ("excluded" if res["entitled_to_no"] else "NOT excluded")
        + ", and the repo's 2% tie band is "
        + ("excluded" if res["tie_band_excluded"] else "far outside reach")
        + (" -- but only just, and this null is the shakiest of the three"
           if res["fragile"] else ""))
    print(f"\n    -> {res['verdict']}")
    for c in res["caveats"]:
        print(f"       caveat: {c}")
    out["probe_vs_constant"] = res


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/results/equivalence.json"))
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["rank1", "duration", "probe"])
    args = ap.parse_args()

    out: dict = {"alpha": ALPHA, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                 "margins_precommitted": {
                     "rank1_fraction_of_full_transfer": FULL_TRANSFER_FRACTION,
                     "duration_fraction_of_low_k_gain": DUR_MARGIN_FRACTION,
                     "probe_tie_band": PROBE_TIE_BAND,
                     "probe_rival_control_ratio": PROBE_RIVAL_MARGIN}}
    if "rank1" not in args.skip:
        rank1_patch(out)
    if "duration" not in args.skip:
        duration_supply(out)
    if "probe" not in args.skip:
        probe_vs_constant(out)

    entitled = {k: v.get("entitled_to_no") for k, v in out.items()
                if isinstance(v, dict) and "entitled_to_no" in v}
    out["entitled_to_no"] = entitled
    print("\nMay the paper write 'no effect'?")
    for k, v in entitled.items():
        print(f"  {k:22s} {'yes' if v else 'NO -- report the bound instead'}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
