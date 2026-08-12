#!/usr/bin/env python3
"""n=6 checkpoints are n=3 families. Model that, instead of apologising for it.

Seven reviewers across three rounds made the same objection and they were right:
`analysis/checkpoint_level.py` treats six checkpoints as six replicates when
three of them are Llasa and two are Qwen3-TTS, `analysis/family_level.py` fixes
the counting but replaces inference with a range, and the paper covered the gap
between them with a sentence about the exact signed-rank floor at n=6. A floor
on a *test* is not an answer about an *estimate*. The reviewers asked for the
three things that are: a model with architecture as a random effect, a
hierarchical posterior that partially pools the six checkpoints into the three
families, and a robustness envelope over the forking choices rather than one
preferred number. This script is those three things.

What it does, and why each piece is here:

**0. Reproduction first.** The trap this repo has fallen into twice is a new
script that disagrees with an old one because it is quietly fitting different
rows. So before anything new is computed, the per-checkpoint and per-family
contrasts are recomputed from `src.common.population.panel` and asserted equal
to `checkpoint_level.json` and `family_level.json`. If that assertion fails,
nothing below it is trustworthy and the script stops.

**1. Mixed-effects and cluster-robust models.** Outcome is whether a generation
is rendered exactly correct; fixed effects are arm and log2 k; random effects
are architecture family, checkpoint nested in family, stimulus crossed with
both, and --- the one that matters --- a family-varying arm slope, which is what
forces the arm effect's standard error to be paid for out of three families
rather than 982 generations. Fitted as a linear probability model so the
coefficient is already in percentage points, with a binomial GEE as a check that
the link function is not doing the work. Five different standard errors are
reported, including the ones that make the effect look weakest, because with
three clusters the variance component sits on its boundary and any single
interval would be a choice dressed as a result.

**2. A hierarchical Bayesian model.** Partial pooling is the honest answer to
"six non-independent checkpoints": the model is told about the nesting and
returns a posterior whose width reflects it. Binomial likelihood, logit scale,
family and checkpoint offsets on both the control level and the gap. Sampled
twice by two independent implementations --- NUTS via numpyro, and a hand-rolled
blocked random-walk Metropolis that depends on nothing but numpy --- and the two
must agree or the script says so. The quantity to quote is not the panel mean
but the *predictive* gap for an architecture family not in the panel, which is
the claim the paper actually makes.

**3. A specification curve.** Every forking choice we made, crossed: which
exclusion rules are on, cycled versus never-cycled controls, pooled versus
checkpoint versus family aggregation, the k threshold, the outcome definition,
and one seed versus three. The populations come from the same `panel()` calls
that `exclusion_sensitivity.py` and `aperiodic_controls.py` use, so the arms
that overlap those scripts reproduce them exactly.

Usage:  python analysis/hierarchical.py            # base env
        python analysis/hierarchical.py --quick    # shorter chains, for a smoke test
"""
from __future__ import annotations

# Statistics only: nothing here needs a GPU, and jax would happily grab one.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import panel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# The clustering the whole script is about. Same map as analysis/family_level.py;
# imported there from nothing, so it is restated here and asserted to match.
FAMILY = {"llasa1b": "Llasa", "llasa3b": "Llasa", "llasa8b": "Llasa",
          "xtts2": "XTTS", "qwen06b": "Qwen3-TTS", "qwen17b": "Qwen3-TTS"}
PAIR = ["word_rep", "control_word"]
DEGENERATE = ["empty", "degenerate"]


# ----------------------------------------------------------------------------
# population
# ----------------------------------------------------------------------------

def load(behavioural: str, kmin: int = 6, **flags) -> pd.DataFrame:
    """The reportable rows, exactly as checkpoint_level.py builds them."""
    d = pd.read_csv(behavioural)
    d = d[d.family.isin(PAIR)]
    d, drop = panel(d, **flags)
    d = d[d.k >= kmin]
    return annotate(d), drop


def annotate(d: pd.DataFrame) -> pd.DataFrame:
    """Add the model matrix columns. `stim` strips the ct_/wr_ prefix, so a
    repeated item and its length-matched control share one stimulus id --- which
    is the pairing the design has and the model should be told about."""
    d = d.copy()
    d["err"] = (d.count_a - d.k) / d.k
    d["exact"] = (d.err == 0).astype(float)
    d["rep"] = (d.family == "word_rep").astype(float)
    d["arch"] = d.model.map(FAMILY)
    d["lk"] = np.log2(d.k.astype(float))
    d["stim"] = (d.item_id.astype(str)
                 .str.replace(r"^(ct|wr)_", "", regex=True))
    d["item"] = d.stim.str.replace(r"_k\d+$", "", regex=True)
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 0. reproduce what is already reported, before reporting anything new
# ----------------------------------------------------------------------------

def per_checkpoint_gaps(d: pd.DataFrame) -> tuple[dict, dict]:
    """(exact-rate gap, median-relative-error gap) per checkpoint, control minus
    repeated, signed so positive means the predicted effect is present."""
    exact, err = {}, {}
    for m, g in d.groupby("model", observed=True):
        r, c = g[g.rep == 1], g[g.rep == 0]
        if len(r) and len(c):
            exact[m] = float(c.exact.mean() - r.exact.mean())
            err[m] = float(c.err.median() - r.err.median())
    return exact, err


def reproduction_check(d: pd.DataFrame, out: dict) -> None:
    """Assert this script's population is the one the paper already reports.

    A new script that disagrees with an old one on the same estimator has almost
    always changed the rows, not the maths (`horizon_forms.py` lost an hour to
    exactly this). So the check runs first and raises rather than warns.
    """
    exact, err = per_checkpoint_gaps(d)
    rec: dict = {"n_rows": int(len(d)),
                 "n_repeated": int((d.rep == 1).sum()),
                 "n_control": int((d.rep == 0).sum()),
                 "n_checkpoints": len(exact), "n_families": d.arch.nunique()}
    # Arithmetic the reader would otherwise have to trust.
    assert rec["n_repeated"] + rec["n_control"] == rec["n_rows"], "arms do not sum"
    assert rec["n_checkpoints"] == 6 and rec["n_families"] == 3, rec
    assert set(d.model.unique()) == set(FAMILY), "panel membership changed"

    ck_path = REPO / "data/results/checkpoint_level.json"
    fl_path = REPO / "data/results/family_level.json"
    agree = {}
    if ck_path.exists():
        ck = json.loads(ck_path.read_text())
        for key, mine in (("exact_rate_gap", exact), ("count_error_gap", err)):
            theirs = ck.get(key, {}).get("per_model", {})
            if not theirs:
                continue
            worst = max(abs(mine[m] - theirs[m]) for m in theirs)
            agree[f"checkpoint_level.{key}"] = dict(
                max_abs_diff=float(worst), matches=bool(worst < 1e-9))
            assert worst < 1e-9, (
                f"{key} disagrees with checkpoint_level.py by {worst:.2e} -- "
                "suspect the population before the maths")
    if fl_path.exists():
        fl = json.loads(fl_path.read_text())
        theirs = fl.get("exact_rate_gap", {}).get("per_family", {})
        if theirs:
            fam = family_means(exact)
            worst = max(abs(fam[f] - theirs[f]) for f in theirs)
            agree["family_level.exact_rate_gap"] = dict(
                max_abs_diff=float(worst), matches=bool(worst < 1e-9))
            assert worst < 1e-9, "family means disagree with family_level.py"
            # family_level.py averages checkpoints within a family; check that
            # the identity actually holds rather than assuming the file is right.
            llasa = np.mean([exact[m] for m in ("llasa1b", "llasa3b", "llasa8b")])
            assert abs(llasa - theirs["Llasa"]) < 1e-9

    rec["agreement"] = agree
    rec["reproduces_existing"] = bool(agree) and all(
        v["matches"] for v in agree.values())
    out["reproduction"] = rec

    print(f"population: {rec['n_rows']} rows "
          f"= {rec['n_repeated']} repeated + {rec['n_control']} control, "
          f"{rec['n_checkpoints']} checkpoints in {rec['n_families']} families")
    for k, v in agree.items():
        print(f"  reproduces {k:38s} max|diff| {v['max_abs_diff']:.2e}  OK")
    if not agree:
        print("  (no existing result files to check against)")


def family_means(per_ckpt: dict[str, float]) -> dict[str, float]:
    fam: dict[str, list[float]] = {}
    for m, v in per_ckpt.items():
        fam.setdefault(FAMILY.get(m, m), []).append(float(v))
    return {f: float(np.mean(v)) for f, v in fam.items()}


# ----------------------------------------------------------------------------
# 1. mixed-effects and cluster-robust models
# ----------------------------------------------------------------------------

def mixed_models(d: pd.DataFrame, out: dict) -> None:
    """The arm effect with architecture treated as what it is: a random effect
    with three levels."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed; skipping the frequentist models")
        out["mixed"] = {"available": False}
        return

    from scipy import stats as sps

    res: dict = {"available": True, "outcome": "exact (0/1), linear probability",
                 "fixed": "arm + log2 k", "sign": "reported as control - repeated"}

    # --- primary: crossed random intercepts plus a family-varying arm slope.
    # statsmodels has one `groups`, so the crossed structure goes in as variance
    # components under a single all-ones group, which is the documented recipe.
    dd = d.assign(one=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("exact ~ rep + lk", dd, groups=dd["one"],
                         vc_formula={"arch": "0+C(arch)",
                                     "ckpt": "0+C(model)",
                                     "stim": "0+C(stim)",
                                     "arch_x_rep": "0+C(arch):rep"})
        fit = md.fit(reml=True)

    b = -float(fit.params["rep"])          # flip: positive = effect present
    se = float(fit.bse["rep"])
    z = 1.959963985
    t2 = float(sps.t.ppf(0.975, df=2))     # three families, two degrees of freedom
    vc = dict(zip(md.exog_vc.names, np.asarray(fit.vcomp, dtype=float).ravel()))
    boundary = {k: bool(v < 1e-6) for k, v in vc.items()}

    res["mixedlm"] = dict(
        arm_effect_pts=100 * b, se_pts=100 * se,
        wald_lo=100 * (b - z * se), wald_hi=100 * (b + z * se),
        t2_lo=100 * (b - t2 * se), t2_hi=100 * (b + t2 * se),
        vcomp={k: float(v) for k, v in vc.items()},
        # the same components as between-unit SDs in percentage points, which is
        # the scale anyone reading the paper is holding in their head
        vcomp_sd_pts={k: 100 * float(np.sqrt(max(v, 0.0))) for k, v in vc.items()},
        vcomp_on_boundary=boundary,
        converged=bool(fit.converged),
        note=("with three families the family-varying arm slope is barely "
              "identified and REML shrinks it hard, so this Wald interval is "
              "narrower than the between-family spread alone would license; the "
              "clustered, family-t and bootstrap intervals below are the widths "
              "to quote"))

    # --- cluster-robust linear probability models, clustered two ways.
    for label, g in (("family", d["arch"]), ("checkpoint", d["model"])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = smf.ols("exact ~ rep + lk", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": g})
        nb = int(pd.Series(g).nunique())
        bb, ss = -float(o.params["rep"]), float(o.bse["rep"])
        tc = float(sps.t.ppf(0.975, df=nb - 1))
        res[f"cluster_{label}"] = dict(
            n_clusters=nb, arm_effect_pts=100 * bb, se_pts=100 * ss,
            lo=100 * (bb - tc * ss), hi=100 * (bb + tc * ss),
            crit=f"t({nb - 1})")

    # --- binomial GEE, to check the linear probability model is not the reason.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gee = smf.gee("exact ~ rep + lk", groups="arch", data=d,
                      family=sm.families.Binomial()).fit()
    lb, ls = -float(gee.params["rep"]), float(gee.bse["rep"])
    res["gee_binomial_family_clustered"] = dict(
        log_odds=lb, se=ls, lo=lb - t2 * ls, hi=lb + t2 * ls,
        odds_ratio=float(np.exp(lb)), crit="t(2)")

    # --- the family-level t interval: three numbers, two degrees of freedom.
    exact, _ = per_checkpoint_gaps(d)
    fam = family_means(exact)
    v = np.array(sorted(fam.values()))
    m, s = float(v.mean()), float(v.std(ddof=1))
    se_f = s / np.sqrt(len(v))
    res["family_t_interval"] = dict(
        per_family={k: 100 * x for k, x in fam.items()},
        n=len(v), mean_pts=100 * m, sd_pts=100 * s,
        lo=100 * (m - t2 * se_f), hi=100 * (m + t2 * se_f), crit="t(2)")

    # --- cluster bootstrap, resampling whole families and whole checkpoints.
    rng = np.random.default_rng(0)
    for label, unit in (("family", "arch"), ("checkpoint", "model")):
        keys = sorted(d[unit].unique())
        idx = {k: d.index[d[unit] == k].to_numpy() for k in keys}
        draws = []
        for _ in range(20000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            rows = np.concatenate([idx[keys[i]] for i in pick])
            s = d.loc[rows]
            r, c = s[s.rep == 1], s[s.rep == 0]
            if len(r) and len(c):
                draws.append(c.exact.mean() - r.exact.mean())
        a = np.array(draws)
        res[f"cluster_bootstrap_{label}"] = dict(
            n_units=len(keys), mean_pts=100 * float(a.mean()),
            lo=100 * float(np.percentile(a, 2.5)),
            hi=100 * float(np.percentile(a, 97.5)),
            frac_positive=float((a > 0).mean()))

    # Least favourable reading across every interval computed here.
    los = [res["mixedlm"]["t2_lo"], res["cluster_family"]["lo"],
           res["cluster_checkpoint"]["lo"], res["family_t_interval"]["lo"],
           res["cluster_bootstrap_family"]["lo"],
           res["cluster_bootstrap_checkpoint"]["lo"]]
    res["worst_case_lower_bound_pts"] = float(min(los))
    res["all_intervals_exclude_zero"] = bool(min(los) > 0)
    out["mixed"] = res

    print("\n1. mixed-effects / cluster-robust, arm effect in percentage points")
    print(f"   {'MixedLM (crossed RE, family-varying arm slope)':46s} "
          f"{res['mixedlm']['arm_effect_pts']:+6.1f}  "
          f"Wald [{res['mixedlm']['wald_lo']:+.1f},{res['mixedlm']['wald_hi']:+.1f}]  "
          f"t(2) [{res['mixedlm']['t2_lo']:+.1f},{res['mixedlm']['t2_hi']:+.1f}]")
    for k, lab in (("cluster_family", "OLS, clustered by family (G=3)"),
                   ("cluster_checkpoint", "OLS, clustered by checkpoint (G=6)"),
                   ("family_t_interval", "family means, t(2)"),
                   ("cluster_bootstrap_family", "cluster bootstrap over families"),
                   ("cluster_bootstrap_checkpoint", "cluster bootstrap over checkpoints")):
        r = res[k]
        val = r.get("arm_effect_pts", r.get("mean_pts"))
        print(f"   {lab:46s} {val:+6.1f}  [{r['lo']:+.1f},{r['hi']:+.1f}]")
    g = res["gee_binomial_family_clustered"]
    print(f"   {'GEE binomial, clustered by family':46s} "
          f"{g['log_odds']:+6.2f} log-odds [{g['lo']:+.2f},{g['hi']:+.2f}] "
          f"(OR {g['odds_ratio']:.0f})")
    print("   random-effect SDs (points): "
          + ", ".join(f"{k}={v:.1f}" for k, v in res["mixedlm"]["vcomp_sd_pts"].items()))
    onb = [k for k, v in res["mixedlm"]["vcomp_on_boundary"].items() if v]
    if onb:
        print(f"   BOUNDARY: {', '.join(onb)} estimated at zero -- with three "
              "families the model\n   cannot resolve that variance, so its Wald "
              "interval is too narrow and the\n   clustered and bootstrap "
              "intervals above are the ones to quote.")
    else:
        print("   the family-varying arm slope is shrunk toward zero "
              f"({res['mixedlm']['vcomp_sd_pts'].get('arch_x_rep', float('nan')):.1f} "
              "pts SD against\n   "
              f"{res['family_t_interval']['sd_pts']:.1f} pts observed across the "
              "three families), so the Wald interval\n   understates the width "
              "three families license; quote the t(2) or bootstrap one.")
    print(f"   least favourable lower bound across all six intervals: "
          f"{res['worst_case_lower_bound_pts']:+.1f} pts -> "
          + ("clears zero" if res["all_intervals_exclude_zero"]
             else "DOES NOT clear zero"))


# ----------------------------------------------------------------------------
# 2. hierarchical Bayes
# ----------------------------------------------------------------------------

def cell_counts(d: pd.DataFrame) -> dict:
    """Six checkpoints x two arms, as successes out of trials."""
    models = sorted(d.model.unique())
    fams = sorted({FAMILY[m] for m in models})
    rows = dict(models=models, families=fams,
                fam_of=[fams.index(FAMILY[m]) for m in models],
                y_ctl=[], n_ctl=[], y_rep=[], n_rep=[])
    for m in models:
        g = d[d.model == m]
        for arm, tag in ((0.0, "ctl"), (1.0, "rep")):
            s = g[g.rep == arm]
            rows[f"y_{tag}"].append(int(s.exact.sum()))
            rows[f"n_{tag}"].append(int(len(s)))
    for tag in ("ctl", "rep"):
        assert all(y <= n for y, n in zip(rows[f"y_{tag}"], rows[f"n_{tag}"]))
    assert sum(rows["n_ctl"]) + sum(rows["n_rep"]) == len(d), "cells lose rows"
    return rows


def _logsig(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def log_post(theta: np.ndarray, C: dict, prior_sd: float) -> float:
    """Log posterior of the non-centred hierarchical binomial model.

    theta = [alpha, Delta, log s_af, log s_ac, log s_df, log s_dc,
             z_af(3), z_ac(6), z_df(3), z_dc(6)]
    """
    nf, nc = len(C["families"]), len(C["models"])
    alpha, Delta = theta[0], theta[1]
    ls = theta[2:6]
    sig = np.exp(ls)
    o = 6
    z_af = theta[o:o + nf]; o += nf
    z_ac = theta[o:o + nc]; o += nc
    z_df = theta[o:o + nf]; o += nf
    z_dc = theta[o:o + nc]

    fi = C["fam_idx"]
    eta_ctl = alpha + sig[0] * z_af[fi] + sig[1] * z_ac
    gap = Delta + sig[2] * z_df[fi] + sig[3] * z_dc
    eta_rep = eta_ctl - gap

    ll = float(np.sum(C["y_ctl"] * _logsig(eta_ctl)
                      + (C["n_ctl"] - C["y_ctl"]) * _logsig(-eta_ctl)
                      + C["y_rep"] * _logsig(eta_rep)
                      + (C["n_rep"] - C["y_rep"]) * _logsig(-eta_rep)))
    lp = (-0.5 * (alpha / 5.0) ** 2 - 0.5 * (Delta / 5.0) ** 2
          - 0.5 * float(np.sum(z_af ** 2)) - 0.5 * float(np.sum(z_ac ** 2))
          - 0.5 * float(np.sum(z_df ** 2)) - 0.5 * float(np.sum(z_dc ** 2))
          # half-normal(prior_sd) on each sigma, with the log-scale Jacobian
          + float(np.sum(-0.5 * (sig / prior_sd) ** 2 + ls)))
    return ll + lp


def metropolis(C: dict, prior_sd: float, n_warm: int, n_draw: int,
               thin: int, chains: int, seed: int) -> np.ndarray:
    """Blocked adaptive random-walk Metropolis. Nothing but numpy, so this runs
    wherever the repo runs, and it exists to be checked against NUTS rather than
    trusted alone."""
    nf, nc = len(C["families"]), len(C["models"])
    dim = 6 + 2 * nf + 2 * nc
    blocks = [np.arange(0, 2), np.arange(2, 6),
              np.arange(6, 6 + nf), np.arange(6 + nf, 6 + nf + nc),
              np.arange(6 + nf + nc, 6 + 2 * nf + nc),
              np.arange(6 + 2 * nf + nc, dim)]
    keep = np.empty((chains, n_draw // thin, dim))
    for ch in range(chains):
        rng = np.random.default_rng(seed + 1000 * ch)
        th = np.zeros(dim)
        th[0] = rng.normal(2.0, 1.0)      # dispersed inits, so R-hat can bite
        th[1] = rng.normal(3.0, 1.5)
        th[2:6] = rng.normal(-0.5, 0.5, 4)
        th[6:] = rng.normal(0, 0.5, dim - 6)
        lpost = log_post(th, C, prior_sd)
        step = np.full(len(blocks), 0.3)
        acc = np.zeros(len(blocks)); tries = np.zeros(len(blocks))
        j = 0
        for it in range(n_warm + n_draw):
            for bi, blk in enumerate(blocks):
                prop = th.copy()
                prop[blk] = th[blk] + step[bi] * rng.standard_normal(len(blk))
                lp = log_post(prop, C, prior_sd)
                tries[bi] += 1
                if np.log(rng.random()) < lp - lpost:
                    th, lpost = prop, lp
                    acc[bi] += 1
            if it < n_warm and (it + 1) % 100 == 0:
                rate = acc / np.maximum(tries, 1)
                step *= np.exp(np.clip(rate - 0.30, -0.5, 0.5))
                acc[:] = 0; tries[:] = 0
            if it >= n_warm and (it - n_warm) % thin == 0 and j < keep.shape[1]:
                keep[ch, j] = th
                j += 1
    return keep


def split_rhat_ess(x: np.ndarray) -> tuple[float, float]:
    """Split-Rhat and bulk ESS for one parameter, chains x draws."""
    m, n = x.shape
    if n < 4:
        return float("nan"), float("nan")
    h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    m2, n2 = s.shape
    means, varis = s.mean(1), s.var(1, ddof=1)
    B = n2 * means.var(ddof=1)
    W = varis.mean()
    if W <= 0:
        return float("nan"), float("nan")
    var_hat = (n2 - 1) / n2 * W + B / n2
    rhat = float(np.sqrt(var_hat / W))
    # autocorrelation via FFT, Geyer's initial positive sequence
    acov = np.zeros(n2)
    for c in range(m2):
        y = s[c] - means[c]
        f = np.fft.rfft(y, 2 * n2)
        a = np.fft.irfft(f * np.conjugate(f), 2 * n2)[:n2] / n2
        acov += a
    acov /= m2
    rho = 1.0 - (W - acov) / var_hat
    t, s_sum = 1, 0.0
    while t + 1 < n2:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        s_sum += p
        t += 2
    tau = max(1.0, 1.0 + 2.0 * s_sum)
    return rhat, float(m2 * n2 / tau)


def derive(draws: np.ndarray, C: dict, seed: int = 7) -> dict:
    """Turn posterior draws into the quantities the paper would quote."""
    nf, nc = len(C["families"]), len(C["models"])
    flat = draws.reshape(-1, draws.shape[-1])
    alpha, Delta = flat[:, 0], flat[:, 1]
    sig = np.exp(flat[:, 2:6])
    o = 6
    z_af = flat[:, o:o + nf]; o += nf
    z_ac = flat[:, o:o + nc]; o += nc
    z_df = flat[:, o:o + nf]; o += nf
    z_dc = flat[:, o:o + nc]
    fi = C["fam_idx"]

    eta_ctl = alpha[:, None] + sig[:, 0:1] * z_af[:, fi] + sig[:, 1:2] * z_ac
    gap_lo = Delta[:, None] + sig[:, 2:3] * z_df[:, fi] + sig[:, 3:4] * z_dc
    eta_rep = eta_ctl - gap_lo
    p_ctl, p_rep = 1 / (1 + np.exp(-eta_ctl)), 1 / (1 + np.exp(-eta_rep))
    gap_ck = p_ctl - p_rep                       # draws x checkpoints

    def ci(a):
        return dict(mean=float(np.mean(a)),
                    lo=float(np.percentile(a, 2.5)),
                    hi=float(np.percentile(a, 97.5)),
                    p_gt0=float(np.mean(a > 0)))

    res: dict = {}
    res["delta_log_odds"] = ci(Delta)
    res["per_checkpoint_gap_pts"] = {
        m: {k: (100 * v if k != "p_gt0" else v)
            for k, v in ci(gap_ck[:, i]).items()}
        for i, m in enumerate(C["models"])}
    res["per_family_gap_pts"] = {}
    for j, f in enumerate(C["families"]):
        cols = [i for i in range(nc) if fi[i] == j]
        res["per_family_gap_pts"][f] = {
            k: (100 * v if k != "p_gt0" else v)
            for k, v in ci(gap_ck[:, cols].mean(1)).items()}

    # "A typical checkpoint of a typical family": random effects at zero.
    typ = 1 / (1 + np.exp(-alpha)) - 1 / (1 + np.exp(-(alpha - Delta)))
    res["panel_gap_pts"] = {k: (100 * v if k != "p_gt0" else v)
                            for k, v in ci(typ).items()}

    # The claim the paper actually makes is about models of this kind, not about
    # these six. That is a posterior *predictive* for an unobserved family, and
    # it is wider than the panel mean by exactly the amount three families
    # license -- which is the number to lead with.
    rng = np.random.default_rng(seed)
    n = flat.shape[0]
    e_new = (alpha + sig[:, 0] * rng.standard_normal(n)
             + sig[:, 1] * rng.standard_normal(n))
    g_new = (Delta + sig[:, 2] * rng.standard_normal(n)
             + sig[:, 3] * rng.standard_normal(n))
    pred = 1 / (1 + np.exp(-e_new)) - 1 / (1 + np.exp(-(e_new - g_new)))
    res["new_family_gap_pts"] = {k: (100 * v if k != "p_gt0" else v)
                                 for k, v in ci(pred).items()}
    res["sigma"] = {name: float(np.mean(sig[:, i])) for i, name in enumerate(
        ["family_level", "checkpoint_level", "family_gap", "checkpoint_gap"])}
    return res


def fit_numpyro(C: dict, prior_sd: float, n_warm: int, n_draw: int,
                chains: int, seed: int):
    """NUTS on the identical model. Returns draws in this script's layout so the
    two samplers can be compared parameter for parameter."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except Exception:
        return None, None
    numpyro.set_host_device_count(chains)
    nf, nc = len(C["families"]), len(C["models"])
    fi = jnp.array(C["fam_idx"])

    def model():
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
        Delta = numpyro.sample("Delta", dist.Normal(0.0, 5.0))
        s = numpyro.sample("sigma", dist.HalfNormal(prior_sd).expand([4]))
        z_af = numpyro.sample("z_af", dist.Normal(0, 1).expand([nf]))
        z_ac = numpyro.sample("z_ac", dist.Normal(0, 1).expand([nc]))
        z_df = numpyro.sample("z_df", dist.Normal(0, 1).expand([nf]))
        z_dc = numpyro.sample("z_dc", dist.Normal(0, 1).expand([nc]))
        eta_ctl = alpha + s[0] * z_af[fi] + s[1] * z_ac
        gap = Delta + s[2] * z_df[fi] + s[3] * z_dc
        numpyro.sample("y_ctl", dist.Binomial(jnp.array(C["n_ctl"]),
                                              logits=eta_ctl),
                       obs=jnp.array(C["y_ctl"]))
        numpyro.sample("y_rep", dist.Binomial(jnp.array(C["n_rep"]),
                                              logits=eta_ctl - gap),
                       obs=jnp.array(C["y_rep"]))

    mcmc = MCMC(NUTS(model), num_warmup=n_warm, num_samples=n_draw,
                num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    s = mcmc.get_samples(group_by_chain=True)
    draws = np.concatenate(
        [np.asarray(s["alpha"])[:, :, None], np.asarray(s["Delta"])[:, :, None],
         np.log(np.asarray(s["sigma"])), np.asarray(s["z_af"]),
         np.asarray(s["z_ac"]), np.asarray(s["z_df"]), np.asarray(s["z_dc"])],
        axis=-1)
    from numpyro.diagnostics import summary as npsummary
    diag = npsummary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
    rhat = max(float(np.max(v["r_hat"])) for v in diag.values())
    ess = min(float(np.min(v["n_eff"])) for v in diag.values())
    return draws, dict(engine="numpyro NUTS", rhat_max=rhat, ess_min=ess)


def bayes(d: pd.DataFrame, out: dict, quick: bool) -> None:
    C0 = cell_counts(d)
    C = dict(C0)
    C["fam_idx"] = np.array(C0["fam_of"])
    for k in ("y_ctl", "n_ctl", "y_rep", "n_rep"):
        C[k] = np.array(C0[k], dtype=float)

    prior_sd = 1.0
    n_warm, n_draw, chains = (400, 800, 4) if quick else (1500, 3000, 4)
    res: dict = {"model": ("binomial likelihood, logit scale; control level and "
                           "arm gap each carry a family offset and a checkpoint "
                           "offset nested in it"),
                 "priors": {"alpha": "Normal(0,5)", "Delta": "Normal(0,5)",
                            "sigma": f"HalfNormal({prior_sd})"},
                 "cells": {m: dict(y_ctl=int(C0["y_ctl"][i]), n_ctl=int(C0["n_ctl"][i]),
                                   y_rep=int(C0["y_rep"][i]), n_rep=int(C0["n_rep"][i]))
                           for i, m in enumerate(C0["models"])}}

    nuts, nuts_diag = fit_numpyro(C, prior_sd, n_warm, n_draw, chains, seed=0)

    mh_warm, mh_draw, mh_thin = (4000, 8000, 4) if quick else (20000, 60000, 10)
    mh = metropolis(C, prior_sd, mh_warm, mh_draw, mh_thin, chains, seed=11)
    rh = [split_rhat_ess(mh[:, :, p]) for p in range(mh.shape[-1])]
    mh_diag = dict(engine="hand-rolled blocked Metropolis",
                   rhat_max=float(np.nanmax([r for r, _ in rh])),
                   ess_min=float(np.nanmin([e for _, e in rh])),
                   draws=int(mh.shape[0] * mh.shape[1]))

    primary, primary_diag = (nuts, nuts_diag) if nuts is not None else (mh, mh_diag)
    primary_diag["converged"] = bool(primary_diag["rhat_max"] < 1.01
                                     and primary_diag["ess_min"] > 400)
    res["diagnostics"] = primary_diag
    res["crosscheck_diagnostics"] = mh_diag if nuts is not None else None
    res.update(derive(primary, C))

    if nuts is not None:
        alt = derive(mh, C)
        deltas = {
            "delta_log_odds_mean": abs(alt["delta_log_odds"]["mean"]
                                       - res["delta_log_odds"]["mean"]),
            "panel_gap_mean_pts": abs(alt["panel_gap_pts"]["mean"]
                                      - res["panel_gap_pts"]["mean"]),
            "new_family_gap_mean_pts": abs(alt["new_family_gap_pts"]["mean"]
                                           - res["new_family_gap_pts"]["mean"]),
        }
        res["crosscheck"] = dict(
            engine="hand-rolled Metropolis vs numpyro NUTS",
            max_abs_diff=deltas,
            agrees=bool(deltas["delta_log_odds_mean"] < 0.15
                        and deltas["panel_gap_mean_pts"] < 1.5
                        and deltas["new_family_gap_mean_pts"] < 3.0),
            metropolis=dict(delta_log_odds=alt["delta_log_odds"],
                            panel_gap_pts=alt["panel_gap_pts"],
                            new_family_gap_pts=alt["new_family_gap_pts"]))

    # Priors on three group-level SDs are doing real work when there are three
    # groups, so the sensitivity is part of the result, not an appendix.
    sens = {}
    for sd in (0.5, 3.0):
        alt_draws, _ = fit_numpyro(C, sd, n_warm, n_draw, chains, seed=0)
        if alt_draws is None:
            alt_draws = metropolis(C, sd, mh_warm, mh_draw, mh_thin, chains, seed=11)
        a = derive(alt_draws, C)
        sens[f"HalfNormal({sd})"] = dict(
            delta_log_odds=a["delta_log_odds"],
            panel_gap_pts=a["panel_gap_pts"],
            new_family_gap_pts=a["new_family_gap_pts"])
    res["prior_sensitivity"] = sens
    out["bayes"] = res

    print(f"\n2. hierarchical Bayes ({primary_diag['engine']}, "
          f"R-hat max {primary_diag['rhat_max']:.3f}, "
          f"ESS min {primary_diag['ess_min']:.0f}"
          + ("" if primary_diag["converged"] else ", NOT CONVERGED") + ")")
    if not primary_diag["converged"]:
        print("   R-hat above 1.01 or ESS below 400: lengthen the chains before "
              "quoting these.")
    pg, ng = res["panel_gap_pts"], res["new_family_gap_pts"]
    dl = res["delta_log_odds"]
    print(f"   gap in log-odds        {dl['mean']:+6.2f} "
          f"[{dl['lo']:+.2f},{dl['hi']:+.2f}]   P(>0)={dl['p_gt0']:.4f}")
    print(f"   panel gap (pts)        {pg['mean']:+6.1f} "
          f"[{pg['lo']:+.1f},{pg['hi']:+.1f}]   P(>0)={pg['p_gt0']:.4f}")
    print(f"   NEW family gap (pts)   {ng['mean']:+6.1f} "
          f"[{ng['lo']:+.1f},{ng['hi']:+.1f}]   P(>0)={ng['p_gt0']:.4f}"
          "   <- the generalising claim")
    print("   by family:")
    for f, v in res["per_family_gap_pts"].items():
        print(f"     {f:12s} {v['mean']:+6.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
              f"  P(>0)={v['p_gt0']:.4f}")
    print("   by checkpoint (partially pooled):")
    for m, v in res["per_checkpoint_gap_pts"].items():
        print(f"     {m:12s} {v['mean']:+6.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
              f"  P(>0)={v['p_gt0']:.4f}")
    if res.get("crosscheck"):
        cc = res["crosscheck"]
        print(f"   cross-check {cc['engine']}: "
              f"agrees={cc['agrees']}, "
              f"max |diff| on the panel gap "
              f"{cc['max_abs_diff']['panel_gap_mean_pts']:.2f} pts")
    print("   prior sensitivity on the group SDs:")
    for k, v in sens.items():
        print(f"     {k:18s} new-family gap {v['new_family_gap_pts']['mean']:+.1f} "
              f"[{v['new_family_gap_pts']['lo']:+.1f},"
              f"{v['new_family_gap_pts']['hi']:+.1f}]  "
              f"P(>0)={v['new_family_gap_pts']['p_gt0']:.4f}")


# ----------------------------------------------------------------------------
# 3. specification curve
# ----------------------------------------------------------------------------

EXCLUSION_ARMS = {
    # exactly the arms analysis/exclusion_sensitivity.py defines, so the
    # overlapping cells of this curve reproduce that script rather than
    # re-deciding what "panel" means
    "panel": dict(bad_templates=True, cap_hits=True, degenerate=False),
    "plus_bad_templates": dict(bad_templates=False, cap_hits=True, degenerate=False),
    "plus_cap_hits": dict(bad_templates=True, cap_hits=False, degenerate=False),
    "plus_degenerate": dict(bad_templates=True, cap_hits=True, degenerate=True),
    "everything": dict(bad_templates=False, cap_hits=False, degenerate=True),
}


def spec_population(raw: pd.DataFrame, apc: pd.DataFrame | None, *,
                    excl: str, kmin: int, control: str,
                    seeds: str) -> pd.DataFrame | None:
    flags = EXCLUSION_ARMS[excl]
    d, _ = panel(raw, ablations=False, **flags)
    if flags["degenerate"]:
        d = d.copy()
        d.loc[d.outcome.isin(DEGENERATE), "count_a"] = 0
    d = annotate(d[d.k >= kmin])

    if control == "aperiodic":
        if apc is None:
            return None
        a, _ = panel(apc, ablations=False, **flags)
        if flags["degenerate"]:
            a = a.copy()
            a.loc[a.outcome.isin(DEGENERATE), "count_a"] = 0
        a = annotate(a[a.k >= kmin])
        a = a[a.rep == 0]
        if not len(a):
            return None
        # pair the never-cycled controls against the same repeated items the
        # main ladder already scored, at the same k and the same checkpoints --
        # only the control side changes (analysis/aperiodic_controls.py)
        r = d[(d.rep == 1) & d.k.isin(a.k.unique()) & d.model.isin(a.model.unique())]
        d = pd.concat([r, a], ignore_index=True)

    if seeds == "seed0":
        d = d[d.seed == 0]
    if not len(d):
        return None
    return d


def spec_estimate(d: pd.DataFrame, outcome: str, agg: str) -> float | None:
    """One number: the headline gap under one set of forking choices."""
    def cell(s: pd.DataFrame) -> float | None:
        r, c = s[s.rep == 1], s[s.rep == 0]
        if len(r) < 5 or len(c) < 5:
            return None
        if outcome == "exact_rate":
            return 100.0 * float(c.exact.mean() - r.exact.mean())
        return 100.0 * float(c.err.median() - r.err.median())

    if agg == "pooled":
        return cell(d)
    per = {m: cell(g) for m, g in d.groupby("model", observed=True)}
    per = {m: v for m, v in per.items() if v is not None}
    if len(per) < 2:
        return None
    if agg == "checkpoint":
        return float(np.mean(list(per.values())))
    fam = family_means(per)
    return float(np.mean(list(fam.values())))


def spec_reproduction_check(specs: list[dict], kmin: int) -> dict:
    """The cells this curve shares with the single-axis scripts must equal them.

    `exclusion_sensitivity.py` is the exclusions axis at kmin, pooled;
    `aperiodic_controls.py` is the never-cycled control arm. If either drifts,
    the two scripts are looking at different rows and the curve is not an
    envelope around the reported number but around some other number.
    """
    def find(**kw):
        for s in specs:
            if all(s[k] == v for k, v in kw.items()):
                return s["gap"]
        return None

    agree: dict = {}
    es = REPO / "data/results/exclusion_sensitivity.json"
    if es.exists():
        j = json.loads(es.read_text())
        if j.get("kmin") == kmin:
            for arm, v in j["arms"].items():
                mine = find(outcome="exact_rate", kmin=kmin, control="cycled",
                            seeds="all", aggregation="pooled", exclusions=arm)
                if mine is None:
                    continue
                diff = abs(mine - 100 * v["exact_gap"])
                agree[f"exclusion_sensitivity.{arm}"] = float(diff)
                assert diff < 1e-9, (
                    f"exclusion arm {arm} disagrees by {diff:.3g} pts -- "
                    "suspect the population before the maths")
    ac = REPO / "data/results/aperiodic_controls.json"
    if ac.exists():
        j = json.loads(ac.read_text())
        row = next((v for k, v in j.items()
                    if isinstance(v, dict) and "re-generated" in k), None)
        mine = find(outcome="exact_rate", control="aperiodic", exclusions="panel",
                    seeds="all", aggregation="pooled", kmin=1)
        if row and mine is not None:
            diff = abs(mine - 100 * row["exact_gap"])
            agree["aperiodic_controls.re-generated"] = float(diff)
            assert diff < 1e-9, (
                f"the never-cycled control arm disagrees by {diff:.3g} pts")
    return dict(max_abs_diff_pts=agree,
                matches=bool(agree) and all(v < 1e-9 for v in agree.values()))


def spec_curve(out: dict, kmin_default: int) -> None:
    raw = pd.read_csv(REPO / "data/results/behavioural_ctc.csv")
    raw = raw[raw.family.isin(PAIR)]
    apc_path = REPO / "data/results/behavioural_aperiodic.csv"
    apc = pd.read_csv(apc_path) if apc_path.exists() else None
    if apc is not None:
        apc = apc[apc.family.isin(PAIR)]

    axes = dict(exclusions=list(EXCLUSION_ARMS),
                kmin=[1, 6, 8, 12, 16],
                control=["cycled", "aperiodic"],
                seeds=["all", "seed0"],
                aggregation=["pooled", "checkpoint", "family"],
                outcome=["exact_rate", "median_rel_error"])

    specs: list[dict] = []
    infeasible = 0
    duplicates = 0
    seen: set = set()
    for excl, kmin, control, seeds in itertools.product(
            axes["exclusions"], axes["kmin"], axes["control"], axes["seeds"]):
        d = spec_population(raw, apc, excl=excl, kmin=kmin,
                            control=control, seeds=seeds)
        if d is not None:
            # The never-cycled controls only exist for k=12..32, so every kmin
            # below 12 selects the identical rows. Counting those four times
            # would pad the curve with copies and make the sign look better
            # supported than it is, so identical populations are collapsed.
            fp = (excl, control, seeds, tuple(sorted(d.k.unique())),
                  len(d), float(d.exact.sum()))
            if fp in seen:
                duplicates += len(axes["aggregation"]) * len(axes["outcome"])
                continue
            seen.add(fp)
        for agg, outcome in itertools.product(axes["aggregation"], axes["outcome"]):
            if d is None:
                infeasible += 1
                continue
            v = spec_estimate(d, outcome, agg)
            if v is None:
                infeasible += 1
                continue
            specs.append(dict(exclusions=excl, kmin=kmin, control=control,
                              seeds=seeds, aggregation=agg, outcome=outcome,
                              n=int(len(d)), gap=float(v)))

    res: dict = {"axes": axes, "n_specifications": len(specs),
                 "n_infeasible": infeasible,
                 "n_duplicate_populations_collapsed": duplicates,
                 "specs": specs}

    for outcome in axes["outcome"]:
        v = np.array([s["gap"] for s in specs if s["outcome"] == outcome])
        res[outcome] = dict(
            n=int(v.size), min=float(v.min()), max=float(v.max()),
            median=float(np.median(v)),
            q05=float(np.percentile(v, 5)), q95=float(np.percentile(v, 95)),
            n_positive=int((v > 0).sum()), n_zero=int((v == 0).sum()),
            n_negative=int((v < 0).sum()),
            frac_sign_preserved=float((v > 0).mean()))
        assert (res[outcome]["n_positive"] + res[outcome]["n_zero"]
                + res[outcome]["n_negative"] == v.size), "spec signs do not sum"

    allv = np.array([s["gap"] for s in specs])
    res["all_specifications"] = dict(
        n=int(allv.size), n_positive=int((allv > 0).sum()),
        n_zero=int((allv == 0).sum()), n_negative=int((allv < 0).sum()),
        frac_sign_preserved=float((allv > 0).mean()))

    # Which fork moves the number, and by how much: the useful reading of a
    # specification curve is not the range but which axis owns it.
    prim = [s for s in specs if s["outcome"] == "exact_rate"]
    infl = {}
    for ax in ("exclusions", "kmin", "control", "seeds", "aggregation"):
        by = {}
        for lvl in axes[ax]:
            v = [s["gap"] for s in prim if s[ax] == lvl]
            if v:
                by[str(lvl)] = float(np.median(v))
        infl[ax] = dict(median_by_level=by,
                        spread=float(max(by.values()) - min(by.values())) if by else 0.0)
    res["axis_influence_exact_rate"] = infl

    # The single worst specification, named, so it can be quoted rather than
    # discovered by a reviewer.
    worst = min(prim, key=lambda s: s["gap"])
    res["worst_specification_exact_rate"] = worst
    res["reproduces"] = spec_reproduction_check(specs, kmin_default)
    out["spec_curve"] = res

    e, m = res["exact_rate"], res["median_rel_error"]
    print(f"\n3. specification curve: {len(specs)} specifications "
          f"({infeasible} infeasible, {duplicates} duplicate populations collapsed)")
    print(f"   exact-rate gap        range [{e['min']:+.1f}, {e['max']:+.1f}] pts, "
          f"median {e['median']:+.1f}, 5-95% [{e['q05']:+.1f}, {e['q95']:+.1f}]")
    print(f"                         sign preserved in {e['n_positive']}/{e['n']} "
          f"({100 * e['frac_sign_preserved']:.1f}%), "
          f"{e['n_zero']} exactly zero, {e['n_negative']} negative")
    print(f"   median-rel-error gap  range [{m['min']:+.1f}, {m['max']:+.1f}] pts, "
          f"median {m['median']:+.1f}")
    print(f"                         sign preserved in {m['n_positive']}/{m['n']} "
          f"({100 * m['frac_sign_preserved']:.1f}%), "
          f"{m['n_zero']} exactly zero, {m['n_negative']} negative")
    a = res["all_specifications"]
    print(f"   across all {a['n']} specifications: {a['n_positive']} positive, "
          f"{a['n_zero']} zero, {a['n_negative']} negative "
          f"({100 * a['frac_sign_preserved']:.1f}% preserve the sign)")
    print("   which fork owns the range (median exact-rate gap by level):")
    for ax, v in infl.items():
        lv = ", ".join(f"{k}={x:+.1f}" for k, x in v["median_by_level"].items())
        print(f"     {ax:12s} spread {v['spread']:5.1f}   {lv}")
    print(f"   weakest single specification: {worst['gap']:+.1f} pts at "
          f"{worst['exclusions']}, k>={worst['kmin']}, {worst['control']} control, "
          f"{worst['seeds']}, {worst['aggregation']}")
    rp = res["reproduces"]
    if rp["max_abs_diff_pts"]:
        print(f"   reproduces the single-axis scripts exactly on their "
              f"{len(rp['max_abs_diff_pts'])} shared cells "
              f"(max |diff| "
              f"{max(rp['max_abs_diff_pts'].values()):.2e} pts)")


# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default=str(REPO / "data/results/behavioural_ctc.csv"))
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="short chains, smoke test only")
    ap.add_argument("--out", default=str(REPO / "data/results/hierarchical.json"))
    args = ap.parse_args()

    out: dict = {"kmin": args.kmin,
                 "question": ("six checkpoints are three architecture families; "
                              "what does the arm effect look like when the "
                              "clustering is modelled instead of apologised for")}

    d, drop = load(args.behavioural, args.kmin)
    out["population_note"] = (f"{drop['n_input']} -> {drop['n_output']} rows before "
                              f"the k>={args.kmin} cut; "
                              "src/common/population.py, panel() defaults")
    reproduction_check(d, out)
    mixed_models(d, out)
    bayes(d, out, args.quick)
    spec_curve(out, args.kmin)

    # The least favourable reading of every part, gathered in one place so a
    # reader does not have to assemble it from three sections to find out where
    # the result is weakest. Computed, not typed.
    lo_freq = out["mixed"].get("worst_case_lower_bound_pts", float("nan"))
    ng = out["bayes"]["new_family_gap_pts"]
    e = out["spec_curve"]["exact_rate"]
    priors = {"HalfNormal(1.0)": ng} | {
        k: v["new_family_gap_pts"] for k, v in out["bayes"]["prior_sensitivity"].items()}
    worst_prior = min(priors, key=lambda k: priors[k]["lo"])
    lf = {
        "frequentist_lower_bound_pts": float(lo_freq),
        "bayes_new_family_lo_pts": float(min(v["lo"] for v in priors.values())),
        "bayes_new_family_worst_prior": worst_prior,
        "bayes_new_family_worst_p_gt0": float(priors[worst_prior]["p_gt0"]),
        "spec_curve_min_pts": float(e["min"]),
        "spec_curve_worst": out["spec_curve"]["worst_specification_exact_rate"],
    }
    lf["every_reading_positive"] = bool(
        lf["frequentist_lower_bound_pts"] > 0 and lf["spec_curve_min_pts"] > 0)
    lf["a_credible_interval_crosses_zero"] = bool(lf["bayes_new_family_lo_pts"] <= 0)
    out["least_favourable"] = lf

    out["verdict"] = (
        f"arm effect {out['mixed']['mixedlm']['arm_effect_pts']:.1f} pts; the "
        f"least favourable of six frequentist intervals has lower bound "
        f"{lo_freq:.1f} pts; the posterior predictive gap for an architecture "
        f"family outside the panel is {ng['mean']:.1f} pts "
        f"[{ng['lo']:.1f}, {ng['hi']:.1f}] with P(gap>0)={ng['p_gt0']:.3f}; the "
        f"exact-rate gap ranges {e['min']:.1f} to {e['max']:.1f} pts across "
        f"{e['n']} specifications and keeps its sign in "
        f"{100 * e['frac_sign_preserved']:.0f}% of them.")
    out["clears_zero"] = bool(lo_freq > 0 and ng["lo"] > 0)

    print("\nleast favourable reading of each part:")
    print(f"   frequentist        lower bound {lf['frequentist_lower_bound_pts']:+.1f} pts")
    print(f"   Bayesian           new-family 95% CrI lower bound "
          f"{lf['bayes_new_family_lo_pts']:+.1f} pts under {worst_prior}, "
          f"P(gap>0)={lf['bayes_new_family_worst_p_gt0']:.3f}")
    print(f"   specification curve weakest specification "
          f"{lf['spec_curve_min_pts']:+.1f} pts")
    if lf["a_credible_interval_crosses_zero"]:
        print("   NOTE: under the widest prior on the group-level SDs the "
              "predictive interval for\n   an unseen architecture family "
              "includes zero. Six checkpoints in three families\n   cannot rule "
              "out an architecture with no deficit; what they establish is a "
              "large\n   effect in every family observed and a high posterior "
              "probability, not certainty.")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n{out['verdict']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
