#!/usr/bin/env python3
r"""Score a completed human listening audit against the CTC judge.

WHAT THIS SCRIPT IS FOR. Five review rounds have asked the same question in
different words: has the CTC judge (`src/common/score_counts.py`) ever been
checked against a human listening to *real generated* audio, as opposed to
concatenative audio with a known ground-truth count
(`analysis/ctc_validation.py`)? Until a completed `listening_sheet.csv` exists,
the honest answer is no. This script is what turns a filled-in sheet into that
number -- and its interpretation is written down here, before any human count
exists, so the read given to a completed audit cannot be chosen after seeing it.

PRE-COMMITTED INTERPRETATION. `d = ctc_count - human_count` on every audited
clip, same sign convention as `analysis/independent_judge.py`'s primary-minus-
independent comparison: **negative d means the CTC judge reports FEWER
repetitions than the human heard** -- the direction the reviewers' objection
actually predicts, since a CTC decoder's blank/repeat-collapse rule is a
mechanism for losing repeated material, not control fillers.

  asymmetry A = mean(d | repeated arm) - mean(d | control arm)

is the number that answers the reviewers, for the same reason
`independent_judge.py` leads with it: a judge-side artifact specific to
counting repetition would show up as the CTC judge under-stating the repeated
arm relative to a human while roughly agreeing with the human on controls,
i.e. A substantially negative. A judge that is merely noisy, or noisy in a
way that has nothing to do with periodicity, moves both arms together and
leaves A near zero.

  PRIMARY JUDGE CONFIRMED   exact_agreement_rate >= 0.70 (all audited clips),
                            mean_abs_diff <= 1.0 count, and A > A_ARTIFACT
                            (-1.0 counts) -- the judge is not measurably worse
                            at repeated items than at controls, relative to a
                            human.
  ARM-ASYMMETRIC ARTIFACT   A <= A_ARTIFACT counts, OR the repeated arm's
                            mean_abs_diff is at least 2x the control arm's
                            AND its mean signed diff is negative (the judge
                            disagrees with humans specifically on repeated
                            items, and specifically by under-counting them).
                            This is the pattern that would mean the judge, not
                            the model, manufactures part of the paper's gap.
  UNINTERPRETABLE           fewer than N_MIN_PER_ARM (15) usable clips in
                            either arm after dropping abstentions, or the
                            bootstrap CI on exact_agreement_rate has half-width
                            > 0.25 (too wide for either verdict above to be
                            trustworthy), or more than UNCLEAR_MAX_FRAC (30%)
                            of the sheet was marked `unclear` (the audit failed
                            to produce enough usable measurements to say
                            anything, which is itself worth reporting --
                            *not* silently averaged away).

Between CONFIRMED and ARTIFACT with n=15-30/arm there is real room for a result
that is technically outside both bands; that is reported as INCONCLUSIVE,
verbatim, rather than rounded toward whichever the author would prefer.

THE SAMPLE THIS RUNS ON IS NOT A POPULATION ESTIMATE. `listening_sheet.csv` is
drawn from the 165-clip *stratified* release sample
(`scripts/make_audio_sample.py`), which itself takes ~1 clip per (model,
family, k band, outcome) cell -- it is not proportional to how often those
cells occur in the full corpus. So "the paper's headline gap, recomputed on
these clips" is reported as exactly that: the same statistic, on the same
audited clips, CTC counts vs. human counts side by side. It is evidence about
whether substituting a human for the judge on these specific clips would have
changed the qualitative story (whether it stays positive, whether the sign of
the shift matches or opposes the reviewers' worry); it is emphatically not a
re-estimate of the 76.7-point population number in `analysis/independent_judge.py`,
and this script never claims it is.

REPORT AT THE LEAST FAVOURABLE READING. Where a statistic can be computed
trimmed (excluding runaway loops, which both a strict and a lenient reader
would call wrong regardless of the exact count) or untrimmed, both are printed,
and the verdict logic above reads the more conservative one, matching the
`independent_judge.py` precedent.

BLANK-ROW GUARD. A sheet with any row that has neither a `human_count` nor
`unclear=y` is an unfinished audit, and this script refuses to score it -- it
prints exactly which `row_id`s are unfinished and exits nonzero, rather than
silently dropping them and reporting a smaller n.

Usage:
  python scripts/score_listening.py
  python scripts/score_listening.py --sheet path/to/sheet.csv --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

# Pre-committed thresholds -- see the module docstring for what each protects
# against. Fixed here, before any listening_sheet.csv has ever had a real
# human_count in it.
A_ARTIFACT = -1.0          # counts; asymmetry floor for "artifact detected"
CONFIRM_EXACT_MIN = 0.70   # exact agreement rate
CONFIRM_MAD_MAX = 1.0      # mean absolute difference, counts
RATIO_ARTIFACT = 2.0       # repeated/control mean|d| ratio, artifact reading
N_MIN_PER_ARM = 15         # below this, the split is not trustworthy
CI_HALFWIDTH_MAX = 0.25    # on exact_agreement_rate
UNCLEAR_MAX_FRAC = 0.30
RUNAWAY = 10               # |d| beyond this is a runaway-loop mismatch, not a
                           # counting disagreement -- see independent_judge.py
N_BOOT = 5000


def read_sheet_with_header(path: Path) -> pd.DataFrame:
    """`listening_sheet.csv` carries `#`-prefixed context lines above the real
    CSV header (row count, time estimate, a reminder not to open the key).
    Those are for the human, not for pandas."""
    return pd.read_csv(path, comment="#", dtype={"row_id": str})


def parse_unclear(v) -> bool:
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"y", "yes", "1", "true"}


def check_complete(sheet: pd.DataFrame) -> pd.DataFrame:
    """Refuse to run on a half-finished sheet. A row is done if it has a valid
    non-negative integer `human_count`, or is explicitly flagged `unclear`
    (with or without a best-guess count). Anything else -- blank count, no
    flag -- is unfinished, and finishing it is the whole point of the guard:
    a partial audit that runs anyway can silently produce a number nobody
    would sign off on if they saw which rows were missing."""
    sheet = sheet.copy()
    sheet["unclear_flag"] = sheet["unclear"].map(parse_unclear)
    has_count = sheet["human_count"].apply(
        lambda v: not pd.isna(v) and str(v).strip() != "")
    bad_count = has_count & ~sheet["human_count"].apply(_is_nonneg_int)
    if bad_count.any():
        rows = ", ".join(sheet.loc[bad_count, "row_id"])
        raise SystemExit(
            f"listening sheet has {bad_count.sum()} row(s) with a human_count "
            f"that is not a non-negative integer: {rows}. Fix these before "
            f"scoring -- a typo here is not something to silently coerce.")
    unfinished = ~has_count & ~sheet["unclear_flag"]
    if unfinished.any():
        rows = ", ".join(sheet.loc[unfinished, "row_id"])
        raise SystemExit(
            f"listening sheet is unfinished: {unfinished.sum()} of {len(sheet)} "
            f"row(s) have neither a human_count nor unclear=y: {rows}\n"
            f"Finish these rows (or mark them unclear) before scoring -- a "
            f"half-finished audit does not get to produce a number.")
    sheet["has_count"] = has_count
    return sheet


def _is_nonneg_int(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f >= 0 and float(f).is_integer()


def boot_ci(x: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [stat(rng.choice(x, x.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def agg(sub: pd.DataFrame) -> dict:
    """Agreement statistics for one slice of audited, non-abstained clips."""
    n = len(sub)
    if n == 0:
        return dict(n=0)
    d = sub["d"].to_numpy(dtype=float)
    keep = np.abs(d) <= RUNAWAY
    lo, hi = boot_ci(d, lambda x: float(np.mean(x == 0)))
    out = dict(
        n=n,
        exact_agreement_rate=float(np.mean(d == 0)),
        exact_agreement_ci=[lo, hi],
        mean_abs_diff=float(np.mean(np.abs(d))),
        median_abs_diff=float(np.median(np.abs(d))),
        mean_signed_diff=float(np.mean(d)),
        median_signed_diff=float(np.median(d)),
        mean_abs_diff_excl_runaway=float(np.mean(np.abs(d[keep]))) if keep.any() else None,
        mean_signed_diff_excl_runaway=float(np.mean(d[keep])) if keep.any() else None,
        n_runaway=int((~keep).sum()),
        ctc_higher=int((d > 0).sum()),
        human_higher=int((d < 0).sum()),
        ties=int((d == 0).sum()),
    )
    return out


def cohen_kappa(a: np.ndarray, b: np.ndarray, weights=None) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(cohen_kappa_score(a, b, weights=weights))
        except Exception:
            return float("nan")


def exact_rate_gap(sub: pd.DataFrame, count_col: str) -> dict:
    """`analysis/independent_judge.py`'s published-headline statistic --
    control exact rate minus repeated exact rate -- recomputed on the audited
    clips with `count_col` in the role of the count. Called once with
    `ctc_count`, once with `human_count`; see the module docstring for why
    this is a same-clips comparison and not a population re-estimate."""
    exact = (sub[count_col] == sub["requested"]).astype(int)
    r = exact[sub.arm == "repeated"]
    c = exact[sub.arm == "control"]
    if len(r) == 0 or len(c) == 0:
        return dict(n_repeated=len(r), n_control=len(c), gap=float("nan"))
    gap = float(c.mean() - r.mean())
    diffs = sub.assign(exact=exact)

    def boot_gap(idx: np.ndarray) -> float:
        s = diffs.iloc[idx]
        rr = s.exact[s.arm == "repeated"]
        cc = s.exact[s.arm == "control"]
        if len(rr) == 0 or len(cc) == 0:
            return float("nan")
        return float(cc.mean() - rr.mean())

    rng = np.random.default_rng(0)
    n = len(diffs)
    boots = [boot_gap(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if len(boots) > 20 else (float("nan"), float("nan"))
    return dict(n_repeated=len(r), n_control=len(c),
               repeated_exact=float(r.mean()), control_exact=float(c.mean()),
               gap=gap, gap_ci=[lo, hi])


def rel_error_by_arm(sub: pd.DataFrame, count_col: str) -> dict:
    """The paper's other headline quantity (`analysis/count_error.py`):
    median relative count error (count - k) / k, by arm."""
    out = {}
    for arm, g in sub.groupby("arm"):
        rel = (g[count_col] - g["requested"]) / g["requested"]
        out[arm] = dict(median=float(rel.median()), n=len(g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="data/listening/listening_sheet.csv")
    ap.add_argument("--key", default="data/listening/listening_key.csv")
    ap.add_argument("--out", default="data/listening/listening_results.json")
    args = ap.parse_args()

    sheet_path, key_path = REPO / args.sheet, REPO / args.key
    if not sheet_path.exists() or not key_path.exists():
        raise SystemExit(f"missing {sheet_path if not sheet_path.exists() else key_path}. "
                         f"Run scripts/make_listening_sheet.py first.")

    sheet = read_sheet_with_header(sheet_path)
    key = pd.read_csv(key_path, dtype={"row_id": str})

    if set(sheet.row_id) != set(key.row_id):
        missing = set(key.row_id) - set(sheet.row_id)
        extra = set(sheet.row_id) - set(key.row_id)
        raise SystemExit(f"sheet/key row_id mismatch -- missing {missing}, "
                         f"extra {extra}. Did the sheet get hand-edited?")

    sheet = check_complete(sheet)   # raises SystemExit if unfinished

    m = key.merge(sheet[["row_id", "human_count", "unclear_flag", "has_count"]],
                  on="row_id", validate="one_to_one")
    n_total = len(m)
    n_unclear = int((~m.has_count).sum())
    if n_total and n_unclear / n_total > UNCLEAR_MAX_FRAC:
        print(f"[warning] {n_unclear}/{n_total} clips ({100*n_unclear/n_total:.0f}%) "
              f"were marked unclear with no count -- above the "
              f"{100*UNCLEAR_MAX_FRAC:.0f}% pre-committed ceiling. Proceeding, "
              f"but see the verdict below: this alone can make the audit "
              f"UNINTERPRETABLE.")

    scored = m[m.has_count].copy()
    scored["human_count"] = scored["human_count"].astype(float).astype(int)
    scored["d"] = scored["ctc_count"] - scored["human_count"]

    print(f"listening audit: {n_total} clips, {len(scored)} scored, "
          f"{n_unclear} abstained (unclear, no count)")

    res: dict = {"n_total": n_total, "n_scored": len(scored),
                "n_unclear_abstained": n_unclear,
                "unclear_frac": n_unclear / n_total if n_total else float("nan")}

    # ---- overall -----------------------------------------------------
    res["overall"] = agg(scored)
    o = res["overall"]
    print(f"\noverall (n={o['n']}): exact agreement {100*o['exact_agreement_rate']:.1f}% "
          f"[{100*o['exact_agreement_ci'][0]:.1f}, {100*o['exact_agreement_ci'][1]:.1f}], "
          f"mean|d|={o['mean_abs_diff']:.2f}, mean d={o['mean_signed_diff']:+.2f}")

    ck = scored["ctc_count"].to_numpy()
    hc = scored["human_count"].to_numpy()
    res["cohen_kappa_nominal"] = cohen_kappa(ck, hc)
    res["cohen_kappa_linear_weighted"] = cohen_kappa(ck, hc, weights="linear")
    max_count = int(max(ck.max(), hc.max())) if len(ck) and len(hc) else 0
    print(f"Cohen's kappa: nominal={res['cohen_kappa_nominal']:.3f}, "
          f"linear-weighted={res['cohen_kappa_linear_weighted']:.3f} "
          f"(counts run to {max_count}, so many sparse categories -- "
          f"exact-agreement + bootstrap CI above is the primary statistic, "
          f"kappa is supplementary)")

    # ---- by arm --------------------------------------------------------
    res["by_arm"] = {arm: agg(g) for arm, g in scored.groupby("arm")}
    print(f"\nby arm:")
    for arm, v in res["by_arm"].items():
        if not v.get("n"):
            continue
        print(f"  {arm:<9s} n={v['n']:3d}  exact={100*v['exact_agreement_rate']:5.1f}%  "
              f"mean|d|={v['mean_abs_diff']:.2f}  mean d={v['mean_signed_diff']:+.2f}")

    rep = res["by_arm"].get("repeated", {})
    ctl = res["by_arm"].get("control", {})
    A = (rep.get("mean_signed_diff", float("nan"))
         - ctl.get("mean_signed_diff", float("nan")))
    A_trim = (rep.get("mean_signed_diff_excl_runaway", float("nan"))
             - ctl.get("mean_signed_diff_excl_runaway", float("nan"))
             if rep.get("mean_signed_diff_excl_runaway") is not None
             and ctl.get("mean_signed_diff_excl_runaway") is not None
             else float("nan"))
    ratio = (rep.get("mean_abs_diff", float("nan"))
            / ctl.get("mean_abs_diff", float("nan"))
            if ctl.get("mean_abs_diff") else float("nan"))
    res["asymmetry"] = dict(A=A, A_excl_runaway=A_trim, mean_abs_diff_ratio=ratio)
    print(f"\nasymmetry A = mean(d|repeated) - mean(d|control) = {A:+.3f} counts "
          f"({A_trim:+.3f} excluding runaway |d|>{RUNAWAY})")
    print(f"repeated/control mean|d| ratio = {ratio:.2f}")

    # ---- by k band -------------------------------------------------------
    res["by_k_band"] = {kb: agg(g) for kb, g in scored.groupby("k_band", observed=True)}
    res["by_arm_k_band"] = {
        arm: {kb: agg(g) for kb, g in sub.groupby("k_band", observed=True)}
        for arm, sub in scored.groupby("arm")}
    print(f"\nby arm x k band (n / exact% / mean|d| / mean d):")
    kbands = ["low", "mid", "high"]
    for arm in ("repeated", "control"):
        cells = res["by_arm_k_band"].get(arm, {})
        line = f"  {arm:<9s}"
        for kb in kbands:
            v = cells.get(kb, {})
            if v.get("n"):
                line += (f"  {kb}: n={v['n']:2d} {100*v['exact_agreement_rate']:5.1f}% "
                         f"|d|={v['mean_abs_diff']:.2f} d={v['mean_signed_diff']:+.2f}")
            else:
                line += f"  {kb}: n=0"
        print(line)

    # ---- headline-gap substitution ----------------------------------------
    res["headline_exact_rate_gap"] = dict(
        ctc=exact_rate_gap(scored, "ctc_count"),
        human=exact_rate_gap(scored, "human_count"))
    g_ctc = res["headline_exact_rate_gap"]["ctc"]
    g_hum = res["headline_exact_rate_gap"]["human"]
    print(f"\nheadline exact-rate gap (control exact - repeated exact) on these "
          f"{len(scored)} audited clips ONLY -- not a population estimate:")
    print(f"  CTC counts:   {100*g_ctc['gap']:+.1f} pts "
          f"[{100*g_ctc['gap_ci'][0]:+.1f}, {100*g_ctc['gap_ci'][1]:+.1f}]  "
          f"(repeated exact {100*g_ctc['repeated_exact']:.1f}%, "
          f"control exact {100*g_ctc['control_exact']:.1f}%)")
    print(f"  human counts: {100*g_hum['gap']:+.1f} pts "
          f"[{100*g_hum['gap_ci'][0]:+.1f}, {100*g_hum['gap_ci'][1]:+.1f}]  "
          f"(repeated exact {100*g_hum['repeated_exact']:.1f}%, "
          f"control exact {100*g_hum['control_exact']:.1f}%)")
    try:
        sys.path.insert(0, str(REPO / "analysis"))
        from independent_judge import PUBLISHED_GAP  # noqa: E402
        res["published_population_gap_for_reference"] = PUBLISHED_GAP
        print(f"  (for reference only, NOT comparable at this n: the published "
              f"population-level checkpoint gap is {100*PUBLISHED_GAP:+.1f} pts)")
    except Exception:
        pass

    res["headline_rel_error"] = dict(
        ctc=rel_error_by_arm(scored, "ctc_count"),
        human=rel_error_by_arm(scored, "human_count"))
    print(f"\nmedian relative count error (count-k)/k, by arm, on audited clips:")
    for arm in ("repeated", "control"):
        c = res["headline_rel_error"]["ctc"].get(arm, {})
        h = res["headline_rel_error"]["human"].get(arm, {})
        if c and h:
            print(f"  {arm:<9s} CTC {c['median']:+.3f}  human {h['median']:+.3f}  "
                  f"(n={c['n']})")

    # ---- verdict -----------------------------------------------------
    n_rep, n_ctl = rep.get("n", 0), ctl.get("n", 0)
    ci = o.get("exact_agreement_ci", [float("nan"), float("nan")])
    ci_halfwidth = (ci[1] - ci[0]) / 2 if all(np.isfinite(ci)) else float("inf")
    uninterpretable_reasons = []
    if n_rep < N_MIN_PER_ARM or n_ctl < N_MIN_PER_ARM:
        uninterpretable_reasons.append(
            f"fewer than {N_MIN_PER_ARM} usable clips in an arm "
            f"(repeated={n_rep}, control={n_ctl})")
    if ci_halfwidth > CI_HALFWIDTH_MAX:
        uninterpretable_reasons.append(
            f"exact-agreement bootstrap CI half-width {ci_halfwidth:.2f} exceeds "
            f"{CI_HALFWIDTH_MAX}")
    if n_total and n_unclear / n_total > UNCLEAR_MAX_FRAC:
        uninterpretable_reasons.append(
            f"{100*n_unclear/n_total:.0f}% of the sheet was abstained "
            f"(unclear), above the {100*UNCLEAR_MAX_FRAC:.0f}% ceiling")

    A_read = A_trim if np.isfinite(A_trim) else A
    artifact_reasons = []
    if np.isfinite(A_read) and A_read <= A_ARTIFACT:
        artifact_reasons.append(
            f"asymmetry A={A_read:+.2f} counts <= {A_ARTIFACT:+.1f}: the CTC "
            f"judge under-counts the repeated arm relative to a human by that "
            f"much more than it does controls")
    if (np.isfinite(ratio) and ratio >= RATIO_ARTIFACT
            and rep.get("mean_signed_diff", 0) < 0):
        artifact_reasons.append(
            f"repeated-arm mean|d| is {ratio:.1f}x the control arm's and "
            f"signed negative (CTC under-counts on repeats specifically)")

    if uninterpretable_reasons:
        verdict = "UNINTERPRETABLE"
        reasons = uninterpretable_reasons
    elif artifact_reasons:
        verdict = "ARM-ASYMMETRIC ARTIFACT"
        reasons = artifact_reasons
    elif (o.get("exact_agreement_rate", 0) >= CONFIRM_EXACT_MIN
          and o.get("mean_abs_diff", np.inf) <= CONFIRM_MAD_MAX
          and (not np.isfinite(A_read) or A_read > A_ARTIFACT)):
        verdict = "PRIMARY JUDGE CONFIRMED"
        reasons = [f"exact agreement {100*o['exact_agreement_rate']:.1f}% >= "
                  f"{100*CONFIRM_EXACT_MIN:.0f}%, mean|d|={o['mean_abs_diff']:.2f} "
                  f"<= {CONFIRM_MAD_MAX}, asymmetry A={A_read:+.2f} > "
                  f"{A_ARTIFACT:+.1f}"]
    else:
        verdict = "INCONCLUSIVE"
        reasons = [f"lands between the pre-committed bands: exact agreement "
                  f"{100*o.get('exact_agreement_rate', float('nan')):.1f}%, "
                  f"mean|d|={o.get('mean_abs_diff', float('nan')):.2f}, "
                  f"A={A_read:+.2f}"]

    res["verdict"] = dict(label=verdict, reasons=reasons)
    print(f"\nVERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
