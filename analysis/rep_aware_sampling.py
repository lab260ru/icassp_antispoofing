#!/usr/bin/env python3
"""Does VALL-E 2's Repetition Aware Sampling close the count deficit?

=============================================================================
PRE-COMMITTED INTERPRETATION. The rules below are a copy of the ones fixed in
`src/common/ras.py`, which was written and on disk before a single RAS waveform
existed. Nothing above the POST-HOC line may be edited after looking at a
result; a rule that turned out to be wrong is reported as wrong, not rewritten.
=============================================================================

THE OBJECTION. A reviewer found prior art the paper had not engaged with. The
paper claims the deficit survives "the field's standard mitigations", but what
was swept is repetition-penalty *magnitude* (xtts2norp/rp2/rp3/rp8,
qwen06brp10/15/30) and greedy-versus-sampled (qwen06bgreedy). A
repetition-history-aware sampler is a different axis, it is published, and it is
engineered around precisely the variable this paper isolates: RAS "refines the
original nucleus sampling process by accounting for token repetition in the
decoding history. It not only stabilizes the decoding but also circumvents the
infinite loop issue." This file tests that axis.

THE CHECKPOINT. Qwen3-TTS-12Hz-0.6B-Base. Largest deficit among the checkpoints
that already carry a decoding-rule sweep to be compared against, and the
cleanest decoding path to modify honestly -- its talker is a stock
`GenerationMixin`, so the rule goes inside the real sampling loop. Reasons in
full in `src/common/ras.py`.

THE ARMS.
  qwen06brasoff  RAS at its no-op threshold (t_r = 2.0, unreachable). Bitwise
                 identical to the stock decoder, and therefore both the
                 full-scale sanity gate and the baseline the RAS arms are read
                 against. It exists because the stored `qwen06b` panel audio
                 does NOT reproduce byte-for-byte in today's environment (see
                 `analysis/ras_gate.py`): transformers moved under us since the
                 panel was generated.
  qwen06bras     K = 10, t_r = 0.1 over the checkpoint's shipped nucleus.
  qwen06brasgr   K = 10, t_r = 0.1 over a top-p = 0.0 nucleus -- the small-v
                 regime VALL-E 2 emphasises, where RAS is what makes a
                 near-greedy nucleus safe. The mitigation at its strongest.

THE MEASUREMENT. The paper's own statistic, by the paper's own code path: over
k >= 6, after `population.panel()`, with the CTC judge, the fraction of items
whose counted occurrences equal k, for repeated items and for their
length-matched controls, and the control-minus-repeated difference in points.

---------------------------------------------------------------- SANITY GATE
Read from `data/results/ras_sanity_gate.json`; see `analysis/ras_gate.py`.
PASS requires (A) the stock path is bitwise reproducible in this environment and
(B) RAS at its no-op threshold reproduces it exactly. B failing makes everything
here unreportable; A failing makes the gate VOID and the arm
inconclusive-by-construction.

------------------------------------------------------------ ENGAGEMENT GATE
  ENGAGED   RAS fires on >= 1% of decode steps on repeated items at k >= 6, AND
            fires more often on repeated items than on their controls.
  VACUOUS   otherwise: whatever the gap does, this rule did not do it.

--------------------------------------------------------- INFORMATIVENESS GATE
  INFORMATIVE    control exact rate >= 50% (the bar `analysis/crosslingual_es.py`
                 pre-registered, for the same reason: a mitigation that breaks
                 both families shrinks the gap for a reason unrelated to
                 counting).
  UNINFORMATIVE  below it.

------------------------------------------------------------------- VERDICT
The band is the existing arm structure on this checkpoint. The published
decoding-rule sweep spans exact-rate gaps of +66.7 to +92.1 points; those five
numbers were computed from already-landed CSVs before `src/common/ras.py` was
written, are hardcoded below as `SWEEP_GAPS`, and are recomputed at run time so
a drift cannot go unnoticed.

  SURVIVES     gap >= +66.7. The deficit stays inside the range the
               penalty/greedy sweep already spans: the robustness claim extends
               to the repetition-history-aware axis and the paper says so,
               naming RAS.
  MITIGATES    gap <= +33.4, i.e. at most half the smallest gap any existing
               sweep arm reaches. Reported first and loudly; the robustness
               claim is retracted for this axis.
  INCONCLUSIVE strictly between. Licenses one sentence -- reduced but not
               closed, both numbers given -- and licenses neither extending the
               robustness claim nor calling RAS a fix.

Standing rule: where a choice exists, take the one that SHRINKS the measured
gap. Every ambiguity is resolved in the mitigation's favour, so a surviving
deficit is a lower bound and whatever RAS does is an upper bound.

Usage:  python analysis/rep_aware_sampling.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.common.population import describe, panel  # noqa: E402

DATA_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts")

# The decoding-rule sweep that already exists on this checkpoint, as exact-rate
# gaps in points (k >= 6, CTC judge, panel exclusions). Computed from landed
# CSVs before any RAS audio existed; recomputed at run time and compared.
SWEEP_GAPS = {
    "qwen06brp30": ("repetition_penalty 3.0", 66.7, "behavioural_qwen_penalty.csv"),
    "qwen06brp15": ("repetition_penalty 1.5", 70.0, "behavioural_qwen_penalty.csv"),
    "qwen06bgreedy": ("greedy (do_sample=False)", 90.0, "behavioural_greedy.csv"),
    "qwen06brp10": ("repetition_penalty 1.0", 90.0, "behavioural_qwen_penalty.csv"),
    "qwen06b": ("repetition_penalty 1.05 (shipped, published, 3 seeds)", 92.1,
                "behavioural_ctc.csv"),
}
SURVIVES_AT = 66.7        # the smallest gap the existing sweep reaches
MITIGATES_AT = 33.4       # half of it
MIN_FIRE_RATE = 0.01
MIN_CONTROL_EXACT = 0.50
PRIMARY = "qwen06bras"
BASELINE = "qwen06brasoff"
SECONDARY = "qwen06brasgr"


def exact_rate(d: pd.DataFrame, fam: str) -> tuple[float, int]:
    """The paper's exact rate: counted occurrences equal to k.

    Identical to `analysis/crosslingual_es.exact_rate` and
    `analysis/exclusion_sensitivity.stats`, deliberately -- a reimplementation
    would make this number incomparable to the ones it is printed beside.
    """
    s = d[d.family == fam]
    if not len(s):
        return float("nan"), 0
    err = (s.count_a - s.k) / s.k
    return float((err == 0).mean()), int(len(s))


def gap_of(d: pd.DataFrame, model: str, kmin: int) -> dict:
    s = d[(d.model == model) & (d.k >= kmin)]
    s, drop = panel(s, ablations=True)
    rep, n_rep = exact_rate(s, "word_rep")
    ctl, n_ctl = exact_rate(s, "control_word")
    return dict(model=model, rep=rep, n_rep=n_rep, ctl=ctl, n_ctl=n_ctl,
                gap_points=100.0 * (ctl - rep), n=int(len(s)),
                dropped={k: v for k, v in drop.items() if v and k != "excluded_templates"})


def boot_gap_ci(d: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the gap, resampling items within each family."""
    rep = d[d.family == "word_rep"]
    ctl = d[d.family == "control_word"]
    if len(rep) < 3 or len(ctl) < 3:
        return float("nan"), float("nan")
    r = ((rep.count_a - rep.k) == 0).to_numpy(dtype=float)
    c = ((ctl.count_a - ctl.k) == 0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    gaps = [100.0 * (rng.choice(c, c.size, replace=True).mean()
                     - rng.choice(r, r.size, replace=True).mean())
            for _ in range(n_boot)]
    return tuple(float(x) for x in np.percentile(gaps, [2.5, 97.5]))


def load_meta(model: str) -> pd.DataFrame:
    p = DATA_ROOT / "tokens" / f"{model}_meta.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.open()])


def fire_stats(model: str, kmin: int) -> dict:
    """Fire rate per family, over decode steps, from the generation ledger."""
    m = load_meta(model)
    out: dict = {}
    if not len(m) or "ras_steps" not in m.columns:
        return out
    for fam in ("word_rep", "control_word"):
        s = m[(m.family == fam) & (m.k >= kmin)]
        if not len(s):
            continue
        steps = float(s.ras_steps.sum())
        out[fam] = dict(
            n_items=int(len(s)), steps=int(steps),
            fired=int(s.ras_fired.sum()),
            fire_rate=float(s.ras_fired.sum() / steps) if steps else float("nan"),
            fire_rate_alt_reading=float(s.ras_fired_alt.sum() / steps) if steps else float("nan"),
            items_never_fired=int((s.ras_fired == 0).sum()))
    return out


def cap_rates(model: str) -> dict:
    """hit_cap by family, from the ledger, BEFORE panel() removes those rows."""
    m = load_meta(model)
    if not len(m) or "hit_cap" not in m.columns:
        return {}
    out = {"all": dict(n=int(len(m)), rate=float(m.hit_cap.mean()))}
    for fam, s in m.groupby("family"):
        out[str(fam)] = dict(n=int(len(s)), rate=float(s.hit_cap.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--csv", default=str(REPO / "data/results/behavioural_ras.csv"))
    ap.add_argument("--gate", default=str(REPO / "data/results/ras_sanity_gate.json"))
    ap.add_argument("--out", default=str(REPO / "data/results/rep_aware_sampling.json"))
    args = ap.parse_args()

    # Provenance of the pre-commitment, recorded so a reader does not have to
    # take "pre-committed" on trust. `src/common/ras.py` carries the binding
    # copy of every threshold and gate below and was on disk at 16:38:13 UTC on
    # 2026-08-14; the first RAS waveform was written at 16:42:50 UTC. This file
    # was written at 16:46, by which time six smoke-test waveforms existed
    # (k = 1 and k = 2 on each arm, outside the k >= 6 region the verdict is
    # read at, and never transcribed or scored). The thresholds here are a copy
    # of the 16:38 file's, not a revision of them.
    provenance = dict(
        precommit_file="src/common/ras.py",
        precommit_written_utc="2026-08-14T16:38:13",
        first_ras_waveform_utc="2026-08-14T16:42:50",
        this_file_written_utc="2026-08-14T16:46:36",
        note=("thresholds and gates are a verbatim copy of the pre-commit file's; "
              "six k<=2 smoke waveforms existed when this file was written and none "
              "had been transcribed or scored"))

    res: dict = {"kmin": args.kmin, "checkpoint": "Qwen3-TTS-12Hz-0.6B-Base",
                 "precommit_provenance": provenance,
                 "method": "Repetition Aware Sampling, VALL-E 2 (arXiv:2406.05370), "
                           "our implementation of the published description",
                 "thresholds": dict(survives_at=SURVIVES_AT, mitigates_at=MITIGATES_AT,
                                    min_fire_rate=MIN_FIRE_RATE,
                                    min_control_exact=MIN_CONTROL_EXACT)}

    # ---------------- Sanity gate ----------------------------------------
    print("=== Sanity gate: does the no-op setting reproduce the stock decoder? ===")
    gate = json.loads(Path(args.gate).read_text()) if Path(args.gate).exists() else {}
    res["sanity_gate"] = {k: v for k, v in gate.items() if k != "items"}
    res["sanity_gate"]["n_items"] = gate.get("n_items")
    gate_state = gate.get("gate", "MISSING")
    print(f"  A determinism control : {gate.get('A_pass')}")
    print(f"  B no-op equivalence   : {gate.get('B_pass')}  "
          f"(token sequences AND wav bytes, {gate.get('n_items')} items)")
    print(f"  wiring check          : {gate.get('wiring_pass')}")
    print(f"  no-op never fired     : {gate.get('noop_never_fired')}")
    print(f"  stored panel audio reproduced today: {gate.get('stored_matches_fresh_all')}")
    print(f"  -> {gate_state}")
    if gate_state != "PASS":
        res["verdict"] = ("UNREPORTABLE" if gate_state == "UNREPORTABLE"
                          else "INCONCLUSIVE-BY-CONSTRUCTION")
        res["reason"] = gate.get("reason", "sanity gate did not run")
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nVERDICT: {res['verdict']} -- {res['reason']}")
        return

    # ---------------- The reference band ----------------------------------
    print("\n=== The decoding-rule sweep this checkpoint already carries ===")
    ref: dict = {}
    for model, (label, hardcoded, csv) in SWEEP_GAPS.items():
        p = REPO / "data/results" / csv
        live = float("nan")
        if p.exists():
            d = pd.read_csv(p)
            if model in set(d.model):
                live = gap_of(d, model, args.kmin)["gap_points"]
        ref[model] = dict(label=label, gap_points_precommitted=hardcoded,
                          gap_points_recomputed=live,
                          drifted=bool(abs(live - hardcoded) > 0.15) if live == live else None)
        print(f"  {label:<48s} {hardcoded:+6.1f}  (recomputed {live:+.1f})")
    res["reference_sweep"] = ref
    drifted = [k for k, v in ref.items() if v["drifted"]]
    if drifted:
        print(f"  WARNING: recomputed gaps drifted from the pre-committed values: {drifted}")
    res["reference_band"] = [SURVIVES_AT, max(v["gap_points_precommitted"] for v in ref.values())]

    # ---------------- The RAS arms ----------------------------------------
    raw = pd.read_csv(args.csv)
    arms = [BASELINE, PRIMARY, SECONDARY]
    present = [m for m in arms if m in set(raw.model)]
    res["arms_present"] = present
    res["n_generations"] = {m: int((raw.model == m).sum()) for m in present}
    res["seeds"] = {m: sorted(int(s) for s in raw[raw.model == m].seed.unique())
                    for m in present}

    print("\n=== Generation ceiling (hit_cap), from the ledger, before exclusions ===")
    res["cap_hits"] = {}
    for m in present:
        c = cap_rates(m)
        res["cap_hits"][m] = c
        bits = "  ".join(f"{k}={100*v['rate']:.1f}%(n={v['n']})"
                         for k, v in c.items() if k in ("all", "word_rep", "control_word"))
        print(f"  {m:<16s} {bits}")

    # Outcome mix before any exclusion. RAS's escape branch samples from an
    # untruncated distribution, so "did it wreck the audio?" has to be visible
    # as a rate rather than inferred from the gap.
    print("\n=== Outcome mix (k >= kmin, before exclusions) ===")
    res["outcomes"] = {}
    for m in present:
        s = raw[(raw.model == m) & (raw.k >= args.kmin)]
        c = s.outcome.value_counts().to_dict()
        res["outcomes"][m] = {str(k): int(v) for k, v in c.items()}
        res["outcomes"][m]["degenerate_or_empty_rate"] = float(
            s.outcome.isin(["empty", "degenerate"]).mean()) if len(s) else float("nan")
        print(f"  {m:<16s} {res['outcomes'][m]}")

    print(f"\n=== Exact-rate gap, k >= {args.kmin}, CTC judge ===")
    print(f"  {'arm':<16s} {'rep exact':>10s} {'ctl exact':>10s} {'gap':>8s} "
          f"{'95% CI':>18s} {'n':>6s}")
    res["exact"] = {}
    for m in present:
        g = gap_of(raw, m, args.kmin)
        s = raw[(raw.model == m) & (raw.k >= args.kmin)]
        s, _ = panel(s, ablations=True)
        g["gap_ci95"] = list(boot_gap_ci(s))
        g["fire"] = fire_stats(m, args.kmin)
        res["exact"][m] = g
        print(f"  {m:<16s} {100*g['rep']:9.1f}% {100*g['ctl']:9.1f}% {g['gap_points']:+8.1f} "
              f"[{g['gap_ci95'][0]:+6.1f},{g['gap_ci95'][1]:+6.1f}] {g['n']:6d}")
    for model, (label, hardcoded, _c) in sorted(SWEEP_GAPS.items(), key=lambda kv: kv[1][1]):
        print(f"  {model:<16s} {'':>10s} {'':>10s} {hardcoded:+8.1f} "
              f"{'(pre-committed)':>18s}         {label}")

    print("\n  by k (primary arm against the regenerated baseline):")
    res["by_k"] = defaultdict(dict)
    for k in sorted(raw[raw.k >= args.kmin].k.unique()):
        line = f"    k={int(k):<3d}"
        for m in present:
            s = raw[(raw.model == m) & (raw.k == k)]
            s, _ = panel(s, ablations=True)
            r_, nr = exact_rate(s, "word_rep")
            c_, nc = exact_rate(s, "control_word")
            res["by_k"][m][int(k)] = dict(rep=r_, n_rep=nr, ctl=c_, n_ctl=nc,
                                          gap_points=100.0 * (c_ - r_))
            line += f"  {m[7:]:>6s} rep {100*r_:5.1f}% ctl {100*c_:5.1f}%"
        print(line)
    res["by_k"] = dict(res["by_k"])

    print("\n=== RAS firing (the engagement gate) ===")
    for m in present:
        f = res["exact"][m]["fire"]
        for fam, v in f.items():
            print(f"  {m:<16s} {fam:<14s} fired {v['fired']:7d} / {v['steps']:7d} steps "
                  f"= {100*v['fire_rate']:5.2f}%   (other reading of the printed ratio: "
                  f"{100*v['fire_rate_alt_reading']:5.2f}%)  items never firing: "
                  f"{v['items_never_fired']}/{v['n_items']}")

    # ---------------- Paired item-level comparison -------------------------
    # The RAS arms share an RNG stream with the baseline by construction (see
    # `src/common/ras.py`, RNG discipline), so the same (item, seed) cell is a
    # paired observation and the flip table is meaningful rather than merely
    # suggestive.
    print("\n=== Paired flips against the regenerated baseline (word_rep, k >= kmin) ===")
    res["paired"] = {}
    base = raw[(raw.model == BASELINE) & (raw.k >= args.kmin)]
    base, _ = panel(base, ablations=True)
    for m in present:
        if m == BASELINE:
            continue
        arm = raw[(raw.model == m) & (raw.k >= args.kmin)]
        arm, _ = panel(arm, ablations=True)
        j = base.merge(arm, on=["item_id", "seed", "family", "k"], suffixes=("_b", "_a"))
        j = j[j.family == "word_rep"]
        ok_b = (j.count_a_b == j.k)
        ok_a = (j.count_a_a == j.k)
        cell = dict(n_paired=int(len(j)),
                    fixed=int((~ok_b & ok_a).sum()), broken=int((ok_b & ~ok_a).sum()),
                    both_right=int((ok_b & ok_a).sum()), both_wrong=int((~ok_b & ~ok_a).sum()),
                    median_counted_over_k_baseline=float((j.count_a_b / j.k).median()),
                    median_counted_over_k_arm=float((j.count_a_a / j.k).median()))
        res["paired"][m] = cell
        print(f"  {m:<16s} n={cell['n_paired']:3d}  fixed {cell['fixed']:3d}  "
              f"broken {cell['broken']:3d}  both right {cell['both_right']:3d}  "
              f"both wrong {cell['both_wrong']:3d}   median counted/k "
              f"{cell['median_counted_over_k_baseline']:.2f} -> "
              f"{cell['median_counted_over_k_arm']:.2f}")

    # ---------------- Gates and verdict ------------------------------------
    g = res["exact"][PRIMARY]
    f = g["fire"]
    fr_rep = f.get("word_rep", {}).get("fire_rate", float("nan"))
    fr_ctl = f.get("control_word", {}).get("fire_rate", float("nan"))
    engaged = bool(fr_rep >= MIN_FIRE_RATE and fr_rep > fr_ctl)
    res["gates"] = {"engagement": dict(fire_rate_rep=fr_rep, fire_rate_ctl=fr_ctl,
                                       threshold=MIN_FIRE_RATE, pass_=engaged)}
    print(f"\n=== Engagement gate ===\n  repeated {100*fr_rep:.2f}% vs control "
          f"{100*fr_ctl:.2f}% of decode steps -> {'PASS' if engaged else 'FAIL'}")

    informative = bool(g["ctl"] == g["ctl"] and g["ctl"] >= MIN_CONTROL_EXACT)
    res["gates"]["informative"] = dict(control_exact=g["ctl"],
                                       threshold=MIN_CONTROL_EXACT, pass_=informative)
    print(f"=== Informativeness gate ===\n  control exact {100*g['ctl']:.1f}% "
          f"-> {'PASS' if informative else 'FAIL'}")

    gap = g["gap_points"]
    base_gap = res["exact"][BASELINE]["gap_points"]
    res["headline"] = dict(arm=PRIMARY, gap_points=gap,
                           gap_ci95=g["gap_ci95"],
                           baseline_arm=BASELINE, baseline_gap_points=base_gap,
                           change_points=gap - base_gap)
    if not engaged:
        res["verdict"] = "VACUOUS"
        res["reason"] = (f"RAS fired on {100*fr_rep:.2f}% of decode steps on repeated items; "
                         "the arm did not test the rule and licenses no claim about this axis")
    elif not informative:
        res["verdict"] = "UNINFORMATIVE"
        res["reason"] = (f"the control exact rate under RAS is {100*g['ctl']:.1f}%, below the "
                         f"{100*MIN_CONTROL_EXACT:.0f}% bar: RAS degrades rendering on both "
                         "families, so the gap moves for a reason unrelated to counting")
    elif gap <= MITIGATES_AT:
        res["verdict"] = "RAS MITIGATES"
        res["reason"] = (f"the exact-rate gap under RAS is {gap:+.1f} points, at most half the "
                         f"{SURVIVES_AT:+.1f} smallest gap the existing decoding-rule sweep "
                         "reaches on this checkpoint; the robustness claim is retracted for "
                         "this axis and RAS is reported as a working mitigation")
    elif gap >= SURVIVES_AT:
        res["verdict"] = "DEFICIT SURVIVES RAS"
        res["reason"] = (f"the exact-rate gap under RAS is {gap:+.1f} points, inside the "
                         f"[{SURVIVES_AT:+.1f}, {res['reference_band'][1]:+.1f}] range the "
                         "penalty and greedy arms already span on this checkpoint; the "
                         "robustness claim extends to the repetition-history-aware axis")
    else:
        res["verdict"] = "INCONCLUSIVE"
        res["reason"] = (f"the exact-rate gap under RAS is {gap:+.1f} points: below the "
                         f"{SURVIVES_AT:+.1f} floor of the existing sweep but above the "
                         f"{MITIGATES_AT:+.1f} halving bar. The deficit is reduced and not "
                         "closed; this licenses neither extending the robustness claim to "
                         "this axis nor calling RAS a fix")
    print(f"\nVERDICT: {res['verdict']} -- {res['reason']}")

    # ------------------------------------------------------------------
    # POST-HOC. Everything above was fixed before any RAS audio existed.
    # Everything below was written after seeing the verdict and is labelled as
    # such. It does not change the verdict; it says what weakens it.
    # ------------------------------------------------------------------
    weak: list[str] = []

    # The regenerated stock baseline against the stored panel rows, same items,
    # same seeds, same judge. This started as a drift caveat -- the first
    # version of the gate's stored-audio check compared an in-memory float32
    # array against a PCM-16 file read-back and so reported a mismatch on every
    # item unconditionally. With that fixed the check passes, and this block
    # became the arm's strongest end-to-end validation instead of its largest
    # caveat: if `qwen06brasoff` reproduces the published `qwen06b` numbers, the
    # whole new path -- processor, sampler, generator, judge, scorer -- lands
    # exactly on the paper's landed result before a single rule is switched on.
    drift: dict = {}
    en = REPO / "data/results/behavioural_ctc.csv"
    if en.exists() and BASELINE in present:
        old = pd.read_csv(en)
        old = old[(old.model == "qwen06b") & (old.k >= args.kmin)]
        old, _ = panel(old, ablations=True)
        new = raw[(raw.model == BASELINE) & (raw.k >= args.kmin)]
        new, _ = panel(new, ablations=True)
        j = old.merge(new, on=["item_id", "seed", "family", "k"], suffixes=("_o", "_n"))
        drift = dict(
            n_paired=int(len(j)),
            same_count_a=float((j.count_a_o == j.count_a_n).mean()) if len(j) else float("nan"),
            stored_gap_points=gap_of(pd.read_csv(en), "qwen06b", args.kmin)["gap_points"],
            regenerated_gap_points=res["exact"][BASELINE]["gap_points"])
        drift["gap_shift_points"] = (drift["regenerated_gap_points"]
                                     - drift["stored_gap_points"])
        drift["reproduces_published"] = bool(
            drift["n_paired"] > 0 and drift["same_count_a"] == 1.0
            and abs(drift["gap_shift_points"]) < 0.05)
        print("\n=== The regenerated baseline against the published panel rows ===")
        print(f"  paired items {drift['n_paired']}, identical count_a on "
              f"{100*drift['same_count_a']:.1f}% of them")
        print(f"  exact-rate gap {drift['stored_gap_points']:+.1f} (stored, published) -> "
              f"{drift['regenerated_gap_points']:+.1f} (regenerated) "
              f"= {drift['gap_shift_points']:+.1f} points")
        print("  -> the new path reproduces the paper's landed result exactly"
              if drift["reproduces_published"] else
              "  -> the new path does NOT reproduce the paper's landed result; "
              "the RAS arms are read against the regenerated baseline only")
        res["baseline_vs_published"] = drift

    if drift.get("reproduces_published"):
        res["strengthens"] = [
            "The no-op arm reproduces the published qwen06b rows exactly -- identical "
            f"count_a on {drift['n_paired']} paired items and the same "
            f"{drift['stored_gap_points']:+.1f} point gap -- so the RAS arms differ from "
            "the paper's landed result in the sampling rule and in nothing else."]
    if gate.get("stored_matches_fresh_all") is False:
        weak.append(
            "The stored qwen06b panel audio does not reproduce byte-for-byte in today's "
            "environment (transformers moved under us since the panel run), so the RAS arms "
            "are read against qwen06brasoff, a stock baseline regenerated today, not against "
            "the published +92.1. The published number is quoted only as the band's top. "
            + (f"The drift moves the gap by {drift.get('gap_shift_points', float('nan')):+.1f} "
               f"points ({drift.get('stored_gap_points', float('nan')):+.1f} stored -> "
               f"{drift.get('regenerated_gap_points', float('nan')):+.1f} regenerated) and "
               f"leaves count_a identical on "
               f"{100*drift.get('same_count_a', float('nan')):.1f}% of paired items."
               if drift else ""))
    alt = f.get("word_rep", {}).get("fire_rate_alt_reading", float("nan"))
    weak.append(
        f"The printed repetition ratio sums from k = 0, which includes the token compared "
        f"with itself; taken literally the rule fires when the candidate occurs at least once "
        f"in the previous K, and that is what was run ({100*fr_rep:.1f}% of steps on repeated "
        f"items). Under the other reading (sum from k = 1, so at least twice) it would have "
        f"fired on {100*alt:.1f}% of steps. Only the literal reading drove generation.")
    weak.append(
        "The fallback branch is the literal reading of 'random sampling': the untruncated, "
        "untempered model distribution. That is the flattest available reading and so the one "
        "most able to escape a loop; a tempered fallback would fire the same but escape less.")
    weak.append(
        "VALL-E 2 sweeps top-p from 0.0 to 0.8; we ran two points, the checkpoint's shipped "
        "nucleus and v = 0.0. A minimum of the gap somewhere inside that interval is not "
        "excluded by these arms.")
    weak.append(
        "One checkpoint. RAS was tested on Qwen3-TTS-0.6B only, for the reasons in "
        "src/common/ras.py; it is not a claim about RAS on VALL-E 2's own architecture, which "
        "we do not have.")
    weak.append(
        "Ours is an implementation of a published description, not the authors' code, which "
        "was not released. Section 'WHAT WE IMPLEMENTED' in src/common/ras.py lists every "
        "decision the description leaves open.")
    res["weakens"] = weak
    print("\n=== What weakens this (post-hoc) ===")
    for w in weak:
        print(f"  * {w}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
