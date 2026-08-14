#!/usr/bin/env python3
"""Every number in a supplement table must appear in the JSON it came from.

`scripts/check_numbers.py` protects the main paper: it is built entirely from
macros, so a stale number is impossible there. The supplement is not. Its tables
are hand-written LaTeX, typed from a script's stdout, and nothing has been
stopping one of them from keeping a value after a rerun changed it. That is the
failure this catches, and it is not hypothetical --- the horizon fit returned
4.13 and then 3.48 for the same functional form within one session, because the
two scripts were reading different populations.

The check is deliberately crude: for each result JSON, format each number the way
the supplement formats it and assert the string occurs somewhere in `supp.tex`.
Crude in two directions, and both are the right way round:

  * A number that appears for an unrelated reason passes. Fine --- this is a
    tripwire against staleness, not a parser.
  * A number the supplement legitimately does not quote fails. Also fine: the
    fix is either to quote it or to drop it from the list here, and both are
    decisions worth making deliberately. That is how Llasa-8B's below-horizon
    row got into S12, having been omitted by accident.

Usage:  python scripts/check_supp_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUPP = REPO / "paper/supplementary/supp.tex"


def load(name: str):
    p = REPO / "data/results" / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    if not SUPP.exists():
        print("  SKIP supp.tex not found")
        return 0
    supp = SUPP.read_text()
    missing: list[str] = []

    def want(label: str, value: float, nd: int = 1, suffix: str = "") -> None:
        """Assert the formatted value appears in supp.tex.

        `suffix` narrows the match, and it matters more than it looks. The check
        is a substring search, so a two-digit integer is nearly toothless: a
        percentage of 87 was verified against a supplement that also contains
        61.7, 8.7 and a dozen other strings with those digits in them, and
        perturbing it to 61 still passed. Passing "\\%" pins the number to its
        unit and restores the teeth for exactly the values where the crude
        version had none.
        """
        needle = f"{value:.{nd}f}{suffix}"
        if needle not in supp:
            missing.append(f"{label} = {needle}")

    hf = load("horizon_forms.json")
    if hf:
        for f in hf["forms"]:
            v = hf["pooled"][f]
            want(f"S17 pooled {f} K_rep", v["rep"]["n_star"])
            want(f"S17 pooled {f} K_ctl", v["ctl"]["n_star"])
            want(f"S17 pooled {f} ratio", v["ratio"], 2)
        for m, d in hf["models"].items():
            for f in hf["forms"]:
                want(f"S17 {m} {f} ratio", d[f]["ratio"], 2)

    es = load("exclusion_sensitivity.json")
    if es:
        for name, a in es["arms"].items():
            want(f"S18 {name} gap", 100 * a["exact_gap"])
            want(f"S18 {name} n", a["n"], 0)

    gd = load("greedy_decoding.json")
    if gd:
        for name, a in gd["arms"].items():
            want(f"S22 {name} gap", 100 * a["gap"])
            want(f"S22 {name} exact rep", 100 * a["rep"]["exact"])

    vc = load("vits_config.json")
    if vc:
        for m, a in vc["arms"].items():
            want(f"S13 {m} exact rep", 100 * a["rep_exact"])
            want(f"S13 {m} exact ctl", 100 * a["ctl_exact"])

    ck = load("checkpoint_level.json")
    if ck:
        for m, v in ck["exact_rate_gap"]["per_model"].items():
            want(f"S23 {m} exact gap", 100 * v)
        for m, v in ck["capacity_gap"]["per_model"].items():
            want(f"S23 {m} capacity gap", v)

    ph = load("probe_horizon_compare.json")
    if ph:
        for k, v in ph["rows"].items():
            want(f"S12 {k} MAE", v["mae"], 2)
            want(f"S12 {k} R2", v["r2"], 2)

    # S14's judge replication. This is the objection five review rounds kept
    # returning to, so its numbers are the ones a stale supplement would be
    # most damaging about -- and the per-judge gaps in particular, since the
    # paper quotes their range and a reader checking the weakest judge should
    # find the same figure in both places.
    ij = load("independent_judge.json")
    if ij:
        for name, v in ij.get("all_judges", {}).items():
            short = name.split("/")[-1]
            want(f"S14 {short} gap", 100 * v["mean"])
        sp = ij.get("arm_spread_across_judges", {})
        if sp:
            # The arm-spread contrast is the decisive argument, not decoration:
            # if these two drift apart from the JSON the argument silently dies.
            want("S14 repeated-arm spread", 100 * sp["repeated_spread"])
            want("S14 control-arm spread", 100 * sp["control_spread"])
        vd = ij.get("verdict", {})
        if vd:
            want("S14 asymmetry A", vd["A"], 2)

    # S26's replication table. Every cell of it is derived rather than copied
    # -- the P1 ratio, the worst cell as a fraction of a full transfer, the
    # equivalence bound as the same fraction -- so a rerun that changed any
    # denominator would leave four rows of plausible, wrong percentages behind
    # with nothing complaining. The percentages are checked as integers because
    # that is how the table prints them.
    cs = load("causal_count_second_checkpoint.json")
    if cs:
        for m, c in cs.get("checkpoints", {}).items():
            a = c.get("rank1_patch_published_protocol") or c.get("rank1_patch")
            if not a:
                continue
            want(f"S26 {m} P1 ratio", a["disruption_index"], 2)
            full = a.get("full_transfer_logcount")
            if full:
                want(f"S26 {m} worst cell pct",
                     100 * a["largest_median_abs_shift"] / full, 0)
            # A bound is quoted only for the arms entitled to one. Llasa-8B's
            # verdict is "cannot tell", and printing a bound for an arm that
            # cannot be read would be the exact overclaim the gate exists to
            # prevent -- so it is absent from the table on purpose.
            eq = a.get("equivalence")
            if eq and a.get("verdict") == "readable null":
                want(f"S26 {m} bound pct",
                     100 * eq["bound_as_fraction_of_transfer"], 0)
        # The Llasa-8B diagnostic: the same-k donor must stay at least as
        # disruptive as the cross-k one, or the sentence built on it is false.
        l8 = cs.get("checkpoints", {}).get("llasa8b", {}).get(
            "rank1_patch_published_protocol", {})
        for cell, label in (("diffseed|L16|P128", "same-k"),
                            ("crossk|L16|P128", "cross-k")):
            v = l8.get("cells", {}).get(cell, {}).get("median_abs_shift")
            if v is not None:
                want(f"S26 llasa8b {label} shift", v, 3)

    # S25's worked examples. This table is the one place the supplement quotes
    # individual rows rather than aggregates, so it is the one most exposed to a
    # rescore silently moving a cell: the selection rule is deterministic, which
    # means a changed pipeline returns *different rows under the same rule* and
    # the old ones would sit there looking fine. The duration ratio and RMS are
    # checked too, since those are what `classify` reads and the prose argues
    # from them ("1.73x the length six renditions take", "RMS 0.0003: silence").
    we = load("worked_examples.json")
    if we:
        for e in we.get("examples", []):
            tag = f"S25 {e['outcome']}"
            want(f"{tag} duration ratio", e["duration_ratio"], 2)
            want(f"{tag} rms", e["rms"], 4)
            for f in ("k", "count_a", "count_b"):
                if str(e[f]) not in supp:
                    missing.append(f"{tag} {f} = {e[f]}")

    # S29's repetition-aware sampling arm. Newest section, hand-typed from its
    # JSON, and the third in a row where a number reached the supplement from an
    # agent's report rather than from the artifact -- the fire rate went in at
    # 48.9% against 48.8% on disk. That is small and it is the same failure as
    # the two before it, so the table gets a tripwire like its neighbours.
    ras = load("rep_aware_sampling.json")
    if ras:
        for arm, v in ras.get("exact", {}).items():
            want(f"S29 {arm} rep exact", 100 * v["rep"], 1, r"\%")
            want(f"S29 {arm} gap", v["gap_points"], 1)
            for b in v.get("gap_ci95", []):
                want(f"S29 {arm} CI bound", b, 1)
        g = ras.get("gates", {}).get("engagement", {})
        if g:
            # The two rates that make the null informative rather than vacuous:
            # if the rule stops firing on repeated items, the arm stops being
            # evidence and the section's argument has to be rewritten.
            want("S29 fire rate repeated", 100 * g["fire_rate_rep"], 1, r"\%")
            want("S29 fire rate control", 100 * g["fire_rate_ctl"], 1, r"\%")
        for arm, v in ras.get("paired", {}).items():
            want(f"S29 {arm} fixed", v["fixed"], 0)
            want(f"S29 {arm} broken", v["broken"], 0)

    # S8's naturalness table. This was supporting detail until the body started
    # leaning on it: the limits paragraph now reports that the CONTROL is the
    # less probable text in 87% of pairs, which is what turns the
    # out-of-distribution rival from a conceded confound into one that predicts
    # the wrong arm. A drift in the scorer would quietly reverse that argument.
    for f, tag in (("text_nll.json", "S8 nll panel-backbone"),
                   ("text_nll_independent.json", "S8 nll independent")):
        nll = load(f)
        if not nll:
            continue
        want(f"{tag} rep mean", nll["rep_mean"], 2)
        want(f"{tag} ctl mean", nll["ctl_mean"], 2)
        want(f"{tag} gap", nll["diff_mean"], 2)
        want(f"{tag} ctl-higher pct", 100 * nll["frac_ctl_higher"], 0, "\\%")

    # S8's capacity gain, for the same reason: it is the paper's only positive
    # mechanistic measurement and moved into the body this week.
    cap = load("capacity.json")
    if cap and cap.get("ratio_median") is not None:
        want("S8 capacity ratio", cap["ratio_median"], 2)
    cc = load("capacity_confound.json")
    if cc:
        for k, lbl in (("ratio_raw_median", "raw"),
                       ("ratio_adjusted_median", "adjusted"),
                       ("ratio_correct_only_median", "correct-only")):
            if cc.get(k) is not None:
                want(f"S8 capacity {lbl}", cc[k], 2)

    # S28's odd rungs. Audited by hand once and clean, which is exactly the
    # state a number is in just before it drifts: the ladder has been rescored
    # whenever a checkpoint or rung was added, and the pooled rates here are
    # what the body quotes for the interpolation claim.
    po = load("period_odd.json")
    if po:
        E = po.get("tests", {}).get("exact", {}).get("E_pooled", {})
        for p in ("1", "2", "3", "4", "6", "8"):
            if p in E:
                want(f"S28 odd E(p={p})", 100 * E[p], 1)
        # The adjacent gaps the two interpolation tests are built from. 1->2 is
        # deliberately not among them: it is the published ladder's own first
        # step, quoted in the main-ladder discussion rather than in the odd-rung
        # subsection, and demanding it here would push a number into a section
        # that has no argument for it. Checking it and finding it absent is how
        # this exclusion got made explicitly instead of by omission.
        for k in ("2->3", "3->4", "4->6", "6->8"):
            v = po.get("bootstrap_gaps", {}).get(k)
            if not v:
                continue
            want(f"S28 odd gap {k}", 100 * v["mean"], 1)
            want(f"S28 odd gap {k} lo", 100 * v["lo"], 1)
            want(f"S28 odd gap {k} hi", 100 * v["hi"], 1)
        # The two figures that carry "dropping XTTS-v2 makes both tests
        # cleaner", i.e. the reason it is kept in rather than dropped.
        x = po.get("leave_one_out", {}).get("xtts2", {}).get("E", {})
        for p in ("6", "8"):
            if p in x:
                want(f"S28 odd without-xtts2 E(p={p})", 100 * x[p], 1)

    # S28's shuffled-order control. This section cost the paper its old title, so
    # it is the one a sceptical reviewer will check line by line, and every
    # number in it is hand-typed from a JSON that has been rescored four times.
    # The per-checkpoint table is the load-bearing part: the pooled null is only
    # honest because the split beneath it is reported, and a stale cell there
    # would turn a reported architecture split back into a clean null.
    sh = load("period_shuffled.json")
    if sh:
        rec = sh.get("recovery", {}).get("exact", {}).get("vs_rerendered", {})
        for p in ("2", "4"):
            if p in rec.get("R", {}):
                want(f"S28 shuffle R(p={p})", rec["R"][p], 3)
        for ck, v in rec.get("per_checkpoint", {}).items():
            want(f"S28 shuffle {ck} E_per(4)", 100 * v["E_per"]["4"], 1)
            want(f"S28 shuffle {ck} E_shuf(4)", 100 * v["E_shuf"]["4"], 1)
        bs = sh.get("bootstrap", {})
        if bs:
            want("S28 shuffle P[confirm]", bs["p_confirm"], 3)
        p2 = sh.get("p2_uninformative", {}).get("2", {})
        if p2:
            want("S28 shuffle p=2 adjacency rate",
                 p2["shuffled_adjacent_repeat_rate"], 3)
        # The order-insensitive recount on XTTS-v2, which is what makes its
        # collapse a rendering failure rather than an artifact of the ordered
        # scan. It is the sentence that keeps that checkpoint in the analysis.
        eu = sh.get("rates", {}).get("exact_u", {})
        for a in ("periodic", "shuffled"):
            v = eu.get(a, {}).get("4", {}).get("xtts2")
            if v is not None:
                want(f"S28 xtts2 recount p=4 {a}", 100 * v, 1)

    # S26's positive control, same reasoning: it demoted the causal null, and
    # its matched-n comparison is quoted in the body as well, so a drift here
    # would put the two documents in disagreement -- which has already happened
    # once in this project, in this exact section.
    pc = load("causal_positive_control.json")
    if pc:
        for ck, c in pc.get("checkpoints", {}).items():
            cell = c.get("cells", {}).get("count|crossk|a1", {})
            if cell.get("S1_over_floor") is not None:
                want(f"S26 pc {ck} count ratio", cell["S1_over_floor"], 2)
            car = c.get("cells", {}).get("carrier|crosstemplate|a1", {})
            if car.get("S4_bound_as_fraction_of_transfer") is not None:
                want(f"S26 pc {ck} carrier bound",
                     100 * car["S4_bound_as_fraction_of_transfer"], 1)

    if missing:
        print(f"  FAIL {len(missing)} supplement numbers are not in supp.tex:")
        for m in missing:
            print(f"       {m}")
        print("       Either the supplement is stale, or it legitimately does "
              "not quote these\n       and the list in this script should say so.")
        return 1
    print("  OK   every result-JSON number appears in supp.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
