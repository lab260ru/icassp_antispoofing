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

    def want(label: str, value: float, nd: int = 1) -> None:
        if f"{value:.{nd}f}" not in supp:
            missing.append(f"{label} = {value:.{nd}f}")

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
