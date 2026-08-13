#!/usr/bin/env python3
"""Are the exclusion rules actually blind to the comparison they feed?

The paper asserts "exclusions are blind to arm and score". A reviewer asked how
that is verified rather than declared, which is the right question: an exclusion
rule that reads the outcome it is about to be used to compute is circular, and
the assertion is worth exactly as much as the check behind it. Until now there
was no check.

There are four rules (`src/common/population.py`), and they are not equally
blind. This script establishes what is true of each rather than asserting one
blanket claim.

  1. **Ablations and off-panel item arms.** Functions of the `model` column and
     the item-id prefix. Nothing to test empirically: no scored quantity is in
     scope. Reported for completeness.

  2. **Judge-unmeasurable templates.** The one rule derived from data, so the
     one that could favour an arm. Two different things get called "blind"
     here and only one of them is testable by re-deriving the list per arm:

     *Provenance* is arm-asymmetric and always was, by construction. Deriving
     the list from repeated-arm rows alone returns nothing; from control-arm
     rows alone it returns t2. That is not the rule peeking at the arms --- it
     is the stimulus design. A template's repeated arm scores one target word
     and its control arm scores k fillers, so they carry different scored
     vocabulary, and t2's unmeasurable words (`okay`, `hmm`) are fillers. An
     earlier version of this script called that a blindness failure; it is not.

     *Application* is what matters, and it is symmetric: the rule excludes a
     TEMPLATE, so both arms of t2 go. That is the property tested here --- every
     excluded template must lose all of its rows in both arms, never some.

     *Direction* is the number a reviewer actually needs, so it is printed
     rather than argued: this exclusion removes control rows the judge was
     failing, which raises the control's exact rate and therefore INFLATES the
     gap. The figures printed below are this one rule's effect on raw pooled
     rows, with no other rule applied and no k threshold, so they are smaller
     than and not comparable to the paper's specification-curve figures, which
     move every rule at once at k>=6. Both say the same thing: the exclusions
     help us, the effect survives without them, and the size is disclosed.

  3. **Items cut off by our own token budget.** `hit_cap` is written by the
     generation harness before any transcript exists, so it cannot depend on
     the judge. That is testable: the flags must be identical row-for-row
     across two different judges' scored tables, because they come from the
     same metadata files either way.

  4. **Degenerate and empty audio.** This one is NOT blind to scoring and the
     script says so instead of pretending. `classify` reads duration, RMS,
     spectral flatness and whether the transcript is empty. Those are audio and
     transcript properties, not count properties --- the rule cannot see whether
     a generation counted correctly --- but it is computed by the scorer, so
     the honest statement is "blind to the count, not blind to the transcript",
     and the paper reports the degenerate rate as its own outcome rather than
     folding it in.

Exit status is non-zero if a rule that should be blind is not.

Usage:  python scripts/check_exclusions_blind.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.common.population import (ABLATIONS, NON_PANEL_ITEM_PREFIXES,  # noqa: E402
                                   excluded_templates)

CTC = REPO / "data/results/behavioural_ctc.csv"
WHISPER = REPO / "data/results/behavioural.csv"
AUDIT = REPO / "analysis/judge_vocab_audit.py"

REPEATED = {"word_rep", "sentence_rep"}
CONTROL = {"control_word", "control_sentence"}


def audit_on(rows: pd.DataFrame, tag: str) -> list[str]:
    """Re-derive the unmeasurable-template list from one arm's rows only."""
    with tempfile.NamedTemporaryFile("w", suffix=f"_{tag}.csv", delete=False) as f:
        rows.to_csv(f.name, index=False)
        src = f.name
    with tempfile.NamedTemporaryFile("w", suffix=f"_{tag}.json", delete=False) as f:
        dst = f.name
    r = subprocess.run(
        [sys.executable, str(AUDIT), "--behavioural", src, "--out", dst],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"  FAIL judge_vocab_audit.py failed on the {tag} arm:\n{r.stderr}")
    return sorted(json.loads(Path(dst).read_text()).get("excluded_templates", []))


def main() -> int:
    fail = 0
    d = pd.read_csv(CTC)

    print(f"  rule 1  ablations={len(ABLATIONS)} models, off-panel item prefixes="
          f"{list(NON_PANEL_ITEM_PREFIXES)}: functions of model/item id, no scored "
          f"quantity in scope")

    # ---- rule 2: application symmetry is the testable claim
    published = sorted(excluded_templates())
    rep = audit_on(d[d.family.isin(REPEATED)], "rep")
    ctl = audit_on(d[d.family.isin(CONTROL)], "ctl")
    print(f"  rule 2  excluded templates {published or '(none)'}; evidence comes "
          f"from the control arm ({ctl or '(none)'}) and not the repeated arm "
          f"({rep or '(none)'}), because the two arms score different vocabulary")
    for t in published:
        rows = d[d.template == t]
        r_n, c_n = int(rows.family.isin(REPEATED).sum()), int(rows.family.isin(CONTROL).sum())
        if r_n and c_n:
            print(f"          t{t[1:]}: drops all {r_n} repeated and all {c_n} "
                  f"control rows -- applied to the template, not to an arm")
        else:
            print(f"  FAIL   {t} exists in only one arm ({r_n} repeated, {c_n} "
                  f"control): excluding it is not symmetric")
            fail = 1
    # The direction, printed rather than argued about.
    keep = d[~d.template.isin(published)]
    for name, frame in (("with the rule", keep), ("without it", d)):
        g = frame[frame.family.isin(REPEATED)].correct.mean() * 100
        c = frame[frame.family.isin(CONTROL)].correct.mean() * 100
        print(f"          {name:14s} repeated {g:5.1f}%  control {c:5.1f}%  "
              f"gap {c - g:5.1f} points")

    # ---- rule 3: cap flags must not depend on the judge
    if WHISPER.exists():
        w = pd.read_csv(WHISPER)
        key = ["model", "item_id", "seed"]
        m = d[key + ["hit_cap"]].merge(w[key + ["hit_cap"]], on=key,
                                       suffixes=("_ctc", "_whisper"))
        bad = int((m.hit_cap_ctc != m.hit_cap_whisper).sum())
        if bad == 0:
            print(f"  rule 3  hit_cap identical under both judges on all {len(m)} "
                  f"shared rows: written by the harness, not by the scorer")
        else:
            print(f"  FAIL   hit_cap differs between judges on {bad}/{len(m)} rows; "
                  f"it is not a pre-scoring property")
            fail = 1
    else:
        print("  rule 3  SKIP second judge's table not present")

    # ---- rule 4: state what it is, and what it is not
    reps = d[d.family.isin(REPEATED)]
    ctls = d[d.family.isin(CONTROL)]
    dgn = {"empty", "degenerate"}
    print(f"  rule 4  degeneracy reads duration/RMS/flatness/empty-transcript --- "
          f"blind to the count, not blind to the transcript. Drops "
          f"{int(reps.outcome.isin(dgn).sum())}/{len(reps)} repeated and "
          f"{int(ctls.outcome.isin(dgn).sum())}/{len(ctls)} control rows; the "
          f"paper reports this rate rather than folding it in")

    print("  " + ("FAIL some exclusion rule is not blind" if fail else
                  "OK   every rule that should be blind to arm and score is"))
    return fail


if __name__ == "__main__":
    sys.exit(main())
