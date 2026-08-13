#!/usr/bin/env python3
"""Show the scoring pipeline doing its job, on rows nobody chose.

Three review rounds have made a version of the same objection: the phenomenon is
audible, the evidence is a number produced by an automated pipeline, and the
paper never once shows that pipeline at work on a concrete generation. S9 gives
the judge's accuracy on synthetic ground truth and S14 its noise floor against a
second recogniser, but neither lets a reader see a stimulus, the audio's
transcript, and the count that came out the other end, together, for a single
item.

The objection is fair and the fix is cheap. What makes it worth anything is that
the examples must not be chosen. So the selection rule is fixed here, in code,
and takes whatever it lands on:

  * restrict to the reportable panel (`src/common/population.py`), with the
    period arm and the ablations out, exactly as every headline number does;
  * sort by (model, item_id, seed) --- a total order fixed by the stimulus file
    and the seed schedule, neither of which was chosen with this table in mind;
  * walk the outcome labels in a fixed order and the checkpoints in sorted
    order together, so outcome $i$ is drawn from checkpoint $i \\bmod N$;
  * take the FIRST row of that pair.

The checkpoint cycle is there because the obvious rule --- first row per
outcome --- puts all six examples on Llasa-1B, which sorts first. That is a
property of the alphabet, not of the pipeline, and a table showing one
checkpoint answers the objection less well while being no more honest. Cycling
keeps the choice out of our hands and spreads the evidence.

The rule is blind to whether the example flatters us: if the first `correct`
row happens to be one where the judge got lucky, or the first `loop` row is one
a reader would call a judging error, it goes in the table anyway. Reading the
rule and reading the table are the same act.

Both counts are shown, not just the one the paper uses: `count_a` is the CTC
transcript count, `count_b` the duration-based estimate, and `count_final` what
the conjunction rule returned. A row where they disagree is more informative
than one where they agree, so nothing suppresses those.

Usage:  python analysis/worked_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.common.population import panel  # noqa: E402

# Fixed order. `correct` first because it is the least interesting and the most
# important to show: a reader who suspects the pipeline of manufacturing
# failures should see what a pass looks like before seeing a failure.
OUTCOMES = ["correct", "undercount", "overcount", "loop", "degenerate", "empty"]
MAXCHARS = 64


def main() -> None:
    d = pd.read_csv(REPO / "data/results/behavioural_ctc.csv")
    d, drop = panel(d, degenerate=True)          # keep degenerate: it is a row type here
    d = d.sort_values(["model", "item_id", "seed"], kind="mergesort")

    models = sorted(d.model.unique())
    rows = []
    for i, label in enumerate(OUTCOMES):
        want_model = models[i % len(models)]
        sub = d[(d.outcome == label) & (d.model == want_model)]
        fallback = sub.empty
        if fallback:
            # That checkpoint never produced this outcome, which is itself worth
            # recording rather than papering over. Fall back to the first row
            # across the panel and say in the artifact that we did.
            sub = d[d.outcome == label]
        if sub.empty:
            continue
        r = sub.iloc[0]
        tr = r.transcript if isinstance(r.transcript, str) else ""
        rows.append(dict(
            model=r.model, item_id=r.item_id, seed=int(r.seed), family=r.family,
            k=int(r.k), expected=int(r.expected_count),
            duration_s=float(r.duration_s),
            # The two quantities `classify` actually reads, so the label in the
            # last column can be checked against the row rather than trusted.
            duration_ratio=float(r.duration_ratio), rms=float(r.rms),
            transcript=tr, transcript_chars=len(tr),
            count_a=int(r.count_a), count_b=int(r.count_b),
            agree=bool(r.agree), count_final=int(r.count_final),
            outcome=str(r.outcome), correct=int(r.correct),
            assigned_model=want_model, fell_back=bool(fallback)))

    out = {"selection": "outcome i from checkpoint i mod N, first row under "
                        "(model, item_id, seed) order",
           "checkpoint_cycle": models,
           "population": drop, "n_rows_considered": int(len(d)), "examples": rows}
    p = REPO / "data/results/worked_examples.json"
    p.write_text(json.dumps(out, indent=2))

    for r in rows:
        t = r["transcript"]
        t = (t[:MAXCHARS] + "...") if len(t) > MAXCHARS else (t or "(none)")
        print(f"{r['outcome']:11s} {r['model']:9s} {r['item_id']:18s} s{r['seed']} "
              f"k={r['k']:2d} A={r['count_a']:3d} B={r['count_b']:3d} "
              f"final={r['count_final']:3d} {r['duration_s']:6.2f}s "
              f"dr={r['duration_ratio']:5.2f} rms={r['rms']:.4f}  {t}")
    print(f"\nwrote {p} ({len(rows)} examples from {len(d)} panel rows)")


if __name__ == "__main__":
    main()
