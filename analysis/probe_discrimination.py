#!/usr/bin/env python3
r"""Does the probe distinguish our account from the leading rival? It does not.

Two accounts predict the behaviour we measure. Ours says the decoder's state
stops carrying the count, so no readout -- including the model's own stop head --
can recover it. The rival, reported for text LMs, says the count survives in the
residual stream and only the output policy fails to act on it.

A linear probe on late-layer states is the experiment that could separate them,
and the paper cited it as support: retention (late R^2 over early R^2) is lower
on repeated text than on its control in every checkpoint. A round-8 reviewer
pointed out that this does not show what we claimed, and they are right. Two
things spoil it:

* **The count is still decodable late.** Late R^2 on repeated items sits around a
  half. A state that had lost the count would not support that, and a rival that
  says the count survives predicts exactly it.
* **Retention above 1 is common.** In two checkpoints the count is *more*
  decodable late than early, by a wide margin. "Degraded, not erased" describes
  the median and hides that.

What survives is the *relative* claim: repeated text retains less than its
matched control, in every checkpoint. That is consistent with our account and
also with a rival in which the policy degrades faster on repeated text, so it
does not discriminate. This script reports both halves so the paper can say so
rather than quote the half that flatters it.

Usage:  python analysis/probe_discrimination.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.population import ABLATIONS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/results/probe.json")
    ap.add_argument("--out", default="data/results/probe_discrimination.json")
    args = ap.parse_args()

    models = {k: v for k, v in json.loads(Path(args.probe).read_text())["models"].items()
              if k not in ABLATIONS}

    rows: dict = {}
    print(f"{'model':10s} {'rep late R2':>12s} {'rep ret':>8s} {'ctl ret':>8s} "
          f"{'rep<ctl':>8s}")
    for m, v in sorted(models.items()):
        r, c = v.get("word_rep"), v.get("control_word")
        if not r or not c:
            continue
        row = dict(rep_early=r["early"]["best"]["r2"], rep_late=r["late"]["best"]["r2"],
                   ctl_early=c["early"]["best"]["r2"], ctl_late=c["late"]["best"]["r2"],
                   rep_retention=r.get("retention", np.nan),
                   ctl_retention=c.get("retention", np.nan))
        row["rep_below_ctl"] = bool(row["rep_retention"] < row["ctl_retention"])
        rows[m] = row
        print(f"{m:10s} {row['rep_late']:12.3f} {row['rep_retention']:8.3f} "
              f"{row['ctl_retention']:8.3f} {str(row['rep_below_ctl']):>8s}")

    late = np.array([v["rep_late"] for v in rows.values()])
    ret = np.array([v["rep_retention"] for v in rows.values()])
    res = dict(
        models=rows, n_models=len(rows),
        n_rep_below_ctl=int(sum(v["rep_below_ctl"] for v in rows.values())),
        rep_late_r2_median=float(np.median(late)),
        rep_late_r2_lo=float(late.min()), rep_late_r2_hi=float(late.max()),
        n_retention_above_one=int((ret > 1).sum()),
        rep_retention_median=float(np.median(ret)),
        rep_retention_lo=float(ret.min()), rep_retention_hi=float(ret.max()))

    print(f"\nrepeated-item late R^2: median {res['rep_late_r2_median']:.2f} "
          f"(range {res['rep_late_r2_lo']:.2f}-{res['rep_late_r2_hi']:.2f}) "
          f"-- the count is still substantially decodable late")
    print(f"retention above 1 in {res['n_retention_above_one']} of {res['n_models']} "
          f"checkpoints (range {res['rep_retention_lo']:.2f}-"
          f"{res['rep_retention_hi']:.2f})")
    print(f"repeated retention below its own control in "
          f"{res['n_rep_below_ctl']} of {res['n_models']}")
    res["discriminates"] = False
    print("\nThe relative effect is consistent; the absolute one is not what we\n"
          "claimed. A state that had lost the count could not support a late R^2\n"
          "near a half, and the rival account predicts precisely that persistence.\n"
          "This test therefore does not discriminate between the two, and the\n"
          "paper should not cite it as though it does.")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
