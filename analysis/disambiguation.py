#!/usr/bin/env python3
"""Does a minimal local break let the decoder individuate identical spans?

The companion to `analysis/period_ladder.py`, testing the same account from the
other side. The period ladder asks what happens when the repeated tokens stop
being identical; this asks what happens when they stay identical and only the
*boundary between them* changes:

    bare    very very very ...          the published repeated arm
    comma   very, very, very, ...       phrase break, word count unchanged
    stop    Very. Very. Very. ...       sentence break, word count unchanged
    and     very and very and very ...  a lexical break, word count 2k-1

The counted unit and the count are the same in all four. The pre-committed
reading lives in `data/stimuli/make_stimuli_disambig.py`, written before any
disambiguator audio existed, and is restated here so the thresholds cannot
drift:

    R = (E(dis) - E(bare)) / (E(8) - E(bare))       the recovered fraction

    R >= 0.50  -> INDIVIDUATION SUPPORTED
    R <= 0.15  -> INDIVIDUATION FALSIFIED
    otherwise  -> PARTIAL, and no mechanism section should be written on it

R is quoted on `comma` and `stop`, the length-matched cells. `and` is reported
beside them and never averaged in. Ordering prediction: `stop` >= `comma` > 0.

Four ways this could be a lie, all checked here:

1. **A front-end that strips punctuation.** Then `comma` is `bare` and its null
   is the pipeline's, not the model's. Generation is seeded per item, so the
   check is mechanical: if the two waveforms are identical the front-end saw
   the same string. Reported as UNINTERPRETABLE, never as a null.
2. **The judge.** The score is occurrences of the target word, and the inserted
   material changes the acoustics around it. Per-arm delivery of the target is
   audited before any rate is read.
3. **Length.** `and` carries k-1 extra words, and length hurts counting, so a
   recovery there is conservative and a null there is not interpretable alone.
4. **Prosody is not individuation.** Pauses lengthen the audio. If recovery
   tracks added duration rather than the break, "the model needed more time" is
   the right description and "individuation" is not. Duration per arm is
   reported, and `and` -- which adds tokens rather than silence -- separates
   them.

Usage:  python analysis/disambiguation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.common.population import panel, describe  # noqa: E402
from src.common.score_counts import normalise, count_occurrences, DATA_ROOT  # noqa: E402

VARIANTS = ["bare", "comma", "stop", "and"]
MATCHED = ["comma", "stop"]        # the length-matched cells R is quoted on


def load_stimuli(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.open()]
    return pd.DataFrame([
        dict(item_id=r["item_id"], variant=r["variant"], n_chars=len(r["text"]),
             stim_words=r["expected_words"], target=r["target_unit"])
        for r in rows])


def audio_identity(d: pd.DataFrame, res: dict) -> set[tuple[str, str, int, int]]:
    """Cells where a disambiguated waveform is byte-identical to its bare twin.

    Generation seeds per item, so identical audio means the text front-end
    normalised the disambiguator away and the model never saw it. Those cells
    measure the pipeline and are excluded from R.
    """
    h: dict[tuple, str] = {}
    for r in d.itertuples():
        p = Path(DATA_ROOT) / "audio" / r.model / f"{r.item_id}_s{r.seed}.wav"
        if p.exists():
            h[(r.model, r.template, r.k, r.seed, r.variant)] = \
                hashlib.md5(p.read_bytes()).hexdigest()
    dead: set = set()
    per_variant: dict[str, list[int]] = {v: [] for v in VARIANTS if v != "bare"}
    for (m, t, k, s, v), digest in h.items():
        if v == "bare":
            continue
        b = h.get((m, t, k, s, "bare"))
        if b is None:
            continue
        same = int(digest == b)
        per_variant[v].append(same)
        if same:
            dead.add((m, t, k, s))
    print("\n--- front-end check: is the disambiguated audio the bare audio? ---")
    rows = []
    for v, vals in per_variant.items():
        if not vals:
            continue
        rate = float(np.mean(vals))
        flag = "  <-- FRONT-END STRIPS IT, CELL UNINTERPRETABLE" if rate > 0.5 else ""
        print(f"  {v:6s} identical to bare in {100*rate:5.1f}% of {len(vals)} cells{flag}")
        rows.append(dict(variant=v, identical_rate=rate, n=len(vals)))
    res["front_end_check"] = rows
    return dead


def judge_audit(d: pd.DataFrame, res: dict) -> None:
    """Target-word delivery per arm: is the recogniser reading the arms alike?

    The score is occurrences of the target word, so a per-arm difference in how
    often the judge renders that word *at all* would masquerade as a difference
    in counting. Measured as the fraction of generations whose transcript
    contains the target at least once -- a criterion no amount of miscounting
    can fail, so a gap here is the recogniser's.
    """
    hit = [int(count_occurrences(normalise(str(t) if isinstance(t, str) else ""), u) > 0)
           for t, u in zip(d.transcript, d.target)]
    d = d.assign(delivered=hit)
    print("\n--- judge audit: does the target word come back at all? ---")
    rows = []
    for v in VARIANTS:
        g = d[d.variant == v]
        if not len(g):
            continue
        rows.append(dict(variant=v, delivery=float(g.delivered.mean()), n=int(len(g))))
        print(f"  {v:6s} target delivered in {100*g.delivered.mean():5.1f}% of "
              f"{len(g):4d} generations")
    res["judge_audit"] = rows
    spread = max(r["delivery"] for r in rows) - min(r["delivery"] for r in rows)
    if spread > 0.10:
        print(f"  WARNING: {100*spread:.1f} pt spread in target delivery across arms; "
              "part of any difference below is the recogniser, not the decoder.")
    res["judge_delivery_spread"] = float(spread)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_period.csv")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli_disambig.jsonl")
    ap.add_argument("--period", default="data/results/period_ladder.json",
                    help="supplies E(8), the no-deficit ceiling R is scaled by")
    ap.add_argument("--out", default="data/results/disambiguation.json")
    args = ap.parse_args()

    stim = load_stimuli(REPO / args.stimuli)
    d = pd.read_csv(REPO / args.behavioural)
    d = d[d.item_id.isin(set(stim.item_id))]
    d, drop = panel(d, period_arm=True)
    print(describe(drop))
    d = d.merge(stim, on="item_id", how="inner", validate="many_to_one")
    if not len(d):
        raise SystemExit("no disambiguator rows scored yet")
    d = d.assign(exact=(d.count_a == d.k))
    print(f"checkpoints: {', '.join(sorted(d.model.unique()))}   n={len(d)}")

    res: dict = {"n": int(len(d))}
    dead = audio_identity(d, res)
    judge_audit(d, res)

    # --- the table.
    t = d.pivot_table(index="model", columns="variant", values="exact", aggfunc="mean")
    n = d.pivot_table(index="model", columns="variant", values="exact", aggfunc="size")
    cols = [v for v in VARIANTS if v in t.columns]
    t, n = t.reindex(columns=cols), n.reindex(columns=cols)
    print("\nexact rate by disambiguator (period 1 throughout; k = 16, 24, 32)")
    print(f"  {'checkpoint':11s} " + "  ".join(f"{v:>6s}" for v in cols) + "     (n per cell)")
    for m in t.index:
        print(f"  {m:11s} " + "  ".join(f"{100*t.loc[m, v]:6.1f}" for v in cols)
              + "     (" + ",".join(str(int(n.loc[m, v])) for v in cols) + ")")
    print(f"  {'MEAN':11s} " + "  ".join(f"{100*t[v].mean():6.1f}" for v in cols))
    res["exact_by_variant"] = json.loads(t.to_json())
    res["n_by_variant"] = json.loads(n.to_json())

    # --- R, against the period ladder's own period-8 ceiling.
    pj = REPO / args.period
    ceil = {}
    if pj.exists():
        ceil = json.loads(pj.read_text()).get("exact_by_period_scan", {}).get("8", {})
    if not ceil:
        print("\nno period-ladder ceiling available; run analysis/period_ladder.py first")
        return
    print(f"\nrecovered fraction R = (E(dis) - E(bare)) / (E(8) - E(bare))")
    print(f"  {'checkpoint':11s} {'E(8)':>6s} {'E(bare)':>8s} " +
          "  ".join(f"R[{v}]" for v in cols if v != "bare"))
    rows = []
    for m in t.index:
        if m not in ceil:
            continue
        e8, eb = float(ceil[m]), float(t.loc[m, "bare"])
        den = e8 - eb
        r = {v: (float(t.loc[m, v]) - eb) / den if abs(den) > 1e-9 else float("nan")
             for v in cols if v != "bare"}
        rows.append(dict(model=m, E8=e8, E_bare=eb, **{f"R_{v}": r[v] for v in r}))
        print(f"  {m:11s} {100*e8:5.1f}% {100*eb:7.1f}% " +
              "  ".join(f"{r[v]:+.3f}" for v in cols if v != "bare"))
    res["recovered_fraction"] = rows

    matched = [v for v in MATCHED if v in cols]
    if rows and matched:
        vals = [np.mean([r[f"R_{v}"] for v in matched]) for r in rows]
        R = float(np.mean(vals))
        call = ("INDIVIDUATION SUPPORTED" if R >= 0.50 else
                "INDIVIDUATION FALSIFIED" if R <= 0.15 else "PARTIAL")
        res["R_matched"] = R
        res["verdict"] = call
        print(f"\n  mean R over the length-matched arms ({', '.join(matched)}): "
              f"{R:+.3f}  ->  {call}")
        if "and" in cols:
            Ra = float(np.mean([r["R_and"] for r in rows]))
            res["R_and"] = Ra
            print(f"  `and` (not length-matched, k-1 extra words, confound points "
                  f"down): R = {Ra:+.3f}")
        if "stop" in cols and "comma" in cols:
            ok = np.mean([r["R_stop"] for r in rows]) >= np.mean([r["R_comma"] for r in rows])
            res["ordering_as_predicted"] = bool(ok)
            print(f"  ordering stop >= comma: {'yes' if ok else 'NO -- unpredicted shape'}")
    if dead:
        print(f"\n  {len(dead)} cells had disambiguated audio identical to bare; "
              "those are pipeline nulls, not model nulls")
        res["identical_cells"] = len(dead)

    # --- duration, to separate "a break" from "more time".
    print("\n--- per-arm length and duration ---")
    rows = []
    for v in cols:
        g = d[d.variant == v]
        rows.append(dict(variant=v, words=float(g.stim_words.mean()),
                         chars=float(g.n_chars.mean()),
                         duration_s=float(g.duration_s.mean())))
        print(f"  {v:6s} {rows[-1]['words']:5.1f} words  {rows[-1]['chars']:6.1f} chars  "
              f"{rows[-1]['duration_s']:6.2f} s")
    res["per_arm_length"] = rows

    out = REPO / args.out
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
