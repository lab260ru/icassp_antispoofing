#!/usr/bin/env python3
"""Emit every number the paper quotes as a LaTeX macro, plus the result tables.

The paper never hard-codes an experimental value: `paper/main.tex` inputs
`numbers.tex`, which this script regenerates from the result CSVs. A stale
number in the PDF is therefore impossible as long as the build runs this first.

Usage:  python analysis/make_numbers.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {"llasa1b": "Llasa-1B", "llasa3b": "Llasa-3B", "llasa8b": "Llasa-8B",
         "xtts2": "XTTS-v2", "qwen06b": "Qwen3-TTS-0.6B", "qwen17b": "Qwen3-TTS-1.7B"}
PARAMS = {"llasa1b": "1B", "llasa3b": "3B", "llasa8b": "8B",
          "xtts2": "0.4B", "qwen06b": "0.6B", "qwen17b": "1.7B"}
ORDER = ["llasa1b", "llasa3b", "llasa8b", "xtts2", "qwen06b", "qwen17b"]
# Re-runs of a panel member under a changed decoding setting. Reported on their
# own, never pooled into panel statistics -- pooling inflates the panel with a
# non-independent copy, which a reviewer caught us doing.
ABLATIONS = {"xtts2norp"}
# LaTeX macro names cannot contain digits, so each model key gets an explicit
# spelled-out tag. An explicit map beats string munging: it is what the .tex
# files are written against, and a silent mismatch shows up as an undefined
# control sequence at build time rather than a wrong number.
TAG = {"llasa1b": "LlasaOneB", "llasa3b": "LlasaThreeB", "llasa8b": "LlasaEightB",
       "xtts2": "XttsTwo", "qwen06b": "QwenZeroSixB", "qwen17b": "QwenOneSevenB"}

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def fmt(x, nd=2, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{nd}f}"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, in percent.

    Wilson rather than normal-approximation because several cells sit at or near
    0% and 100%, where the normal interval is degenerate or runs outside [0,1].
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * min(c + h, 1.0)


def prop_macros(macros: dict, name: str, sub, col: str = "correct") -> None:
    """Emit percentage, Wilson CI bounds and n for one proportion."""
    n = int(len(sub))
    k = int(sub[col].sum()) if n else 0
    lo, hi = wilson(k, n)
    macros[name] = fmt(100 * k / n, 1) if n else "--"
    macros[name + "Lo"] = fmt(lo, 1)
    macros[name + "Hi"] = fmt(hi, 1)
    macros[name + "N"] = str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv",
                    help="CTC-judged by default; the Whisper table exists only for "
                         "the instrument audit and must not feed the paper")
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--summary", default="data/results/summary.json")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--outdir", default="paper")
    args = ap.parse_args()

    out = Path(args.outdir)
    macros: dict[str, str] = {}

    # ---- stimuli -------------------------------------------------------
    stim = [json.loads(l) for l in open(args.stimuli)]
    macros["NStimuli"] = str(len(stim))
    macros["NInstrumented"] = str(sum(1 for s in stim if s.get("instrumented")))
    macros["NControls"] = str(sum(1 for s in stim if s["family"] == "control_word"))
    macros["KMax"] = str(max(s["k"] for s in stim))

    beh = pd.read_csv(args.behavioural) if Path(args.behavioural).exists() else pd.DataFrame()
    state = pd.read_csv(args.state) if Path(args.state).exists() else pd.DataFrame()
    summ = json.loads(Path(args.summary).read_text()) if Path(args.summary).exists() else {}

    models = [m for m in ORDER if len(beh) and m in set(beh.model)] or []
    macros["NModels"] = WORDS.get(len(models), str(len(models)))
    macros["NModelsNum"] = str(len(models))
    fams = {m.split("0")[0][:5] for m in models}
    n_fam = len({"llasa" if m.startswith("llasa") else "xtts" if m.startswith("xtts")
                 else "qwen" for m in models})
    macros["NFamilies"] = WORDS.get(n_fam, str(n_fam))
    if len(beh):
        macros["NSeeds"] = WORDS.get(int(beh.seed.nunique()), str(beh.seed.nunique()))
        macros["NGenerations"] = str(len(beh))
        low = beh[beh.k <= 4]
        macros["AgreeLowK"] = fmt(100 * low.agree.mean(), 1) if len(low) else "--"
        macros["AuditPct"] = fmt(100 * (1 - beh.agree.mean()), 1)

    # ---- per-model headline numbers ------------------------------------
    for m in models:
        e = summ.get("models", {}).get(m, {})
        tag = TAG.get(m)
        if tag is None:
            continue
        macros[f"kStar{tag}"] = fmt(e.get("k_star"), 0)

    # Panel-level q_hat aggregates and the P2 regression macros were removed:
    # both estimators were abandoned (see results, 'Two things we could not
    # establish'). Leaving them defined invites quoting a number the paper no
    # longer stands behind. Preserved in git history and analysis/pooled_q.py.

    # ---- capacity confound check -----------------------------------------
    cc_path = Path("data/results/capacity_confound.json")
    if cc_path.exists():
        cc = json.loads(cc_path.read_text())
        macros["CapRawRatio"] = fmt(cc.get("ratio_raw_median"), 2)
        macros["CapAdjRatio"] = fmt(cc.get("ratio_adjusted_median"), 2)
        macros["CapCorrRatio"] = fmt(cc.get("ratio_correct_only_median"), 2)
        macros["CapCorrN"] = str(cc.get("ratio_correct_only_n", 0))

    # ---- capacity: the central state measurement -------------------------
    cap_path = Path("data/results/capacity.json")
    if cap_path.exists():
        cap = json.loads(cap_path.read_text())
        macros["CapRatio"] = fmt(cap.get("ratio_median"), 2)
        # How many checkpoints still gain states under repetition. A decoder at
        # the theorem's fixed point would gain none, so this is the number the
        # "slows, does not halt" sentence rests on.
        macros["CapPosModels"] = str(sum(
            1 for m, v in cap.get("models", {}).items()
            if m not in ABLATIONS and v.get("repeated", {}).get("lo", -1) > 0))
        macros["CapRatioPct"] = fmt(100 * cap.get("ratio_median", np.nan), 0)
        macros["NCapSep"] = str(cap.get("n_separated", 0))
        macros["NCapModels"] = str(cap.get("n_models", 0))
        n_sep, n_tot = cap.get("n_separated", 0), cap.get("n_models", 0)
        # Reads as prose either way: "all six" when the panel is unanimous,
        # "4 of 6" when it is not.
        macros["CapSepPhrase"] = (f"all {WORDS.get(n_tot, n_tot)}"
                                  if n_sep == n_tot and n_tot
                                  else f"{n_sep} of {n_tot}")
        gaps = [v["paired"]["gap"] for v in cap["models"].values()
                if np.isfinite(v["paired"].get("gap", np.nan))]
        fpos = [v["paired"]["frac_pos"] for v in cap["models"].values()
                if np.isfinite(v["paired"].get("frac_pos", np.nan))]
        if gaps:
            macros["CapGapMedian"] = fmt(np.median(gaps), 1)
        if fpos:
            macros["CapFracPos"] = fmt(100 * np.median(fpos), 0)
        for m, v in cap["models"].items():
            tag = TAG.get(m)
            if tag is None:
                continue
            macros[f"GainRep{tag}"] = fmt(v["repeated"]["gain"], 1)
            macros[f"GainCtl{tag}"] = fmt(v["control"]["gain"], 1)
            macros[f"GainRepLo{tag}"] = fmt(v["repeated"]["lo"], 1)
            macros[f"GainRepHi{tag}"] = fmt(v["repeated"]["hi"], 1)
            macros[f"GainCtlLo{tag}"] = fmt(v["control"]["lo"], 1)
            macros[f"GainCtlHi{tag}"] = fmt(v["control"]["hi"], 1)
            macros[f"SatRep{tag}"] = fmt(v["repeated"]["sat"], 0)
            macros[f"SatCtl{tag}"] = fmt(v["control"]["sat"], 0)

    # ---- Lemma B coverage --------------------------------------------------
    # Text attention is not separable in every architecture, so the dilution
    # evidence covers a subset of the panel. State the subset rather than
    # implying panel-wide support.
    if len(state) and "attn_block_entropy" in state.columns:
        have = set(state[state.attn_block_entropy.notna()].model.unique()) - ABLATIONS
        macros["LemBModels"] = WORDS.get(len(have), str(len(have)))
        macros["LemBModelsNum"] = str(len(have))

    # ---- count error: the behavioural measurement (CTC judge) -------------
    ce_path = Path("data/results/count_error.json")
    if ce_path.exists():
        ce = json.loads(ce_path.read_text())
        for key, tag in (("pooled_rep", "Rep"), ("pooled_ctl", "Ctl")):
            v = ce.get(key, {})
            macros[f"Err{tag}"] = fmt(100 * v.get("median", np.nan), 1)
            macros[f"Err{tag}Lo"] = fmt(100 * v.get("lo", np.nan), 1)
            macros[f"Err{tag}Hi"] = fmt(100 * v.get("hi", np.nan), 1)
            macros[f"Err{tag}N"] = str(v.get("n", 0))
        n_sep, n_tot = ce.get("n_separated", 0), ce.get("n_models", 0)
        macros["ErrSepPhrase"] = (f"all {WORDS.get(n_tot, n_tot)}"
                                  if n_sep == n_tot and n_tot else f"{n_sep} of {n_tot}")
        macros["ErrNSep"] = str(n_sep)
        macros["ErrNModels"] = str(n_tot)
        for m, v in ce.get("models", {}).items():
            tag = TAG.get(m)
            if tag is None:
                continue
            macros[f"Err{tag}"] = fmt(100 * v["rep"]["median"], 1)
            macros[f"ErrCtl{tag}"] = fmt(100 * v["ctl"]["median"], 1)
        # Checkpoint-level median of the per-checkpoint medians. The paper
        # argues in Section 4 that the checkpoint is the unit of replication,
        # then quoted an item-pooled figure in the abstract; a reviewer noticed
        # the two disagree by half their own size (-12.5 against -8.3). Both are
        # emitted so the text can show them together.
        _p = {m: v for m, v in ce.get("models", {}).items() if m not in ABLATIONS}
        _meds = [v["rep"]["median"] for v in _p.values()
                 if np.isfinite(v["rep"]["median"])]
        if _meds:
            macros["ErrRepCk"] = fmt(100 * float(np.median(_meds)), 1)
        # worst and best panel members, for the two-regime sentence
        panel = {m: v for m, v in ce.get("models", {}).items()
                 if m not in ABLATIONS and np.isfinite(v["rep"]["median"])}
        if panel:
            worst = min(panel, key=lambda m: panel[m]["rep"]["median"])
            best = max(panel, key=lambda m: panel[m]["rep"]["median"])
            macros["ErrWorstModel"] = LABEL.get(worst, worst)
            macros["ErrWorstVal"] = fmt(100 * panel[worst]["rep"]["median"], 1)
            macros["ErrBestModel"] = LABEL.get(best, best)
            macros["ErrBestVal"] = fmt(100 * panel[best]["rep"]["median"], 1)

    # ---- exactly-right rates ---------------------------------------------
    # The median relative error understates the contrast, because a control's
    # median of zero and a control that is *always* zero are different claims and
    # the paper previously ran them together ("not a single miscount", which was
    # false: 22% of control generations carried a nonzero error before the
    # judge-vocabulary exclusion, 5.7% after). The fraction of generations that
    # come back exactly right states it without room for that.
    beh_path = Path("data/results/behavioural_ctc.csv")
    if beh_path.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.common.population import panel as restrict
        raw = pd.read_csv(beh_path)
        raw = raw[raw.family.isin(["word_rep", "control_word"])]
        pop, _ = restrict(raw)
        pop = pop[pop.k >= 6]
        pop = pop.assign(err=(pop.count_a - pop.k) / pop.k)
        for fam, tag in (("word_rep", "Rep"), ("control_word", "Ctl")):
            g = pop[pop.family == fam]
            if len(g):
                macros[f"Exact{tag}"] = fmt(100 * (g.err == 0).mean(), 1)
                macros[f"Exact{tag}N"] = str(len(g))
        # exact-rate at the ends of the ladder: the count stays proportional to k
        # while the chance of being exactly right collapses, and those are the
        # two halves of the result.
        g = pop[pop.family == "word_rep"]
        for kk, tag in ((6, "KLo"), (32, "KHi")):
            s = g[g.k == kk]
            if len(s):
                macros[f"ExactRep{tag}"] = fmt(100 * (s.err == 0).mean(), 1)
        cm = pop[pop.family == "control_word"]
        if len(cm):
            macros["CtlMissPct"] = fmt(100 * (cm.err != 0).mean(), 1)
        macros["ExactRepKLoVal"] = "6"
        macros["ExactRepKHiVal"] = "32"

    # ---- shape of the deficit: horizon or proportional? -------------------
    hs_path = Path("data/results/horizon_shape.json")
    if hs_path.exists():
        hs = json.loads(hs_path.read_text())
        p = hs.get("pooled", {})
        macros["ShapeSatSSE"] = fmt(p.get("saturating", {}).get("sse"), 1)
        macros["ShapePropSSE"] = fmt(p.get("proportional", {}).get("sse"), 1)
        macros["ShapePropSlope"] = fmt(p.get("proportional", {}).get("param"), 3)
        macros["ShapePropPct"] = fmt(100 * p.get("proportional", {}).get("param", np.nan), 0)
        macros["ShapeNStar"] = fmt(p.get("soft_horizon", {}).get("n_star"), 0)
        macros["ShapeKMax"] = str(hs.get("kmin", 6))
        macros["ShapePropModels"] = str(hs.get("n_proportional", 0))
        ns = [v.get("soft_horizon", {}).get("n_star", np.nan)
              for v in hs.get("models", {}).values()]
        ns = [x for x in ns if np.isfinite(x)]
        if ns:
            macros["ShapeNStarMin"] = fmt(min(ns), 0)
            macros["ShapeNStarMax"] = fmt(max(ns), 0)
    # ---- the extension ladder: where tracking stops, and what repetition
    # does to that point. The ratio is the reportable quantity: both families
    # saturate, because these decoders have a general utterance-length ceiling,
    # and only the ratio divides that ceiling out.
    he_path = Path("data/results/horizon_ext.json")
    if he_path.exists():
        he = json.loads(he_path.read_text())
        p_ = he.get("pooled", {})
        for tag, key in (("Rep", "rep"), ("Ctl", "ctl")):
            v = p_.get(key, {})
            macros[f"HzN{tag}"] = fmt(v.get("n_star"), 0)
            macros[f"HzN{tag}Lo"] = fmt(v.get("lo"), 0)
            macros[f"HzN{tag}Hi"] = fmt(v.get("hi"), 0)
        macros["HzRatio"] = fmt(p_.get("ratio"), 1)
        names = he.get("model_names", [])
        labs = [LABEL.get(m, m) for m in names]
        macros["HzModelNames"] = (", ".join(labs[:-1]) + " and " + labs[-1]
                                  if len(labs) > 1 else (labs[0] if labs else "--"))
        npc = he.get("n_per_cell", {}).get("rep", {})
        if npc:
            vals = [int(v) for v in npc.values()]
            macros["HzNCellLo"] = str(min(vals))
            macros["HzNCellHi"] = str(max(vals))
        ex = he.get("excluded", {})
        macros["HzExclCap"] = str(ex.get("cap_hits", 0))
        macros["HzExclDegen"] = str(ex.get("degenerate", 0))
        macros["HzNGen"] = str(ex.get("n_generated", 0))
        macros["HzNModels"] = str(he.get("n_models", 0))
        sep = sum(1 for v in he.get("models", {}).values()
                  if np.isfinite(v["rep"].get("hi", np.nan))
                  and np.isfinite(v["ctl"].get("lo", np.nan))
                  and v["rep"]["hi"] < v["ctl"]["lo"])
        macros["HzNSep"] = str(sep)
        # Which checkpoints separate, and which reverse. With four models the
        # panel is no longer unanimous, and naming the exception is more use to
        # a reader than a bare count.
        mods = he.get("models", {})
        sep_names = [LABEL.get(m, m) for m, v in mods.items()
                     if v["rep"]["hi"] < v["ctl"]["lo"]]
        rev_names = [LABEL.get(m, m) for m, v in mods.items()
                     if v.get("ratio", 1) < 1]
        macros["HzSepNames"] = " and ".join(sep_names) if sep_names else "none"
        macros["HzRevNames"] = " and ".join(rev_names) if rev_names else "none"
        macros["HzNRev"] = str(len(rev_names))
        dc = he.get("decline", {})
        if dc:
            macros["HzDeclinePeak"] = fmt(dc.get("c_peak"), 0)
            macros["HzDeclineLast"] = fmt(dc.get("c_last"), 0)
            macros["HzDeclineKPeak"] = str(dc.get("k_peak", ""))
            macros["HzDeclineKLast"] = str(dc.get("k_last", ""))
        mk = he.get("median_by_k", {})
        for tag, key in (("Rep", "rep"), ("Ctl", "ctl")):
            for kk in (48, 128):
                v = mk.get(key, {}).get(str(kk), mk.get(key, {}).get(kk))
                if v is not None:
                    macros[f"HzMed{tag}K{'Lo' if kk == 48 else 'Hi'}"] = fmt(v, 0)

    hse_path = Path("data/results/horizon_shape_ext.json")
    if hse_path.exists():
        hse = json.loads(hse_path.read_text())
        p = hse.get("pooled", {})
        macros["ExtWinner"] = str(p.get("winner", "--")).replace("_", " ")
        macros["ExtSatSSE"] = fmt(p.get("saturating", {}).get("sse"), 1)
        macros["ExtPropSSE"] = fmt(p.get("proportional", {}).get("sse"), 1)
        macros["ExtPropSlope"] = fmt(p.get("proportional", {}).get("param"), 3)
        macros["ExtNStar"] = fmt(p.get("soft_horizon", {}).get("n_star"), 0)
        macros["ExtRatios"] = ", ".join(
            f"{int(k)}: {v:.2f}" for k, v in sorted(
                (int(a), b) for a, b in hse.get("ratio_by_k", {}).items()))

    # ---- checkpoint-level inference --------------------------------------
    # Replaces the vote counts ("5 of 6") that round-5 reviewers objected to:
    # the checkpoint is the unit the paper's claim generalises over, so the
    # estimate and its interval are computed across checkpoints, not
    # generations. The exact signed-rank floor at n=6 is 0.031, and the macros
    # below are quoted with that stated so nobody reads more resolution into
    # them than six checkpoints can carry.
    ck_path = Path("data/results/checkpoint_level.json")
    if ck_path.exists():
        ck = json.loads(ck_path.read_text())
        for key, tag in (("count_error_gap", "CkErr"), ("exact_rate_gap", "CkExact"),
                         ("capacity_gap", "CkCap")):
            v = ck.get(key)
            if not v:
                continue
            scale = 100 if key != "capacity_gap" else 1
            nd = 1 if key != "capacity_gap" else 1
            macros[tag] = fmt(scale * v["mean"], nd)
            macros[tag + "Lo"] = fmt(scale * v["lo"], nd)
            macros[tag + "Hi"] = fmt(scale * v["hi"], nd)
            macros[tag + "P"] = fmt(v["wilcoxon_p"], 3)
            macros[tag + "NPos"] = str(v["n_positive"])
            macros[tag + "N"] = str(v["n"])
            macros[tag + "SD"] = fmt(scale * v["sd"], 1)

    # ---- does dilution explain item-level failure? (it does not) ----------
    ds_path = Path("data/results/dilution_sufficiency.json")
    if ds_path.exists():
        ds = json.loads(ds_path.read_text())
        rep = ds.get("measures", {}).get("word_rep", {})
        ctl = ds.get("measures", {}).get("control_word", {})
        v = rep.get("attn_share_max", {})
        macros["DilR"] = fmt(v.get("r"), 2)
        macros["DilRLo"] = fmt(v.get("lo"), 2)
        macros["DilRHi"] = fmt(v.get("hi"), 2)
        macros["DilCells"] = str(v.get("n_cells", 0))
        macros["DilItems"] = str(v.get("n_items", 0))
        macros["DilNAgainst"] = str(ds.get("n_against", 0))
        c = ctl.get("attn_share_max", {})
        macros["DilCtlR"] = fmt(c.get("r"), 2)
        macros["DilCtlLo"] = fmt(c.get("lo"), 2)
        macros["DilCtlHi"] = fmt(c.get("hi"), 2)

    # ---- family-level clustering, and genuinely aperiodic controls -------
    fl_path = Path("data/results/family_level.json")
    if fl_path.exists():
        fl = json.loads(fl_path.read_text())
        for key, tag in (("exact_rate_gap", "FamExact"), ("capacity_gap", "FamCap")):
            v = fl.get(key)
            if not v:
                continue
            scale = 100 if key != "capacity_gap" else 1
            macros[tag] = fmt(scale * v["mean"], 1)
            macros[tag + "Lo"] = fmt(scale * v["lo"], 1)
            macros[tag + "Hi"] = fmt(scale * v["hi"], 1)
            macros[tag + "N"] = str(v["n_families"])

    apc_path = Path("data/results/aperiodic_controls.json")
    if apc_path.exists():
        ac = json.loads(apc_path.read_text())
        # The re-generated k=12..32 controls, against the cycled ones at the same
        # k. A reviewer estimated the confound cost two thirds of the effect by
        # comparing k<=8 with k>8, which conflates it with the effect's own
        # k-dependence; run at matched k it costs about thirteen points.
        mid = next((v for k, v in ac.items()
                    if isinstance(v, dict) and "re-generated" in k), None)
        per = next((v for k, v in ac.items()
                    if isinstance(v, dict) and "period-8" in k), None)
        if mid:
            macros["ApMidGap"] = fmt(100 * mid["exact_gap"], 1)
            macros["ApMidRep"] = fmt(100 * mid["exact_rep"], 1)
            macros["ApMidCtl"] = fmt(100 * mid["exact_ctl"], 1)
            macros["ApMidN"] = str(mid["n_ctl"])
        if mid and per:
            macros["ApCycGap"] = fmt(100 * per["exact_gap"], 1)
            macros["ApCost"] = fmt(100 * (per["exact_gap"] - mid["exact_gap"]), 1)
        lo = next((v for k, v in ac.items()
                   if isinstance(v, dict) and "aperiodic" in k), None)
        hi = next((v for k, v in ac.items()
                   if isinstance(v, dict) and "146-word" in k), None)
        if lo:
            macros["ApLoGap"] = fmt(100 * lo["exact_gap"], 1)
            macros["ApLoRep"] = fmt(100 * lo["exact_rep"], 1)
            macros["ApLoCtl"] = fmt(100 * lo["exact_ctl"], 1)
        if hi:
            macros["ApHiGap"] = fmt(100 * hi["exact_gap"], 1)
            macros["ApHiMedRep"] = fmt(hi["median_rep"], 2)
            macros["ApHiMedCtl"] = fmt(hi["median_ctl"], 2)

    # ---- the non-autoregressive baseline ---------------------------------
    # The architectural control the paper's own title asks for, and -- because a
    # non-AR decoder renders the same repeated strings through the same judge --
    # the only control that can show the recogniser is not manufacturing the gap.
    nb_path = Path("data/results/nonar_baseline.json")
    if nb_path.exists():
        nb = json.loads(nb_path.read_text())
        for key, tag in (("nonar", "NonAR"), ("ar_pooled", "ARPool")):
            v = nb.get(key, {})
            macros[f"{tag}Rep"] = fmt(100 * v.get("exact_rep", np.nan), 1)
            macros[f"{tag}Ctl"] = fmt(100 * v.get("exact_ctl", np.nan), 1)
            macros[f"{tag}Gap"] = fmt(100 * v.get("exact_gap", np.nan), 1)
        macros["NonARAbove"] = str(nb.get("n_ar_above_nonar", 0))
        macros["NonARN"] = str(nb.get("n_ar", 0))

    # ---- is the deficit just the decoding-time repetition penalty? --------
    # A round-8 reviewer noted the three families ship penalties differing by an
    # order of magnitude, which is a decoding-level rival to the whole account.
    # The data already answered it; the paper had not said so.
    pc_path = Path("data/results/penalty_confound.json")
    if pc_path.exists():
        pc = json.loads(pc_path.read_text())
        x = pc.get("by_arch", {}).get("XTTS-v2", {})
        if x:
            macros["PenRepLo"] = fmt(100 * x["exact_rep_lo"], 1)
            macros["PenRepHi"] = fmt(100 * x["exact_rep_hi"], 1)
            macros["PenArms"] = str(x["n_usable"])
            macros["PenFold"] = fmt(x["fold_change"], 0)
            macros["PenSlope"] = fmt(100 * x["slope_per_unit"], 2)
        np_ = pc.get("no_penalty_summary", {})
        if np_:
            macros["PenNoneN"] = WORDS.get(np_["n_models"], str(np_["n_models"]))
            macros["PenNoneRep"] = fmt(100 * np_["exact_rep"], 1)
            macros["PenNoneCtl"] = fmt(100 * np_["exact_ctl"], 1)
            macros["PenNoneGap"] = fmt(100 * np_["exact_gap"], 1)
        nr = pc.get("arms", {}).get("xtts2norp", {})
        if nr:
            macros["PenOffCtl"] = fmt(100 * nr["exact_ctl"], 1)

    # ---- CTC judge on real generated audio (field validation) ------------
    fv_path = Path("data/results/ctc_field_validation.json")
    if fv_path.exists():
        fv = json.loads(fv_path.read_text())
        sec = fv.get("check4_second_recogniser", {}).get("overall", {})
        macros["HubertDiff"] = fmt(sec.get("mean_abs_diff"), 2)
        macros["HubertExact"] = fmt(100 * sec.get("exact_match_rate", np.nan), 1)

    # ---- CTC judge validation --------------------------------------------
    cv_path = Path("data/results/ctc_validation.json")
    if cv_path.exists():
        cv = json.loads(cv_path.read_text()).get("summary", {})
        macros["CtcValPer"] = fmt(cv.get("periodic_ctc_kge4"), 2)
        macros["CtcValDis"] = fmt(cv.get("distinct_ctc_kge4"), 2)
        macros["WhisperValPer"] = fmt(cv.get("periodic_whisper_kge4"), 2)
        macros["WhisperValDis"] = fmt(cv.get("distinct_whisper_kge4"), 2)

    # ---- ASR instrument audit ---------------------------------------------
    ar_path = Path("data/results/asr_reliability.json")
    if ar_path.exists():
        macros["AsrAuditPresent"] = "1"

    # ---- unit invariance --------------------------------------------------
    ui_path = Path("data/results/unit_invariance.json")
    if ui_path.exists():
        ui = json.loads(ui_path.read_text())
        if "mean_abs_k_diff" in ui:
            macros["UnitKDiff"] = fmt(ui["mean_abs_k_diff"], 1)
            macros["UnitTokRatio"] = fmt(ui["mean_tok_ratio"], 2)
            macros["UnitNModels"] = str(ui.get("n_models", 0))

    # ---- XTTS repetition-penalty ablation --------------------------------
    # XTTS-v2 ships repetition_penalty=5.0 on acoustic tokens, which acts
    # directly against the behaviour under study. If the dissociation survives
    # with it disabled, that decoding-time intervention is not what produces it.
    if len(beh) and {"xtts2", "xtts2norp"} <= set(beh.model):
        for key, tag in (("xtts2", "Rp"), ("xtts2norp", "NoRp")):
            b = beh[beh.model == key]
            rep = b[(b.family == "word_rep") & (b.k >= 6)]
            ctl = b[(b.family == "control_word") & (b.k >= 6)]
            if len(rep):
                macros[f"Abl{tag}Rep"] = fmt(100 * rep.correct.mean(), 1)
            if len(ctl):
                macros[f"Abl{tag}Ctl"] = fmt(100 * ctl.correct.mean(), 1)
            acc = b[b.family == "word_rep"].groupby("k")["correct"].mean()
            above = acc.index[acc > 0.5]
            macros[f"Abl{tag}KStar"] = str(int(above.max())) if len(above) else "0"

    # ---- mitigation sweep --------------------------------------------------
    ms_path = Path("data/results/mitigation_sweep.json")
    if ms_path.exists():
        ms = json.loads(ms_path.read_text())
        arms = ms.get("arms", {})
        usable = [a for a in arms.values()
                  if np.isfinite(a["ctl"]["median"]) and abs(a["ctl"]["median"]) < 0.05]
        if usable:
            lo_p = min(a["penalty"] for a in usable)
            hi_p = max(a["penalty"] for a in usable)
            errs = [a["rep"]["median"] for a in usable]
            macros["MitNArms"] = WORDS.get(len(usable), str(len(usable)))
            macros["MitPenLo"] = fmt(lo_p, 0)
            macros["MitPenHi"] = fmt(hi_p, 0)
            macros["MitErrBest"] = fmt(100 * max(errs), 1)
            macros["MitErrWorst"] = fmt(100 * min(errs), 1)
            macros["MitSpan"] = fmt(100 * (max(errs) - min(errs)), 1)
        if np.isfinite(ms.get("slope_per_penalty_unit", np.nan)):
            macros["MitSlope"] = fmt(100 * ms["slope_per_penalty_unit"], 1)
        broken = [a for a in arms.values()
                  if np.isfinite(a["ctl"]["median"]) and abs(a["ctl"]["median"]) >= 0.05]
        if broken:
            macros["MitBrokenPen"] = fmt(min(a["penalty"] for a in broken), 0)

    # ---- naturalness control -------------------------------------------
    # Two scorers: a panel backbone (Llasa-1B) and an independent LM outside the
    # panel. The paper quotes the independent one, because a referee drawn from
    # the tested models is not a referee.
    for path, pre in (("data/results/text_nll.json", "Nll"),
                      ("data/results/text_nll_independent.json", "NllInd")):
        p = Path(path)
        if not p.exists():
            continue
        nll = json.loads(p.read_text())
        if "diff_mean" not in nll:
            continue
        macros[pre + "Diff"] = fmt(nll["diff_mean"], 2)
        macros[pre + "CtlHigherPct"] = fmt(100 * nll["frac_ctl_higher"], 0)
        macros[pre + "Rep"] = fmt(nll["rep_mean"], 2)
        macros[pre + "Ctl"] = fmt(nll["ctl_mean"], 2)
        macros[pre + "Pairs"] = str(nll["n_pairs"])

    # ---- probe -----------------------------------------------------------
    probe_path = Path("data/results/probe.json")
    if probe_path.exists():
        pr = {k: v for k, v in
              json.loads(probe_path.read_text()).get("models", {}).items()
              if k not in ABLATIONS}
        rets_r, rets_c, late_r, n_below = [], [], [], 0
        for v in pr.values():
            r, c = v.get("word_rep"), v.get("control_word")
            if r and np.isfinite(r.get("retention", np.nan)):
                rets_r.append(r["retention"])
                late_r.append(r["late"]["best"]["r2"])
            if c and np.isfinite(c.get("retention", np.nan)):
                rets_c.append(c["retention"])
            if r and c and np.isfinite(r.get("retention", np.nan)) \
                    and np.isfinite(c.get("retention", np.nan)) \
                    and r["retention"] < c["retention"]:
                n_below += 1
        if rets_r:
            macros["ProbeRetRep"] = fmt(np.median(rets_r), 2)
            macros["ProbeLateRTwo"] = fmt(np.median(late_r), 2)
        if rets_c:
            macros["ProbeRetCtl"] = fmt(np.median(rets_c), 2)
        macros["ProbeNBelow"] = str(n_below)
        macros["ProbeNModels"] = str(len(pr))
        macros["ProbePhrase"] = (f"all {WORDS.get(len(pr), len(pr))}"
                                 if n_below == len(pr) and pr
                                 else f"{n_below} of {len(pr)}")


    # ---- what the probe does and does not discriminate --------------------
    pd_path = Path("data/results/probe_discrimination.json")
    if pd_path.exists():
        pdc = json.loads(pd_path.read_text())
        macros["ProbeLateMed"] = fmt(pdc["rep_late_r2_median"], 2)
        macros["ProbeLateLo"] = fmt(pdc["rep_late_r2_lo"], 2)
        macros["ProbeLateHi"] = fmt(pdc["rep_late_r2_hi"], 2)
        macros["ProbeAboveOne"] = str(pdc["n_retention_above_one"])
        macros["ProbeNBelowCtl"] = str(pdc["n_rep_below_ctl"])
        macros["ProbeNCk"] = str(pdc["n_models"])

    # ---- the dissociation, pooled over the PANEL (ablations excluded) -----
    if len(beh):
        panel_beh = beh[~beh.model.isin(ABLATIONS)]
        hi = panel_beh[panel_beh.k >= 6]
        prop_macros(macros, "AccRepHigh", hi[hi.family == "word_rep"])
        prop_macros(macros, "AccCtlHigh", hi[hi.family == "control_word"])
        prop_macros(macros, "AccRepLow",
                    panel_beh[(panel_beh.family == "word_rep") & (panel_beh.k <= 3)])
        loops = panel_beh[(panel_beh.family == "word_rep") & (panel_beh.k >= 8)]
        if len(loops):
            macros["LoopPct"] = fmt(100 * loops.outcome.isin(["loop", "overcount"]).mean(), 1)
            macros["TruncPct"] = fmt(
                100 * loops.outcome.isin(["truncation", "undercount"]).mean(), 1)
            macros["LoopN"] = str(len(loops))

    # ---- attention dilution (Lemma B) -----------------------------------
    # Measured on the softmax renormalised over the k repeated columns, which is
    # what the lemma actually bounds. Predictions: share ~ 1/k (log-log slope
    # -1), block entropy ~ log k (slope 1 in nats), deviation from uniform ~ 1/k.
    if len(state):
        base = state[(state.layer_frac > 0.6) & (state.k >= 2)]

        def loglog_slope(col: str) -> float:
            if col not in base.columns:
                return np.nan
            s = base[base[col].notna() & (base[col] > 0)]
            if len(s) < 8:
                return np.nan
            g = s.groupby("k")[col].median()
            if len(g) < 4:
                return np.nan
            return float(np.polyfit(np.log(g.index.to_numpy(float)),
                                    np.log(g.to_numpy()), 1)[0])

        def lin_slope(col: str) -> float:
            if col not in base.columns:
                return np.nan
            s = base[base[col].notna()]
            if len(s) < 8:
                return np.nan
            g = s.groupby("k")[col].median()
            if len(g) < 4:
                return np.nan
            return float(np.polyfit(np.log(g.index.to_numpy(float)),
                                    g.to_numpy(), 1)[0])

        macros["AttnShareSlope"] = fmt(loglog_slope("attn_share"), 2)
        macros["AttnDevSlope"] = fmt(loglog_slope("attn_unif_dev"), 2)
        macros["BlockEntropySlope"] = fmt(lin_slope("attn_block_entropy"), 2)
        macros["TextEntropySlope"] = fmt(lin_slope("attn_text_entropy"), 2)
        # The entropy gap and the most-attended share bound different halves of
        # the logit spread: log k - H constrains the average, while the largest
        # single share constrains the extreme. Reporting both under one symbol
        # made them look inconsistent (0.06 against 0.64 nats), so each is
        # emitted separately and the results text says which the lemma uses.
        if {"attn_block_entropy", "attn_share_max"} <= set(base.columns):
            g = base[(base.k >= 6) & base.attn_block_entropy.notna()]
            if len(g):
                macros["DeltaEntropy"] = fmt(
                    float((np.log(g.k.astype(float)) - g.attn_block_entropy).max()), 2)
            g2 = base[(base.k >= 6) & base.attn_share_max.notna()]
            if len(g2):
                # worst case, not typical: the lemma needs an upper bound
                macros["DeltaMax"] = fmt(
                    float(np.log((g2.attn_share_max * g2.k.astype(float)).max())), 2)
                macros["ShareMaxK"] = fmt(
                    float((g2.attn_share_max * g2.k.astype(float)).max()), 1)
                macros["ShareMedK"] = fmt(
                    float((g2.attn_share_max * g2.k.astype(float)).median()), 1)
                macros["DeltaMedian"] = fmt(
                    float(np.log((g2.attn_share_max * g2.k.astype(float)).median())), 2)
        if "attn_block_entropy" in base.columns:
            hi = base[(base.k >= 16) & base.attn_block_entropy.notna()]
            if len(hi):
                macros["BlockEntropyHi"] = fmt(hi.attn_block_entropy.median(), 2)
                macros["BlockEntropyMaxHi"] = fmt(
                    float(np.log(hi.k.median())), 2)

    # ---- write numbers.tex ----------------------------------------------
    lines = ["% AUTO-GENERATED by analysis/make_numbers.py -- do not edit.", ""]
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        raise SystemExit(
            f"macro names must be letters only (LaTeX restriction); offenders: {bad}")
    for k, v in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{k}}}{{{v}\xspace}}" if False
                     else rf"\newcommand{{\{k}}}{{{v}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'numbers.tex'} ({len(macros)} macros)")

    # ---- Table 1: per-model summary --------------------------------------
    capm = (json.loads(cap_path.read_text())["models"] if cap_path.exists() else {})
    ce_models = (json.loads(ce_path.read_text())["models"] if ce_path.exists() else {})
    rows = []
    for m in models:
        e = summ.get("models", {}).get(m, {})
        c = capm.get(m, {})
        b = beh[beh.model == m]
        rep = b[(b.family == "word_rep") & (b.k >= 6)]
        ctl = b[(b.family == "control_word") & (b.k >= 6)]
        ok = ((ctl.outcome.isin(["correct", "overcount", "undercount"])
               & (ctl.duration_ratio > 0.7) & (ctl.spectral_flatness < 0.35)).mean()
              if len(ctl) else np.nan)
        gr, gc = c.get("repeated", {}), c.get("control", {})
        ce_m = ce_models.get(m, {})
        rows.append((
            LABEL.get(m, m), PARAMS.get(m, "--"),
            fmt(100 * ce_m.get("rep", {}).get("median", np.nan), 1),
            fmt(100 * ce_m.get("ctl", {}).get("median", np.nan), 1),
            fmt(gr.get("gain"), 1), fmt(gc.get("gain"), 1),
            fmt(c.get("ratio"), 2),
        ))
    tbl = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & Par. & \multicolumn{2}{c}{count err.\ $k\!\ge\!6$ (\%)} "
        r"& \multicolumn{3}{c}{capacity gain $\mathrm{d}\mathcal{N}_{\mathrm{eff}}/\mathrm{d}\log k$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-7}",
        r" & & rep. & ctl. & rep. & ctl. & ratio \\", r"\midrule",
    ]
    for r in rows:
        tbl.append(" & ".join(r) + r" \\")
    tbl += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1.tex").write_text("\n".join(tbl) + "\n")
    print(f"wrote {out/'table1.tex'} ({len(rows)} models)")

    # ---- Table 2: spectral proxies, faithful vs hallucinated ---------------
    if len(state) and len(beh):
        lab = (beh[beh.seed == 0].set_index(["model", "item_id"])["outcome"].to_dict())
        st = state[state.layer_frac > 0.6].copy()
        st["outcome"] = [lab.get((r.model, r.item_id)) for r in st.itertuples()]
        st["halluc"] = ~st.outcome.isin(["correct"])
        t2 = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Model & \multicolumn{2}{c}{$\mathcal{N}_{\mathrm{eff}}$} "
              r"& \multicolumn{2}{c}{$\alpha$} & \multicolumn{2}{c}{$\log_{10} Kf$} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
              r" & faith. & halluc. & faith. & halluc. & faith. & halluc. \\", r"\midrule"]
        n2 = 0
        for m in models:
            s = st[(st.model == m) & st.n_eff.notna()]
            if s.empty:
                continue
            f_, h_ = s[~s.halluc], s[s.halluc]
            if f_.empty or h_.empty:
                continue
            n2 += 1
            t2.append(" & ".join([
                LABEL.get(m, m),
                fmt(f_.n_eff.median(), 1), fmt(h_.n_eff.median(), 1),
                fmt(f_.alpha.median(), 2), fmt(h_.alpha.median(), 2),
                fmt(f_.log_kf.median(), 2), fmt(h_.log_kf.median(), 2),
            ]) + r" \\")
        t2 += [r"\bottomrule", r"\end{tabular}"]
        if n2:
            (out / "table2.tex").write_text("\n".join(t2) + "\n")
            print(f"wrote {out/'table2.tex'} ({n2} models)")


if __name__ == "__main__":
    main()
