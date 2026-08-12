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
# The single definition, imported rather than copied: this file kept its own
# stale list of one entry while the panel grew four ablation arms and two non-AR
# baselines, so any macro filtering on it was quietly including them.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.common.population import ABLATIONS  # noqa: E402
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

    # ---- main-ladder exclusion accounting ---------------------------------
    # The extension ladder reports its exclusions in the main text and the main
    # ladder did not, which a reviewer read as selective accounting. Same
    # numbers, same place.
    beh_all = Path("data/results/behavioural_ctc.csv")
    if beh_all.exists():
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.common.population import panel as _panel
        _raw = pd.read_csv(beh_all)
        _raw = _raw[_raw.family.isin(["word_rep", "control_word"])]
        _kept, _drop = _panel(_raw, degenerate=True)
        _final, _ = _panel(_raw)
        macros["MainNGen"] = str(_drop["n_input"] - _drop.get("ablations", 0))
        macros["MainExclTmpl"] = str(_drop.get("bad_templates", 0))
        macros["MainExclCap"] = str(_drop.get("cap_hits", 0))
        macros["MainExclDegen"] = str(len(_kept) - len(_final))
        macros["MainKept"] = str(len(_final))

    # ---- do the exclusions fall evenly on the two arms? -------------------
    # A reviewer noted that an exclusion concentrated on one arm could
    # manufacture the contrast it is used to support. The template rule is
    # balanced by construction; the budget rule is not, and its asymmetry runs
    # against the effect, which is worth saying rather than leaving to be found.
    if beh_all.exists():
        from src.common.population import excluded_templates, cap_flags
        _b = pd.read_csv(beh_all)
        _b = _b[_b.family.isin(["word_rep", "control_word"])
                & ~_b.model.isin(ABLATIONS)]
        _bad, _flags = excluded_templates(), cap_flags()
        _b = _b.assign(cap=[_flags.get((r.model, r.item_id, r.seed), False)
                            for r in _b.itertuples()])
        for fam, tag in (("word_rep", "Rep"), ("control_word", "Ctl")):
            g = _b[_b.family == fam]
            macros[f"ExclTmpl{tag}"] = fmt(100 * g.template.isin(_bad).mean(), 1)
            macros[f"ExclCap{tag}"] = fmt(100 * g.cap.mean(), 1)

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
        # The reversing checkpoint's exact-rate gap. It is the one checkpoint
        # with no median-error deficit, which a reader can easily misread as
        # "no deficit at all" -- it has one, and it is large. Derived from the
        # same reversal set as HzRevNames so the two cannot drift apart.
        rev_keys = [m for m, v in mods.items() if v.get("ratio", 1) < 1]
        ck_early = json.loads(Path("data/results/checkpoint_level.json").read_text()) \
            if Path("data/results/checkpoint_level.json").exists() else {}
        per = ck_early.get("exact_rate_gap", {}).get("per_model", {})
        if len(rev_keys) == 1 and rev_keys[0] in per:
            macros["CkRevExact"] = fmt(100 * per[rev_keys[0]], 1)
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
        # The two baselines disagree, so each is reported by name; pooling them
        # would hide the only thing this experiment established.
        for m, tag in (("vits", "Vits"), ("f5tts", "Fivetts")):
            v = nb.get("nonar_per_model", {}).get(m, {})
            if v:
                macros[f"{tag}Rep"] = fmt(100 * v["exact_rep"], 1)
                macros[f"{tag}Ctl"] = fmt(100 * v["exact_ctl"], 1)
                macros[f"{tag}Gap"] = fmt(100 * v["exact_gap"], 1)
        macros["NonARAbove"] = str(nb.get("n_ar_above_nonar", 0))
        # The mid band is quoted for the baseline that shows NO dissociation,
        # where the floor-effect question actually arises.
        mid = nb.get("bands", {}).get("6-8", {})
        nulls = nb.get("baselines_without_dissociation", [])
        if mid and nulls and mid.get(nulls[0]):
            m0 = mid[nulls[0]]
            macros["NonARMidRep"] = fmt(100 * m0["exact_rep"], 1)
            macros["NonARMidCtl"] = fmt(100 * m0["exact_ctl"], 1)
        if mid.get("ar"):
            macros["ARMidGap"] = fmt(100 * mid["ar"]["gap"], 0)
        macros["NonARNull"] = ", ".join(nulls) if nulls else "none"
        macros["NonARShows"] = ", ".join(nb.get("baselines_with_dissociation", []))
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
        q = pc.get("by_arch", {}).get("Qwen3-TTS-0.6B", {})
        if q:
            macros["PenQRepLo"] = fmt(100 * q["exact_rep_lo"], 1)
            macros["PenQRepHi"] = fmt(100 * q["exact_rep_hi"], 1)
            macros["PenQFold"] = fmt(q["fold_change"], 0)
        # The cleanest arm in the whole argument: a model that ships a penalty,
        # run with it disabled, still shows the gap at full size.
        z = pc.get("arms", {}).get("qwen06brp10", {})
        if z:
            macros["PenZeroRep"] = fmt(100 * z["exact_rep"], 1)
            macros["PenZeroCtl"] = fmt(100 * z["exact_ctl"], 1)
        macros["PenNArch"] = str(len(pc.get("by_arch", {})))
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


    # ---- the judge's noise floor, in the units it errs in ------------------
    nf_path = Path("data/results/noise_floor.json")
    if nf_path.exists():
        nf = json.loads(nf_path.read_text())
        f, e = nf["floor"], nf["effect"]
        macros["FloorTwoRec"] = fmt(f["two_recognisers_mean_abs_diff"], 2)
        macros["FloorTwoRecHi"] = fmt(f["two_recognisers_mean_abs_diff_high_k"], 2)
        macros["FloorNoise"] = fmt(f["noise_mean_abs_delta"], 2)
        macros["FloorOursHigher"] = str(f["ours_higher"])
        macros["FloorNDisagree"] = str(f["n_disagreements"])
        macros["MissMedian"] = fmt(e["median_missing"], 1)
        macros["MissMean"] = fmt(e["mean_missing"], 2)
        macros["EffectOverFloor"] = fmt(nf["effect_over_floor"], 1)

    # ---- the duration intervention (the paper's only intervention) --------
    di_path = Path("data/results/duration_intervention.json")
    if di_path.exists():
        di = json.loads(di_path.read_text())
        macros["DurSplit"] = str(di["split"])
        macros["DurLoFree"] = fmt(100 * di["low_k"]["free"]["exact"], 1)
        macros["DurLoFixed"] = fmt(100 * di["low_k"]["fixed"]["exact"], 1)
        macros["DurLoN"] = str(di["low_k"]["free"]["n"])
        macros["DurHiFree"] = fmt(100 * di["high_k"]["free"]["exact"], 1)
        macros["DurHiFixed"] = fmt(100 * di["high_k"]["fixed"]["exact"], 1)
        macros["DurHiN"] = str(di["high_k"]["free"]["n"])
        macros["DurLoGain"] = fmt(100 * di["low_k_gain"], 1)
        macros["DurHiGain"] = fmt(100 * di["high_k_gain"], 1)

    # ---- the released audio sample ---------------------------------------
    smp = Path("data/audio_sample/manifest.csv")
    if smp.exists():
        macros["SampleN"] = str(len(pd.read_csv(smp)))

    # ---- is the horizon exception arbitrary, or the weakest checkpoint? ----
    # A reviewer asked for a substantive account of the one checkpoint that
    # shows no repeated-side saturation, rather than "unexplained exception".
    # It turns out to rank last on every panel measure we have, which is an
    # account: the exception is where the effect is smallest, not arbitrary.
    hf_early = Path("data/results/horizon_forms.json")
    if ck_path.exists() and hf_early.exists():
        ck2 = json.loads(ck_path.read_text())
        rev = [m for m, v in json.loads(hf_early.read_text()).get("models", {}).items()
               if v.get("soft_horizon", {}).get("ratio", 1) < 1]
        keys = ["exact_rate_gap", "count_error_gap", "capacity_gap"]
        if len(rev) == 1:
            last = [k for k in keys
                    if min(ck2[k]["per_model"], key=ck2[k]["per_model"].get) == rev[0]]
            macros["RevWeakestN"] = str(len(last))
            macros["RevWeakestOf"] = str(len(keys))
            macros["RevWeakestAll"] = ("all" if len(last) == len(keys)
                                       else f"{len(last)} of {len(keys)}")

    # ---- does the deficit survive greedy decoding? ------------------------
    gd_path = Path("data/results/greedy_decoding.json")
    if gd_path.exists():
        gd = json.loads(gd_path.read_text())
        macros["GreedyGap"] = fmt(100 * gd["gap_greedy"], 1)
        macros["GreedySampGap"] = fmt(100 * gd["gap_sampled_pooled"], 1)
        macros["GreedyRep"] = fmt(100 * gd["arms"]["greedy"]["rep"]["exact"], 1)
        macros["GreedyCtl"] = fmt(100 * gd["arms"]["greedy"]["ctl"]["exact"], 1)
        macros["GreedyN"] = str(gd["arms"]["greedy"]["n"])

    # ---- is VITS immune, or is its stock configuration immune? ------------
    vc_path = Path("data/results/vits_config.json")
    if vc_path.exists():
        vc = json.loads(vc_path.read_text())
        macros["VitsCfgN"] = str(vc["n_arms"])
        macros["VitsCfgWord"] = WORDS.get(vc["n_arms"], str(vc["n_arms"]))
        macros["VitsCfgStock"] = fmt(100 * vc["arms"]["vits"]["gap"], 1)
        macros["VitsCfgWorst"] = fmt(
            100 * max(v["gap"] for v in vc["arms"].values()), 1)

    # ---- do our own exclusions manufacture the gap? -----------------------
    es_path = Path("data/results/exclusion_sensitivity.json")
    if es_path.exists():
        es = json.loads(es_path.read_text())
        macros["ExSensPanel"] = fmt(100 * es["gap_panel"], 1)
        macros["ExSensAll"] = fmt(100 * es["gap_everything"], 1)
        macros["ExSensMin"] = fmt(100 * es["gap_min"], 1)
        macros["ExSensN"] = str(es["arms"]["everything"]["n"])

    # ---- is the horizon ratio a property of the data or of the curve? -----
    hf_path = Path("data/results/horizon_forms.json")
    if hf_path.exists():
        hf = json.loads(hf_path.read_text())
        macros["HzFormLo"] = fmt(hf["pooled_ratio_min"], 1)
        macros["HzFormHi"] = fmt(hf["pooled_ratio_max"], 1)
        macros["HzFormN"] = str(len(hf["forms"]))
        macros["HzFormNCk"] = str(hf["n_models"])
        macros["HzFormStable"] = ("does not" if hf["direction_stable"]
                                  else "does")

    # ---- the probe past the horizon: the theorem's own prediction ---------
    ph_path = Path("data/results/probe_horizon_compare.json")
    if ph_path.exists():
        phj = json.loads(ph_path.read_text())
        ph = phj["rows"]
        # The checkpoint that loses the count past the horizon, and the one that
        # does not. Both are named from the result rather than fixed here, so a
        # rerun that flipped them could not leave the paper quoting the wrong
        # one as the confirmatory case.
        lost = phj.get("lost_past_horizon") or []
        kept = phj.get("kept_past_horizon") or []
        macros["PhNLost"] = str(len(lost))
        macros["PhNCk"] = str(phj.get("n_checkpoints", 0))
        def _join(ms):
            names = [LABEL.get(m, m) for m in ms]
            return (names[0] if len(names) == 1
                    else " and ".join([", ".join(names[:-1]), names[-1]]))
        if lost:
            m = lost[0]
            # With more than one checkpoint on a side the sentence must name them
            # all; picking the first would quietly drop a result.
            macros["PhLostName"] = _join(lost)
            macros["PhPastMae"] = fmt(ph[f"past:{m}"]["mae"], 2)
            macros["PhPastConst"] = fmt(ph[f"past:{m}"]["const_mae"], 2)
            macros["PhPastRTwo"] = fmt(ph[f"past:{m}"]["r2"], 2)
            macros["PhPastN"] = str(ph[f"past:{m}"]["n"])
            if f"below:{m}" in ph:
                macros["PhBelowMae"] = fmt(ph[f"below:{m}"]["mae"], 2)
                macros["PhBelowConst"] = fmt(ph[f"below:{m}"]["const_mae"], 2)
                macros["PhBelowRTwo"] = fmt(ph[f"below:{m}"]["r2"], 2)
        if kept:
            m = kept[0]
            macros["PhKeptName"] = _join(kept)
            # The checkpoint whose margin actually matches the control's. Named
            # separately from PhKeptName, which lists everything that kept the
            # count -- including one that only ties the trivial predictor.
            clear = phj.get("clearly_kept") or []
            if clear:
                macros["PhClearName"] = _join(clear)
            macros["PhKeptMae"] = fmt(ph[f"past:{m}"]["mae"], 2)
            macros["PhKeptRTwo"] = fmt(ph[f"past:{m}"]["r2"], 2)
            # Ratios against the constant predictor, listed for every checkpoint
            # that kept the count. A single "kept" number would hide that one of
            # them beats the trivial baseline by less than one percent.
            macros["PhKeptRatios"] = " and ".join(
                fmt(ph[f"past:{k}"]["mae_ratio"], 2) for k in kept)
        # Every checkpoint's ratio against the constant predictor, in one
        # macro, so the paper can report them symmetrically. Detailing the
        # checkpoint that confirms and summarising the ones that do not is a
        # presentation asymmetry a reviewer caught, and separate macros invite
        # it back.
        order = [m for m in ORDER if f"past:{m}" in ph]
        macros["PhAllRatios"] = ", ".join(
            fmt(ph[f"past:{m}"]["mae_ratio"], 2) for m in order)
        macros["PhAllNames"] = ", ".join(LABEL.get(m, m) for m in order)
        if lost:
            macros["PhLostRatio"] = fmt(ph[f"past:{lost[0]}"]["mae_ratio"], 2)
        tied = phj.get("indistinguishable") or []
        if tied:
            macros["PhTieName"] = _join(tied)
            macros["PhTieRatio"] = fmt(ph[f"past:{tied[0]}"]["mae_ratio"], 2)
            macros["PhTieBand"] = fmt(100 * phj.get("tie_band", 0), 0)
            macros["PhNNull"] = str(len(set(lost) | set(tied)))
        # Control items over the identical k range. If the probe reads the count
        # off these while failing on the repeated ones, "the range is too narrow"
        # is dead as an explanation -- the range is the same.
        ctl_order = [m for m in ORDER if f"control:{m}" in ph]
        ctl_key = f"control:{ctl_order[0]}" if ctl_order else None
        if ctl_key:
            macros["PhCtlMae"] = fmt(ph[ctl_key]["mae"], 2)
            macros["PhCtlRTwo"] = fmt(ph[ctl_key]["r2"], 2)
            macros["PhCtlRatio"] = fmt(ph[ctl_key]["mae_ratio"], 2)
            # One control arm excuses the range on one checkpoint. Listing them
            # all is what lets the rebuttal be stated for the panel rather than
            # for whichever checkpoint happened to have a control arm first.
            macros["PhNCtl"] = str(len(ctl_order))
            macros["PhCtlNames"] = ", ".join(LABEL.get(m, m) for m in ctl_order)
            macros["PhCtlRatios"] = ", ".join(
                fmt(ph[f"control:{m}"]["mae_ratio"], 2) for m in ctl_order)
            worst = max(ctl_order, key=lambda m: ph[f"control:{m}"]["mae_ratio"])
            macros["PhCtlRatioWorst"] = fmt(ph[f"control:{worst}"]["mae_ratio"], 2)
            macros["PhCtlWorstName"] = LABEL.get(worst, worst)
            macros["PhNCtlBeat"] = str(len(phj.get("controls_beating_constant")
                                           or []))

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

    # ---- hierarchical inference -------------------------------------------
    # Six checkpoints are three architecture families, and every earlier draft
    # met that by apologising for it ("too small for significance testing").
    # Seven of nine reviewers across three rounds asked for the model instead of
    # the apology, so analysis/hierarchical.py fits it: a mixed-effects linear
    # probability model with family as a random effect, a hierarchical Bayesian
    # partial pooling whose posterior predictive answers "what about a fourth
    # architecture", and a specification curve over every forking choice we made.
    hier_path = Path("data/results/hierarchical.json")
    if hier_path.exists():
        h = json.loads(hier_path.read_text())
        mx, by, sc = h.get("mixed", {}), h.get("bayes", {}), h.get("spec_curve", {})
        if mx:
            macros["MixArm"] = fmt(mx["mixedlm"]["arm_effect_pts"], 1)
            # The widest of the six intervals, not the narrowest. The MixedLM
            # Wald interval is the tightest and its variance component is shrunk
            # well below the observed between-family spread, so quoting it would
            # be quoting the least defensible number in the set.
            wide = mx["cluster_checkpoint"]
            macros["MixLo"] = fmt(wide["lo"], 1)
            macros["MixHi"] = fmt(wide["hi"], 1)
            macros["MixWorst"] = fmt(mx["worst_case_lower_bound_pts"], 1)
            macros["MixAllClear"] = ("all" if mx.get("all_intervals_exclude_zero")
                                     else "not all")
        if by:
            nf = by["new_family_gap_pts"]
            macros["BayesNewFam"] = fmt(nf["mean"], 1)
            macros["BayesNewFamLo"] = fmt(nf["lo"], 1)
            macros["BayesNewFamHi"] = fmt(nf["hi"], 1)
            macros["BayesNewFamP"] = fmt(nf["p_gt0"], 3)
            # The prior under which that interval does touch zero. Reporting the
            # favourable prior alone would be exactly the kind of forking choice
            # the specification curve exists to expose.
            ps = by.get("prior_sensitivity", {}).get("HalfNormal(3.0)", {})
            if ps:
                macros["BayesWidePriorLo"] = fmt(ps["new_family_gap_pts"]["lo"], 1)
                macros["BayesWidePriorP"] = fmt(ps["new_family_gap_pts"]["p_gt0"], 3)
        if sc:
            # The headline gap is computed over k>=6, and a reviewer correctly
            # objected that conditioning on the regime where the effect is
            # largest, without saying so, inflates what the abstract appears to
            # claim. The unconditioned full-ladder gap comes from the same
            # specification grid, at otherwise identical settings.
            for s in sc.get("specs", []):
                if (s.get("kmin") == 1 and s.get("outcome") == "exact_rate"
                        and s.get("exclusions") == "panel"
                        and s.get("control") == "cycled"
                        and s.get("seeds") == "all"
                        and s.get("aggregation") == "family"):
                    macros["FullLadderGap"] = fmt(s["gap"], 1)
                    macros["FullLadderN"] = str(s["n"])
                    break
            er, allsp = sc["exact_rate"], sc["all_specifications"]
            macros["SpecN"] = str(allsp["n"])
            macros["SpecLo"] = fmt(er["min"], 1)
            macros["SpecHi"] = fmt(er["max"], 1)
            macros["SpecNRev"] = str(allsp["n_negative"])

    # ---- the judge, re-run ------------------------------------------------
    # Eight of nine reviewers in r15-r17 and two more in r18 made the same
    # objection: the CTC judge was validated on spliced concatenative audio and
    # never on the real generated failure audio, and CTC blank-collapse -- which
    # merges adjacent identical words -- is precisely the confound under study.
    # analysis/independent_judge.py re-scores the same audio with three further
    # recognisers, including one (Whisper) with no blank-collapse mechanism.
    ij_path = Path("data/results/independent_judge.json")
    if ij_path.exists():
        ij = json.loads(ij_path.read_text())
        alljs = ij.get("all_judges", {})
        if alljs:
            gaps = [v["mean"] for v in alljs.values()]
            macros["IndNJudges"] = WORDS.get(len(alljs), str(len(alljs)))
            macros["IndGapLo"] = fmt(100 * min(gaps), 1)
            macros["IndGapHi"] = fmt(100 * max(gaps), 1)
            macros["IndNCkPos"] = str(min(v["n_positive"] for v in alljs.values()))
            macros["IndNCk"] = str(max(v["n_checkpoints"] for v in alljs.values()))
        sp = ij.get("arm_spread_across_judges", {})
        if sp:
            # The decisive number. Swapping the scorer moves the arm the models
            # get RIGHT and leaves the arm they get WRONG almost fixed; a
            # judge-side collapse of repeated material predicts the opposite.
            macros["IndRepSpread"] = fmt(100 * sp["repeated_spread"], 1)
            macros["IndCtlSpread"] = fmt(100 * sp["control_spread"], 1)
            macros["IndRepLo"] = fmt(100 * sp["repeated_exact_min"], 1)
            macros["IndRepHi"] = fmt(100 * sp["repeated_exact_max"], 1)
        vd = ij.get("verdict", {})
        if vd:
            macros["IndAsym"] = fmt(vd["A"], 2)
            macros["IndRetained"] = fmt(100 * vd["gap_retained_fraction"], 0)
        th = ij.get("thresholds", {})
        if th:
            macros["IndAsymThresh"] = fmt(th["artifact_asymmetry"], 1)

    # ---- CosyVoice 2: a fourth family, and a held-out test of the model ----
    # Reviewers called an alignment-supervised AR system "the single most
    # informative missing experiment". It is also, by accident of timing, an
    # out-of-sample check on the hierarchical fit: the posterior predictive for
    # an architecture outside the panel was committed (6f34945, 17:28) ninety
    # minutes before this checkpoint's data existed (18:58). Nothing here is
    # pooled into the panel -- see the cosyvoice2 entry in population.py.
    cosy_path = Path("data/results/behavioural_cosyvoice.csv")
    if cosy_path.exists():
        cz = pd.read_csv(cosy_path)
        cz = cz[cz.family.isin(["word_rep", "control_word"])]
        cz_pop, _ = restrict(cz.assign(model=cz.model.replace("cosyvoice2", "_cosy")))
        cz_pop = cz_pop[cz_pop.k >= 6]
        cz_pop = cz_pop.assign(err=(cz_pop.count_a - cz_pop.k) / cz_pop.k)
        rep = cz_pop[cz_pop.family == "word_rep"]
        ctl = cz_pop[cz_pop.family == "control_word"]
        if len(rep) and len(ctl):
            r_ex = 100 * (rep.err == 0).mean()
            c_ex = 100 * (ctl.err == 0).mean()
            macros["CosyRep"] = fmt(r_ex, 1)
            macros["CosyCtl"] = fmt(c_ex, 1)
            macros["CosyGap"] = fmt(c_ex - r_ex, 1)
            macros["CosyN"] = str(len(rep))
            # The direction is the new information: it over-produces rather
            # than truncating, so it makes roughly the right amount of speech
            # and still loses the count.
            macros["CosyMedErr"] = fmt(100 * rep.err.median(), 1)
            macros["CosyCapHit"] = fmt(100 * rep.hit_cap.mean(), 1)

    # ---- the contraction premise, finally measured -------------------------
    # Two earlier estimators failed and the paper had to say the premise was
    # untested. This one passes its own gates -- a synthetic Jacobian recovered
    # to 4e-4, a time-reversed control returning exactly zero, exact JVP against
    # finite differences to 2e-5 -- so the paper can say something much stronger
    # and much worse for the theorem: the premise is false in this decoder.
    jq_path = Path("data/results/jacobian_q.json")
    if jq_path.exists():
        jq = json.loads(jq_path.read_text())
        summ_j = jq.get("summary", {})
        hl = summ_j.get("headline", {})
        cell = summ_j.get("by_cell", {}).get("tau|substack0", {})
        if hl and cell:
            macros["JacQRep"] = fmt(hl["q_repeated"], 1)
            macros["JacQCtl"] = fmt(hl["q_control"], 1)
            macros["JacQRepLo"] = fmt(cell["repeated"]["ci_lo"], 1)
            macros["JacQRepHi"] = fmt(cell["repeated"]["ci_hi"], 1)
            macros["JacQN"] = str(cell["repeated"]["n_items"])
            # Zero of 29, in every cell. The paper quotes the count, not the
            # fraction: "q<1 in none of them" is the sentence that lands.
            macros["JacQNBelow"] = str(int(round(
                cell["repeated"]["frac_below_1"] * cell["repeated"]["n_items"])))
            p = hl["paired"]
            macros["JacQPairedP"] = fmt(p["wilcoxon_p"], 3)
            macros["JacQNRepLower"] = str(int(round(
                p["frac_repeated_lower"] * p["n_pairs"])))
        lf = summ_j.get("least_favourable", {})
        if lf:
            macros["JacQWorst"] = fmt(lf["q_repeated_ci_hi"], 1)
        band = summ_j.get("q_band_reproducing_observed_nstar", {})
        if band:
            # What q would have had to be for the theorem to explain the
            # saturation we actually observe. Nothing measured is near it.
            macros["JacQNeeded"] = fmt(band["q_for_nstar"]["point"], 2)

    # ---- the causal intervention, second pass -----------------------------
    # The first pass spliced whole donor states and was too disruptive to read:
    # an unrelated donor moved the count 1.34x as much as one differing in k. A
    # rank-1 patch that transfers only the probe's count coordinate removes the
    # damage entirely, which turns an uninterpretable result into a readable
    # null -- a stronger statement, and the one the paper now makes.
    cf_path = Path("data/results/causal_count_followup.json")
    if cf_path.exists():
        cf = json.loads(cf_path.read_text())
        rk = cf.get("rank1_patch", {})
        cell = rk.get("cells", {}).get("crossk|L7|P128", {})
        if cell:
            macros["CausalShift"] = fmt(cell["median"], 2)
            macros["CausalShiftLo"] = fmt(cell["ci_lo"], 2)
            macros["CausalShiftHi"] = fmt(cell["ci_hi"], 2)
            macros["CausalN"] = str(cell["n"])
        if rk:
            macros["CausalDegen"] = fmt(rk["patched_degenerate_pct"], 1)
            macros["CausalNoop"] = str(rk["n_noop"])

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

    # Per-checkpoint exact-match rates. Three review rounds charged asymmetric
    # reporting because the paper's most-quoted pair of numbers -- 18.2% against
    # 94.3% -- was the one quantity Table 1 broke out for nobody, so it could not
    # be audited per checkpoint the way every other number here can. Same
    # population as the pooled macros above: panel(), k>=6, exact means err==0.
    exact_by_model: dict[str, tuple[float, float]] = {}
    if beh_path.exists():
        _t = pd.read_csv(beh_path)
        _t = _t[_t.family.isin(["word_rep", "control_word"])]
        _t, _ = restrict(_t)
        _t = _t[_t.k >= 6]
        _t = _t.assign(err=(_t.count_a - _t.k) / _t.k)
        for m in models:
            g = _t[_t.model == m]
            r = g[g.family == "word_rep"]
            c = g[g.family == "control_word"]
            exact_by_model[m] = (
                100 * (r.err == 0).mean() if len(r) else np.nan,
                100 * (c.err == 0).mean() if len(c) else np.nan,
            )
        # The table must agree with the pooled macros it sits beside. A
        # per-model column that does not reconcile with the headline is worse
        # than no column at all, so check it here rather than by eye.
        for fam, tag in (("word_rep", "Rep"), ("control_word", "Ctl")):
            g = _t[_t.family == fam]
            if len(g):
                assert fmt(100 * (g.err == 0).mean(), 1) == macros[f"Exact{tag}"], (
                    f"Table 1 exact-rate population disagrees with \\Exact{tag}")

    rows = []
    for m in models:
        c = capm.get(m, {})
        ce_m = ce_models.get(m, {})
        er, ec = exact_by_model.get(m, (np.nan, np.nan))
        rows.append((
            LABEL.get(m, m), PARAMS.get(m, "--"),
            fmt(100 * ce_m.get("rep", {}).get("median", np.nan), 1),
            fmt(100 * ce_m.get("ctl", {}).get("median", np.nan), 1),
            fmt(er, 1), fmt(ec, 1),
            fmt(c.get("ratio"), 2),
        ))
    tbl = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & Par. & \multicolumn{2}{c}{count err.\ (\%)} "
        r"& \multicolumn{2}{c}{exactly right (\%)} & cap. \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
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
