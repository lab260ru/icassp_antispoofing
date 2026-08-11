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
    ap.add_argument("--behavioural", default="data/results/behavioural.csv")
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

    # ---- panel-level aggregates ----------------------------------------
    if summ.get("models"):
        qr = [e.get("q_rep", {}).get("q") for e in summ["models"].values()]
        qc = [e.get("q_ctl", {}).get("q") for e in summ["models"].values()]
        qr = [v for v in qr if v is not None and np.isfinite(v)]
        qc = [v for v in qc if v is not None and np.isfinite(v)]
        if qr:
            macros["qRepMin"], macros["qRepMax"] = fmt(min(qr), 3), fmt(max(qr), 3)
            macros["NContracting"] = str(sum(1 for v in qr if v < 1.0))
            macros["NContractingWord"] = WORDS.get(sum(1 for v in qr if v < 1.0),
                                                   str(sum(1 for v in qr if v < 1.0)))
        if qr and qc:
            macros["NqRepBelowCtl"] = str(sum(
                1 for e in summ["models"].values()
                if np.isfinite(e.get("q_rep", {}).get("q", np.nan))
                and np.isfinite(e.get("q_ctl", {}).get("q", np.nan))
                and e["q_rep"]["q"] < e["q_ctl"]["q"]))
        ks = [e.get("k_star") for e in summ["models"].values()]
        ks = [v for v in ks if v is not None and np.isfinite(v)]
        if ks:
            macros["kStarMin"], macros["kStarMax"] = fmt(min(ks), 0), fmt(max(ks), 0)

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

    # LaTeX macro names may not contain digits, so "r2" must be spelled out.
    p2 = summ.get("p2", {})
    for k, name, nd in (("r2", "RTwo", 2), ("spearman", "Spearman", 2),
                        ("beta", "Beta", 2), ("alpha", "Alpha", 2)):
        if k in p2:
            macros["PTwo" + name] = fmt(p2[k], nd)
    if "n" in p2:
        macros["PTwoN"] = str(p2["n"])

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
