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

    # ---- capacity: the central state measurement -------------------------
    cap_path = Path("data/results/capacity.json")
    if cap_path.exists():
        cap = json.loads(cap_path.read_text())
        macros["CapRatio"] = fmt(cap.get("ratio_median"), 2)
        macros["CapRatioPct"] = fmt(100 * cap.get("ratio_median", np.nan), 0)
        macros["NCapSep"] = str(cap.get("n_separated", 0))
        macros["NCapModels"] = str(cap.get("n_models", 0))
        macros["NCapSepWord"] = WORDS.get(cap.get("n_separated", 0),
                                          str(cap.get("n_separated", 0)))
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

    p2 = summ.get("p2", {})
    for k, nd in (("r2", 2), ("spearman", 2), ("beta", 2), ("alpha", 2)):
        if k in p2:
            macros["PTwo" + k.capitalize()] = fmt(p2[k], nd)
    if "n" in p2:
        macros["PTwoN"] = str(p2["n"])

    # ---- the dissociation, pooled over the panel ------------------------
    if len(beh):
        hi = beh[beh.k >= 6]
        rep = hi[hi.family == "word_rep"]
        ctl = hi[hi.family == "control_word"]
        if len(rep):
            macros["AccRepHigh"] = fmt(100 * rep.correct.mean(), 1)
        if len(ctl):
            ok = (ctl.outcome.isin(["correct", "overcount", "undercount"])
                  & (ctl.duration_ratio > 0.7) & (ctl.spectral_flatness < 0.35))
            macros["AccCtlHigh"] = fmt(100 * ok.mean(), 1)
        lo = beh[(beh.family == "word_rep") & (beh.k <= 3)]
        if len(lo):
            macros["AccRepLow"] = fmt(100 * lo.correct.mean(), 1)
        loops = beh[(beh.family == "word_rep") & (beh.k >= 8)]
        if len(loops):
            macros["LoopPct"] = fmt(100 * loops.outcome.isin(["loop", "overcount"]).mean(), 1)
            macros["TruncPct"] = fmt(
                100 * loops.outcome.isin(["truncation", "undercount"]).mean(), 1)

    # ---- attention dilution slopes --------------------------------------
    if len(state) and "attn_per_occurrence" in state.columns:
        s = state[(state.layer_frac > 0.6) & (state.k >= 2)
                  & state.attn_per_occurrence.notna() & (state.attn_per_occurrence > 0)]
        if len(s) > 6:
            g = s.groupby("k")["attn_per_occurrence"].median()
            if len(g) > 3:
                sl = np.polyfit(np.log(g.index.to_numpy(float)), np.log(g.to_numpy()), 1)[0]
                macros["AttnSlope"] = fmt(sl, 2)
        s2 = state[(state.layer_frac > 0.6) & (state.k >= 2) & state.attn_entropy.notna()]
        if len(s2) > 6:
            g2 = s2.groupby("k")["attn_entropy"].median()
            if len(g2) > 3:
                sl2 = np.polyfit(np.log(g2.index.to_numpy(float)), g2.to_numpy(), 1)[0]
                macros["EntropySlope"] = fmt(sl2, 2)

    # ---- write numbers.tex ----------------------------------------------
    lines = ["% AUTO-GENERATED by analysis/make_numbers.py -- do not edit.", ""]
    for k, v in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{k}}}{{{v}\xspace}}" if False
                     else rf"\newcommand{{\{k}}}{{{v}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'numbers.tex'} ({len(macros)} macros)")

    # ---- Table 1: per-model summary --------------------------------------
    capm = (json.loads(cap_path.read_text())["models"] if cap_path.exists() else {})
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
        rows.append((
            LABEL.get(m, m), PARAMS.get(m, "--"),
            fmt(e.get("k_star"), 0),
            fmt(100 * rep.correct.mean(), 1) if len(rep) else "--",
            fmt(100 * ok, 1),
            fmt(gr.get("gain"), 1), fmt(gc.get("gain"), 1),
            fmt(c.get("ratio"), 2),
        ))
    tbl = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Model & Par. & $k^\ast$ & \multicolumn{2}{c}{acc.\ $k\!\ge\!6$ (\%)} "
        r"& \multicolumn{3}{c}{capacity gain $\mathrm{d}\mathcal{N}_{\mathrm{eff}}/\mathrm{d}\log k$} \\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-8}",
        r" & & & rep. & ctl. & rep. & ctl. & ratio \\", r"\midrule",
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
