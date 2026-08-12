#!/usr/bin/env python3
"""Generate the paper's figures.

Fig 1  the dissociation: counting accuracy vs k for repeated text and for
       length-matched non-repetitive controls
Fig 2  the mechanism: boundary-state distances d_m decaying geometrically, and
       the cross-model regression of k* on 1/log(1/q_hat) that Theorem A predicts
Fig 3  attention dilution: per-occurrence mass ~ 1/k and entropy ~ log k

Usage:  python analysis/figures.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ICASSP two-column: a single-column figure is 3.35in wide.
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "lines.markersize": 3.2, "legend.frameon": False,
})

ORDER = ["llasa1b", "llasa3b", "llasa8b", "xtts2", "qwen06b", "qwen17b"]
LABEL = {"llasa1b": "Llasa-1B", "llasa3b": "Llasa-3B", "llasa8b": "Llasa-8B",
         "xtts2": "XTTS-v2", "qwen06b": "Qwen3-TTS-0.6B", "qwen17b": "Qwen3-TTS-1.7B"}
COLOR = {"llasa1b": "#4C72B0", "llasa3b": "#DD8452", "llasa8b": "#55A868",
         "xtts2": "#C44E52", "qwen06b": "#8172B3", "qwen17b": "#937860"}
# Ablation re-runs of a panel member. Kept out of the main figure for the same
# reason they are kept out of pooled statistics: they are not extra checkpoints.
ABLATIONS = {"xtts2norp"}


def ordered(models, drop_ablations: bool = True) -> list[str]:
    ms = set(models) - (ABLATIONS if drop_ablations else set())
    return [m for m in ORDER if m in ms] + sorted(ms - set(ORDER))


def fig_main(beh: pd.DataFrame, state: pd.DataFrame, cap: dict, out: Path) -> None:
    """The paper's single figure: behaviour, the control contrast, and the
    capacity mechanism, in one full-width row.

    Four panels in a row rather than two stacked two-panel figures: an ICASSP
    page is 4 pages and two figure environments cost roughly a third of one.
    """
    models = ordered(beh.model.unique())
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.20))

    # Relative count error rather than exact-match accuracy: signed, so
    # premature stopping separates from looping, and scale-free, so k=4 and
    # k=32 are on the same axis.
    beh = beh.copy()
    beh["rel_err"] = (beh.count_a - beh.k) / beh.k
    ok = ~beh.outcome.isin(["empty", "degenerate"])

    ax = axes[0]
    for m in models:
        s = beh[(beh.model == m) & (beh.family == "word_rep") & ok]
        if s.empty:
            continue
        g = s.groupby("k")["rel_err"].median().sort_index()
        ax.plot(g.index, 100 * g.values, "o-", color=COLOR.get(m), label=LABEL.get(m, m))
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"repetitions $k$")
    ax.set_ylabel("count error (\%)")
    ax.set_title("(a) undercount")
    ax.legend(fontsize=4.6, loc="lower left", handlelength=1.2)

    ax = axes[1]
    for m in models:
        for fam, ls, mk, al in (("word_rep", "-", "o", 1.0),
                                ("control_word", "--", "s", 0.55)):
            s = beh[(beh.model == m) & (beh.family == fam) & ok]
            if s.empty:
                continue
            g = s.groupby("k")["rel_err"].median().sort_index()
            ax.plot(g.index, 100 * g.values, ls, marker=mk, color=COLOR.get(m), alpha=al)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.plot([], [], "ko-", label="repeated")
    ax.plot([], [], "ks--", alpha=0.55, label="control")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("count error (\%)")
    ax.set_title("(b) periodicity, not length")
    ax.legend(fontsize=5, loc="lower left", handlelength=1.4)

    st = state[state.layer_frac > 0.6]
    ax = axes[2]
    for m in ordered(st.model.unique()):
        for fam, ls, mk, al in (("word_rep", "-", "o", 1.0),
                                ("control_word", "--", "s", 0.55)):
            g = st[(st.model == m) & (st.family == fam)].groupby("k")["n_eff"].median()
            if g.empty:
                continue
            ax.plot(g.index, g.values, ls, marker=mk, color=COLOR.get(m), alpha=al)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathcal{N}_{\mathrm{eff}}$")
    ax.set_title("(c) states saturate")

    ax = axes[3]
    cm = ordered(list(cap["models"].keys()))  # ablations dropped
    xs = np.arange(len(cm), dtype=float)
    w = 0.34
    for off, key, lab, col in ((-w / 2, "repeated", "rep.", "#C44E52"),
                               (w / 2, "control", "ctrl.", "#4C72B0")):
        v = [cap["models"][m][key]["gain"] for m in cm]
        lo = [cap["models"][m][key]["lo"] for m in cm]
        hi = [cap["models"][m][key]["hi"] for m in cm]
        err = np.array([[max(a - b, 0) for a, b in zip(v, lo)],
                        [max(b - a, 0) for a, b in zip(v, hi)]])
        ax.bar(xs + off, v, w, label=lab, color=col, alpha=1.0 if key == "repeated" else 0.8)
        ax.errorbar(xs + off, v, yerr=err, fmt="none", ecolor="0.2", elinewidth=0.6,
                    capsize=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(m, m) for m in cm], rotation=32, ha="right", fontsize=4.8)
    ax.set_ylabel(r"$\mathrm{d}\mathcal{N}_{\mathrm{eff}}/\mathrm{d}\log k$")
    ax.set_title("(d) capacity gain")
    ax.legend(fontsize=5, loc="upper right", handlelength=1.2)

    fig.tight_layout(pad=0.25, w_pad=0.7)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def fig_dissociation(beh: pd.DataFrame, out: Path) -> None:
    """Counting accuracy against k, with the length-matched control overlaid.

    The control is the load-bearing comparison: it holds word count and syntax
    fixed and removes only the repetition, so a gap between the two curves is
    attributable to periodicity rather than to sequence length.
    """
    models = ordered(beh.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15))

    ax = axes[0]
    for m in models:
        s = beh[(beh.model == m) & (beh.family.isin(["word_rep", "sentence_rep"]))]
        if s.empty:
            continue
        acc = s.groupby("k")["correct"].mean().sort_index()
        ax.plot(acc.index, acc.values, "o-", color=COLOR.get(m), label=LABEL.get(m, m))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"requested repetitions $k$")
    ax.set_ylabel("counting accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) repetition ladder")
    ax.legend(ncol=2, loc="upper right")

    ax = axes[1]
    for m in models:
        rep = beh[(beh.model == m) & (beh.family == "word_rep")]
        ctl = beh[(beh.model == m) & (beh.family == "control_word")]
        if rep.empty or ctl.empty:
            continue
        # both curves ask the same question: were all k units rendered?
        r = rep.groupby("k")["correct"].mean().sort_index()
        ok = (ctl.outcome.isin(["correct", "overcount", "undercount"])
              & (ctl.duration_ratio > 0.7) & (ctl.spectral_flatness < 0.35))
        c = ctl.assign(ok=ok.astype(float)).groupby("k")["ok"].mean().sort_index()
        ax.plot(r.index, r.values, "o-", color=COLOR.get(m), label=f"{LABEL.get(m,m)} rep.")
        ax.plot(c.index, c.values, "s--", color=COLOR.get(m), alpha=0.55,
                label=f"{LABEL.get(m,m)} ctrl.")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$k$ (identical vs. distinct words)")
    ax.set_ylabel("fraction fully rendered")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) periodicity, not length")
    ax.legend(ncol=2, loc="lower left")

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def fig_mechanism(state: pd.DataFrame, cap: dict, out: Path) -> None:
    """The capacity result: N_eff against k, and the fitted gains with CIs."""
    st = state[state.layer_frac > 0.6]
    models = ordered(st.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

    # ---- (a) N_eff vs k, repeated vs control ---------------------------
    ax = axes[0]
    for m in models:
        for fam, ls, mk, alpha in (("word_rep", "-", "o", 1.0),
                                   ("control_word", "--", "s", 0.55)):
            g = st[(st.model == m) & (st.family == fam)].groupby("k")["n_eff"].median()
            if g.empty:
                continue
            ax.plot(g.index, g.values, ls, marker=mk, color=COLOR.get(m), alpha=alpha,
                    label=f"{LABEL.get(m,m)} {'rep.' if fam=='word_rep' else 'ctrl.'}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"repetitions $k$")
    ax.set_ylabel(r"effective rank $\mathcal{N}_{\mathrm{eff}}$")
    ax.set_title("(a) states visited saturate")
    ax.legend(ncol=2, loc="upper left", fontsize=5.4)

    # ---- (b) capacity gain with bootstrap CIs --------------------------
    ax = axes[1]
    xs = np.arange(len(models), dtype=float)
    w = 0.34
    for off, key, lab, col in ((-w / 2, "repeated", "repeated", "#C44E52"),
                               (w / 2, "control", "control", "#4C72B0")):
        vals = [cap["models"].get(m, {}).get(key, {}).get("gain", np.nan) for m in models]
        los = [cap["models"].get(m, {}).get(key, {}).get("lo", np.nan) for m in models]
        his = [cap["models"].get(m, {}).get(key, {}).get("hi", np.nan) for m in models]
        err = np.array([[max(v - l, 0) for v, l in zip(vals, los)],
                        [max(h - v, 0) for v, h in zip(vals, his)]])
        ax.bar(xs + off, vals, w, label=lab, color=col,
               alpha=1.0 if key == "repeated" else 0.8)
        ax.errorbar(xs + off, vals, yerr=err, fmt="none", ecolor="0.2",
                    elinewidth=0.7, capsize=1.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(m, m) for m in models], rotation=20, ha="right")
    ax.set_ylabel(r"$\mathrm{d}\mathcal{N}_{\mathrm{eff}}/\mathrm{d}\log k$")
    ax.set_title("(b) capacity gain, 95% CI")
    ax.legend(loc="upper right")

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def fig_decay(state: pd.DataFrame, out: Path) -> None:
    """Layer profile of the capacity gap: where in depth the saturation lives."""
    d = state[state.n_eff.notna()]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    plotted = False
    for m in ordered(d.model.unique()):
        s = d[d.model == m]
        prof = []
        for lf in sorted(s.layer_frac.unique()):
            a = s[(s.layer_frac == lf) & (s.family == "word_rep") & (s.k >= 8)]
            b = s[(s.layer_frac == lf) & (s.family == "control_word") & (s.k >= 8)]
            if len(a) < 3 or len(b) < 3 or b.n_eff.median() <= 0:
                continue
            prof.append((lf, a.n_eff.median() / b.n_eff.median()))
        if len(prof) < 3:
            continue
        plotted = True
        ax.plot([p[0] for p in prof], [p[1] for p in prof], "o-",
                color=COLOR.get(m), label=LABEL.get(m, m))
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("normalised depth")
    ax.set_ylabel(r"$\mathcal{N}_{\mathrm{eff}}$ ratio (rep./ctrl.)")
    ax.legend(loc="lower left", fontsize=5.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def fig_attention(state: pd.DataFrame, out: Path) -> None:
    """Attention dilution: Lemma 1's two observable consequences."""
    s = state[(state.layer_frac > 0.6) & state.get("attn_per_occurrence").notna()] \
        if "attn_per_occurrence" in state.columns else pd.DataFrame()
    if s.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05))
    models = ordered(s.model.unique())

    ax = axes[0]
    for m in models:
        g = s[(s.model == m) & (s.k >= 2)].groupby("k")["attn_per_occurrence"].median()
        if g.empty:
            continue
        ax.plot(g.index, g.values, "o-", color=COLOR.get(m), label=LABEL.get(m, m))
    ks = np.array(sorted(s[s.k >= 2].k.unique()), dtype=float)
    if ks.size:
        ref = s[s.k == ks[0]]["attn_per_occurrence"].median() * ks[0]
        ax.plot(ks, ref / ks, "k:", lw=0.9, label=r"$\propto 1/k$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"repetitions $k$")
    ax.set_ylabel("median attention per occurrence")
    ax.set_title("(a) dilution")
    ax.legend(loc="lower left", ncol=2)

    ax = axes[1]
    for m in models:
        g = s[(s.model == m) & (s.k >= 2)].groupby("k")["attn_text_entropy"].median()
        if g.empty:
            continue
        ax.plot(np.log(g.index.to_numpy(dtype=float)), g.values, "o-",
                color=COLOR.get(m), label=LABEL.get(m, m))
    ax.set_xlabel(r"$\log k$")
    ax.set_ylabel("text-attention entropy (nats)")
    ax.set_title("(b) entropy grows with " + r"$\log k$")
    ax.legend(loc="lower right", ncol=1)

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--state", default="data/results/state.csv")
    ap.add_argument("--summary", default="data/results/summary.json")
    ap.add_argument("--outdir", default="paper/figs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("figures:")

    beh = pd.read_csv(args.behavioural) if Path(args.behavioural).exists() else None
    state = pd.read_csv(args.state) if Path(args.state).exists() else None
    cap_path = Path("data/results/capacity.json")
    cap = json.loads(cap_path.read_text()) if cap_path.exists() else None

    if beh is not None and state is not None and cap is not None:
        fig_main(beh, state, cap, outdir / "fig_main.pdf")
    # supplementary figures
    if beh is not None:
        fig_dissociation(beh, outdir / "fig1_dissociation.pdf")
    if state is not None:
        if cap is not None:
            fig_mechanism(state, cap, outdir / "fig2_mechanism.pdf")
        fig_decay(state[state.layer_frac.notna()], outdir / "fig3_depth.pdf")
        fig_attention(state, outdir / "fig4_attention.pdf")


if __name__ == "__main__":
    main()
