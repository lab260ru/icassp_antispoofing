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
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ICASSP two-column: a single-column figure is 3.35in wide.
plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "lines.markersize": 3.2, "legend.frameon": False,
    # A bar in the main figure is 6pt wide; the 1pt default hatch stroke reads
    # as a second colour rather than as texture at that size.
    "hatch.linewidth": 0.35,
})

ORDER = ["llasa1b", "llasa3b", "llasa8b", "xtts2", "qwen06b", "qwen17b"]
LABEL = {"llasa1b": "Llasa-1B", "llasa3b": "Llasa-3B", "llasa8b": "Llasa-8B",
         "xtts2": "XTTS-v2", "qwen06b": "Qwen3-TTS-0.6B", "qwen17b": "Qwen3-TTS-1.7B"}
# Short forms for the one place a tick label has to carry the checkpoint name:
# panel (c) gives each of six models about 9pt of x-axis.
SHORT = {"llasa1b": "Llasa-1B", "llasa3b": "Llasa-3B", "llasa8b": "Llasa-8B",
         "xtts2": "XTTS-v2", "qwen06b": "Qwen-0.6B", "qwen17b": "Qwen-1.7B"}
COLOR = {"llasa1b": "#4C72B0", "llasa3b": "#DD8452", "llasa8b": "#55A868",
         "xtts2": "#C44E52", "qwen06b": "#8172B3", "qwen17b": "#937860"}
# Repeated vs. control is the figure's one recurring contrast, so it gets one
# encoding everywhere: colour *and* line style *and* marker in (b), colour *and*
# fill lightness *and* hatch in (c), so neither greyscale printing nor a
# red-green deficiency can erase it.
C_REP, C_CTL = "#C44E52", "#4C72B0"
# Ablation re-runs of a panel member -- the penalty sweep and the no-penalty
# arm. Kept out of the main figure for the same reason they are kept out of
# pooled statistics: they are one checkpoint under altered decoding, not extra
# checkpoints. The canonical list lives with the population filter; falling back
# to a local copy only keeps this script runnable outside the repo.
try:  # pragma: no cover - import path convenience
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.common.population import ABLATIONS
except Exception:  # pragma: no cover
    ABLATIONS = {"xtts2norp", "xtts2rp2", "xtts2rp3", "xtts2rp8"}


def ordered(models, drop_ablations: bool = True) -> list[str]:
    ms = set(models) - (ABLATIONS if drop_ablations else set())
    return [m for m in ORDER if m in ms] + sorted(ms - set(ORDER))


# The figure is drawn at the size it is printed at: 246.3 x 65.5pt, one ICASSP
# column wide. Nothing here may change that size. LaTeX scales whatever it is
# handed to the width in results_body.tex, so drawing wider only shrinks the
# type (a 7in draw would land 6pt labels on the page at 2pt), and drawing
# taller costs page budget the body does not have. Every panel is therefore
# laid out in points against this fixed canvas rather than by tight_layout,
# which cannot honour a height this small and silently lets titles, legends
# and tick labels collide -- which is exactly how it used to fail.
FIG_W_PT, FIG_H_PT = 246.315, 65.473
PT = 1.0 / 72.0


def _rect(x, y, w, h):
    """Points from the bottom-left of the canvas -> a figure-fraction rect."""
    return [x / FIG_W_PT, y / FIG_H_PT, w / FIG_W_PT, h / FIG_H_PT]


def _int_ticks(ax, axis: str, ticks) -> None:
    """Label an axis with the integers themselves.

    A base-2 log axis defaults to mathtext powers ($2^{5}$), which at 5.5pt in
    a 0.9in-tall figure prints as a lone digit sitting above another digit --
    the single most-reported defect in this figure's review history, read by
    reviewers as a log-exponent extraction artifact. Every quantity on these
    axes is a small integer count; print the integers.
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mticker.FixedLocator(ticks))
    a.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(mticker.NullLocator())


def _fit_ylabels(fig, axes) -> None:
    """Nudge y labels that are longer than their axis back onto the canvas.

    "count error (%)" is 46pt of type and the axes are 36pt tall, but the
    label's own column is empty from the top of the canvas to the bottom, so
    the label needs moving, not shrinking. Matplotlib recomputes only the
    horizontal position of a y label at draw time, so the vertical nudge set
    here survives the save.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        lbl = ax.yaxis.label
        if not lbl.get_text():
            continue
        bb = lbl.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted())
        y0, y1 = bb.y0 * 72.0, bb.y1 * 72.0
        dy = max(1.0 - y0, 0.0) - max(y1 - (FIG_H_PT - 1.0), 0.0)
        if dy:
            x, y = lbl.get_position()
            lbl.set_position((x, y + dy / (ax.get_position().height * FIG_H_PT)))


def fig_main(beh: pd.DataFrame, state: pd.DataFrame, cap: dict, out: Path,
             beh_ext: pd.DataFrame | None = None) -> None:
    """The paper's single figure: behaviour, the control contrast, and the
    capacity mechanism, in one full-width row.

    Three panels, not four. The old panel (a) plotted repeated-item error
    against k, which panel (b) already contains as its solid curves; dropping
    it buys the remaining three enough width to stay legible.
    """
    models = [m for m in ordered(beh.model.unique()) if m in ORDER]

    fig = plt.figure(figsize=(FIG_W_PT * PT, FIG_H_PT * PT))
    # Vertical budget, in points: 27 under the axes -- set by the six rotated
    # checkpoint names of panel (c), the tallest thing below any axis -- 2.5
    # above, and 36 of plot in between. The panel letters go *inside* the axes:
    # a title row would cost a fifth of the plot height.
    bot, top_pad = 27.0, 2.5
    axh = FIG_H_PT - bot - top_pad
    # Horizontal budget: each panel keeps its own gutter for a rotated y label
    # plus its widest tick label.
    gut = (29.0, 27.0, 23.0)
    axw = (FIG_W_PT - sum(gut) - 4.0) / 3.0
    x0 = [gut[0], gut[0] + axw + gut[1], gut[0] + axw + gut[1] + axw + gut[2]]
    axes = [fig.add_axes(_rect(x, bot, axw, axh)) for x in x0]
    for ax, letter in zip(axes, "abc"):
        ax.tick_params(labelsize=5.5, pad=1.2, length=1.8, width=0.5)
        ax.text(0.035, 0.955, f"({letter})", transform=ax.transAxes, ha="left",
                va="top", fontsize=6)

    # ---- (a) relative count error against k ---------------------------
    # Relative count error rather than exact-match accuracy: signed, so
    # premature stopping separates from looping, and scale-free, so k=4 and
    # k=32 are on the same axis.
    beh = beh.copy()
    beh["rel_err"] = (beh.count_a - beh.k) / beh.k
    ok = ~beh.outcome.isin(["empty", "degenerate"])

    ax = axes[0]
    for m in models:
        for fam, ls, mk, al in (("word_rep", "-", "o", 1.0),
                                ("control_word", "--", "s", 0.55)):
            s = beh[(beh.model == m) & (beh.family == fam) & ok]
            if s.empty:
                continue
            g = s.groupby("k")["rel_err"].median().sort_index()
            ax.plot(g.index, 100 * g.values, ls, marker=mk, color=COLOR.get(m),
                    alpha=al, lw=0.8, ms=1.9, mew=0)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.plot([], [], "k-", marker="o", ms=1.9, mew=0, lw=0.8, label="repeated")
    ax.plot([], [], "k--", marker="s", ms=1.9, mew=0, lw=0.8, alpha=0.55,
            label="control")
    ax.set_xscale("log", base=2)
    _int_ticks(ax, "x", [1, 8, 32])
    ax.set_yticks([-100, -50, 0])
    # Headroom below the data for the legend, and above it for the panel letter;
    # -100% is the floor of the measure, so nothing is hidden by either.
    ax.set_ylim(-196, 58)
    ax.set_xlabel(r"repetitions $k$", fontsize=6, labelpad=0.8)
    ax.set_ylabel("count error (%)", fontsize=6, labelpad=1.0)
    # The band under -100% is empty by construction -- the measure floors at
    # -100% -- so the legend can sit in it without a patch and without covering
    # the one curve that runs along the floor.
    ax.legend(fontsize=5, loc="lower right", handlelength=1.5, handletextpad=0.4,
              labelspacing=0.1, borderpad=0.1, borderaxespad=0.1)

    # ---- (b) the extension ladder -------------------------------------
    # This is where the count actually stops tracking the request, and the
    # control curve is what stops that being read as counting collapse when it
    # is partly a general utterance-length ceiling.
    ax = axes[1]
    if beh_ext is not None and len(beh_ext):
        e = beh_ext.copy()
        eok = ~e.outcome.isin(["empty", "degenerate"])
        ks = np.array(sorted(e.k.unique()), dtype=float)
        # Linear on both axes, where the old panel was log-log: the requested
        # counts span less than two octaves, so the log axes bought nothing and
        # cost a set of power-of-two tick labels. Linear also makes y=x the
        # straight diagonal a reader expects.
        ax.plot([ks[0] - 20, ks[-1]], [ks[0] - 20, ks[-1]], ":", color="0.35",
                lw=0.7)
        for fam, ls, mk, col in (("word_rep", "-", "o", C_REP),
                                 ("control_word", "--", "s", C_CTL)):
            g = e[(e.family == fam) & eok].groupby("k")["count_a"].median().sort_index()
            if g.empty:
                continue
            ax.plot(g.index, g.values, ls, marker=mk, color=col, lw=0.9,
                    ms=2.0, mew=0)
        _int_ticks(ax, "x", [48, 64, 96, 128])
        _int_ticks(ax, "y", [0, 64, 128])
        ax.set_xlim(ks[0] - 8, ks[-1] + 8)
        ax.set_ylim(0, 148)
        ax.set_xlabel(r"requested $k$", fontsize=6, labelpad=0.8)
        ax.set_ylabel("rendered count", fontsize=6, labelpad=1.0)
        # The two families and the reference line are named on themselves. A
        # legend box would have to sit in the wedge above the diagonal, which
        # is the only free space this panel has and is not big enough for it.
        ax.text(0.5, 0.02, "repeated", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=5, color=C_REP)
        ax.text(0.97, 0.57, "control", transform=ax.transAxes, ha="right",
                va="center", fontsize=5, color=C_CTL)
        p0 = ax.transData.transform((ks[0], ks[0]))
        p1 = ax.transData.transform((ks[-1], ks[-1]))
        ax.text(0.5 * (ks[0] + ks[-1]), 0.5 * (ks[0] + ks[-1]) + 7, r"$y=x$",
                ha="center", va="bottom", fontsize=5, color="0.25",
                rotation=np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])),
                rotation_mode="anchor")
    else:
        ax.set_axis_off()

    # ---- (c) fitted capacity gain, 95% bootstrap CIs -------------------
    ax = axes[2]
    cm = [m for m in ordered(list(cap["models"].keys())) if m in ORDER]
    xs = np.arange(len(cm), dtype=float)
    w = 0.36
    # Light hatched fill against dark flat fill, not two saturated hues: the
    # bars carry the repeated/control contrast with no line style available to
    # back the colour up, and #C44E52 and #4C72B0 have the same greyscale value.
    for off, key, face, edge, hatch in (
            (-w / 2, "repeated", C_REP, C_REP, ""),
            (w / 2, "control", "#AEC8E0", C_CTL, "//")):
        v = [cap["models"][m][key]["gain"] for m in cm]
        lo = [cap["models"][m][key]["lo"] for m in cm]
        hi = [cap["models"][m][key]["hi"] for m in cm]
        err = np.array([[max(a - b, 0) for a, b in zip(v, lo)],
                        [max(b - a, 0) for a, b in zip(v, hi)]])
        ax.bar(xs + off, v, w, color=face, hatch=hatch, edgecolor=edge,
               linewidth=0.4)
        ax.errorbar(xs + off, v, yerr=err, fmt="none", ecolor="0.15",
                    elinewidth=0.5, capsize=0.9, capthick=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(m, m) for m in cm], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=5.0)
    ax.tick_params(axis="x", pad=0.6)
    ax.set_xlim(-0.6, len(cm) - 0.4)
    ax.set_yticks([0, 25, 50])
    ax.set_ylim(0, 88)     # room above the tallest interval for the letter
    ax.set_ylabel(r"$\mathrm{d}\mathcal{N}_{\mathrm{eff}}/\mathrm{d}\log k$",
                  fontsize=5.6, labelpad=1.0)

    _fit_ylabels(fig, axes)
    # Not bbox_inches="tight" (the rcParam default): a tight box would resize
    # the output to whatever overflowed, and the printed size is the one thing
    # this figure is not allowed to change.
    fig.savefig(out, bbox_inches=fig.bbox_inches, pad_inches=0.0)
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
    ap.add_argument("--behavioural-ext", nargs="+",
                    default=["data/results/behavioural_ext_llasa.csv",
                             "data/results/behavioural_ext.csv"],
                    help="extension-ladder tables; missing ones are skipped")
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

    ext_frames = [pd.read_csv(p) for p in args.behavioural_ext if Path(p).exists()]
    beh_ext = pd.concat(ext_frames, ignore_index=True) if ext_frames else None
    if beh_ext is not None:
        # Same exclusions as every other analysis: budget truncations are ours,
        # not the model's, and would read as a counting failure.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.common.population import cap_flags
        flags = cap_flags()
        beh_ext = beh_ext[[not flags.get((r.model, r.item_id, r.seed), False)
                           for r in beh_ext.itertuples()]]

    if beh is not None and state is not None and cap is not None:
        fig_main(beh, state, cap, outdir / "fig_main.pdf", beh_ext=beh_ext)
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
