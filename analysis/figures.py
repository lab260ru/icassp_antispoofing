#!/usr/bin/env python3
"""Generate the paper's figures.

Fig 1  the dissociation: counting accuracy vs k for repeated text and for
       length-matched non-repetitive controls, beside the period ladder that
       says the deficit is graded in the period rather than switched on by
       verbatim token identity
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
         "xtts2": "XTTS-v2", "qwen06b": "Qwen-0.6B", "qwen17b": "Qwen-1.7B",
         # Not a panel member (see population.ABLATIONS), but it can appear in
         # the period ladder, whose legend is the one place its name would show.
         "cosyvoice2": "CosyVoice2"}
COLOR = {"llasa1b": "#4C72B0", "llasa3b": "#DD8452", "llasa8b": "#55A868",
         "xtts2": "#C44E52", "qwen06b": "#8172B3", "qwen17b": "#937860"}
# Panel (b) separates checkpoints from one another rather than repeated from
# control, so hue is the only encoding it inherits -- and hue alone is exactly
# what a greyscale printer and a red-green deficiency both destroy (#C44E52 and
# #4C72B0 have nearly the same luminance). Each curve therefore also gets a dash
# pattern and a marker, assigned by position in the plotted list so that a
# checkpoint arriving in period_ladder.json tomorrow gets the next unused pair
# instead of a KeyError.
CURVE_STYLES = [("-", "o"), ((0, (3.0, 1.3)), "s"), ((0, (1.0, 1.2)), "^"),
                ((0, (4.5, 1.2, 1.0, 1.2)), "D"), ((0, (2.0, 1.0, 0.6, 1.0)), "v"),
                ((0, (5.5, 1.4)), "P")]
# Hues for a checkpoint that is not in COLOR (a fourth architecture landing in
# the ladder before it lands in ORDER).
SPARE_COLORS = ["#4C72B0", "#8172B3", "#C44E52", "#55A868", "#DD8452", "#937860"]
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


def _period_panel(ax, period: dict) -> None:
    """Exact rate against period p, one line per checkpoint.

    Reads `exact_by_period_scan` (the published ordered-scan rule) and
    `n_by_period` out of period_ladder.json rather than naming checkpoints:
    the ladder gains rungs and checkpoints between drafts, and a hardcoded list
    would either crash or silently drop the new one.

    Two things this panel is not allowed to imply:

    * **p=k is not the far end of the ladder.** p in {2,4,8} are rotations of
      one 8-word pool, so they are lexically identical to each other; p=k draws
      a 146-word pool and runs ~215 characters against ~188 for the interior,
      so it is neither vocabulary- nor length-matched to them and the verdict
      in the text is computed without it. It is drawn detached, past a rule,
      with hollow markers and no connecting segment, so no eye reads a
      trend through it.
    * **No error bars.** The stored bootstrap is over (template, k) cells of the
      *strict conjunction* rule, while these points are the ordered scan; the
      two rules differ by up to 28 points at p=2, so borrowing the interval
      would draw points outside their own bars. The interval that belongs to
      these points is not computed here, so none is drawn.
    """
    scan = period.get("exact_by_period_scan") or {}
    if not scan:
        ax.set_axis_off()
        return
    nby = period.get("n_by_period") or {}

    def rate(arm: str, m: str):
        """The cell's rate, or None if the cell is absent or empty.

        A (checkpoint, arm) cell can exist in the rate table with a null while
        that checkpoint's audio is still generating; plotting it would join a
        line across a rung nobody measured.
        """
        v = scan.get(arm, {}).get(m)
        n = nby.get(arm, {}).get(m, 1) if nby else 1
        return None if v is None or not n else 100 * float(v)

    arms = [a for a in ("1", "2", "4", "8", "k") if a in scan]
    interior = [a for a in arms if a != "k"]
    models = ordered({m for a in arms for m in scan[a]}, drop_ablations=False)
    # Ladder rungs are equally spaced, not placed at p: the interesting
    # comparison is rung-to-rung, and a linear p axis would crush p=1,2,4 into
    # the left eighth of the panel while a log axis would relabel them as powers.
    xpos = {a: float(i) for i, a in enumerate(interior)}
    if "k" in arms:
        xpos["k"] = float(len(interior)) + 0.55   # detached: see docstring

    for i, m in enumerate(models):
        ls, mk = CURVE_STYLES[i % len(CURVE_STYLES)]
        col = COLOR.get(m, SPARE_COLORS[i % len(SPARE_COLORS)])
        pts = [(xpos[a], rate(a, m)) for a in interior]
        pts = [(x, y) for x, y in pts if y is not None]
        ax.plot([x for x, _ in pts], [y for _, y in pts], ls=ls, marker=mk,
                color=col, lw=0.9, ms=2.0, mew=0, label=SHORT.get(m, m))
        yk = rate("k", m) if "k" in arms else None
        if yk is not None:
            # Hollow, unconnected: a point on the same axes, not a fifth rung.
            ax.plot([xpos["k"]], [yk], ls="none", marker=mk,
                    mfc="none", mec=col, mew=0.6, ms=2.4)
    if "k" in arms:
        ax.axvline(0.5 * (xpos[interior[-1]] + xpos["k"]), color="0.55",
                   lw=0.5, ls=(0, (1.2, 1.4)))

    ax.set_xticks([xpos[a] for a in arms])
    ax.set_xticklabels(arms, fontsize=5.5)
    ax.set_xlim(-0.35, max(xpos.values()) + 0.35)
    _int_ticks(ax, "y", [0, 50, 100])
    # Headroom above 100% for the panel letter, and below 0% so the p=1 markers
    # do not sit on the spine.
    ax.set_ylim(-8, 124)
    ax.set_xlabel(r"period $p$", fontsize=6, labelpad=0.8)
    ax.set_ylabel("exactly right (%)", fontsize=6, labelpad=1.0)
    # Lower right is empty by the shape of the result -- the curves are high
    # everywhere except p=1, which is at the far left -- so the legend needs no
    # box and hides nothing.
    ax.legend(fontsize=5, loc="lower right", ncol=2 if len(models) > 3 else 1,
              handlelength=1.6, handletextpad=0.4, labelspacing=0.12,
              columnspacing=0.8, borderpad=0.1, borderaxespad=0.15)


def fig_main(beh: pd.DataFrame, out: Path, period: dict | None = None) -> None:
    """The paper's single figure: the dissociation, and the period ladder.

    Two panels, not three. The figure used to spend (b) on the k<=128 extension
    ladder and (c) on the fitted rate of effective-rank gain -- both exploratory,
    and the body now compresses both into one sentence saying neither
    distinguishes a counting horizon from any saturating process. Meanwhile the
    period ladder, which is the paper's answer to its own title, had no figure at
    all. Two panels at the same total width is the trade: the headline result
    gets drawn, and the two survivors get half a column each instead of a third.
    """
    models = [m for m in ordered(beh.model.unique()) if m in ORDER]

    fig = plt.figure(figsize=(FIG_W_PT * PT, FIG_H_PT * PT))
    # Vertical budget, in points. The canvas is the same 246.3 x 65.5pt it has
    # always been -- it is printed at this size, not scaled -- but the 27pt that
    # used to sit under the axes existed for one thing only: the six rotated
    # checkpoint names of the old panel (c). Both surviving panels label their
    # x axis with short horizontal integers, so 15pt covers a tick row and a
    # label, and the 12pt released goes straight into plot height. The panel
    # letters go *inside* the axes: a title row would cost a quarter of it back.
    bot, top_pad = 15.0, 2.5
    axh = FIG_H_PT - bot - top_pad
    # Horizontal budget: each panel keeps its own gutter for a rotated y label
    # plus its widest tick label ("-100" in (a), "100" in (b)).
    gut = (29.0, 25.0)
    axw = (FIG_W_PT - sum(gut) - 4.0) / 2.0
    x0 = [gut[0], gut[0] + axw + gut[1]]
    axes = [fig.add_axes(_rect(x, bot, axw, axh)) for x in x0]
    for ax, letter in zip(axes, "ab"):
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
    # -100% is the floor of the measure, so nothing is hidden by either. The
    # band was -196 when the axes were 36pt tall; at 48pt the same legend needs
    # proportionally less of it, so the data get the space back.
    ax.set_ylim(-158, 44)
    ax.set_xlabel(r"repetitions $k$", fontsize=6, labelpad=0.8)
    ax.set_ylabel("count error (%)", fontsize=6, labelpad=1.0)
    # The band under -100% is empty by construction -- the measure floors at
    # -100% -- so the legend can sit in it without a patch and without covering
    # the one curve that runs along the floor.
    ax.legend(fontsize=5, loc="lower right", handlelength=1.5, handletextpad=0.4,
              labelspacing=0.1, borderpad=0.1, borderaxespad=0.1)

    # ---- (b) the period ladder ----------------------------------------
    # The paper's answer to its own title. Carrier, word count and requested
    # count are held fixed and only the period p varies, so the shape of this
    # curve is the whole argument: a *step* between p=1 and p>1 would say the
    # effect is verbatim token identity, and a graded monotone rise says it is
    # the period. Draw it as a line per checkpoint precisely so the reader can
    # see gradedness holding within each one, not only in a pooled mean.
    ax = axes[1]
    if period:
        _period_panel(ax, period)
    else:
        ax.set_axis_off()

    _fit_ylabels(fig, axes)
    # Not bbox_inches="tight" (the rcParam default): a tight box would resize
    # the output to whatever overflowed, and the printed size is the one thing
    # this figure is not allowed to change.
    fig.savefig(out, bbox_inches=fig.bbox_inches, pad_inches=0.0)
    plt.close(fig)
    print(f"  {out}")


def fig_extension(beh_ext: pd.DataFrame, out: Path) -> None:
    """The k<=128 extension ladder: rendered count against requested count.

    This was panel (b) of the main figure until the period ladder took the
    slot. It is exploratory -- the body now gives it one clause -- so it lives
    here, at supplementary scale, rather than spending a third of a column.
    """
    e = beh_ext.copy()
    eok = ~e.outcome.isin(["empty", "degenerate"])
    ks = np.array(sorted(e.k.unique()), dtype=float)
    if not ks.size:
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.plot([ks[0] - 20, ks[-1]], [ks[0] - 20, ks[-1]], ":", color="0.35", lw=0.7,
            label=r"$y=x$")
    for fam, ls, mk, col, lab in (("word_rep", "-", "o", C_REP, "repeated"),
                                  ("control_word", "--", "s", C_CTL, "control")):
        g = e[(e.family == fam) & eok].groupby("k")["count_a"].median().sort_index()
        if g.empty:
            continue
        ax.plot(g.index, g.values, ls, marker=mk, color=col, lw=0.9, label=lab)
    _int_ticks(ax, "x", [48, 64, 96, 128])
    _int_ticks(ax, "y", [0, 64, 128])
    ax.set_xlim(ks[0] - 8, ks[-1] + 8)
    ax.set_ylim(0, 148)
    ax.set_xlabel(r"requested $k$")
    ax.set_ylabel("rendered count")
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.3)
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
    ap.add_argument("--behavioural-ext", nargs="+",
                    default=["data/results/behavioural_ext_llasa.csv",
                             "data/results/behavioural_ext.csv"],
                    help="extension-ladder tables; missing ones are skipped")
    ap.add_argument("--period", default="data/results/period_ladder.json",
                    help="period ladder table; panel (b) is dropped if missing")
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

    per_path = Path(args.period)
    period = json.loads(per_path.read_text()) if per_path.exists() else None

    if beh is not None:
        fig_main(beh, outdir / "fig_main.pdf", period=period)
    # supplementary figures
    if beh is not None:
        fig_dissociation(beh, outdir / "fig1_dissociation.pdf")
    if beh_ext is not None and len(beh_ext):
        fig_extension(beh_ext, outdir / "fig5_extension.pdf")
    if state is not None:
        if cap is not None:
            fig_mechanism(state, cap, outdir / "fig2_mechanism.pdf")
        fig_decay(state[state.layer_frac.notna()], outdir / "fig3_depth.pdf")
        fig_attention(state, outdir / "fig4_attention.pdf")


if __name__ == "__main__":
    main()
