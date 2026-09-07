#!/usr/bin/env python3
"""Redraw fig_main for the rewrite with a legend that decodes both panels.

The submitted figure named the six checkpoints only in panel (b)'s inside
legend; panel (a) told repeated from control but left the colours undecoded.
Here one shared legend above both panels names every checkpoint, so neither
panel depends on the other to be read. The canvas grows from 65.5pt to 92pt
tall to pay for it -- affordable now that the theory section is gone.

Colours are family ramps (validated for CVD separation): blues = Llasa with
darker = larger, warm = Qwen3-TTS with darker = larger, green = XTTS-v2.

Run from the repo root:  python3 paper/ICASSP_paper/main/make_fig.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import figures as F  # noqa: E402  (analysis/figures.py)
from src.common.population import panel  # noqa: E402

COLOR = {"llasa1b": "#67BAE9", "llasa3b": "#2C86CB", "llasa8b": "#215196",
         "xtts2": "#00976D", "qwen06b": "#DD8F00", "qwen17b": "#B04500"}

# ICASSP 2027 requires >= 9pt type throughout the paper, in figures
# included. Every size below is 9.0, and the canvas grew from 92pt to
# fit it: taller for the two-row legend and the bigger tick band,
# wider gutters for the 9pt y-labels.
# 243.78pt is spconf's exact \columnwidth ((178-6)/2 mm), so the figure
# is included at width=\columnwidth with scale 1.0 and the 9pt type
# inside it stays 9pt on the page rather than shrinking to 8.9.
FIG_W_PT, FIG_H_PT = 243.78, 170.0
FS = 9.0
PT = 1.0 / 72.0
# The module's layout helpers read its canvas globals; align them with ours so
# _fit_ylabels nudges against the right canvas height.
F.FIG_W_PT, F.FIG_H_PT = FIG_W_PT, FIG_H_PT
F.COLOR.clear()
F.COLOR.update(COLOR)


def _enforce_min_type(ax) -> None:
    """Re-set every text size on `ax` to FS.

    `analysis/figures.py` draws panel (b) with 5-6pt type, and that module is
    shared with the frozen submitted build, so it is not edited: the sizes are
    overridden here after it has drawn.
    """
    ax.tick_params(labelsize=FS, pad=2.0, length=2.6, width=0.6)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontsize(FS)
    ax.xaxis.label.set_fontsize(FS)
    ax.yaxis.label.set_fontsize(FS)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 2.0
    for txt in ax.texts:
        txt.set_fontsize(FS)


def _rect(x, y, w, h):
    return [x / FIG_W_PT, y / FIG_H_PT, w / FIG_W_PT, h / FIG_H_PT]


def main() -> None:
    beh = pd.read_csv(REPO / "data/results/behavioural_ctc.csv")
    # The submitted figure drew every row; the paper's numbers never did. The
    # rows the statistics exclude (the judge-unmeasurable template, budget
    # truncations, degenerate audio) put -100% medians at low k that no
    # reported claim contains -- so the figure now draws the same population
    # every number comes from.
    beh, _ = panel(beh)
    period = json.loads((REPO / "data/results/period_ladder.json").read_text())
    out = Path(__file__).resolve().parent / "figs" / "fig_main.pdf"

    models = [m for m in F.ordered(beh.model.unique()) if m in F.ORDER]

    fig = plt.figure(figsize=(FIG_W_PT * PT, FIG_H_PT * PT))
    # Vertical budget: 15pt bottom (ticks + x label), 15pt top for the shared
    # legend's two rows, the rest is plot.
    bot, leg_band = 27.0, 48.0
    axh = FIG_H_PT - bot - leg_band
    # 9pt y-labels need more gutter than 6pt ones: at 32pt the (b) label
    # sat against (a)'s right spine.
    gut = (40.0, 40.0)
    axw = (FIG_W_PT - sum(gut) - 4.0) / 2.0
    x0 = [gut[0], gut[0] + axw + gut[1]]
    axes = [fig.add_axes(_rect(x, bot, axw, axh)) for x in x0]
    for ax, letter in zip(axes, "ab"):
        ax.tick_params(labelsize=FS, pad=2.0, length=2.6, width=0.6)
        ax.text(0.035, 0.955, f"({letter})", transform=ax.transAxes, ha="left",
                va="top", fontsize=FS)

    # ---- (a) exactly-right rate against k -----------------------------
    # The headline metric, not the median error: medians hug zero at k>=6
    # and made every model look near-perfect, which inverted the message.
    # Exact rates separate by ~76 points and match Tables 1-2 directly.
    beh = beh.copy()
    beh["exact"] = (beh.count_a == beh.k).astype(float)

    ax = axes[0]
    for m in models:
        for fam, ls, mk, al in (("word_rep", "-", "o", 1.0),
                                ("control_word", "--", "s", 0.55)):
            s = beh[(beh.model == m) & (beh.family == fam)]
            if s.empty:
                continue
            g = s.groupby("k")["exact"].mean().sort_index()
            ax.plot(g.index, 100 * g.values, ls, marker=mk, color=COLOR.get(m),
                    alpha=al, lw=1.1, ms=2.8, mew=0)
    ax.plot([], [], "k-", marker="o", ms=2.8, mew=0, lw=1.1, label="repeated")
    ax.plot([], [], "k--", marker="s", ms=2.8, mew=0, lw=1.1, alpha=0.55,
            label="control")
    ax.set_xscale("log", base=2)
    # The prose pivots on k=6 three times; ticks at every rung's octave plus a
    # light guide at 6 let the reader find it.
    F._int_ticks(ax, "x", [1, 2, 4, 8, 16, 32])
    ax.axvline(6, color="0.6", lw=0.5, ls=(0, (1.0, 1.6)), zorder=0)
    _int = F._int_ticks
    _int(ax, "y", [0, 50, 100])
    ax.set_ylim(-16, 124)
    ax.set_xlabel(r"repetitions $k$", fontsize=FS, labelpad=1.5)
    ax.set_ylabel("exactly right (%)", fontsize=FS, labelpad=2.0)
    # The two italic verdict labels this panel used to carry ("controls stay
    # exact" / "repeated collapses") cannot be set at 9pt inside an 84pt panel
    # without landing on the curves; the caption states both instead.

    # ---- (b) the period ladder ----------------------------------------
    ax = axes[1]
    F._period_panel(ax, period)
    _enforce_min_type(ax)
    # The checkpoint names move to the shared legend; keep the panel clean.
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # ---- shared checkpoint legend above both panels -------------------
    # Each handle carries the checkpoint's colour AND its panel-(b) dash and
    # marker, so the key decodes both panels at once (in (a) the colours
    # repeat with the arm styles of that panel's own legend).
    handles = []
    for i, m in enumerate(models):
        ls, mk = F.CURVE_STYLES[i % len(F.CURVE_STYLES)]
        full = dict(F.SHORT, qwen06b="Qwen3-TTS-0.6B", qwen17b="Qwen3-TTS-1.7B")
        handles.append(plt.Line2D([], [], color=COLOR[m], ls=ls, marker=mk,
                                  lw=1.1, ms=2.8, mew=0, label=full.get(m, m)))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=FS,
               frameon=False, handlelength=1.9, handletextpad=0.4,
               labelspacing=0.15, columnspacing=1.0, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.0))
    # Third row: which arm is which, in (a). Black, because in (a) the colours
    # are already spent on checkpoints.
    arm_handles = [
        plt.Line2D([], [], color="k", ls="-", marker="o", lw=1.1, ms=2.8,
                   mew=0, label="repeated"),
        plt.Line2D([], [], color="k", ls="--", marker="s", lw=1.1, ms=2.8,
                   mew=0, alpha=0.55, label="control"),
    ]
    fig.legend(handles=arm_handles, loc="upper center", ncol=2, fontsize=FS,
               frameon=False, handlelength=1.9, handletextpad=0.4,
               columnspacing=1.6, borderaxespad=0.0,
               bbox_to_anchor=(0.5, 1.0 - 2.0 * (FS + 3.0) / FIG_H_PT))

    F._fit_ylabels(fig, axes)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches=fig.bbox_inches, pad_inches=0.0)
    fig.savefig(out.with_suffix(".png"), dpi=400,
                bbox_inches=fig.bbox_inches, pad_inches=0.0)
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()
