"""Locate repetition boundaries in a decoder's generation trajectory.

The theory is about the hidden state *at repetition boundaries*: the state the
decoder is in each time it has finished rendering one more copy of the repeated
phrase. To test it we need to say, for a given generated sequence, which step
index corresponds to boundary m.

We read that off the model's own text attention rather than off the audio. For
each occurrence j of the repeated unit we know its column in the text span
(exactly, from the tokenizer's character offsets); the step at which that column
receives peak attention is the step at which the decoder is rendering it. This
is self-contained — it needs no ASR, no alignment model, and no assumption that
the audio was rendered at a constant rate — and the attention trace is data we
capture anyway for the dilution measurement.

Fallback: if attention is unavailable or uninformative, boundaries are placed by
equal division of the generated sequence. That is unbiased with respect to the
hypothesis (it assumes constant tempo, which if anything *inflates* apparent
regularity), and every result carries which method produced it.
"""
from __future__ import annotations

import re

import numpy as np


def _char_to_column(offsets, c0: int) -> int | None:
    for ti, (a, b) in enumerate(offsets):
        if a <= c0 < b or (a == b == c0):
            return ti
    return None


def unit_columns(tokenizer, text: str, units: list[str]) -> list[int]:
    """Text-span column of the first token of each successive unit.

    `units` is the ordered list of words delimiting repetition boundaries: for a
    repeated item, k copies of the target; for its length-matched control, the k
    distinct fillers. Matching walks left to right and never rewinds, so both
    item types get boundaries by the identical procedure — which is what makes
    their decay rates comparable.
    """
    if not units:
        return []
    # `tokenizer` is an OffsetTokenizer (src/common/tokenizers.py), which hides
    # the difference between HF fast tokenizers and raw `tokenizers` files, and
    # applies each model's own text transform before tokenizing. Units must be
    # searched in the transformed string, not the raw one.
    text = tokenizer.prepare(text)
    offsets = tokenizer.offsets(text)
    cols: list[int] = []
    pos = 0
    for u in units:
        m = re.compile(rf"\b{re.escape(u)}\b", re.IGNORECASE).search(text, pos)
        if m is None:
            break
        col = _char_to_column(offsets, m.start())
        if col is None:
            break
        cols.append(col)
        pos = m.end()
    return cols


def occurrence_columns(tokenizer, text: str, unit: str, k: int) -> list[int]:
    """Columns of the first `k` occurrences of a single repeated `unit`."""
    return unit_columns(tokenizer, text, [unit] * k)


def read_head(attn: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Attention centroid per generation step: where in the text the decoder is.

    Per-column argmax is the obvious way to locate an occurrence, and it fails
    here for exactly the reason the paper is about: when the k spans are
    near-identical, their attention columns are near-identical too, so the argmax
    over steps collapses to the same step for every occurrence. The centroid
    `sum_j j p(t,j)` degrades gracefully instead --- it still advances through the
    text even when no single column dominates.
    """
    a = np.asarray(attn, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return np.array([], dtype=np.float32)
    p = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-8)
    c = (p * np.arange(a.shape[1], dtype=np.float32)).sum(axis=1)
    if smooth > 1 and c.size > smooth:
        k = np.ones(smooth, dtype=np.float32) / smooth
        c = np.convolve(c, k, mode="same")
    return c


def boundary_steps_from_attention(attn: np.ndarray, cols: list[int]) -> np.ndarray:
    """First step at which the read head has reached each occurrence column.

    Monotone by construction, and robust to the flat attention profiles that
    repeated text produces.
    """
    if attn.ndim != 2 or not cols:
        return np.array([], dtype=int)
    c = read_head(attn)
    if c.size == 0:
        return np.array([], dtype=int)
    # The centroid is pulled toward the middle of the attended mass, so it spans
    # a compressed range; rescale the target columns onto the range it actually
    # traverses rather than assuming it reaches the raw column index.
    lo_c, hi_c = float(np.percentile(c, 2)), float(np.percentile(c, 98))
    if hi_c - lo_c < 1e-6:
        return np.array([], dtype=int)
    span = max(max(cols), 1)
    targets = [lo_c + (hi_c - lo_c) * (col / span) for col in cols]
    steps: list[int] = []
    t = 0
    for tgt in targets:
        hits = np.nonzero(c[t:] >= tgt)[0]
        if hits.size == 0:
            break
        t = int(hits[0]) + t
        steps.append(t)
    return np.asarray(steps, dtype=int)


def boundary_steps_uniform(n_steps: int, k: int) -> np.ndarray:
    """Equal-division fallback: assume a constant rendering tempo."""
    if k < 2 or n_steps < k:
        return np.array([], dtype=int)
    edges = np.linspace(0, n_steps - 1, k + 1)
    return ((edges[:-1] + edges[1:]) / 2).astype(int)


def boundary_distances(hidden: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """d_m = ||h(t_{m+1}) - h(t_m)||_2 per probe layer.

    Returns [n_gaps, n_probe_layers]. Under the contraction hypothesis these
    decay geometrically in m at rate q.
    """
    if steps.size < 3:
        return np.zeros((0, hidden.shape[1]), dtype=np.float32)
    h = hidden[steps].astype(np.float32)          # [k, P, d]
    return np.linalg.norm(np.diff(h, axis=0), axis=-1)  # [k-1, P]


def fit_geometric(d: np.ndarray, min_points: int = 4) -> dict:
    """Fit log d_m = a + m log q by least squares.

    Returns q_hat (the per-repetition contraction factor), the fit quality, and
    the implied horizon. q_hat < 1 is the contracting regime of Theorem A;
    q_hat >= 1 is the non-contracting regime, where the theorem's conclusion does
    not apply and counting can in principle persist.
    """
    d = np.asarray(d, dtype=np.float64)
    good = np.isfinite(d) & (d > 0)
    if good.sum() < min_points:
        return dict(q_hat=np.nan, r2=np.nan, n=int(good.sum()), d1=np.nan)
    m = np.arange(d.size)[good]
    y = np.log(d[good])
    slope, intercept = np.polyfit(m, y, 1)
    pred = slope * m + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return dict(q_hat=float(np.exp(slope)), r2=float(r2), n=int(good.sum()),
                d1=float(np.exp(intercept)))


def horizon_scale(q_hat: float) -> float:
    """The theory's depth predictor, `1 / log(1/q)`.

    Theorem A gives `N* = log(mu / (2LC)) / log q`, i.e. `N*` is proportional to
    `1 / log(1/q)` with a constant set by the readout margin `mu`, its Lipschitz
    constant `L`, and the orbit scale `C`. None of those three is separately
    observable from activations, but their *product* is a single unknown shared
    across items of one model. Reporting `1/log(1/q_hat)` therefore states the
    theory's prediction with exactly one free constant, which is what the
    cross-model regression in `analysis/horizon_fit.py` fits and tests.
    """
    if not (np.isfinite(q_hat) and 0 < q_hat < 1):
        return float("inf")
    return float(1.0 / np.log(1.0 / q_hat))


def horizon(q_hat: float, d1: float, floor: float) -> float:
    """Absolute horizon under an explicit choice of decision margin.

    Instantiates Theorem A(iv) with the fitted orbit scale `d1` in the role of
    `2LC` and a measured `floor` in the role of the margin `mu`. Reported as a
    secondary, assumption-laden number; the primary analysis uses
    `horizon_scale`, which needs no such choice.
    """
    if not (np.isfinite(q_hat) and 0 < q_hat < 1):
        return float("inf")
    if not (np.isfinite(d1) and np.isfinite(floor)) or d1 <= 0 or floor <= 0:
        return float("nan")
    if floor >= d1:
        return 0.0
    return float(np.log(floor / d1) / np.log(q_hat))
