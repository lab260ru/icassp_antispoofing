"""State-space dispersion: the boundary-free measurement of contraction.

Why not measure boundary distances directly. Theorem A is about the states at
repetition boundaries, so the obvious experiment is to locate those boundaries
and difference the states. We tried that. Locating them requires a monotone
text read-head, and the deep-layer attention centroid of these decoders is not
monotone (it advances on barely half of steps), so boundary estimates collapse
onto near-duplicate steps and the resulting distances are dominated by
localisation error. The failure is not incidental: the flatter the attention
over repeated spans -- the very effect Lemma B predicts -- the worse any
attention-based localiser gets.

What we measure instead. Theorem A(ii) says the states visited late in a
repeated generation become mutually indistinguishable. That is a statement about
the *spread* of the visited set, which needs no boundary labels at all. We tile
the trajectory into equal windows, measure the mean pairwise distance between
normalised states inside each window, and fit its decay. A contraction with
factor `q` per repetition shrinks the spread by `q` per repetition, so the fitted
per-window rate converts to a per-repetition `q_hat` by the ratio of window
length to repetition length.

States are normalised before differencing so the measurement is about direction,
not residual-stream magnitude; magnitude is near-constant along these
trajectories anyway (~130 for Llasa-1B), so this changes little, but it makes the
quantity comparable across models with different activation scales.
"""
from __future__ import annotations

import numpy as np

N_WINDOWS = 8
MIN_WINDOW = 8


def normalise_states(h: np.ndarray) -> np.ndarray:
    x = np.asarray(h, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def dispersion_profile(h: np.ndarray, n_windows: int = N_WINDOWS,
                       min_window: int = MIN_WINDOW) -> np.ndarray:
    """Mean pairwise distance among normalised states, per equal-length window.

    Returns an array of length `n_windows` with NaN where a window was too short
    to support a spread estimate.
    """
    hn = normalise_states(h)
    T = hn.shape[0]
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    out = np.full(n_windows, np.nan, dtype=np.float64)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        w = hn[a:b]
        if w.shape[0] < min_window:
            continue
        g = w @ w.T
        iu = np.triu_indices(w.shape[0], 1)
        out[i] = float(np.sqrt(np.maximum(2.0 - 2.0 * g[iu], 0.0)).mean())
    return out


def fit_dispersion(sigma: np.ndarray, k: int, n_windows: int = N_WINDOWS) -> dict:
    """Fit log sigma(w) = a + w * s and convert `s` to a per-repetition factor.

    One window spans `T/n_windows` steps and one repetition spans `T/k`, so a
    repetition is `n_windows/k` windows long and `log q = s * n_windows / k`.
    The conversion is what makes `q_hat` comparable across items with different
    `k` and different generation lengths.
    """
    s = np.asarray(sigma, dtype=np.float64)
    good = np.isfinite(s) & (s > 1e-6)
    if good.sum() < 4 or k < 2:
        return dict(q_hat=np.nan, slope=np.nan, r2=np.nan, n=int(good.sum()),
                    sigma_early=np.nan, sigma_late=np.nan, sigma_ratio=np.nan)
    w = np.arange(s.size, dtype=np.float64)[good]
    y = np.log(s[good])
    slope, intercept = np.polyfit(w, y, 1)
    pred = slope * w + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    vals = s[good]
    half = max(1, vals.size // 3)
    early, late = float(vals[:half].mean()), float(vals[-half:].mean())
    return dict(
        q_hat=float(np.exp(slope * n_windows / k)),
        slope=float(slope),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
        n=int(good.sum()),
        sigma_early=early, sigma_late=late,
        sigma_ratio=float(late / early) if early > 1e-9 else np.nan,
    )


def dispersion_stats(h: np.ndarray, k: int) -> dict:
    """Convenience: profile then fit."""
    return fit_dispersion(dispersion_profile(h), k)
