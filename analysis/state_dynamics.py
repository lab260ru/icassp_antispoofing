#!/usr/bin/env python3
"""Measure the repetition-boundary state dynamics (predictions P1, P3, P4).

For every instrumented generation:
  P1  boundary distances d_m and their geometric fit -> q_hat, R^2, horizon N*
  P3  attention mass per repeated occurrence, and text-attention entropy vs k
  P4  spectral proxies of the generated trajectory: effective rank, spectral
      slope alpha, Kirchhoff index -- the observables of the companion Whisper
      paper, carried over so the two studies are directly comparable

The noise floor that turns q_hat into a horizon is measured, not assumed: it is
the median boundary distance on the *control* items, which have the same length
and carrier but no repetition. That is the scale below which a state difference
is indistinguishable from ordinary rendering variation.

Usage:
  python analysis/state_dynamics.py --models llasa1b llasa3b --out data/results/state.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402

from common.boundaries import (  # noqa: E402
    boundary_distances, boundary_steps_from_attention, boundary_steps_uniform,
    fit_geometric, horizon, horizon_scale, unit_columns,
)
from common.dispersion import dispersion_profile, fit_dispersion  # noqa: E402
from common.registry import BY_KEY, DATA_ROOT  # noqa: E402


MAX_SVD_STEPS = 512   # SVD cost is O(min(T,d)^2 max(T,d)); cap the step axis


def spectral_proxies(h: np.ndarray, min_steps: int = 24) -> dict:
    """Effective rank, spectral slope and Kirchhoff index of a trajectory.

    `h` is [T, d] for one probe layer. These are the observables used by the
    companion Whisper study; computing them the same way here is what lets the
    two papers' tables be read side by side.

    Trajectories longer than `MAX_SVD_STEPS` are uniformly subsampled. The
    proxies are properties of the spectrum's shape, which subsampling leaves
    intact, and the alternative is an O(T d^2) SVD per layer per item.
    """
    out = dict(n_eff=np.nan, alpha=np.nan, log_kf=np.nan)
    if h.shape[0] < min_steps:
        return out
    if h.shape[0] > MAX_SVD_STEPS:
        h = h[np.linspace(0, h.shape[0] - 1, MAX_SVD_STEPS).astype(int)]
    x = h.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return out
    s = s[s > 0]
    if s.size < 4:
        return out
    p = s / s.sum()
    out["n_eff"] = float(np.exp(-(p * np.log(p)).sum()))
    # slope of the spectrum tail in log-log coordinates
    lo, hi = 2, max(6, int(0.8 * s.size))
    idx = np.arange(lo, min(hi, s.size))
    if idx.size >= 4:
        out["alpha"] = float(-np.polyfit(np.log(idx + 1.0), np.log(s[idx]), 1)[0])
    lam = s ** 2
    lam = lam[lam > 1e-10 * lam.max()]
    out["log_kf"] = float(np.log10((1.0 / lam).sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/state.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spectral-layers", type=int, default=4,
                    help="how many probe layers get the SVD-based proxies")
    args = ap.parse_args()

    stim = {}
    for line in open(args.stimuli):
        it = json.loads(line)
        stim[it["item_id"]] = it

    from common import offset_tok as tokmod

    rows: list[dict] = []
    for model in args.models:
        spec = BY_KEY[model]
        act_dir = Path(DATA_ROOT) / "activations" / model
        if not act_dir.exists():
            print(f"[{model}] no activations, skipping")
            continue
        tok = tokmod.load(spec)
        if tok is None:
            print(f"[{model}] no offset tokenizer; uniform boundaries only")

        files = sorted(act_dir.glob(f"*_s{args.seed}.npz"))
        print(f"[{model}] {len(files)} instrumented items", flush=True)
        for f in files:
            item_id = f.stem.rsplit("_s", 1)[0]
            it = stim.get(item_id)
            if it is None:
                continue
            try:
                z = np.load(f)
            except Exception:  # noqa: BLE001
                continue
            hidden = z["hidden"]                       # [T, P, d]
            probes = z["probe_layers"].tolist()
            T = hidden.shape[0]
            attn_keys = sorted([k for k in z.files if k.startswith("attn_l")],
                               key=lambda s: int(s[6:]))
            # deepest probed layer: the one closest to the readout
            attn = z[attn_keys[-1]] if attn_keys else None

            k = int(it["k"])
            units = it.get("boundary_units") or [it["target_unit"]] * k
            cols: list[int] = []
            if tok is not None:
                try:
                    cols = unit_columns(tok, it["text"], units)
                except Exception:  # noqa: BLE001
                    cols = []

            method = "attention"
            steps = (boundary_steps_from_attention(attn, cols)
                     if (attn is not None and cols) else np.array([], dtype=int))
            if steps.size < 3:
                steps = boundary_steps_uniform(T, k)
                method = "uniform"
            d = (boundary_distances(hidden, steps) if steps.size >= 3
                 else np.zeros((0, hidden.shape[1]), dtype=np.float32))
            base = dict(model=model, item_id=item_id, family=it["family"],
                        template=it["template"], k=k, n_steps=T,
                        n_boundaries=int(steps.size), method=method,
                        control_of=it.get("control_of"))

            # --- P3: attention dilution ---------------------------------
            if attn is not None and attn.size:
                a = attn.astype(np.float32)
                tot = a.sum(axis=1, keepdims=True)
                p = a / np.maximum(tot, 1e-8)
                ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
                base["attn_entropy"] = float(np.median(ent))
                base["attn_text_mass"] = float(np.median(tot))
                if cols:
                    per = [float(np.median(a[:, c])) for c in cols if c < a.shape[1]]
                    base["attn_per_occurrence"] = float(np.mean(per)) if per else np.nan
                    base["n_occ_cols"] = len(per)

            # --- P1 + P4, per probe layer -------------------------------
            # The geometric fit is cheap and wanted at every probe layer; the
            # spectral proxies cost an SVD each, so they run on an evenly
            # spaced subset (P4 is a cross-check against the companion ASR
            # study, not the load-bearing measurement).
            n_sp = max(1, min(args.spectral_layers, len(probes)))
            sp_idx = set(np.linspace(0, len(probes) - 1, n_sp).astype(int).tolist())
            for pi, layer in enumerate(probes):
                # primary: boundary-free dispersion decay (see common/dispersion)
                disp = fit_dispersion(dispersion_profile(hidden[:, pi, :]), k)
                # secondary: the boundary-difference estimator, kept for the
                # supplementary comparison of the two operationalisations
                bfit = (fit_geometric(d[:, pi]) if d.size
                        else dict(q_hat=np.nan, r2=np.nan, n=0, d1=np.nan))
                sp = (spectral_proxies(hidden[:, pi, :]) if pi in sp_idx
                      else dict(n_eff=np.nan, alpha=np.nan, log_kf=np.nan))
                rows.append(dict(
                    **base, layer=int(layer), layer_frac=pi / max(len(probes) - 1, 1),
                    q_hat=disp["q_hat"], r2=disp["r2"], slope=disp["slope"],
                    sigma_early=disp["sigma_early"], sigma_late=disp["sigma_late"],
                    sigma_ratio=disp["sigma_ratio"],
                    q_bnd=bfit["q_hat"], r2_bnd=bfit["r2"], d1=bfit["d1"],
                    d_median=float(np.median(d[:, pi])) if d.size else np.nan,
                    **sp,
                ))

    if not rows:
        print("no rows produced")
        return

    # --- noise floor from controls, per (model, layer) ------------------
    floors: dict[tuple, float] = {}
    for key in {(r["model"], r["layer"]) for r in rows}:
        vals = [r["d_median"] for r in rows
                if (r["model"], r["layer"]) == key and r["family"] == "control_word"
                and np.isfinite(r["d_median"])]
        floors[key] = float(np.median(vals)) if vals else np.nan
    for r in rows:
        fl = floors.get((r["model"], r["layer"]), np.nan)
        r["floor"] = fl
        r["horizon_scale"] = horizon_scale(r["q_hat"])
        r["horizon"] = horizon(r["q_hat"], r["d1"], fl)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols_out = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out}")

    # --- console summary -----------------------------------------------
    for model in args.models:
        sub = [r for r in rows if r["model"] == model and r["family"] == "word_rep"
               and np.isfinite(r["q_hat"]) and r["r2"] > 0.5]
        ctl = [r for r in rows if r["model"] == model and r["family"] == "control_word"
               and np.isfinite(r["q_hat"]) and r["r2"] > 0.5]
        if not sub:
            continue
        deep = [r for r in sub if r["layer_frac"] > 0.6]
        deep_c = [r for r in ctl if r["layer_frac"] > 0.6]
        q = np.median([r["q_hat"] for r in deep]) if deep else np.nan
        qc = np.median([r["q_hat"] for r in deep_c]) if deep_c else np.nan
        hz = np.median([r["horizon"] for r in deep if np.isfinite(r["horizon"])]) if deep else np.nan
        print(f"\n[{model}] deep layers, R2>0.5")
        print(f"  repeated q_hat = {q:.3f}   (n={len(deep)})")
        print(f"  control  q_hat = {qc:.3f}   (n={len(deep_c)})")
        print(f"  implied horizon N* = {hz:.1f}")
        print(f"  boundary method: {set(r['method'] for r in sub)}")


if __name__ == "__main__":
    main()
