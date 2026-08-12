#!/usr/bin/env python3
"""What does the decoder's stop decision actually read?

The formal bound said the stop head *cannot* read the repetition count past a
horizon. Its premise was never measured and its one directional prediction came
back sign-reversed, so it is not the explanation. The replacement question is
narrower and measurable: at each decoding step, what does the EOS logit track?

The hypothesis under test is that it tracks **elapsed duration / position**, not
**count**. That would explain a result already in hand: handing F5-TTS the correct
total duration repairs most of the k<12 deficit (43.3% -> 76.7% exact) and does
exactly nothing at k>=12 (+0.0 pts, n=60). At low k, duration and count are nearly
interchangeable, so a duration-driven stop policy gets the count right by accident;
at high k they decouple and it cannot.

--------------------------------------------------------------------------------
PRE-COMMITTED INTERPRETATION.  Written before any number below was looked at.
--------------------------------------------------------------------------------

*The target.* log P(EOS | state) at every generated step, recovered exactly rather
than approximated. The instrumented pass stored the post-final-norm hidden state,
which is what `lm_head` consumes, so the logits come back from one matrix product
without re-running the model; the reconstruction is checked against the
`entropy`/`top1` arrays that pass computed from the real logits
(`verify_reconstruction`). Raw EOS logit is carried as a secondary target because
log P is shift-corrected and the raw logit is not.

*The predictor blocks.* Three, each given a flexible basis so no block loses by
being modelled more crudely than its rivals:

  DUR   elapsed position (tokens generated so far; at 50 Hz this is elapsed
        seconds up to a constant, so "duration" and "position" are ONE predictor
        here and are never counted twice). Restricted cubic spline, 5 knots.
  CNT   the count as the states carry it: `probe_count.py`'s ridge probe
        (alpha=1, dual solve, layer chosen by leave-one-template-out R^2 on the
        item-level task) applied to the per-step state, always out of fold.
        Restricted cubic spline, 5 knots.
  K     the requested count, i.e. the conditioning the text supplies. Entered as
        dummies -- one free parameter per k level -- which is the most generous
        treatment available and therefore the conservative one, since our own
        hypothesis is the one that says K should not matter much.

A second count read-out, `CNT2`, is fitted as a robustness arm and declared here
rather than after the fact: the same ridge probe trained directly on *per-step*
states instead of on terminal-window means, which removes the train/apply
mismatch in CNT. It is reported separately and never silently swapped in, because
a probe trained per-step can exploit position to predict k and would then be a
disguised duration predictor -- which is precisely the confusion this script
exists to avoid.

*What is deliberately NOT in the decomposition.* "True count rendered so far" is
not derivable. Locating repetition boundaries needs a monotone text read-head and
this decoder does not have one (implementation-notes s4: the deep-layer attention
centroid advances on ~51% of steps; the pooled boundary fit returns q~1.00,
R^2~0). Every closed-form substitute is an oracle: k * t / T_item, or fraction
elapsed t / T_item, uses the realised stop time T_item, which is the very event
the regression is trying to explain -- it equals 1.0 at the stop step by
construction. Such a predictor would post a large R^2 and mean nothing. It is
computed and reported as an explicitly-labelled ORACLE CEILING and never enters
the commonality analysis.

*Population.* Repeated (`word_rep`) items, seed 0, k in {2..32} from the main
ladder plus {48,64,96,128} from the instrumented extension. `control_word` is
carried as a secondary arm. `hit_cap` items -- generation stopped on our token
budget, not on the model's stop decision -- are excluded from the primary
analysis (population rule 4) and reported as a robustness arm, because an item
that never stopped has no stop decision to explain. Steps are weighted 1/T_item
so an 8192-step runaway does not outvote a 271-step item; unweighted is reported
alongside.

*What would support "the stop head reads duration, not count".*
  - DUR unique Delta-R^2 is the largest of the three blocks in BOTH k bands, and
  - CNT unique Delta-R^2 is small (< 0.05) in both bands, and
  - K's unique contribution shrinks from low k to high k -- the duration target
    stops growing with the requested count exactly where the intervention stops
    working.

*What would support the opposite ("it reads the count").*
  - CNT unique Delta-R^2 comparable to or exceeding DUR's, in either band; or
  - CNT unique staying large at high k, i.e. the count still driving the stop
    decision in the regime where behaviour collapses.

*What would make the result UNINTERPRETABLE, declared in advance.*
  - COLLINEARITY. If |Pearson r| between elapsed position and the probe-decoded
    count exceeds 0.95, or the first canonical correlation between the DUR and
    CNT bases exceeds 0.95, in the analysis population, the two predictors are
    one predictor and no decomposition of them is meaningful. Threshold fixed at
    0.95 before looking. Between 0.90 and 0.95 the decomposition is reported but
    called weakly identified.
  - POWER. The positive control plants a count-only relationship on the REAL
    design matrix. If the machinery cannot then recover CNT unique R^2 >= 0.20
    when count is the sole true driver, the design cannot detect a count-driven
    stop head even if one exists, and a null is uninformative rather than
    evidence.

*Amendment, logged before any real number was read.* A dry run of the control
routine on a purely SYNTHETIC design with r(pos, count) = 0.85 recovered unique
R^2 of only 0.06-0.14 for the block that was actually planted, while assigning
+0.000 to the blocks that were not. Attribution was correct; the magnitude was
far below the 0.20 floor. The floor was fixed in advance against no calibration
at all, so it is kept as the criterion the verdict uses -- that is the least
favourable reading -- and a second DOMINANCE criterion is reported beside it:
the planted block's unique R^2 must exceed every unplanted block's by more than
0.02, with unplanted blocks below 0.02. Both are printed and stored; neither was
chosen after seeing a real EOS logit.

*Positive control.* Four synthetic targets built on the real design matrix (real
positions, real k, real probe-count values) with known planted structure:
duration-only, count-only, k-only, and a 50/50 duration+k mixture. The
decomposition must attribute each to its planted block. This checks the
commonality arithmetic AND the design's power at the same time.

*Reporting.* At the least favourable reading. Every n is printed. Where a
predictor is an oracle it is labelled one.

Usage:
  python analysis/stop_head.py --models llasa1b llasa8b --gpu 3
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402

from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices  # noqa: E402

SPEECH_END = "<|SPEECH_GENERATION_END|>"
TAIL_FRAC = 1.0 / 3.0
COLLINEAR_FATAL = 0.95
COLLINEAR_WARN = 0.90
CONTROL_MIN_RECOVERY = 0.20
CNT_SMALL = 0.05
CACHE = Path(DATA_ROOT) / "stop_head_cache"

STIMULI = ["data/stimuli/stimuli.jsonl",
           "data/stimuli/stimuli_ext_instr.jsonl",
           "data/stimuli/stimuli_ext_instr_ctl.jsonl"]


# ---------------------------------------------------------------- model head --

def snapshot(hf_id: str) -> str:
    org, name = hf_id.split("/")
    hits = glob.glob(f"{os.environ['HF_HOME']}/hub/models--{org}--{name}/snapshots/*")
    if not hits:
        raise FileNotFoundError(hf_id)
    return hits[0]


def load_head(hf_id: str):
    """`lm_head.weight`, the final RMSNorm gain and the EOS id, without loading
    the model.

    We need one matrix product, not a forward pass: the stored hidden state at
    the last probe layer is already post-final-norm (transformers appends
    `self.norm(h)` as the last element of `hidden_states`), which is exactly what
    `lm_head` consumes. Reading two tensors out of the safetensors shards keeps
    this off the GPUs the sibling jobs are using.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    snap = snapshot(hf_id)
    cfg = json.loads(Path(snap + "/config.json").read_text())
    key = ("model.embed_tokens.weight" if cfg.get("tie_word_embeddings", False)
           else "lm_head.weight")
    idx_path = Path(snap + "/model.safetensors.index.json")
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
        shard = lambda k: snap + "/" + wmap[k]                        # noqa: E731
    else:
        shard = lambda k: snap + "/model.safetensors"                 # noqa: E731
    with safe_open(shard(key), "pt") as f:
        W = f.get_tensor(key)
    with safe_open(shard("model.norm.weight"), "pt") as f:
        gain = f.get_tensor("model.norm.weight").float().numpy()
    eos = int(AutoTokenizer.from_pretrained(snap).convert_tokens_to_ids(SPEECH_END))
    return W, gain, eos, float(cfg.get("rms_norm_eps", 1e-5)), int(cfg["num_hidden_layers"])


# ---------------------------------------------------------- per-step scalars --

def item_scalars(h_last: np.ndarray, Wg, w_eos: np.ndarray, device: str,
                 chunk: int = 512):
    """(eos_logit, eos_logprob, entropy, top1) for every step of one item."""
    import torch

    x32 = h_last.astype(np.float32)
    logit = x32 @ w_eos                                   # exact, fp32, one row
    xg = torch.from_numpy(x32).to(device=device, dtype=torch.float16)
    lse, ent, top1 = [], [], []
    with torch.no_grad():
        for i in range(0, xg.shape[0], chunk):
            lg = (xg[i:i + chunk] @ Wg.T).float()
            lse.append(torch.logsumexp(lg, dim=-1).cpu().numpy())
            lp = torch.log_softmax(lg, dim=-1)
            pr = lp.exp()
            ent.append((-(pr * lp).sum(-1)).cpu().numpy())
            top1.append(pr.max(-1).values.cpu().numpy())
            del lg, lp, pr
    lse = np.concatenate(lse)
    return logit, logit - lse, np.concatenate(ent), np.concatenate(top1)


def logit_lens_eos(h: np.ndarray, gain: np.ndarray, w_eos: np.ndarray,
                   eps: float) -> np.ndarray:
    """EOS logit read off every probe layer through the final norm + unembedding.

    Layers before the last are stored pre-norm, so the norm is applied here; the
    last probe layer is already post-norm and passes through untouched. Looped
    over layers because materialising [T, P, d] in fp32 costs a gigabyte at
    T=8192.
    """
    T, P, _ = h.shape
    out = np.empty((T, P), dtype=np.float32)
    for p in range(P):
        x = h[:, p, :].astype(np.float32)
        if p == P - 1:
            out[:, p] = x @ w_eos
        else:
            rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
            out[:, p] = ((x / rms) * gain[None, :]) @ w_eos
    return out


# ------------------------------------------------------------- count probing --

def ridge_dual(Xtr: np.ndarray, ytr: np.ndarray, alpha: float = 1.0):
    """probe_count.py's estimator, returned as an explicit (w, mu, ybar)."""
    mu = Xtr.mean(0, keepdims=True)
    A = Xtr - mu
    ybar = float(ytr.mean())
    K = A @ A.T + alpha * np.eye(A.shape[0])
    w = A.T @ np.linalg.solve(K, ytr - ybar)
    return w, mu[0], ybar


def l2norm(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


# ------------------------------------------------------------------- bases ----

def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell's parameterisation."""
    k = np.asarray(knots, dtype=float)
    K = len(k)
    if K < 3:
        return x[:, None]
    cols = [x]
    den = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        t = (np.maximum(x - k[j], 0) ** 3
             - np.maximum(x - k[K - 2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[K - 2])
             + np.maximum(x - k[-1], 0) ** 3 * (k[K - 2] - k[j]) / (k[-1] - k[K - 2]))
        cols.append(t / den)
    return np.column_stack(cols)


def spline_basis(x: np.ndarray, n_knots: int = 5) -> np.ndarray:
    q = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_knots)))
    return x[:, None] if len(q) < 3 else rcs(x, q)


def dummies(x: np.ndarray) -> np.ndarray:
    lev = np.unique(x)
    if len(lev) < 2:
        return np.zeros((len(x), 0))
    return np.column_stack([(x == v).astype(float) for v in lev[1:]])


def standardise(B: np.ndarray) -> np.ndarray:
    if B.shape[1] == 0:
        return B
    sd = B.std(0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (B - B.mean(0, keepdims=True)) / sd


# ------------------------------------------------------------ weighted OLS ----

def wr2(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6) -> float:
    """Weighted R^2 of an OLS fit with intercept. Empty X -> 0."""
    ybar = float((w * y).sum() / w.sum())
    tot = float((w * (y - ybar) ** 2).sum())
    if tot < 1e-12 or X.shape[1] == 0:
        return 0.0
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(y)), X]) * sw[:, None]
    b = y * sw
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
    return float(1.0 - float(((b - A @ coef) ** 2).sum()) / tot)


def commonality(blocks: dict, y: np.ndarray, w: np.ndarray) -> dict:
    """Full commonality decomposition plus each block's unique Delta-R^2."""
    names = list(blocks)
    subs = {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            subs["+".join(combo)] = wr2(np.column_stack([blocks[c] for c in combo]),
                                        y, w)
    full = subs["+".join(names)]
    out = {"r2_full": full, "r2_alone": {n: subs[n] for n in names}, "unique": {}}
    for n in names:
        rest = [m for m in names if m != n]
        out["unique"][n] = full - (subs["+".join(rest)] if rest else 0.0)
    if len(names) == 3:
        a, b, c = names
        g = lambda *ks: subs["+".join(ks)]                            # noqa: E731
        out["common_pairs"] = {
            f"{a}&{b}": g(a, c) + g(b, c) - g(a, b, c) - g(c),
            f"{a}&{c}": g(a, b) + g(b, c) - g(a, b, c) - g(b),
            f"{b}&{c}": g(a, b) + g(a, c) - g(a, b, c) - g(a),
        }
        out["common_all3"] = (g(a) + g(b) + g(c) - g(a, b) - g(a, c) - g(b, c)
                              + g(a, b, c))
    out["subsets"] = subs
    return out


def canon_corr(A: np.ndarray, B: np.ndarray) -> float:
    """First canonical correlation between two centred bases."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    qa = np.linalg.qr(A - A.mean(0, keepdims=True))[0]
    qb = np.linalg.qr(B - B.mean(0, keepdims=True))[0]
    return float(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False)[0], 0, 1))


def loto_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray, grp: np.ndarray) -> float:
    """Leave-one-template-out weighted R^2: the house cross-validation."""
    if X.shape[1] == 0:
        return 0.0
    pred = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te, tr = grp == g, grp != g
        if tr.sum() < 50 or te.sum() < 5:
            continue
        sw = np.sqrt(w[tr])
        A = np.column_stack([np.ones(int(tr.sum())), X[tr]]) * sw[:, None]
        coef = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]),
                               A.T @ (y[tr] * sw))
        pred[te] = np.column_stack([np.ones(int(te.sum())), X[te]]) @ coef
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return float("nan")
    ybar = float((w[ok] * y[ok]).sum() / w[ok].sum())
    tot = float((w[ok] * (y[ok] - ybar) ** 2).sum())
    res = float((w[ok] * (y[ok] - pred[ok]) ** 2).sum())
    return float(1.0 - res / tot) if tot > 1e-12 else float("nan")


# ------------------------------------------------------------------ loading ---

def load_stimuli() -> dict:
    stim = {}
    for f in STIMULI:
        p = REPO / f
        if p.exists():
            for line in open(p):
                it = json.loads(line)
                stim[it["item_id"]] = it
    return stim


def load_meta(model: str) -> dict:
    out = {}
    for line in open(Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"):
        try:
            r = json.loads(line)
        except Exception:                                            # noqa: BLE001
            continue
        if r.get("seed") == 0:
            out[r["item_id"]] = r
    return out


def verify_reconstruction(z, ent, top1) -> dict:
    """Our logits against the ones the instrumented pass computed from the model.

    The stored `entropy`/`top1` came from `logits[plen-1:-1]` -- the distributions
    that PRODUCED each generated token -- while the stored hidden states are
    `hidden_states[plen:]`, the states left AFTER each generated token. They are
    offset by exactly one step: our step t is their step t+1. That offset is a
    real trap, so it is asserted here rather than assumed, and the unshifted
    error is reported next to it.
    """
    ref_e, ref_t = z["entropy"], z["top1"]
    n = min(len(ent) - 1, len(ref_e) - 1)
    if n < 8:
        return {}
    return dict(ent_maxabs=float(np.abs(ent[:n] - ref_e[1:n + 1]).max()),
                top1_maxabs=float(np.abs(top1[:n] - ref_t[1:n + 1]).max()),
                ent_maxabs_noshift=float(np.abs(ent[:n] - ref_e[:n]).max()))


def obj_array(seq):
    a = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        a[i] = v
    return a


def fit_count_probes(rows, act_dir, P, step_stride=8):
    """Per-family: choose a layer, then decode the count out of fold, per step.

    Two probes, both `probe_count.py`'s estimator (ridge alpha=1, dual solve,
    leave-one-template-out):
      cnt   -- trained on terminal-window means, the published probe, applied to
               single-step states;
      cnt2  -- trained directly on (subsampled) single-step states, which removes
               the train/apply mismatch but can launder position into a "count".
    """
    info = {}
    for fam in ("word_rep", "control_word"):
        sel = [r for r in rows if r["family"] == fam]
        if len(sel) < 8:
            continue
        Xl = np.stack([r["tailmean"] for r in sel])                  # [n, P, d]
        y = np.log2(np.array([r["k"] for r in sel], dtype=float))
        tm = np.array([r["template"] for r in sel])
        best, best_r2, by_layer = 0, -9.0, []
        for p in range(P):
            pred = np.full(len(y), np.nan)
            for t in np.unique(tm):
                te, tr = tm == t, tm != t
                if tr.sum() < 6 or te.sum() < 1:
                    continue
                w_, mu_, yb_ = ridge_dual(Xl[tr, p, :].astype(np.float64), y[tr])
                pred[te] = (Xl[te, p, :].astype(np.float64) - mu_) @ w_ + yb_
            ok = np.isfinite(pred)
            r2 = (1 - ((y[ok] - pred[ok]) ** 2).sum()
                  / max(((y[ok] - y[ok].mean()) ** 2).sum(), 1e-9)) if ok.sum() > 3 else -9.0
            by_layer.append(float(r2))
            if r2 > best_r2:
                best_r2, best = float(r2), p
        info[fam] = dict(layer=int(best), loto_r2=best_r2, by_layer=by_layer,
                         n_items=len(sel))

        # One disk pass. The activations live on a shared spinning disk and each
        # re-read of this family costs ~10 GB, so the per-step features for the
        # chosen layer are held in RAM (sum(T) x d x 4 bytes, ~1 GB) rather than
        # loaded once to train the step-level probe and again to apply it.
        feats = {}
        step_X, step_y, step_t = [], [], []
        for r in sel:
            f = l2norm(np.load(act_dir / f"{r['item']}_s0.npz")["hidden"]
                       [:, best, :].astype(np.float32))
            feats[r["item"]] = f
            step_X.append(f[::step_stride])
            step_y.append(np.full(len(f[::step_stride]), np.log2(r["k"])))
            step_t.append(np.full(len(f[::step_stride]), r["template"]))
        step_X = np.concatenate(step_X).astype(np.float64)
        step_y = np.concatenate(step_y)
        step_t = np.concatenate(step_t)

        for t in np.unique(tm):
            tr = tm != t
            if tr.sum() < 6:
                continue
            w1, mu1, yb1 = ridge_dual(Xl[tr, best, :].astype(np.float64), y[tr])
            m = step_t != t
            # dual solve is n-by-n; subsample the step training set to keep it so
            idx = np.where(m)[0]
            if len(idx) > 1200:
                idx = idx[np.linspace(0, len(idx) - 1, 1200).astype(int)]
            w2, mu2, yb2 = ridge_dual(step_X[idx], step_y[idx])
            for r in [s for s in sel if s["template"] == t]:
                feat = feats[r["item"]].astype(np.float64)
                r["cnt"] = ((feat - mu1) @ w1 + yb1).astype(np.float32)
                r["cnt2"] = ((feat - mu2) @ w2 + yb2).astype(np.float32)
        del feats
    return info


def build_cache(model: str, gpu: int, force: bool = False) -> dict:
    """Per-step EOS scalars and per-step decoded counts for every instrumented item."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / f"{model}.npz"
    if out_path.exists() and not force:
        d = np.load(out_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import torch

    spec = BY_KEY[model]
    stim, meta = load_stimuli(), load_meta(model)
    W, gain, eos, eps, n_layers = load_head(spec.hf_id)
    probes = probe_layer_indices(n_layers, spec.probe_layers)
    w_eos = W[eos].float().numpy()
    Wg = W.to(device=f"cuda:{gpu}", dtype=torch.float16)
    del W

    act_dir = Path(DATA_ROOT) / "activations" / model
    rows, checks = [], []
    for f in sorted(act_dir.glob("*_s0.npz")):
        iid = f.stem[:-3]
        it, m = stim.get(iid), meta.get(iid)
        if it is None or m is None or it["family"] not in ("word_rep", "control_word"):
            continue
        if it["k"] < 2:
            continue
        z = np.load(f)
        h = z["hidden"]
        T = h.shape[0]
        if T < 24 or h.shape[1] != len(probes):
            continue
        logit, logprob, ent, top1 = item_scalars(h[:, -1, :], Wg, w_eos, f"cuda:{gpu}")
        chk = verify_reconstruction(z, ent, top1)
        if chk:
            chk["item"] = iid
            checks.append(chk)
        w = max(4, int(T * TAIL_FRAC))
        rows.append(dict(item=iid, k=int(it["k"]), family=it["family"],
                         template=it["template"], T=T, hit_cap=bool(m["hit_cap"]),
                         logit=logit.astype(np.float32),
                         logprob=logprob.astype(np.float32),
                         lens=logit_lens_eos(h, gain, w_eos, eps),
                         tailmean=l2norm(h[-w:].astype(np.float32).mean(0))))
        del h, z
        print(f"  [{model}] {iid:24s} T={T:5d} k={rows[-1]['k']:3d} "
              f"cap={int(rows[-1]['hit_cap'])}", flush=True)
    del Wg
    torch.cuda.empty_cache()

    probe_info = fit_count_probes(rows, act_dir, len(probes))
    keep = [r for r in rows if "cnt" in r]
    packed = dict(
        item=np.array([r["item"] for r in keep]),
        k=np.array([r["k"] for r in keep]),
        family=np.array([r["family"] for r in keep]),
        template=np.array([r["template"] for r in keep]),
        T=np.array([r["T"] for r in keep]),
        hit_cap=np.array([r["hit_cap"] for r in keep]),
        logit=obj_array([r["logit"] for r in keep]),
        logprob=obj_array([r["logprob"] for r in keep]),
        lens=obj_array([r["lens"] for r in keep]),
        cnt=obj_array([r["cnt"] for r in keep]),
        cnt2=obj_array([r["cnt2"] for r in keep]),
        probe=np.array(json.dumps(probe_info)),
        checks=np.array(json.dumps(checks)),
        probe_layers=np.array(probes),
    )
    np.savez(out_path, **packed)
    print(f"  cached {len(keep)} items -> {out_path}", flush=True)
    return packed


# ------------------------------------------------------------------ analysis --

def flatten(pk: dict, family: str, kband: str, drop_cap: bool,
            tail_frac: float = 1.0) -> dict:
    """Step-level table for one (family, k band)."""
    cols = {n: [] for n in ("pos", "cnt", "cnt2", "k", "y", "ylogit", "frac", "w")}
    grp, itm = [], []
    n_items, ks = 0, []
    for i in range(len(pk["item"])):
        if pk["family"][i] != family:
            continue
        k = int(pk["k"][i])
        if (kband == "low" and k >= 12) or (kband == "high" and k < 12):
            continue
        if drop_cap and bool(pk["hit_cap"][i]):
            continue
        T = int(pk["T"][i])
        idx = np.arange(int(T * (1 - tail_frac)), T)
        if len(idx) < 16:
            continue
        n_items += 1
        ks.append(k)
        cols["pos"].append(idx + 1.0)
        cols["cnt"].append(np.asarray(pk["cnt"][i])[idx])
        cols["cnt2"].append(np.asarray(pk["cnt2"][i])[idx])
        cols["k"].append(np.full(len(idx), float(k)))
        cols["y"].append(np.asarray(pk["logprob"][i])[idx])
        cols["ylogit"].append(np.asarray(pk["logit"][i])[idx])
        cols["frac"].append((idx + 1.0) / T)
        cols["w"].append(np.full(len(idx), 1.0 / len(idx)))
        grp.append(np.full(len(idx), str(pk["template"][i])))
        itm.append(np.full(len(idx), str(pk["item"][i])))
    if not n_items:
        return {}
    d = {n: np.concatenate(v) for n, v in cols.items()}
    d.update(grp=np.concatenate(grp), item=np.concatenate(itm),
             n_items=n_items, k_levels=sorted(set(ks)))
    return d


def make_blocks(d: dict, count_key: str = "cnt") -> dict:
    return {"DUR": standardise(spline_basis(d["pos"])),
            "CNT": standardise(spline_basis(d[count_key])),
            "K": standardise(dummies(d["k"]))}


def analyse(d: dict, weighted: bool = True, target: str = "y",
            count_key: str = "cnt") -> dict:
    y = d[target]
    w = d["w"] if weighted else np.ones_like(d["w"])
    B = make_blocks(d, count_key)
    res = commonality(B, y, w)
    res["n_steps"] = int(len(y))
    res["n_items"] = int(d["n_items"])
    res["k_levels"] = d["k_levels"]
    res["collinearity"] = dict(
        r_pos_cnt=float(np.corrcoef(d["pos"], d[count_key])[0, 1]),
        r_pos_k=float(np.corrcoef(d["pos"], d["k"])[0, 1]),
        r_cnt_k=float(np.corrcoef(d[count_key], d["k"])[0, 1]),
        cc_dur_cnt=canon_corr(B["DUR"], B["CNT"]),
        cc_dur_k=canon_corr(B["DUR"], B["K"]),
        cc_cnt_k=canon_corr(B["CNT"], B["K"]),
    )
    res["loto_r2"] = {n: loto_r2(B[n], y, w, d["grp"]) for n in B}
    res["loto_r2"]["FULL"] = loto_r2(np.column_stack(list(B.values())), y, w, d["grp"])
    res["oracle_frac_elapsed_r2"] = wr2(standardise(spline_basis(d["frac"])), y, w)
    return res


def layerwise(pk: dict, family: str, kband: str, probe_layers) -> dict:
    """Logit-lens decomposition: where along the stack does the EOS signal enter,
    and what is it reading when it does?

    `lens[:, p]` is the EOS logit read off probe layer p through the final norm
    and the unembedding. Two regressions per layer: on the LEVEL, which says what
    the running EOS logit tracks at that depth, and on the INCREMENT from the
    previous probe layer, which is that layer block's own direct contribution to
    the EOS direction. A layer block that writes duration into the stop head
    shows up as DUR-unique on its increment.
    """
    d = flatten(pk, family, kband, True)
    if not d:
        return {}
    idx = {}
    off = 0
    for i in range(len(pk["item"])):
        if pk["family"][i] != family:
            continue
        k = int(pk["k"][i])
        if (kband == "low" and k >= 12) or (kband == "high" and k < 12):
            continue
        if bool(pk["hit_cap"][i]):
            continue
        L = np.asarray(pk["lens"][i])
        idx[i] = (off, off + L.shape[0], L)
        off += L.shape[0]
    if off != len(d["pos"]):
        return {}
    P = len(probe_layers)
    lens = np.empty((off, P), dtype=np.float32)
    for _, (a, b, L) in idx.items():
        lens[a:b] = L
    B = make_blocks(d)
    w = d["w"]
    out = {"probe_layers": [int(p) for p in probe_layers], "level": [], "increment": []}
    for p in range(P):
        out["level"].append(commonality(B, lens[:, p].astype(np.float64), w)["unique"])
        prev = lens[:, p - 1] if p else np.zeros(off, dtype=np.float32)
        out["increment"].append(
            commonality(B, (lens[:, p] - prev).astype(np.float64), w)["unique"])
    out["r_lens_final_vs_logit"] = float(np.corrcoef(lens[:, -1], d["ylogit"])[0, 1])
    return out


def eos_spike(pk: dict, family: str) -> dict:
    """Shape of the target, measured rather than assumed.

    Reported because it decides how much the step-level regression can possibly
    mean. If log P(EOS) sits on a floor for the whole generation and rises only
    at the step where the model actually stops, then pooled step-level variance
    is mostly floor, and an R^2 computed over all steps is not a measure of the
    stop decision. Items that hit our token budget are the built-in negative
    case: they never stopped, so they should never spike.
    """
    fin_stop, fin_cap, floor, pct = [], [], [], []
    for i in range(len(pk["item"])):
        if pk["family"][i] != family:
            continue
        lp = np.asarray(pk["logprob"][i])
        if bool(pk["hit_cap"][i]):
            fin_cap.append(float(lp[-1]))
        else:
            fin_stop.append(float(lp[-1]))
            pct.append(float((lp[-1] > lp).mean()))
        floor.append(float(np.median(lp[:-5])))
    med = lambda v: float(np.median(v)) if v else None                # noqa: E731
    return dict(n_stopped=len(fin_stop), n_cap=len(fin_cap),
                final_logp_stopped=med(fin_stop), final_logp_cap=med(fin_cap),
                floor_logp=med(floor), final_percentile_within_item=med(pct))


def stop_time_model(pk: dict, family: str, kband: str, rng: np.random.Generator,
                    stim: dict, n_boot: int = 2000) -> dict:
    """Does the stop TIME track the requested count?

    The step-level regression cannot answer this cleanly, and the reason is
    structural rather than statistical: within one generation the stop always
    happens at the largest elapsed position there is, so "elapsed duration
    predicts the stop" is partly true by construction. The free quantity is
    across items -- how long the model chooses to talk for a given k. A slope of
    1 on log2(k) means the stop time tracks the request; 0 means it ignores it.
    Bootstrapped over templates, which is the resampling unit everywhere else in
    this project.

    Two regressors, and the second is the one to quote. Slope on log2(k) is
    mechanically attenuated at low k by the fixed carrier: k=2 -> k=8 adds six
    words to an eleven-word sentence, so total length can only grow ~50% while
    the request grows 4x, and the slope is depressed for a reason that has
    nothing to do with counting. Slope on log2(expected words) removes that: it
    asks whether the model talks for as long as the text it was given, where 1.0
    is proportional rendering and below 1.0 is an utterance shorter than the text
    asks for. The matched control is the reference, since these decoders have a
    general utterance-length ceiling that applies to both arms.
    """
    T, k, wds, tm = [], [], [], []
    for i in range(len(pk["item"])):
        if pk["family"][i] != family or bool(pk["hit_cap"][i]):
            continue
        kk = int(pk["k"][i])
        if (kband == "low" and kk >= 12) or (kband == "high" and kk < 12):
            continue
        it = stim.get(str(pk["item"][i]))
        if it is None:
            continue
        T.append(float(pk["T"][i]))
        k.append(float(kk))
        wds.append(float(it["expected_words"]))
        tm.append(str(pk["template"][i]))
    if len(T) < 6 or len(set(k)) < 3:
        return {}
    y, tm = np.log2(T), np.array(tm)
    tmps = sorted(set(tm.tolist()))
    out = dict(n_items=len(T), n_k_levels=len(set(k)), n_templates=len(tmps))
    for name, xv in (("k", np.log2(k)), ("words", np.log2(wds))):
        slope, icept = np.polyfit(xv, y, 1)
        r2 = float(1 - ((y - (slope * xv + icept)) ** 2).sum()
                   / max(((y - y.mean()) ** 2).sum(), 1e-9))
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(tmps, size=len(tmps), replace=True)
            idx = np.concatenate([np.where(tm == t)[0] for t in pick])
            if len(set(xv[idx].tolist())) < 2:
                continue
            boots.append(np.polyfit(xv[idx], y[idx], 1)[0])
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
            if len(boots) > 50 else (float("nan"), float("nan"))
        out[name] = dict(slope=float(slope), r2=r2, ci95=list(ci),
                         excludes_zero=bool(ci[0] > 0),
                         under_proportional=bool(ci[1] < 1.0))
    return out


def positive_control(d: dict, rng: np.random.Generator) -> dict:
    """Plant known structure on the REAL design matrix and try to recover it."""
    B = make_blocks(d)
    w = d["w"]

    def sig(name):
        z = B[name] @ rng.normal(size=B[name].shape[1])
        return (z - z.mean()) / (z.std() + 1e-9)

    plants = {}
    for label, mix in (("duration_only", {"DUR": 1.0}),
                       ("count_only", {"CNT": 1.0}),
                       ("k_only", {"K": 1.0}),
                       ("duration+k", {"DUR": 0.5, "K": 0.5})):
        y = sum(c * sig(n) for n, c in mix.items())
        y = y + rng.normal(scale=0.5 * float(y.std()), size=len(y))   # SNR ~ 4:1
        r = commonality(B, y, w)
        planted = set(mix)
        spurious = max([r["unique"][n] for n in r["unique"] if n not in planted],
                       default=0.0)
        recovered = min(r["unique"][n] for n in planted)
        plants[label] = dict(unique=r["unique"], r2_full=r["r2_full"],
                             r2_alone=r["r2_alone"], recovered=recovered,
                             spurious=spurious, margin=recovered - spurious)
        # additivity of the decomposition, asserted rather than assumed
        tot = (sum(r["unique"].values()) + sum(r["common_pairs"].values())
               + r["common_all3"])
        plants[label]["additivity_err"] = abs(tot - r["r2_full"])
    return plants


def fmt(v) -> str:
    return "   nan" if v is None or not np.isfinite(v) else f"{v:+.3f}"


def stop_profile(pk: dict, family: str, rate_hz: float = 50.0,
                 probe_t=(125, 250, 500, 1000)) -> dict:
    """The same question without a regression, so it can be checked by eye.

    Two things a duration-driven stop head has to show. (a) The realised stop
    time stops growing with the requested count exactly where the behavioural
    deficit becomes severe -- so the slope of log2(stop time) on log2(k) should
    fall from near 1 at low k towards 0 at high k. (b) At a FIXED elapsed time
    the EOS log-probability should barely depend on k, and should depend on it
    less at high k than at low k: the spread across k levels at fixed t is the
    part of the stop decision that the count could be driving.
    """
    per_k: dict = {}
    for i in range(len(pk["item"])):
        if pk["family"][i] != family:
            continue
        k = int(pk["k"][i])
        e = per_k.setdefault(k, dict(T=[], cap=0, n=0, at={t: [] for t in probe_t}))
        e["n"] += 1
        if bool(pk["hit_cap"][i]):
            e["cap"] += 1
        else:
            e["T"].append(int(pk["T"][i]))
        lp = np.asarray(pk["logprob"][i])
        for t in probe_t:
            if len(lp) > t:
                e["at"][t].append(float(lp[t - 1]))
    rows = {}
    for k in sorted(per_k):
        e = per_k[k]
        rows[k] = dict(
            n=e["n"], n_cap=e["cap"], n_stopped=len(e["T"]),
            median_stop_tokens=float(np.median(e["T"])) if e["T"] else None,
            median_stop_s=float(np.median(e["T"]) / rate_hz) if e["T"] else None,
            eos_logprob_at={str(t): (float(np.mean(v)) if v else None)
                            for t, v in e["at"].items()},
            n_at={str(t): len(v) for t, v in e["at"].items()})
    out = {"by_k": rows}
    for band, sel in (("low", [k for k in rows if k < 12]),
                      ("high", [k for k in rows if k >= 12])):
        ks = [k for k in sel if rows[k]["median_stop_tokens"]]
        if len(ks) >= 3:
            x = np.log2(np.array(ks, dtype=float))
            yv = np.log2(np.array([rows[k]["median_stop_tokens"] for k in ks]))
            out[f"slope_logT_logk_{band}"] = float(np.polyfit(x, yv, 1)[0])
            out[f"slope_n_k_levels_{band}"] = len(ks)
        sp = {}
        for t in probe_t:
            v = [rows[k]["eos_logprob_at"][str(t)] for k in sel
                 if rows[k]["eos_logprob_at"][str(t)] is not None]
            if len(v) >= 3:
                sp[str(t)] = float(np.std(v))
        out[f"eos_spread_across_k_{band}"] = sp
    return out


# ------------------------------------------------------------------- verdict --

def verdict(out: dict) -> dict:
    """Apply the pre-committed rules. No new criteria are invented here."""
    lines, flags = [], {}
    for model, m in out["models"].items():
        cells = {kb: m["cells"].get(f"word_rep/{kb}/nocap") for kb in ("low", "high")}
        if not all(cells.values()):
            continue
        sep = max(max(c["collinearity"]["cc_dur_cnt"] for c in cells.values()),
                  max(abs(c["collinearity"]["r_pos_cnt"]) for c in cells.values()))
        ctrl = bool(m.get("control_passed", False))
        ctrl_dom = bool(m.get("control_dominance_passed", False))
        dur_wins = all(c["unique"]["DUR"] >= c["unique"]["CNT"] for c in cells.values())
        cnt_small = all(c["unique"]["CNT"] < CNT_SMALL for c in cells.values())
        k_shrinks = cells["high"]["unique"]["K"] < cells["low"]["unique"]["K"]
        if sep > COLLINEAR_FATAL:
            v = "predictors inseparable"
        elif not ctrl:
            v = ("inconclusive (pre-committed positive control failed"
                 + ("; dominance control passed)" if ctrl_dom else ")"))
        elif dur_wins and cnt_small:
            v = "stop head reads duration, not count"
        elif not dur_wins:
            v = "stop head reads count"
        else:
            v = "inconclusive"
        flags[model] = dict(verdict=v, separability_max=sep, control_passed=ctrl,
                            control_dominance_passed=ctrl_dom,
                            dur_beats_cnt=dur_wins, cnt_unique_small=cnt_small,
                            k_unique_shrinks_low_to_high=bool(k_shrinks),
                            weakly_identified=bool(COLLINEAR_WARN < sep <= COLLINEAR_FATAL))
        lines.append(f"{model}: {v}"
                     + ("  [weakly identified]" if flags[model]["weakly_identified"] else "")
                     + f"\n    max |r| / cc(DUR,CNT) = {sep:.3f}; "
                       f"K unique {'shrinks' if k_shrinks else 'does not shrink'} "
                       f"low->high k")
    vs = {f["verdict"] for f in flags.values()}
    if len(vs) == 1:
        lines.append(f"panel: both checkpoints agree -- {vs.pop()}")
    elif vs:
        lines.append("panel: checkpoints disagree; quote each separately, "
                     "not the one that reads better")
    return dict(per_model=flags, lines=lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llasa1b", "llasa8b"])
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(REPO / "data/results/stop_head.json"))
    args = ap.parse_args()
    if args.gpu not in (2, 3):
        raise SystemExit("this host allocates GPUs 2 and 3 to this project only")

    rng = np.random.default_rng(0)
    out: dict = {"pre_commit": dict(collinear_fatal=COLLINEAR_FATAL,
                                    collinear_warn=COLLINEAR_WARN,
                                    control_min_recovery=CONTROL_MIN_RECOVERY,
                                    cnt_unique_small=CNT_SMALL),
                 "models": {}}

    for model in args.models:
        print(f"\n{'=' * 72}\n=== {model}\n{'=' * 72}", flush=True)
        pk = build_cache(model, args.gpu, args.force)
        checks = json.loads(pk["checks"].item())
        probe = json.loads(pk["probe"].item())
        ent_err = max((c["ent_maxabs"] for c in checks), default=float("nan"))
        top_err = max((c["top1_maxabs"] for c in checks), default=float("nan"))
        ns = float(np.median([c["ent_maxabs_noshift"] for c in checks])) if checks else np.nan
        print(f"\nlogit reconstruction vs the instrumented pass' own logits: "
              f"max |dH| = {ent_err:.4f} nats, max |dtop1| = {top_err:.4f}, "
              f"over {len(checks)} items")
        print(f"  (median error if the one-step offset is NOT applied: {ns:.3f} nats "
              f"-- that offset is the trap)")
        for fam, v in probe.items():
            print(f"count probe [{fam:13s}] layer idx {v['layer']}/{len(v['by_layer']) - 1}"
                  f"  item-level LOTO R^2 {v['loto_r2']:+.3f}  n={v['n_items']}")

        mres: dict = {"reconstruction": dict(n_items=len(checks),
                                             max_entropy_err_nats=ent_err,
                                             max_top1_err=top_err,
                                             median_unshifted_err_nats=ns),
                      "count_probe": probe, "cells": {}}

        for family in ("word_rep", "control_word"):
            for kband in ("low", "high"):
                for drop_cap in (True, False):
                    d = flatten(pk, family, kband, drop_cap)
                    if not d or d["n_items"] < 4:
                        continue
                    tag = f"{family}/{kband}/{'nocap' if drop_cap else 'withcap'}"
                    r = analyse(d)
                    r["unweighted_unique"] = analyse(d, weighted=False)["unique"]
                    r["raw_logit_unique"] = analyse(d, target="ylogit")["unique"]
                    r["cnt2_arm"] = {k: v for k, v in analyse(d, count_key="cnt2").items()
                                     if k in ("unique", "r2_full", "collinearity")}
                    dt = flatten(pk, family, kband, drop_cap, tail_frac=0.25)
                    r["terminal_quarter_unique"] = analyse(dt)["unique"] if dt else None
                    mres["cells"][tag] = r
                    if drop_cap:
                        c = r["collinearity"]
                        print(f"\n[{tag}]  n_items={r['n_items']}  n_steps={r['n_steps']}"
                              f"  k={r['k_levels']}")
                        print(f"   collinearity  r(pos,cnt)={c['r_pos_cnt']:+.3f}   "
                              f"cc(DUR,CNT)={c['cc_dur_cnt']:.3f}   "
                              f"cc(DUR,K)={c['cc_dur_k']:.3f}   "
                              f"cc(CNT,K)={c['cc_cnt_k']:.3f}")
                        print("   R2 alone     " + "  ".join(
                            f"{n}={r['r2_alone'][n]:+.3f}" for n in r["r2_alone"])
                            + f"   FULL={r['r2_full']:+.3f}")
                        print("   unique dR2   " + "  ".join(
                            f"{n}={fmt(r['unique'][n])}" for n in r["unique"]))
                        print("   common       " + "  ".join(
                            f"{n}={v:+.3f}" for n, v in r["common_pairs"].items())
                            + f"   all3={r['common_all3']:+.3f}")
                        print("   LOTO R2      " + "  ".join(
                            f"{n}={fmt(r['loto_r2'][n])}" for n in r["loto_r2"]))
                        print("   CNT2 arm     " + "  ".join(
                            f"{n}={fmt(v)}" for n, v in r["cnt2_arm"]["unique"].items())
                            + f"   r(pos,cnt2)="
                              f"{r['cnt2_arm']['collinearity']['r_pos_cnt']:+.3f}")
                        print(f"   ORACLE fraction-elapsed R2 = "
                              f"{r['oracle_frac_elapsed_r2']:+.3f}  "
                              f"(not a mechanism: it knows the stop time)")

        pls = [int(x) for x in pk["probe_layers"]]
        mres["layerwise"] = {kb: layerwise(pk, "word_rep", kb, pls)
                             for kb in ("low", "high")}
        for kb, lw in mres["layerwise"].items():
            if not lw:
                continue
            print(f"\nlogit lens, word_rep/{kb} -- unique dR2 on the EOS logit read "
                  f"at each depth (level | increment)")
            for j, p in enumerate(lw["probe_layers"]):
                lv, inc = lw["level"][j], lw["increment"][j]
                print(f"   L{p:2d}  level  DUR={fmt(lv['DUR'])} CNT={fmt(lv['CNT'])} "
                      f"K={fmt(lv['K'])}   |   incr  DUR={fmt(inc['DUR'])} "
                      f"CNT={fmt(inc['CNT'])} K={fmt(inc['K'])}")
            print(f"   r(lens at last layer, actual EOS logit) = "
                  f"{lw['r_lens_final_vs_logit']:+.4f}  (sanity: must be 1.000)")

        mres["eos_spike"] = {f: eos_spike(pk, f)
                             for f in ("word_rep", "control_word")}
        sp0 = mres["eos_spike"]["word_rep"]
        print(f"\nshape of the target (word_rep): log P(EOS) sits at "
              f"{sp0['floor_logp']:+.2f} for the whole generation and reaches "
              f"{sp0['final_logp_stopped']:+.2f} at the step the model stops "
              f"(n={sp0['n_stopped']}); items that hit our budget instead end at "
              f"{sp0['final_logp_cap']:+.2f} (n={sp0['n_cap']}).")
        print(f"   the stop step is at percentile "
              f"{sp0['final_percentile_within_item']:.3f} of its own trajectory: "
              f"the target is a floor with one spike, so pooled step-level R^2 is\n"
              f"   mostly a statement about the floor, not about the decision.")

        stim_all = load_stimuli()
        mres["stop_time"] = {f"{f}/{kb}": stop_time_model(pk, f, kb, rng, stim_all)
                             for f in ("word_rep", "control_word")
                             for kb in ("low", "high")}
        print("\nstop TIME (stopped items only). Slope of log2(stop tokens) on "
              "log2(k) and, the one to quote,\non log2(expected words), which "
              "divides out the fixed carrier. 1.0 = proportional rendering.")
        for tag, v in mres["stop_time"].items():
            if v:
                a, b = v["k"], v["words"]
                print(f"   {tag:20s} n={v['n_items']:3d}  "
                      f"on log2(k)     {a['slope']:+.2f} "
                      f"[{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}] R2={a['r2']:+.2f}"
                      f"   |   on log2(words) {b['slope']:+.2f} "
                      f"[{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] R2={b['r2']:+.2f}"
                      + ("  <- under-proportional" if b["under_proportional"] else ""))

        mres["stop_profile"] = {f: stop_profile(pk, f)
                                for f in ("word_rep", "control_word")}
        sp = mres["stop_profile"]["word_rep"]
        print("\nstop profile, word_rep:  k -> median stop (s) [n stopped / n, cap]"
              "   EOS logP at fixed elapsed t")
        for k, v in sp["by_k"].items():
            at = "  ".join(f"t={t}:{('%+.2f' % v['eos_logprob_at'][t]) if v['eos_logprob_at'][t] is not None else '  --  '}"
                           f"(n={v['n_at'][t]})" for t in v["eos_logprob_at"])
            print(f"   k={k:3d}  {str(round(v['median_stop_s'], 1)) if v['median_stop_s'] else '  --':>6s}s "
                  f"[{v['n_stopped']}/{v['n']}, cap {v['n_cap']}]   {at}")
        print(f"   slope log2(stop tokens) on log2(k):  "
              f"low k = {sp.get('slope_logT_logk_low', float('nan')):+.2f}   "
              f"high k = {sp.get('slope_logT_logk_high', float('nan')):+.2f}   "
              f"(1.0 = stop time tracks the requested count, 0.0 = ignores it)")
        print(f"   sd of EOS logP across k levels at fixed t: "
              f"low {sp.get('eos_spread_across_k_low')}  "
              f"high {sp.get('eos_spread_across_k_high')}")

        dc = flatten(pk, "word_rep", "high", True)
        if dc and dc["n_items"] >= 4:
            pc = positive_control(dc, rng)
            mres["positive_control"] = pc
            print("\npositive control -- planted on the real word_rep/high design matrix:")
            for lab, v in pc.items():
                print(f"   plant {lab:14s} R2={v['r2_full']:.3f}   unique: "
                      + "  ".join(f"{n}={fmt(v['unique'][n])}" for n in v["unique"]))
            rec = pc["count_only"]["unique"]["CNT"]
            mres["control_passed"] = bool(
                rec >= CONTROL_MIN_RECOVERY
                and pc["duration_only"]["unique"]["DUR"] >= CONTROL_MIN_RECOVERY
                and pc["k_only"]["unique"]["K"] >= CONTROL_MIN_RECOVERY)
            # The absolute floor above was fixed in advance and calibrated
            # against nothing. A dry run of this same routine on a SYNTHETIC
            # design with r(pos,cnt)=0.85 recovered only 0.06-0.14 unique R^2
            # for the planted block while assigning +0.000 to the blocks that
            # were not planted -- i.e. correct attribution at a magnitude the
            # absolute floor would still reject. Both criteria are therefore
            # reported: the pre-committed absolute one, which is the one the
            # verdict uses, and this dominance one, which asks the question the
            # floor was meant to ask -- does the planted block beat the
            # unplanted blocks. Added after the synthetic dry run and before any
            # real number was looked at.
            mres["control_dominance_passed"] = bool(
                all(v["margin"] > 0.02 and v["spurious"] < 0.02 for v in pc.values()))
            mres["control_additivity_max_err"] = max(v["additivity_err"]
                                                     for v in pc.values())
            print(f"   -> pre-committed control "
                  f"{'PASSED' if mres['control_passed'] else 'FAILED'}"
                  f"  (count-only recovery {rec:+.3f}, absolute floor "
                  f"{CONTROL_MIN_RECOVERY})")
            print(f"   -> dominance control "
                  f"{'PASSED' if mres['control_dominance_passed'] else 'FAILED'}"
                  f"  (planted block beats unplanted in all four plants; "
                  f"worst margin "
                  f"{min(v['margin'] for v in pc.values()):+.3f}, worst spurious "
                  f"{max(v['spurious'] for v in pc.values()):+.3f})")
            print(f"   -> commonality additivity error "
                  f"{mres['control_additivity_max_err']:.2e} (must be ~0)")

        out["models"][model] = mres

    out["verdict"] = verdict(out)
    print("\n" + "=" * 72)
    for line in out["verdict"]["lines"]:
        print(line)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
