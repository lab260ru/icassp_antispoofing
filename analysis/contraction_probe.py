#!/usr/bin/env python3
"""Measure the per-repetition contraction factor q directly, by finite-difference
perturbation propagation.

Every review round has flagged the same gap: the paper measures the theorem's
*consequence* (representational capacity saturates under repetition) but never
measures q, the Lipschitz constant of the per-repetition map F, itself. The
first attempt (`common/boundaries.py`) tried to read q off distances between
hidden states *at repetition boundaries*, located via text attention. That
failed for a structural reason: the deep-layer attention centroid is not
monotone (it advances on ~51% of steps), so boundary estimates collapse onto
near-duplicate positions and the resulting distances measure localisation
error, not state dynamics.

This script needs no boundary localisation at all. It estimates q the way a
Lipschitz constant is actually defined: perturb the state by a small vector
delta, run the decoder forward, and see how big the resulting difference is
tau steps later relative to ||delta||. Sweeping direction, magnitude and lag,
and comparing repeated text against length-matched non-repeated controls, is
what turns that ratio into a number anyone should trust or distrust.

------------------------------------------------------------------------------
WHERE exactly is the perturbation injected and read back -- and why not just
"layer ell, position t" verbatim as the informal recipe puts it
------------------------------------------------------------------------------
A teacher-forced forward pass computes one decoder layer's output for *all*
sequence positions in a single batched matmul, before any forward hook runs.
So a `register_forward_hook` that edits a layer's OUTPUT tensor only at column
t cannot be seen by that SAME layer's own output at any other column -- that
tensor was already finalised. We checked this mechanically (see the
`--selftest` path): editing position t changes that layer's own output at
position t-1 by exactly 0.000000, at position t by an O(||delta||) amount, and
at later positions likewise by exactly 0 (a hook on a layer's *output*, read
back at that *same* layer, is a provable null result, not a small effect).

The one causal channel that survives inside a single one-shot pass is a
layer's own causal self-attention re-reading a perturbed *input*: layer ell's
key/value projections at position t are a function of layer ell's INPUT at t
(= layer ell-1's output), and layer ell's own query at a later position t+tau
attends back over that key. This is mechanically identical to what an
incremental, KV-cached generation would do if layer ell's cache entry at
position t were perturbed -- which is the actual object the theorem's map F
is a proxy for. We therefore inject delta via a `forward_pre_hook` on the
probe layer (equivalently: added to the residual stream immediately upstream
of it) and read the SAME probe layer's output afterward. This is the minimal
technical deviation from the recipe as stated that makes it a non-degenerate
measurement, and it is the one we validated in `--selftest` before running
anything else.

Caveat we cannot engineer away, stated plainly: this design only credits ONE
layer's own attention for carrying information across time. In real
generation, every layer keeps its own cache, and the theorem's state s_m is
best read as a property of the whole stack, not one layer. Reading a shallower
injection point and the SAME deep readout layer (the `--secondary-layer-frac`
run below) lets some of that missing cross-layer channel back in, at the cost
of conflating time-propagation with depth-propagation. We report both and let
the comparison, not either number alone, carry the argument.

------------------------------------------------------------------------------
Item selection
------------------------------------------------------------------------------
tau = n_speech_tokens / k is only a meaningful "per-repetition period" if the
decoder actually rendered close to k repetitions before stopping -- items that
hit the generation cap (hit_cap=True in the token metadata) were truncated
mid-runaway and do not have a well-defined tau. We use items with k >= k_min
(so tau is well resolved and 2*tau still fits) and hit_cap == False, keeping
word_rep items paired with their length- and template-matched control_word
twin (same k) so the repeated-vs-control comparison is apples to apples.

------------------------------------------------------------------------------
Dtype
------------------------------------------------------------------------------
Float32 throughout the perturbation passes: bf16's ~3 decimal digits of
mantissa are the same order of magnitude as the smallest delta we inject
(||delta|| ~ 1e-3 * ||h||), which would make the finite-difference estimate
mostly numerical noise. SDPA's flash/memory-efficient kernels do not support
float32, so we force the memory-efficient backend explicitly (the default
"math" fallback for fp32 materialises the full T x T attention matrix per head
and reliably OOMs at any useful batch size).

Usage:
  python analysis/contraction_probe.py --model llasa1b --gpu 0
  python analysis/contraction_probe.py --model llasa1b --gpu 0 --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from common.registry import BY_KEY, DATA_ROOT  # noqa: E402
from models.llasa_gen import build_prompt, extract_speech_ids  # noqa: E402

# ---- the four controls the task demands, as fixed sweep grids ---------------
DELTAS_REL = [1e-3, 1e-2, 1e-1]      # step-size / linearity check
LAG_FRACS = [0.5, 1.0, 2.0]          # lag control: tau/2, tau, 2*tau
N_DIR = 4                            # direction averaging (mean AND max reported)
N_T = 6                              # anchor positions per item
BATCH_CHUNK = 16                     # forward-pass batch size (see --selftest bench)


# ==============================================================================
# Reconstruction: recorded speech-token ids -> the exact teacher-forcing input
# ==============================================================================
def reconstruct_ids(tok, item: dict, sp_ids: np.ndarray) -> tuple[torch.Tensor, int]:
    """Rebuild [prompt ++ generated] input ids from the recorded integer codec ids.

    `llasa_gen.py` saves only the extracted codec integers (for the separate
    vocoding step), not the raw generated token-id tensor. Reconstructing the
    latter is exact and checkable: mapping each integer back through
    `"<|s_{id}|>"` and re-running the SAME `extract_speech_ids` used at
    recording time must reproduce the saved array byte-for-byte, which we
    assert on every item rather than assuming the round trip is clean.
    """
    prompt_ids = build_prompt(tok, item["text"])
    strs = [f"<|s_{int(i)}|>" for i in sp_ids]
    ids = tok.convert_tokens_to_ids(strs)
    if any(x is None for x in ids):
        raise ValueError(f"{item['item_id']}: unrecognised speech-token id in reconstruction")
    speech_t = torch.tensor(ids, dtype=prompt_ids.dtype).unsqueeze(0)
    full = torch.cat([prompt_ids, speech_t], dim=1)
    # round-trip check: decode back and re-extract, must match exactly
    back = extract_speech_ids(tok.convert_ids_to_tokens(ids))
    if back != sp_ids.tolist():
        raise ValueError(f"{item['item_id']}: reconstruction round-trip mismatch")
    return full, int(prompt_ids.shape[1])


# ==============================================================================
# The perturbation hook
# ==============================================================================
class PositionalPerturbation:
    """forward_pre_hook: add a per-row delta at a per-row position, batched.

    Registered on `model.model.layers[inject_layer]`. Each batch row b gets its
    own target column `positions[b]` and its own vector `deltas[b]`, so one
    forward call covers many (t, direction, magnitude) trials at once -- the
    only way finite-difference sweeps of this size are affordable.

    Returns a NEW tensor (clone + scatter-add) rather than mutating in place,
    so the untouched clean-pass hidden-state tensors we compare against can
    never be silently aliased into this one.
    """

    def __init__(self, positions: torch.Tensor, deltas: torch.Tensor):
        self.positions = positions   # [B] long
        self.deltas = deltas         # [B, d] float32

    def __call__(self, module, args):
        hidden = args[0]
        hidden = hidden.clone()
        b_idx = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[b_idx, self.positions, :] = hidden[b_idx, self.positions, :] + self.deltas
        return (hidden,) + tuple(args[1:])


def forward_hidden(model, ids: torch.Tensor) -> tuple:
    """One teacher-forced pass through the base model (no lm_head: we never
    need logits here, and skipping the vocab projection is the difference
    between fitting a batch of 20 and OOMing on batch 1 -- the vocab is
    ~194k tokens wide). Forces the memory-efficient SDPA backend because the
    default math fallback for float32 materialises O(T^2) attention weights
    per head and blows the memory budget long before it blows the runtime
    budget; see the module docstring."""
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.model(ids, output_hidden_states=True)
    return out.hidden_states  # tuple(n_layers+1) of [B, T, d]; index l+1 = output of layer l


# ==============================================================================
# Item selection
# ==============================================================================
def load_meta(model: str) -> dict:
    path = Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["seed"] == 0:
            out[r["item_id"]] = r
    return out


def select_pairs(stim: dict, meta: dict, k_min: int, max_pairs: int) -> list[tuple[dict, dict]]:
    """(word_rep, matched control_word) pairs with a well-defined tau on both sides.

    hit_cap==False is required on BOTH members: a truncated word_rep item's
    n_speech_tokens does not reflect k intended repetitions (it reflects
    running into the generation cap, which is closer to the failure mode the
    paper is ABOUT than to a clean periodic trajectory to measure tau on), and
    we want the two members of a pair on equal footing regardless.
    """
    pairs = []
    for it in stim.values():
        if it["family"] != "word_rep" or it["k"] < k_min:
            continue
        ctrl_id = None
        for cid, cit in stim.items():
            if cit.get("control_of") == it["item_id"]:
                ctrl_id = cid
                break
        if ctrl_id is None:
            continue
        ctrl = stim[ctrl_id]
        mw, mc = meta.get(it["item_id"]), meta.get(ctrl_id)
        if mw is None or mc is None or mw.get("hit_cap") or mc.get("hit_cap"):
            continue
        pairs.append((it, ctrl))
    pairs.sort(key=lambda p: p[0]["item_id"])
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def anchor_positions(t_gen: int, k: int, tau: float, n_t: int) -> list[int]:
    """n_t local (0-indexed within the generated span) anchor positions with
    room for the largest lag (2*tau) to still land inside the trajectory."""
    hi = int((k - 2) * tau) - 1
    if hi < 1 or tau < 2:
        return []
    n_t = min(n_t, hi + 1)
    return sorted(set(np.linspace(0, hi, n_t).astype(int).tolist()))


# ==============================================================================
# Self-test: prove the hook placement is non-degenerate before trusting it
# ==============================================================================
def selftest(model, tok, device: str, item: dict, sp_ids: np.ndarray, probe_layer: int) -> None:
    full, plen = reconstruct_ids(tok, item, sp_ids)
    full = full.to(device)
    hs_clean = forward_hidden(model, full)
    d = model.config.hidden_size
    rng = np.random.default_rng(0)
    direction = rng.normal(size=d).astype(np.float32)
    direction /= np.linalg.norm(direction)
    g_t = plen + min(50, full.shape[1] - plen - 60)
    ref_norm = float(hs_clean[probe_layer][0, g_t, :].norm())
    delta = torch.tensor(0.01 * ref_norm * direction, device=device)
    hook = PositionalPerturbation(torch.tensor([g_t], device=device), delta.unsqueeze(0))
    handle = model.model.layers[probe_layer].register_forward_pre_hook(hook)
    hs_pert = forward_hidden(model, full)
    handle.remove()
    h_out_clean = hs_clean[probe_layer + 1][0]
    h_out_pert = hs_pert[probe_layer + 1][0]
    before = (h_out_pert[g_t - 1] - h_out_clean[g_t - 1]).norm().item()
    at = (h_out_pert[g_t] - h_out_clean[g_t]).norm().item()
    after = (h_out_pert[g_t + 50] - h_out_clean[g_t + 50]).norm().item()
    dn = float(delta.norm())
    print(f"[selftest] ||delta||={dn:.4f}  diff@t-1={before:.6f} (must be 0)  "
          f"diff@t={at:.6f} (should be O(delta))  diff@t+50={after:.6f} (small, nonzero)")
    assert before < 1e-8, "hook is leaking backward in time -- something is wrong"
    assert at > 0.1 * dn, "perturbation is not registering locally -- hook not firing correctly"
    print("[selftest] PASSED: no acausal leakage; local perturbation registers; "
          "later positions show a small nonzero effect through the layer's own attention.")


# ==============================================================================
# Main sweep
# ==============================================================================
def run_item(model, tok, device: str, item: dict, sp_ids: np.ndarray,
             inject_layer: int, read_layer: int, n_t: int, n_dir: int,
             deltas_rel: list[float], lag_fracs: list[float], batch_chunk: int,
             seed: int) -> list[dict]:
    full, plen = reconstruct_ids(tok, item, sp_ids)
    full = full.to(device)
    T = full.shape[1]
    t_gen = len(sp_ids)
    k = item["k"]
    tau = t_gen / k

    hs_clean = forward_hidden(model, full)
    h_ref = hs_clean[inject_layer][0]        # input to the injection layer (norm reference)
    h_out_clean = hs_clean[read_layer + 1][0].detach().clone()  # output of the read layer
    del hs_clean
    hidden_size = model.config.hidden_size

    anchors = anchor_positions(t_gen, k, tau, n_t)
    if not anchors:
        return []

    # Build every (t, direction, magnitude) trial up front, then chunk into
    # batches. Directions are fixed per (item, t) and reused across magnitudes,
    # so the step-size check compares apples to apples along one direction.
    rng = np.random.default_rng(hash((item["item_id"], inject_layer, seed)) % (2**31))
    trials = []  # each: dict(t_local, g_t, dir_idx, mag, delta[np array])
    for t_local in anchors:
        g_t = plen + t_local
        ref_norm = float(h_ref[g_t].norm())
        dirs = rng.normal(size=(n_dir, hidden_size)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for di in range(n_dir):
            for mag in deltas_rel:
                trials.append(dict(t_local=t_local, g_t=g_t, dir_idx=di, mag=mag,
                                   delta=(mag * ref_norm * dirs[di]).astype(np.float32)))

    records = []
    for c0 in range(0, len(trials), batch_chunk):
        chunk = trials[c0:c0 + batch_chunk]
        B = len(chunk)
        batch_ids = full.repeat(B, 1)
        positions = torch.tensor([tr["g_t"] for tr in chunk], device=device, dtype=torch.long)
        deltas = torch.tensor(np.stack([tr["delta"] for tr in chunk]), device=device)
        hook = PositionalPerturbation(positions, deltas)
        handle = model.model.layers[inject_layer].register_forward_pre_hook(hook)
        hs_pert = forward_hidden(model, batch_ids)
        handle.remove()
        h_out_pert = hs_pert[read_layer + 1]   # [B, T, d]
        del hs_pert, batch_ids

        for i, tr in enumerate(chunk):
            dnorm = float(np.linalg.norm(tr["delta"]))
            g_t = tr["g_t"]
            # lag-0 sanity value: perturbation should register close to g_t itself
            ratio0 = float((h_out_pert[i, g_t, :] - h_out_clean[g_t, :]).norm()) / dnorm
            for lag_frac in lag_fracs:
                read_pos = g_t + int(round(lag_frac * tau))
                if read_pos >= T:
                    continue
                diff = float((h_out_pert[i, read_pos, :] - h_out_clean[read_pos, :]).norm())
                records.append(dict(
                    item_id=item["item_id"], family=item["family"], template=item["template"],
                    k=k, control_of=item.get("control_of"), tau=tau,
                    t=tr["t_local"], dir_idx=tr["dir_idx"], mag_rel=tr["mag"],
                    delta_norm=dnorm, lag_frac=lag_frac, ratio=diff / dnorm, ratio_at_t=ratio0,
                ))
        del h_out_pert
        torch.cuda.empty_cache()

    del h_out_clean, h_ref, full
    torch.cuda.empty_cache()
    return records


# ==============================================================================
# Aggregation / summary statistics
# ==============================================================================
def _distrib(vals: np.ndarray) -> dict:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, q25=np.nan, q75=np.nan,
                    min=np.nan, max=np.nan)
    return dict(n=int(vals.size), mean=float(vals.mean()), median=float(np.median(vals)),
                std=float(vals.std()), q25=float(np.percentile(vals, 25)),
                q75=float(np.percentile(vals, 75)), min=float(vals.min()), max=float(vals.max()))


def summarize(records: list[dict], pairs: list[tuple[dict, dict]]) -> dict:
    import scipy.stats as st

    def col(name, rows):
        return np.array([r[name] for r in rows], dtype=np.float64)

    out: dict = {}

    # --- headline: ratio at lag=tau, mag=1e-2 (the "reasonable middle" step) --
    headline_mag = 1e-2
    lag_tau_mid = [r for r in records if abs(r["lag_frac"] - 1.0) < 1e-9
                   and abs(r["mag_rel"] - headline_mag) < 1e-12]
    for fam in ("word_rep", "control_word"):
        fam_rows = [r for r in lag_tau_mid if r["family"] == fam]
        out[f"headline_{fam}"] = _distrib(col("ratio", fam_rows))
        # per-(item,t): mean and max over the n_dir directions, then pooled
        by_it = {}
        for r in fam_rows:
            by_it.setdefault((r["item_id"], r["t"]), []).append(r["ratio"])
        mean_over_dir = np.array([np.mean(v) for v in by_it.values()])
        max_over_dir = np.array([np.max(v) for v in by_it.values()])
        out[f"headline_{fam}_mean_over_dir"] = _distrib(mean_over_dir)
        out[f"headline_{fam}_max_over_dir"] = _distrib(max_over_dir)

    # --- paired repeated-vs-control comparison, matched by (template, k) -----
    paired_diffs, paired_ratios = [], []
    for wr, ct in pairs:
        wr_rows = [r["ratio"] for r in lag_tau_mid if r["item_id"] == wr["item_id"]]
        ct_rows = [r["ratio"] for r in lag_tau_mid if r["item_id"] == ct["item_id"]]
        if not wr_rows or not ct_rows:
            continue
        mw, mc = float(np.median(wr_rows)), float(np.median(ct_rows))
        paired_diffs.append(mw - mc)
        paired_ratios.append(mw / mc if mc > 1e-12 else np.nan)
    paired_diffs = np.asarray(paired_diffs)
    frac_more_contractive = float((paired_diffs < 0).mean()) if paired_diffs.size else np.nan
    wilcoxon_p = np.nan
    if paired_diffs.size >= 6 and np.any(paired_diffs != 0):
        try:
            wilcoxon_p = float(st.wilcoxon(paired_diffs).pvalue)
        except Exception:  # noqa: BLE001
            wilcoxon_p = np.nan
    out["paired_comparison"] = dict(
        n_pairs=int(paired_diffs.size),
        frac_pairs_word_rep_more_contractive=frac_more_contractive,
        median_diff_word_rep_minus_control=float(np.median(paired_diffs)) if paired_diffs.size else np.nan,
        wilcoxon_p=wilcoxon_p,
        median_ratio_word_rep_over_control=float(np.nanmedian(paired_ratios)) if paired_ratios else np.nan,
    )

    # --- step-size / linearity check, at lag=tau, direction-matched ----------
    by_dir = {}
    for r in records:
        if abs(r["lag_frac"] - 1.0) > 1e-9:
            continue
        key = (r["item_id"], r["t"], r["dir_idx"])
        by_dir.setdefault(key, {})[r["mag_rel"]] = r["ratio"]
    small, mid, large = DELTAS_REL
    f_mid_small, f_large_small = [], []
    for v in by_dir.values():
        if small in v and mid in v and v[small] > 1e-12:
            f_mid_small.append(v[mid] / v[small])
        if small in v and large in v and v[small] > 1e-12:
            f_large_small.append(v[large] / v[small])
    out["step_size_check"] = dict(
        note="ratio(mag)/ratio(smallest_mag) along the same direction; ~1 means "
             "the local-linear (small-delta) regime holds at that step.",
        mid_over_small=_distrib(np.array(f_mid_small)),
        large_over_small=_distrib(np.array(f_large_small)),
    )

    # --- lag control: is there anything special about lag == tau -------------
    lag_summary = {}
    for lag_frac in LAG_FRACS:
        rows = [r["ratio"] for r in records if abs(r["lag_frac"] - lag_frac) < 1e-9
                and abs(r["mag_rel"] - headline_mag) < 1e-12]
        lag_summary[str(lag_frac)] = _distrib(np.array(rows))
    out["lag_check"] = lag_summary

    return out


def verdict_from_summary(summary: dict) -> dict:
    """A terse, mechanical pass/fail read of the four controls. Human judgement
    in the final report should not simply defer to this, but it should not
    contradict it without explanation either."""
    hw = summary.get("headline_word_rep", {})
    hc = summary.get("headline_control_word", {})
    pc = summary.get("paired_comparison", {})
    ssc = summary.get("step_size_check", {})

    q_word_rep = hw.get("median", np.nan)
    q_control = hc.get("median", np.nan)
    contracting = np.isfinite(q_word_rep) and q_word_rep < 1.0
    dissociates = (np.isfinite(pc.get("median_diff_word_rep_minus_control", np.nan))
                   and pc["median_diff_word_rep_minus_control"] < 0
                   and pc.get("frac_pairs_word_rep_more_contractive", 0) > 0.5)
    lin_mid = ssc.get("mid_over_small", {}).get("median", np.nan)
    stable_small_step = np.isfinite(lin_mid) and 0.5 <= lin_mid <= 2.0

    return dict(
        q_word_rep_median=q_word_rep, q_control_median=q_control,
        contracting_regime=bool(contracting),
        repeated_more_contractive_than_control=bool(dissociates),
        wilcoxon_p=pc.get("wilcoxon_p", np.nan),
        step_size_stable=bool(stable_small_step),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llasa1b")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/contraction_probe.json"))
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--n-t", type=int, default=N_T)
    ap.add_argument("--n-dir", type=int, default=N_DIR)
    ap.add_argument("--batch-chunk", type=int, default=BATCH_CHUNK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--secondary-layer-frac", type=float, default=0.5,
                    help="fractional depth for the secondary (depth-compounded) "
                         "injection point; read layer is always the deepest. "
                         "Set <0 to skip the secondary run.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the hook-placement sanity check and exit")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    spec = BY_KEY[args.model]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{spec.key}] loading {spec.hf_id} in float32 on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    read_layer = n_layers - 1
    secondary_layer = (int(round(args.secondary_layer_frac * (n_layers - 1)))
                        if args.secondary_layer_frac >= 0 else None)
    print(f"[{spec.key}] n_layers={n_layers} hidden={model.config.hidden_size} "
          f"read_layer={read_layer} secondary_inject_layer={secondary_layer}", flush=True)

    stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.stimuli)}
    meta = load_meta(args.model)
    pairs = select_pairs(stim, meta, args.k_min, args.max_pairs)
    print(f"[{spec.key}] {len(pairs)} word_rep/control_word pairs, k>={args.k_min}, "
          f"hit_cap=False on both members", flush=True)
    if not pairs:
        print("no eligible items -- nothing to do")
        return

    tok_dir = Path(DATA_ROOT) / "tokens" / args.model

    if args.selftest:
        wr, _ = pairs[0]
        sp_ids = np.load(tok_dir / f"{wr['item_id']}_s0.npy")
        selftest(model, tok, device, wr, sp_ids, read_layer)
        return

    items = [it for pair in pairs for it in pair]
    designs = [("primary", read_layer)]
    if secondary_layer is not None and secondary_layer != read_layer:
        designs.append(("secondary", secondary_layer))

    all_records = {name: [] for name, _ in designs}
    t_start = time.time()
    for idx, it in enumerate(items):
        sp_ids = np.load(tok_dir / f"{it['item_id']}_s0.npy")
        for name, inject_layer in designs:
            try:
                recs = run_item(model, tok, device, it, sp_ids, inject_layer, read_layer,
                                args.n_t, args.n_dir, DELTAS_REL, LAG_FRACS,
                                args.batch_chunk, args.seed)
            except Exception as e:  # noqa: BLE001
                print(f"  [{it['item_id']}/{name}] FAILED: {e}", flush=True)
                continue
            all_records[name].extend(recs)
        if (idx + 1) % 4 == 0 or idx == len(items) - 1:
            el = time.time() - t_start
            print(f"[{spec.key}] {idx+1}/{len(items)} items  {el/(idx+1):.1f}s/item  "
                  f"last={it['item_id']}", flush=True)

    result: dict = dict(
        config=dict(
            model=args.model, hf_id=spec.hf_id, dtype="float32", gpu=args.gpu,
            n_layers=n_layers, hidden_size=model.config.hidden_size,
            read_layer=read_layer, inject_layers=dict(designs),
            deltas_rel=DELTAS_REL, lag_fracs=LAG_FRACS, n_dir=args.n_dir, n_t=args.n_t,
            k_min=args.k_min, n_pairs=len(pairs), n_items=len(items), seed=args.seed,
            item_ids=[it["item_id"] for it in items],
        ),
        designs={},
    )
    for name, _ in designs:
        recs = all_records[name]
        print(f"\n[{spec.key}/{name}] {len(recs)} trial-lag records", flush=True)
        summary = summarize(recs, pairs) if recs else {}
        verdict = verdict_from_summary(summary) if summary else {}
        if verdict:
            print(f"  q (word_rep, median @ lag=tau, mag=1e-2)   = {verdict['q_word_rep_median']:.5f}")
            print(f"  q (control_word, median @ lag=tau, mag=1e-2) = {verdict['q_control_median']:.5f}")
            print(f"  contracting (q<1): {verdict['contracting_regime']}   "
                  f"repeated more contractive than control: "
                  f"{verdict['repeated_more_contractive_than_control']} "
                  f"(Wilcoxon p={verdict['wilcoxon_p']})")
            print(f"  step-size stable at small delta: {verdict['step_size_stable']}")
        # columnar storage: far cheaper than list-of-dicts for ~10^4 rows
        columnar = {}
        if recs:
            for key in recs[0]:
                columnar[key] = [r[key] for r in recs]
        result["designs"][name] = dict(
            inject_layer=dict(designs)[name], records=columnar, summary=summary, verdict=verdict,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=lambda x: None))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
