#!/usr/bin/env python3
"""Does the Llasa-1B causal null survive a second checkpoint, and a second family?

`analysis/causal_count.py` asked whether the decoder *reads* the count coordinate
the ridge probe decodes, and answered on Llasa-1B: a rank-1 patch that transfers
only the probe's count coordinate leaves generation intact (0.0% degenerate,
self-patch bitwise-identical 18/18) and moves the rendered count not at all
(median +0.00 log-count, 90% CI [-0.089, +0.070], n=36 at `crossk|L7|P128`;
`analysis/equivalence.py` turns that into "a shift above 0.089 log-count, 9% of a
full donor-to-receiver transfer, is excluded at 95%"). Four reviewers in one
round objected that single-checkpoint mechanistic results were being stated as
general conclusions. The Jacobian measurement has since been extended to five
checkpoints across two families; the causal patch has not, and is the last
unscoped mechanistic claim in the paper.

This file re-runs both arms --- the rank-1 projection patch and the ridge
steering sweep --- on further checkpoints, prioritising a **different
architecture family** (Qwen3-TTS) over a bigger Llasa, because a second family
is what the objection actually demands.

------------------------------------------------------------------------------
WHAT HAD TO CHANGE, AND WHY IT IS NOT A WEAKENING
------------------------------------------------------------------------------

Llasa-1B's protocol teacher-forced the receiver's own first $P$ generated tokens
as a *prefix*, patched the states underlying them during that prefill, and
resumed free generation. Every patched cell was paired against a **resume
reference** (same prefix, no hook, same seed) because a patched run and a free
baseline consume the sampler's random stream at different offsets --- the
resumed run is handed $P$ tokens for free and starts drawing at step $P$.

Qwen3-TTS cannot be driven that way and the reason is architectural, not
incidental. Its talker's input at each decode step is not an embedding lookup
but a sum of sixteen RVQ-codebook embeddings plus a text term
(`analysis/qwen_codec_dump.py` documents the seventeen-way sum), the codec ids
are never returned by the public API, and `generate_voice_clone` is a
`@torch.no_grad()` path with no way to hand it a pre-populated KV cache. There
is no prefix to force. Reconstructing one would mean intercepting both the
talker's sampler and the code predictor's fifteen interleaved draws inside a
library forward --- a large amount of unverifiable machinery in the load-bearing
part of a causal claim.

So both checkpoints are run under a protocol that needs no prefix at all:

  **the patch is applied at the first $P$ *decode* steps of an ordinary free
  generation, and the reference is that item's free baseline at the same seed.**

This removes the reason the resume reference existed rather than ignoring it.
Under the original protocol the patched run and the free baseline drew from the
sampler at different offsets; here both runs start at step 0 and consume exactly
the same number of draws per step --- for Qwen, fifteen code-predictor draws then
one talker draw, unconditionally, whatever the logits say --- so the two streams
are aligned step-for-step by construction. The claim is not asserted: it is
exactly what the no-op gate below tests, and the gate is bitwise.

The protocol also differs from Llasa-1B's in one direction that matters, and it
is the direction that makes a null *harder* to get. Under the original protocol
the tokens inside the patch window were pinned to the receiver's own, so the
patch could only act on what happened after the window. Here the window's frames
are free to respond to the patch, so the count coordinate can act both through
the residual stream and through the frames the model actually emits. If the
rendered count still does not move, it did not move under a strictly more
permissive intervention.

Because the protocol changed and the checkpoint changed, changing both at once
would confound them. **Llasa-1B is therefore re-run under the new protocol as a
bridge**, at the same layers, window duration and donors as its published run.
That makes the comparison a factorial rather than a substitution: if bridge
Llasa-1B reproduces published Llasa-1B, the protocol is exonerated and any Qwen
difference is the checkpoint's.

Window length is matched in **seconds, not tokens**: 128 Llasa tokens at 50 Hz
is 2.56 s, and 32 Qwen frames at 12.5 Hz is 2.56 s. Patch layers are matched in
**fractional depth** on each checkpoint's stored probe grid: Llasa-1B's
[3, 7, 11] of 16 is 0.19/0.44/0.69, and Qwen-0.6B's [6, 12, 18] of 28 is
0.21/0.43/0.64. The steering layer is chosen by each checkpoint's own rule ---
"the probe's best late layer on word_rep", read from `data/results/probe.json`,
which is layer 13 for Llasa-1B and layer 9 for Qwen-0.6B.

------------------------------------------------------------------------------
PRE-COMMITTED INTERPRETATION.  Written before any result on any new checkpoint
was looked at, and before a single generation was run.
------------------------------------------------------------------------------

**Sanity gate (both arms). If it fails, nothing in this file is reportable and
the verdict for that checkpoint is `pipeline inconclusive`.**

  G1. Free generation is reproducible under a fixed seed, bitwise.
  G2. The hook fires on exactly the positions it claims to.
  G3. The **no-op self-patch is bitwise-identical to the free baseline** --- the
      same audio for Qwen, the same speech-token sequence for Llasa --- in 100%
      of runs, and the rank-1 hook reports max $|\\delta|$ **exactly 0**. The
      numerical subtlety S26 records applies unchanged and is the reason the
      hook is reused rather than rewritten: written as "project out, write in"
      ($h - (h\\cdot v)v + (c_d)v$) the correction perturbs every patched
      position in bfloat16; written as a difference $h + (c_d - c_r)v$ it is
      exact --- but only if both coefficients come from the *same expression* on
      the same memory layout. Computing the donor's coefficient outside the hook
      with a matmul and the receiver's inside with another left them disagreeing
      by 1.5e-3 on a self-patch, which is small, nonzero, and fatal to a bitwise
      gate. `Rank1PatchHook` is imported from `analysis/causal_count.py`
      unmodified so that this cannot silently regress.
  G4. A real donor patch **does** change the continuation --- otherwise G3 is
      satisfied by a hook that does nothing and the null is a null by
      construction.
  G5. $\\alpha=0$ steering with the hook attached reproduces the free baseline
      bitwise, in 100% of runs.

**The donor-state capture path.** Llasa-1B's run recorded donor states on
`generate`'s own prefill and wrote them back into a prefill, after discovering
that a plain `model(...)` forward and a `generate` prefill of the same tokens
disagree in bfloat16 by up to 1.0 at this layer (different attention mask,
different SDPA kernel), which had made the no-op diverge by generation step 26
--- a pipeline that would have reported a spurious causal effect of about the
size being looked for, out of nothing but rounding. Here donor states are
recorded on the **incremental decode path** and written back into the
**incremental decode path**, so the two paths are the same path by construction
rather than by care. G3 is what proves it.

**The readout.** Rendered count from the CTC judge (`src/common/asr_ctc.py`,
greedy, LM-free), counted by `src/common/score_counts.py`'s rule --- occurrences
of the target unit for a repeated item, units-rendered for a control. Measured
on $\\log(1+\\text{count})$, paired within receiver run against that run's own
free baseline at the same seed, summarised by the median paired shift with a
bootstrap CI and a two-sided sign test. Duration, unit count, cap-hit rate,
stop rate and degenerate-audio rate are carried alongside every cell, because an
intervention that merely makes the model babble longer would move the count too
and must not be allowed to look like a count effect.

**Gate P1 --- is the arm readable at all?** The disruption index is the
*unrelated* donor's median $|\\text{shift}|$ over the informative *cross-k*
donor's, at the central cell. A donor that cannot possibly know the receiver's
$k$ --- a repeated-*sentence* item from a different family, no adverb, no
matching carrier --- must move the count **less than half** as much as one that
differs in $k$. Below 0.5 the arm is readable. **If P1 fails the arm is
`too disruptive to interpret` and nothing else in it is reported**, exactly as
in the first pass, where whole-state splicing failed this gate at 1.3.

  Recorded now, because it decided the reading last time and must not be
  invented afterwards: Llasa-1B's published rank-1 arm has disruption index
  1.74, i.e. it *fails* P1 as a ratio --- but both its numerator and its
  denominator sit at the arm's own noise floor (unrelated 0.259, cross-$k$ 0.149
  median $|$shift$|$, against a full transfer of 0.992), so the ratio is noise
  over noise and the published reading is the post-hoc one: every cell at the
  noise floor, nothing moves, informative or not. That escape hatch is available
  to a new checkpoint **only under the same explicit test**, declared here in
  advance: `posthoc_all_cells_at_noise_floor` holds iff the largest median
  $|$shift$|$ over *all* donor kinds is below **25% of a full transfer**. If P1
  fails and that test also fails, the verdict is `too disruptive to interpret`
  and no count claim is made for that checkpoint.

**Conditional on P1, the rank-1 patch is a causal locus iff:**

  P2. the cross-$k$ signed shift (signed toward the donor) is positive with a
      two-sided sign test surviving Holm correction across cells, **and** the
      higher-$k$ and lower-$k$ donors move the count in opposite raw directions;
      AND
  P3. patched degeneracy is no more than 15 points above the reference arm's,
      and the stop rate no more than 15 points below it.

**P1 ∧ ¬P2 is a readable null** and is the outcome expected from Llasa-1B: the
count coordinate transplants cleanly and moving it does not move the rendered
count. A readable null is only reportable *as* a null when the equivalence
bound is tight enough to license it: the arm must exclude, at 95% by TOST on the
bootstrap median (row bootstrap **and** a bootstrap clustered by receiver item,
both agreeing), any shift larger than half a full donor-to-receiver transfer.
If the bound is wider than that the verdict is `underpowered`, not `null`, and
the number of pairs needed is reported instead. **The bound, not a confidence
interval containing zero, is the deliverable.**

**Ridge steering sweep counts as a hit iff:**

  R1. pooled Spearman $\\rho$ between $\\alpha$ and $\\log(1+\\text{count})$ on
      within-item z-scores of the repeated arm, inside the interpretable band,
      with $|\\rho|>0.4$, $p<0.05$, and consistent per-item sign in at least two
      thirds of items; AND
  R2. the same correlation partialling out $\\log(\\text{duration})$ keeps its
      sign with $|\\rho_{\\text{partial}}|>0.25$; AND
  R3. the repeated arm's effect between the extreme in-band $\\alpha$s is at
      least 1.5x the length-matched control arm's.

**Interpretable band**: an $\\alpha$ whose degenerate-plus-empty rate exceeds
25% is out of band and is dropped before the dose-response is fitted. Fewer than
three surviving $\\alpha$ values means `underpowered`, not `null`. $\\alpha$ is
in units of the low-$k$-to-high-$k$ projection gap along the ridge direction, so
$\\alpha=1$ means "one full low-to-high step"; the *relative* magnitude of that
step against the mean per-position state norm is recorded per checkpoint,
because Llasa-1B's difference-in-means direction destroyed the audio at 45% and
its ridge direction stayed interpretable at 15%, and a checkpoint whose ridge
step lands at 45% is being pushed harder than Llasa-1B was and must not be
compared to it as though it were not.

**Uninterpretable** for a checkpoint if: the sanity gate fails; or more than half
the runs in an arm are degenerate, empty or cap-hit; or the bootstrap CI on the
*reference* median $\\log(1+\\text{count})$ is wider than the full baseline gap
between donor-$k$ and receiver-$k$ items, in which case the instrument cannot
resolve an effect of the size being looked for and the arm is `underpowered`.

**The three-way verdict per checkpoint is the one Llasa-1B's run used and no
other: `causal locus` / `readable null` / `too disruptive to interpret`**, with
`underpowered` and `pipeline inconclusive` as the two refusals.

**What would make this whole exercise uninterpretable**, stated so it cannot be
explained away later:

  U1. bridge Llasa-1B under this protocol failing to reproduce published
      Llasa-1B --- same verdict, and a median paired shift inside the published
      90% CI [-0.089, +0.070]. If the protocol change moves Llasa-1B, then the
      protocol is a confound and no cross-checkpoint comparison in this file
      means anything; the Qwen numbers would then describe the protocol.
  U2. the Qwen reference arm being degenerate or cap-hit in more than half its
      runs, which would mean the CTC judge is not measuring the count on this
      checkpoint at all. Qwen's ledger says otherwise (1 cap hit in 108 rows at
      the receiver $k$s) but the arm is checked, not assumed.
  U3. the ridge direction at the chosen layer having a low-to-high separation
      indistinguishable from zero, which would mean $\\alpha$ has no units and
      the sweep is sweeping nothing.

**A result that disagrees with Llasa-1B is the most valuable outcome available.**
If the count moves on another checkpoint that changes the paper's conclusion,
and the run prints `CONTRADICTS-LLASA1B` on the first cell that shows it rather
than waiting for the grid to finish.

Usage (env `qwen` for qwen*, env `base` for llasa*):
  python analysis/causal_count_second.py --model qwen06b --gpu 2 --selftest
  python analysis/causal_count_second.py --model qwen06b --gpu 2 --arm A
  python analysis/causal_count_second.py --model qwen06b --gpu 2 --arm R
  python analysis/causal_count_second.py --model llasa1b --gpu 3 --arm A   # bridge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common.gpus import check_gpu  # noqa: E402
from common.registry import BY_KEY, DATA_ROOT  # noqa: E402

# The hooks are imported, never re-implemented: the bitwise no-op depends on the
# exact expression inside Rank1PatchHook (see G3 above), and a copy would be a
# copy that can drift.
from analysis.causal_count import Rank1PatchHook, SteerHook  # noqa: E402

DATA_ROOT = Path(DATA_ROOT)

# --------------------------------------------------------------- per-checkpoint

# Window is matched in seconds (2.56 s) rather than tokens; patch layers are
# matched in fractional depth on each checkpoint's stored probe grid; the steer
# layer is each checkpoint's own "probe's best late layer on word_rep" read from
# data/results/probe.json.
CONFIG = {
    "llasa1b": dict(family="llasa", patch_layers=[3, 7, 11], central_layer=7,
                    patch_pos=128, steer_layer=13, max_new=3584),
    "llasa8b": dict(family="llasa", patch_layers=[8, 16, 24], central_layer=16,
                    patch_pos=128, steer_layer=16, max_new=3584),
    "qwen06b": dict(family="qwen", patch_layers=[6, 12, 18], central_layer=12,
                    patch_pos=32, steer_layer=9, max_new=2048),
    "qwen17b": dict(family="qwen", patch_layers=[6, 12, 18], central_layer=12,
                    patch_pos=32, steer_layer=27, max_new=2048),
}

TEMPLATES = ["t1", "t3", "t5", "t6"]     # t2 is excluded by the judge-vocab audit
PATCH_TEMPLATES = ["t1", "t3", "t5"]
RECEIVER_KS = [16, 24, 32]
DONOR_KS = {16: [8, 32], 24: [8, 32], 32: [8, 16]}
SEEDS = [0, 1]
DIFFSEED_OFFSET = 100
UNRELATED = ["sr_s1_k16", "sr_s4_k08", "sr_s1_k08", "sr_s2_k08"]

RIDGE_ALPHAS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
RIDGE_KS = [16, 24, 32]
LOW_K = [2, 3, 4]
HIGH_K = [16, 24, 32]

# The published Llasa-1B rank-1 cell, for the bridge check (U1) and for the
# CONTRADICTS banner. Frozen here so a later edit to the results file cannot
# quietly move the goalposts.
PUBLISHED_LLASA1B = dict(cell="crossk|L7|P128", median=0.0,
                         ci_lo=-0.09097177820572666, ci_hi=0.08701137698962969,
                         ci90_lo=-0.089, ci90_hi=0.070, n=36,
                         full_transfer=0.9920656809377557, bound=0.089)


# ---------------------------------------------------------------- hooks


class RecordDecodeHook:
    """Record one layer's output at absolute decode positions [lo, hi).

    Keyed on ``cache_position``, the model's own statement of which absolute
    positions this forward is computing, which is exact under both the prefill
    (many positions at once) and the incremental decode (one position). Kept in
    the model's own dtype so that a donor round-trip through this buffer is
    lossless -- the states go back in on the same path they came off.
    """

    def __init__(self, lo: int, hi: int) -> None:
        self.lo, self.hi = int(lo), int(hi)
        self.buf: dict[int, torch.Tensor] = {}

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
        hs = output[0] if isinstance(output, tuple) else output
        cp = kwargs.get("cache_position")
        if cp is None:
            return output
        sel = (cp >= self.lo) & (cp < self.hi)
        if not bool(sel.any()):
            return output
        pos = [int(p) for p in cp[sel].tolist()]
        vals = hs[0, sel, :].detach().to("cpu").clone()
        for i, p in enumerate(pos):
            self.buf[p] = vals[i]
        return output

    def stack(self) -> torch.Tensor | None:
        if not self.buf:
            return None
        return torch.stack([self.buf[k] for k in sorted(self.buf)])


# ---------------------------------------------------------------- backends


class Run:
    """One generation, in whatever currency the checkpoint's judge consumes."""

    def __init__(self, n_units: int, hit_cap: bool, stopped: bool,
                 units: list[int] | None = None,
                 wav: np.ndarray | None = None, sr: int | None = None) -> None:
        self.n_units, self.hit_cap, self.stopped = n_units, hit_cap, stopped
        self.units, self.wav, self.sr = units, wav, sr


class LlasaBackend:
    family = "llasa"

    def __init__(self, key: str, gpu: int) -> None:
        spec = BY_KEY[key]
        self.spec, self.device = spec, f"cuda:{gpu}"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.models.llasa_gen import SPEECH_END
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(self.device).eval()
        self.eos = self.tok.convert_tokens_to_ids(SPEECH_END)
        self.layers = self.model.model.layers
        self._pc: dict[str, torch.Tensor] = {}

    def n_layers(self) -> int:
        return len(self.layers)

    def prompt(self, text: str) -> torch.Tensor:
        from src.models.llasa_gen import build_prompt
        if text not in self._pc:
            self._pc[text] = build_prompt(self.tok, text).to(self.device)
        return self._pc[text]

    def plen(self, text: str) -> int:
        return int(self.prompt(text).shape[1])

    def generate(self, text: str, seed: int, max_new: int,
                 hooks: dict[int, object] | None = None) -> Run:
        from src.models.llasa_gen import extract_speech_ids
        ids = self.prompt(text)
        handles = []
        for L, h in (hooks or {}).items():
            handles.append(self.layers[L].register_forward_hook(h, with_kwargs=True))
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                out = self.model.generate(
                    ids, max_new_tokens=max_new, eos_token_id=self.eos,
                    do_sample=True, top_p=1.0, temperature=0.8,
                    pad_token_id=self.tok.eos_token_id)
        finally:
            for h in handles:
                h.remove()
        gen = out[0][ids.shape[1]:]
        hit_cap = int(gen.shape[0]) >= max_new
        stopped = bool(gen.numel() and int(gen[-1].item()) == self.eos)
        body = gen[:-1] if stopped else gen
        toks = self.tok.batch_decode(body, skip_special_tokens=True)
        sp = extract_speech_ids(toks)
        return Run(len(sp), hit_cap, stopped, units=sp)


class QwenBackend:
    family = "qwen"

    def __init__(self, key: str, gpu: int) -> None:
        spec = BY_KEY[key]
        self.spec, self.device = spec, f"cuda:{gpu}"
        from qwen_tts import Qwen3TTSModel
        from src.models.qwen_gen import build_prompt, decode_step_count, eos_trim_length
        self._steps, self._trim = decode_step_count, eos_trim_length
        self.tts = Qwen3TTSModel.from_pretrained(
            spec.hf_id, device_map=self.device, dtype=torch.bfloat16,
            attn_implementation="eager")
        self.talker = self.tts.model.talker
        self.layers = self.talker.model.layers
        self.eos = self.talker.config.codec_eos_token_id
        self.prompt_items = build_prompt(self.tts)
        from src.models.qwen_gen import TalkerGenerateCapture
        self.cap = TalkerGenerateCapture(self.talker)
        # The voice-clone prefill is 85 rows for every item and every text: the
        # ICL block is padded/truncated to the reference audio's frame count, so
        # it does not depend on the synthesis text. Asserted per generation.
        self._plen = 85

    def n_layers(self) -> int:
        return len(self.layers)

    def plen(self, text: str) -> int:
        return self._plen

    def generate(self, text: str, seed: int, max_new: int,
                 hooks: dict[int, object] | None = None) -> Run:
        handles = []
        for L, h in (hooks or {}).items():
            handles.append(self.layers[L].register_forward_hook(h, with_kwargs=True))
        try:
            torch.manual_seed(seed)
            self.cap.clear()
            wavs, sr = self.tts.generate_voice_clone(
                text=text, language="English",
                voice_clone_prompt=self.prompt_items, max_new_tokens=max_new)
        finally:
            for h in handles:
                h.remove()
        result = self.cap.result
        # Rendered frames = len(hidden_states) - 1, NOT `eos_trim_length`.
        # `qwen_gen.eos_trim_length` walks `hidden_states[j+1]` for j in
        # range(len(hidden_states)) and so runs off the end of the tuple; on the
        # transformers version installed here it also never fires, because HF
        # halts the instant EOS is sampled and the EOS frame is never
        # materialised. The library's own vocoder drops one row, so the frame
        # count that matches the published ledger -- and the returned waveform,
        # to the sample -- is one less than the number of forward calls. Both
        # identities are asserted rather than trusted.
        n_calls = self._steps(result)
        n_units = max(0, n_calls - 1)
        wav = np.asarray(wavs[0])
        from_wav = int(round(len(wav) / float(sr) * self.spec.token_rate_hz))
        assert abs(from_wav - n_units) <= 1, (
            f"frame count disagrees with the waveform: {n_units} vs {from_wav}")
        plen = int(result.hidden_states[0][0][0].shape[1])
        assert plen == self._plen, f"prefill width moved: {plen} != {self._plen}"
        hit_cap = n_units >= max_new
        return Run(int(n_units), bool(hit_cap), not hit_cap,
                   wav=wav, sr=int(sr))


def make_backend(key: str, gpu: int):
    fam = CONFIG[key]["family"]
    return LlasaBackend(key, gpu) if fam == "llasa" else QwenBackend(key, gpu)


# ---------------------------------------------------------------- directions


def build_directions(model: str, layer: int, stim: dict) -> dict:
    """Count directions at one layer, from the activations already on disk.

    Same construction as `analysis/causal_count.py`'s: difference-in-means of
    terminal states between high-$k$ and low-$k$ repeated items, and the ridge
    weight vector for $\\log_2 k$. The only change is that the stored probe grid
    is not the whole layer stack on every checkpoint (Qwen stores every third
    layer, Llasa-8B every fourth), so the requested layer is looked up in the
    file's own `probe_layers` array instead of being used as a column index --
    silently indexing column 12 of a ten-column array is exactly the kind of
    off-by-a-grid error that would produce a confident direction pointing
    nowhere.
    """
    act = DATA_ROOT / "activations" / model
    H, ks, pos_norm = [], [], []
    col = None
    for iid, it in stim.items():
        if it["family"] != "word_rep":
            continue
        f = act / f"{iid}_s0.npz"
        if not f.exists():
            continue
        try:
            z = np.load(f)
            h, probes = z["hidden"], list(int(p) for p in z["probe_layers"])
        except Exception:  # noqa: BLE001
            continue
        if layer not in probes:
            raise SystemExit(
                f"[dirs] layer {layer} is not on {model}'s stored probe grid "
                f"{probes}; choose a patch/steer layer from that grid")
        c = probes.index(layer)
        col = c if col is None else col
        assert col == c, "probe grid differs between activation files"
        if h.shape[0] < 24:
            continue
        w = max(4, h.shape[0] // 3)
        tail = h[-w:, c, :].astype(np.float64)
        H.append(tail.mean(axis=0))
        pos_norm.append(float(np.linalg.norm(tail, axis=1).mean()))
        ks.append(it["k"])
    H = np.stack(H)
    ks = np.asarray(ks, dtype=float)

    lo, hi = np.isin(ks, LOW_K), np.isin(ks, HIGH_K)
    d_dim = H[hi].mean(0) - H[lo].mean(0)
    v_dim = d_dim / np.linalg.norm(d_dim)

    y = np.log2(ks)
    Xc = H - H.mean(0, keepdims=True)
    A = Xc.T @ Xc + 1.0 * np.eye(Xc.shape[1])
    w_ridge = np.linalg.solve(A, Xc.T @ (y - y.mean()))
    v_ridge = w_ridge / np.linalg.norm(w_ridge)

    out = {}
    for name, v in (("dim", v_dim), ("ridge", v_ridge)):
        proj = H @ v
        delta = float(proj[hi].mean() - proj[lo].mean())
        out[name] = dict(v=v.astype(np.float32), delta=abs(delta),
                         sign=1.0 if delta >= 0 else -1.0,
                         mean_pos_norm=float(np.mean(pos_norm)),
                         rel_magnitude_at_alpha1=abs(delta) / float(np.mean(pos_norm)),
                         n_items=int(H.shape[0]), layer=int(layer))
    return out


# ---------------------------------------------------------------- bookkeeping


class Store:
    """Audio/tokens plus a manifest, in the layout the standard pipeline expects."""

    def __init__(self, key: str, family: str) -> None:
        self.key, self.family = key, family
        self.audio_dir = DATA_ROOT / "audio" / key
        self.tok_dir = DATA_ROOT / "tokens" / key
        (self.audio_dir if family == "qwen" else self.tok_dir).mkdir(
            parents=True, exist_ok=True)
        self.manifest = REPO / "data/results" / f"causal_{key}_manifest.jsonl"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if self.manifest.exists():
            for line in self.manifest.open():
                try:
                    r = json.loads(line)
                    self.done[r["cond_id"]] = r
                except Exception:  # noqa: BLE001
                    pass
        self.f = self.manifest.open("a")

    def has(self, cond_id: str) -> bool:
        return cond_id in self.done

    def put(self, rec: dict, run: Run) -> None:
        if self.family == "qwen":
            import soundfile as sf
            sf.write(str(self.audio_dir / f"{rec['stem']}.wav"), run.wav, run.sr)
        else:
            np.save(self.tok_dir / f"{rec['stem']}.npy",
                    np.asarray(run.units, dtype=np.int32))
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.done[rec["cond_id"]] = rec

    def payload(self, cond_id: str) -> np.ndarray | None:
        """The stored artifact for a condition, for the bitwise no-op gate."""
        r = self.done.get(cond_id)
        if r is None:
            return None
        if self.family == "qwen":
            import soundfile as sf
            p = self.audio_dir / f"{r['stem']}.wav"
            if not p.exists():
                return None
            w, _ = sf.read(str(p), dtype="float32")
            return np.asarray(w)
        p = self.tok_dir / f"{r['stem']}.npy"
        return np.load(p) if p.exists() else None


def stem_for(cond_id: str) -> str:
    return (cond_id.replace("|", "__").replace("+", "p")
            .replace("-", "m").replace(".", "d"))


def load_stimuli() -> dict:
    return {json.loads(l)["item_id"]: json.loads(l)
            for l in (REPO / "data/stimuli/stimuli.jsonl").open()}


def item_id(fam: str, tmpl: str, k: int, stim: dict) -> str | None:
    for iid, it in stim.items():
        if it["family"] == fam and it["template"] == tmpl and it["k"] == k:
            return iid
    return None


def rec_base(**kw) -> dict:
    r = dict(arm=None, kind=None, recv_item=None, recv_k=None, family=None,
             template=None, seed=None, donor_item=None, donor_k=None,
             donor_kind="none", donor_seed=None, layer=None, patch_pos=None,
             patch_pos_eff=None, alpha=None, direction=None, rel_magnitude=None,
             n_gen_tokens=None, n_speech_tokens=None, hit_cap=None, stopped=None,
             noop_identical=None, hook_fired=None, max_abs_delta=None)
    r.update(kw)
    return r


def same_payload(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return False
    return bool(a.shape == b.shape and bool((a == b).all()))


def ledger(model: str) -> dict[tuple[str, int], dict]:
    """The published generation ledger, for the fidelity check (G0)."""
    p = DATA_ROOT / "tokens" / f"{model}_meta.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        try:
            r = json.loads(line)
            out[(r["item_id"], int(r["seed"]))] = r
        except Exception:  # noqa: BLE001
            pass
    return out


def payload_of(be, run: Run) -> np.ndarray:
    """The artifact a bitwise gate compares: the waveform, or the token ids."""
    return run.wav if be.family == "qwen" else np.asarray(run.units, dtype=np.int32)


# ---------------------------------------------------------------- sanity gate


def selftest(be, model: str, stim: dict) -> bool:
    """G1-G5, run alone before any grid is spent."""
    cfg = CONFIG[model]
    L, P, mx = cfg["central_layer"], cfg["patch_pos"], cfg["max_new"]
    rid = item_id("word_rep", "t1", 16, stim)
    did = item_id("word_rep", "t1", 32, stim)
    rtext, dtext = stim[rid]["text"], stim[did]["text"]
    plen = be.plen(rtext)
    ok = True

    def payload(r: Run):
        return payload_of(be, r)

    r1 = be.generate(rtext, 0, mx)
    r2 = be.generate(rtext, 0, mx)
    rep = same_payload(payload(r1), payload(r2))
    print(f"  [G1] free generation reproducible bitwise: {rep}")
    ok &= rep

    # G0 is a DIAGNOSTIC, not a gate, and it was added during implementation
    # rather than pre-committed (the pre-committed gate set is G1-G5). It asks
    # whether the free baseline reproduces the trajectory already on disk.
    #
    # It passes on Qwen-0.6B (113 frames, exactly the ledger's) and fails on
    # Llasa-1B (873 tokens against the ledger's 1231 and the published causal
    # arm's 1528) -- and it fails for a reason outside this file: instantiating
    # `analysis/causal_count.py`'s own `Runner` in the current environment
    # (transformers 4.57.3, torch 2.11.0+cu130) and asking it for the same item
    # at the same seed returns 873 too. The published Llasa-1B causal manifest
    # is no longer re-derivable by the code that wrote it; the trajectories
    # sampled under a fixed seed have moved with the library stack.
    #
    # This is not fatal and it is not swept up: a causal null is a statement
    # about a population of items, not about one trajectory, and every
    # comparison in this file is *internally paired* -- each patched run against
    # that same run's own free baseline, generated in the same process, in the
    # same environment, at the same seed. What it does forbid is treating the
    # Llasa-1B bridge as a bitwise re-run of the published one. U1 was
    # pre-committed as a statistical check (does the bridge median land inside
    # the published 90% CI) and stays one; it cannot be strengthened to bitwise
    # after the fact, and is not.
    led = ledger(model).get((rid, 0))
    if led is None:
        print("  [G0] no ledger row for this item; fidelity unchecked")
    else:
        agree = int(led["n_speech_tokens"]) == int(r1.n_units)
        print(f"  [G0] (diagnostic, not a gate) free baseline reproduces the "
              f"published ledger: {agree} "
              f"(ours {r1.n_units}, ledger {led['n_speech_tokens']})")

    rec = RecordDecodeHook(plen, plen + P)
    r3 = be.generate(rtext, 0, mx, hooks={L: rec})
    st = rec.stack()
    n_rec = 0 if st is None else int(st.shape[0])
    fired = (n_rec == min(P, r1.n_units + 1)) or (n_rec == P)
    print(f"  [G2] record hook captured {n_rec} decode positions "
          f"(window {P}, run emitted {r1.n_units}): {fired}")
    ok &= fired and same_payload(payload(r3), payload(r1))

    v = torch.from_numpy(build_directions(model, L, stim)["ridge"]["v"]).to(be.device)
    h = Rank1PatchHook(st.to(be.device), v, plen)
    r4 = be.generate(rtext, 0, mx, hooks={L: h})
    same = same_payload(payload(r4), payload(r1))
    print(f"  [G3] rank-1 no-op self-patch reproduces the free baseline bitwise: "
          f"{same} (max |delta| {h.max_abs_delta:.3g}, fired {h.fired})")
    ok &= same and h.max_abs_delta == 0.0

    drec = RecordDecodeHook(be.plen(dtext), be.plen(dtext) + P)
    be.generate(dtext, 0, mx, hooks={L: drec})
    dst = drec.stack()
    h2 = Rank1PatchHook(dst.to(be.device), v, plen)
    r5 = be.generate(rtext, 0, mx, hooks={L: h2})
    moved = not same_payload(payload(r5), payload(r1))
    print(f"  [G4] rank-1 donor patch changes the continuation: {moved} "
          f"(max |delta| {h2.max_abs_delta:.3g})")
    ok &= moved

    zero = torch.zeros(v.shape[0], dtype=torch.float32)
    hs = SteerHook(zero.to(be.device), plen)
    r6 = be.generate(rtext, 0, mx, hooks={cfg["steer_layer"]: hs})
    same0 = same_payload(payload(r6), payload(r1))
    print(f"  [G5] alpha=0 steering reproduces the free baseline bitwise: {same0} "
          f"(fired {hs.fired})")
    ok &= same0

    print(f"  SANITY GATE ({model}): {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ---------------------------------------------------------------- arm A


def arm_a(be, model: str, store: Store, stim: dict) -> None:
    cfg = CONFIG[model]
    P, mx = cfg["patch_pos"], cfg["max_new"]
    layers, central = cfg["patch_layers"], cfg["central_layer"]
    cells = [(L, P) for L in layers]
    t0, n_gen = time.time(), 0

    dirs = {L: torch.from_numpy(build_directions(model, L, stim)["ridge"]["v"])
            for L in layers}
    print(f"[A/{model}] ridge directions at layers {sorted(dirs)}; window {P} "
          f"positions ({P / BY_KEY[model].token_rate_hz:.2f} s)", flush=True)

    base_states: dict[tuple[str, int], dict[int, torch.Tensor]] = {}
    base_run: dict[tuple[str, int], Run] = {}

    def baseline(iid: str, seed: int) -> Run:
        """Free generation, with donor states recorded on the same decode path.

        Memoised: the k=8 item of a template donates to all three of that
        template's receivers, and re-sampling it would cost a quarter of the
        arm's budget for a sequence that is deterministic in the seed.
        """
        nonlocal n_gen
        if (iid, seed) in base_run:
            return base_run[(iid, seed)]
        text = stim[iid]["text"]
        plen = be.plen(text)
        recs = {L: RecordDecodeHook(plen, plen + P) for L in layers}
        run = be.generate(text, seed, mx, hooks=recs)
        base_run[(iid, seed)] = run
        base_states[(iid, seed)] = {L: recs[L].stack() for L in layers
                                    if recs[L].stack() is not None}
        cond = f"base|{iid}|s{seed}"
        if not store.has(cond):
            store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="A",
                               kind="baseline", recv_item=iid,
                               recv_k=stim[iid]["k"], family=stim[iid]["family"],
                               template=stim[iid].get("template"), seed=seed,
                               n_gen_tokens=run.n_units,
                               n_speech_tokens=run.n_units, hit_cap=run.hit_cap,
                               stopped=run.stopped), run)
            n_gen += 1
        return run

    contradicted = False
    for tmpl in PATCH_TEMPLATES:
        for rk in RECEIVER_KS:
            rid = item_id("word_rep", tmpl, rk, stim)
            if rid is None:
                continue
            rtext = stim[rid]["text"]
            plen = be.plen(rtext)

            specs: list[tuple[str, str, int, int | None]] = []
            for dk in DONOR_KS[rk]:
                d = item_id("word_rep", tmpl, dk, stim)
                if d:
                    specs.append(("crossk", d, dk, None))
            cid = item_id("control_word", tmpl, rk, stim)
            if cid:
                specs.append(("control", cid, rk, None))
            uid = UNRELATED[PATCH_TEMPLATES.index(tmpl) % len(UNRELATED)]
            if uid in stim:
                specs.append(("unrelated", uid, stim[uid]["k"], None))

            for seed in SEEDS:
                ref = baseline(rid, seed)
                plan: list[tuple[str, str, int, int, int]] = []
                for kind, did, dk, _ in specs:
                    ds = seed
                    baseline(did, ds)
                    for L in ([lay for lay, _ in cells] if kind == "crossk"
                              else [central]):
                        plan.append((kind, did, dk, ds, L))
                dsd = seed + DIFFSEED_OFFSET
                baseline(rid, dsd)
                plan.append(("diffseed", rid, rk, dsd, central))
                plan.append(("self", rid, rk, seed, central))

                for kind, did, dk, ds, L in plan:
                    cond = (f"patch|{rid}|s{seed}|{kind}|{did}|ds{ds}|"
                            f"L{L}|P{P}")
                    if store.has(cond):
                        continue
                    src = base_states.get((did, ds), {}).get(L)
                    if src is None or src.shape[0] == 0:
                        continue
                    p_eff = int(min(P, src.shape[0]))
                    hook = Rank1PatchHook(src[:p_eff].to(be.device),
                                          dirs[L].to(be.device), plen)
                    run = be.generate(rtext, seed, mx, hooks={L: hook})
                    identical = None
                    if kind == "self":
                        # Compared against the baseline *in memory*, never
                        # against the stored wav: soundfile writes float32 input
                        # to a WAV as PCM_16 by default, so a round trip through
                        # the file would quantise the signal and the gate would
                        # measure the container rather than the intervention.
                        identical = same_payload(payload_of(be, run),
                                                 payload_of(be, ref))
                        if not identical:
                            print(f"  !! NO-OP GATE FAILED at {cond}", flush=True)
                    store.put(rec_base(
                        cond_id=cond, stem=stem_for(cond), arm="A", kind="patch",
                        recv_item=rid, recv_k=rk, family="word_rep",
                        template=tmpl, seed=seed, donor_item=did, donor_k=int(dk),
                        donor_kind=kind, donor_seed=int(ds), layer=int(L),
                        patch_pos=int(P), patch_pos_eff=p_eff,
                        n_gen_tokens=run.n_units, n_speech_tokens=run.n_units,
                        hit_cap=run.hit_cap, stopped=run.stopped,
                        noop_identical=identical,
                        max_abs_delta=float(hook.max_abs_delta),
                        hook_fired=int(hook.fired)), run)
                    n_gen += 1
                    # Early warning only: the unit count is not the judged count,
                    # so this cannot decide anything -- it exists so that a gross
                    # disagreement with Llasa-1B surfaces during the run rather
                    # than after it.
                    if kind == "crossk" and not contradicted and ref.n_units:
                        rel = abs(run.n_units - ref.n_units) / ref.n_units
                        if rel > 0.5:
                            contradicted = True
                            print(f"  ** WATCH: {cond} moved unit count "
                                  f"{ref.n_units} -> {run.n_units} "
                                  f"({100 * rel:.0f}%); if the judged count "
                                  f"moves with it this CONTRADICTS-LLASA1B",
                                  flush=True)
                    del hook
                    torch.cuda.empty_cache()
                print(f"[A/{model}] {rid} s{seed}: {n_gen} generations, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[A/{model}] DONE {n_gen} generations in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- arm R


def arm_r(be, model: str, store: Store, stim: dict, seeds: list[int]) -> None:
    cfg = CONFIG[model]
    L, mx = cfg["steer_layer"], cfg["max_new"]
    t0, n_gen = time.time(), 0

    d = build_directions(model, L, stim)["ridge"]
    meta = {k: v for k, v in d.items() if k != "v"}
    meta["alphas"] = RIDGE_ALPHAS
    (REPO / "data/results" / f"causal_{store.key}_direction.json").write_text(
        json.dumps(meta, indent=2))
    print(f"[R/{model}] ridge@L{L}: {json.dumps(meta)}", flush=True)
    if d["delta"] <= 0:
        raise SystemExit("[R] U3: the low-to-high separation is zero; alpha has "
                         "no units and the sweep would be sweeping nothing")

    items: list[str] = []
    for k in RIDGE_KS:
        for tmpl in TEMPLATES:
            for fam in ("word_rep", "control_word"):
                iid = item_id(fam, tmpl, k, stim)
                if iid:
                    items.append(iid)
    print(f"[R/{model}] {len(items)} items x {len(RIDGE_ALPHAS)} alphas x "
          f"{len(seeds)} seeds", flush=True)

    for seed in seeds:
        for iid in items:
            it = stim[iid]
            for a in RIDGE_ALPHAS:
                cond = f"ridge|{iid}|s{seed}|L{L}|a{a:+.2f}"
                if store.has(cond):
                    continue
                vec = torch.from_numpy(
                    d["v"] * np.float32(a * d["delta"] * d["sign"]))
                hook = SteerHook(vec.to(be.device), be.plen(it["text"]))
                run = be.generate(it["text"], seed, mx, hooks={L: hook})
                identical = None
                if a == 0.0:
                    base = be.generate(it["text"], seed, mx)
                    identical = same_payload(payload_of(be, run),
                                             payload_of(be, base))
                    if not identical:
                        print(f"  !! ALPHA-0 GATE FAILED at {cond}", flush=True)
                    del base
                store.put(rec_base(
                    cond_id=cond, stem=stem_for(cond), arm="R", kind="steer",
                    recv_item=iid, recv_k=it["k"], family=it["family"],
                    template=it["template"], seed=seed, layer=int(L),
                    alpha=float(a), direction="ridge",
                    rel_magnitude=float(abs(a) * d["rel_magnitude_at_alpha1"]),
                    n_gen_tokens=run.n_units, n_speech_tokens=run.n_units,
                    hit_cap=run.hit_cap, stopped=run.stopped,
                    noop_identical=identical, hook_fired=int(hook.fired)), run)
                n_gen += 1
                del hook
                torch.cuda.empty_cache()
            if n_gen % 27 < len(RIDGE_ALPHAS):
                print(f"[R/{model}] {n_gen} generations, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[R/{model}] DONE {n_gen} generations in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CONFIG))
    ap.add_argument("--arm", choices=["A", "R"])
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    check_gpu(args.gpu)
    torch.cuda.set_device(args.gpu)
    stim = load_stimuli()
    be = make_backend(args.model, args.gpu)
    be.device = f"cuda:{args.gpu}"
    print(f"[{args.model}] {be.n_layers()} layers; config {CONFIG[args.model]}",
          flush=True)

    if args.selftest:
        raise SystemExit(0 if selftest(be, args.model, stim) else 1)
    if args.arm == "A":
        arm_a(be, args.model, Store(f"pq_{args.model}", be.family), stim)
    elif args.arm == "R":
        arm_r(be, args.model, Store(f"rq_{args.model}", be.family), stim,
              args.seeds)
    else:
        raise SystemExit("--arm A or --arm R (or --selftest)")


if __name__ == "__main__":
    main()
