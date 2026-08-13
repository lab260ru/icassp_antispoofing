#!/usr/bin/env python3
r"""Is the rank-1 count patch capable of changing this decoder's output at all?

WHY THIS FILE EXISTS
--------------------
`analysis/causal_count_second.py` reports a causal null: a rank-1 patch that
transplants only the ridge probe's count coordinate from a donor generation
(different $k$) into a receiver moves the rendered count not at all, with
equivalence bounds of 6--20% of a full donor-to-receiver transfer. The paper
reads that as "the count coordinate transplants cleanly and the decoder ignores
it".

A reviewer named the alternative the paper does not exclude:

  (a) the representation survives and the readout ignores it (our reading), or
  (b) **the patch simply does not carry enough to matter on any checkpoint** ---
      in which case the equivalence bound bounds an intervention too weak to
      test the hypothesis, and Llasa-8B's failure mode (same-$k$ and cross-$k$
      donors moving the count equally) is the tell.

Three facts already on disk sharpen (b) rather than answering it, and they are
recorded here before anything new was generated:

  * On the readable checkpoints `posthoc_all_cells_at_noise_floor` is true:
    informative, uninformative and unrelated donors all move the count by the
    same nothing.
  * Every patched run *does* differ from its reference --- 0/162 bitwise
    identical on `pq_qwen17b`, 0/324 on `rs_llasa8b` --- with a median
    $|\Delta \log \text{duration}|$ of 0.12--0.21. So the write is not a no-op.
  * But the `diffseed` cell --- donor states taken from the *same item at a
    different seed*, i.e. a pure re-roll of the sampler --- produces the same
    footprint (0.149 on qwen17b, 0.219 on llasa8b) and the same nothing in the
    count. **The patch's entire measured effect on the output is
    indistinguishable from resampling.** "The output changed" is therefore not
    evidence that the write carried information; any nonzero perturbation
    changes a sampled trajectory.

So the question this file answers is narrow and is not answered by any existing
artifact: *is a rank-1 write along a probe direction, at the magnitude we use,
capable of changing this decoder's output in a direction-appropriate way?*

WHAT IS HELD FIXED
------------------
Everything except the direction written and a scalar multiplier on the write.
Same checkpoints, same protocol per checkpoint (Qwen3-TTS-1.7B decode-time
paired against the free baseline; Llasa-8B the published resume protocol paired
against the resume reference), same patch layer, same window in seconds, same
receivers, same donors' role, same `Rank1PatchHook` arithmetic, same CTC judge,
same degeneracy flags, same population helper. If any of those moved the control
would no longer speak to the count result.

THE THREE DIRECTIONS.  All are fit with the *same* ridge machinery on the *same*
stored activations at the *same* layer with the *same* unit normalisation as the
count probe (`analysis/causal_count_second.build_directions`, whose ridge vector
this file re-derives and asserts equal to 1e-12 before using anything else).

  count    the paper's own direction: ridge weights for $\log_2 k$ over
           `word_rep` terminal states. Donor: the same template at a different
           $k$ (the paper's `crossk` cell). At $\alpha=1$ with scale exactly
           1.0 this **is** the published intervention, and G6 below checks that
           bitwise against the unmodified hook.
  carrier  ridge weights for a binary carrier-identity target (receiver's
           template = 0, donor's template = 1) over the same `word_rep` states.
           Donor: a `word_rep` item of a *different* template at the same $k$ ---
           a different sentence, a different repeated word. This is direction
           class (1) from the brief: a property the output cannot hide. Whether
           the receiver renders "the dog was very ... big" or "he walked far ...
           into the forest" is visible in the transcript with no inference.
           Leave-one-item-out sign accuracy of the carrier probe is recorded per
           checkpoint; a direction that is not decodable is not a control.
  rand     a fixed random unit direction (`default_rng(20260813 + layer)`),
           donor as for `count`. It has no appropriate direction and no S4; it
           exists to say whether disruption at large $\alpha$ is specific to a
           meaningful direction or is what any rank-1 write of that size does.

MAGNITUDE MATCHING.  The brief asks for "the identical magnitude". A different
direction has a different natural donor--receiver coefficient gap, so identical
magnitude has to be imposed. For each (receiver, seed) let
$M = \mathrm{med}_{\text{count donors}} \mathrm{med}_p |c_d(p) - c_r(p)|$ along
the count direction --- the size of the paper's own write. Every non-count
direction is rescaled by $s = M / \mathrm{med}_p|\delta_u(p)|$ so that its
median per-position write is $M$ too; the count direction is **not** rescaled
($s \equiv 1$) so that $\alpha=1$ reproduces the published patch exactly. The
realised median $|\delta|$ and its ratio to the mean per-position state norm are
recorded for every cell, so the matching is checkable rather than asserted.

THE LADDER.  $\alpha \in \{1, 2, 4, 8, 16\}$ multiplies the write.
$\alpha=1$ is exactly what the paper does.

SANITY GATES.  If any fails, this file reports nothing for that checkpoint and
the verdict is `pipeline inconclusive`.

  G1. Free generation is reproducible under a fixed seed, bitwise.
  G2. The record hook fires on exactly the positions it claims to.
  G3. The no-op self-patch (`count`, $\alpha=1$, donor = receiver) reproduces
      the reference bitwise, with the hook reporting max $|\delta|$ exactly 0.
  G6. `ScaledRank1PatchHook` at scale 1.0 and $\alpha=1$ produces a payload
      **bitwise identical** to the unmodified `Rank1PatchHook` from
      `analysis/causal_count.py` on the same real donor. This is the only new
      arithmetic in the file and it is checked empirically rather than by
      inspection.
  G7. The count ridge direction re-derived here is bitwise the one
      `causal_count_second.build_directions` returns.

------------------------------------------------------------------------------
PRE-COMMITTED INTERPRETATION.  Written before a single generation of this
experiment existed.  Detection statistics and thresholds are fixed here and are
not adjusted afterwards.
------------------------------------------------------------------------------

**The reference** for every cell is the one the checkpoint's own published arm
uses: the free baseline at the same seed (Qwen, decode-time) or the resume
reference at the same seed and window (Llasa-8B, published protocol).

**The noise floor** is the `diffseed` condition run inside this same experiment:
`count` direction, $\alpha=1$, donor = the *same receiver item at seed+100*.
That write carries no information about anything --- same text, same $k$, same
carrier --- so whatever it does to the output is what a re-roll of the sampler
does. Every threshold below is stated against that floor, measured in this run.

**Detection statistics**, computed per (direction, $\alpha$) cell over the pairs
in that cell:

  S1  **mean** over pairs of the word-level Levenshtein distance between the
      patched and reference CTC transcripts, divided by the reference's word
      count. "Did the rendered words change."

      AMENDMENT, and the only one.  S1 was written here as a *median* and is
      now a mean. The change was forced by a degeneracy, not by a result, and
      was made with 35 pilot rows on disk --- two of nine receivers, at most two
      pairs in any cell and exactly one diffseed pair, no cell near the n at
      which any threshold below can be evaluated and no verdict readable. A
      sampler re-roll usually leaves the *words* alone and moves only the
      timing, so the floor's median S1 is exactly 0; "S1 >= 2 x 0" is satisfied
      by any nonzero S1 whatsoever and would have made every cell DETECTABLE by
      construction --- i.e. the degenerate rule biases hard toward the
      conclusion the paper wants, which is the reason to fix it rather than
      live with it. The mean of a normalised word-edit distance cannot collapse
      that way. **No threshold value changed** (still a factor of 2 on S1, still
      25 points on S3); only the functional summarising S1. The median is still
      computed and reported beside the mean, together with a one-sided
      Mann--Whitney U of cell-vs-floor and the fraction of pairs with any word
      change at all, so a reader can apply either rule.
  S2  median over pairs of $|\Delta \log(\text{duration})|$.
  S3  degenerate-or-empty rate, in points, using the scorer's own flag rule.
  S4  the direction-appropriate signed shift, median over pairs, with a
      two-sided sign test and a 20000-sample bootstrap 90% interval from
      `analysis/equivalence.py` (N_BOOT=20000, ALPHA=0.05):
        count    $\Delta \log(1+\text{count})$ signed toward the donor's $k$ ---
                 the paper's own statistic, unchanged.
        carrier  $\Delta$ carrier score, where the carrier score of a transcript
                 is $\mathrm{lev}(\hat{t}, \text{receiver text}) -
                 \mathrm{lev}(\hat{t}, \text{donor text})$ over words, each
                 normalised by the length of the text it is measured against.
                 Positive = the transcript moved toward the donor's sentence.
        rand     none. A random direction has no appropriate direction.

**A full transfer**, the denominator that makes S4 interpretable, is the
baseline gap between donor and receiver items of the same template, measured in
this run's own baselines: for `count`, $|\log(1+\text{count})|$ between the
donor-$k$ and receiver-$k$ baselines; for `carrier`, the carrier score of the
donor's own baseline transcript minus the receiver's (which is close to its
maximum by construction, since each renders its own sentence).

**Thresholds, fixed now:**

  DETECTABLE at (direction, $\alpha$)  iff  S1 $\ge 2 \times$ S1$_0$
                                       or  S3 $\ge$ S3$_0$ + 25 points,
  where S1$_0$, S3$_0$ are the diffseed floor's values in the same arm. The
  factor 2 is chosen because a statistic at its own re-roll floor is by
  construction not evidence of anything; the 25 points is the interpretable-band
  rule the ridge sweep in `causal_count_second.py` already pre-committed.

  DIRECTION-APPROPRIATE at (direction, $\alpha$)  iff  S4 median $> 0$, the
  two-sided sign test survives Holm correction across that direction's five
  $\alpha$ levels at $p<0.05$, **and** the 90% bootstrap interval on the median
  excludes 0, **and** the cell is interpretable (degenerate-or-empty $\le 25$%,
  the same band rule). A shift produced only in cells whose audio has been
  destroyed is not a steer.

  TWO REPAIRS TO THE OPERATIONALISATION of that last clause, made at the same
  moment as the S1 amendment above --- 41 pilot rows, at most three pairs in any
  cell, no verdict readable --- and both of which can only make
  DIRECTION-APPROPRIATE *harder*, never easier. Neither introduces a new
  criterion; each makes the code implement the sentence already written above.

    R1. "destroyed" was operationalised as the scorer's `degenerate` flag, and
        that flag does not fire on the thing that actually happens at large
        $\alpha$: the generation *collapses*, emitting a 0.8 s stub in place of
        a 13 s utterance. 0.8 s is longer than the 0.25 s the flag tests, has
        ordinary RMS and ordinary spectral flatness, and is scored as a clean
        recording of almost nothing. A cell is therefore interpretable iff at
        most 25% of its pairs are degenerate **or** have a duration outside
        $[0.5, 2] \times$ the reference's.
    R2. a collapse *mimics a transfer* under both S4 definitions, and would be
        scored as a steer if the collapse guard ever let it through. Emitting
        nothing drives the count to 0, which is "toward the donor" for every
        lower-$k$ donor; and an empty transcript is equidistant from both
        sentences, which is "toward the donor" relative to a reference that
        renders the receiver's. So for the `count` direction
        DIRECTION-APPROPRIATE additionally requires the published **P2**
        condition, carried over unchanged from
        `analysis/causal_count_second.py`: the higher-$k$ and lower-$k$ donors
        must move the count in **opposite raw directions**. A collapse moves
        both the same way and fails it.

  A THIRD GUARD, for the degenerate case of the DETECTABLE rule: if the floor's
  mean S1 is exactly 0 --- possible if no sampler re-roll in the whole floor
  changes a single word --- the ratio is undefined, and DETECTABLE then requires
  a one-sided Mann--Whitney U of the cell's S1 against the floor's at $p<0.05$
  instead of the ratio. Stated here in advance rather than resolved later.

**VERDICTS.**

  **PATCH IS LIVE --- the null survives.**  At $\alpha=1$, `carrier` (written at
  the identical magnitude, at the identical layer and window, under the
  identical protocol) is DIRECTION-APPROPRIATE, while `count` is not. Then a
  rank-1 write of exactly this size demonstrably steers this decoder, the count
  write demonstrably does not, alternative (b) is excluded, and the paper's
  reading stands. Reportable as: "a magnitude-matched write along a carrier
  direction moves the rendered words by X; the same write along the count
  direction moves the rendered count by less than the bound."

  **PATCH IS INERT --- the null is uninformative.**  At $\alpha=1$ **no**
  direction is DETECTABLE (every cell within 2x the re-roll floor on S1 and
  within 25 points of it on S3), and the ladder's smallest DETECTABLE $\alpha$
  is $\ge 4$ for every direction. Then the intervention we published sits at
  least a factor of four below the magnitude at which this decoder registers a
  rank-1 write at all; the equivalence bound is a bound on an intervention too
  weak to test the hypothesis; the causal null must be demoted from "the decoder
  ignores the count coordinate" to **"underpowered intervention"**, and the
  paper must say so, quoting the factor.

  **INTERMEDIATE.**  Anything else, resolved into exactly these three, each with
  what it licenses:

    I-a  $\alpha=1$ IS DETECTABLE (the write escapes the re-roll floor) but no
         direction is DIRECTION-APPROPRIATE at any interpretable $\alpha$.
         Licenses: the write of the published magnitude measurably perturbs the
         decoder, so it is not inert; but we never positively demonstrated that
         *any* rank-1 write along *any* probe direction steers this decoder, so
         the count null is a null about a perturbation of demonstrated potency
         and undemonstrated specificity. The bound may stand; the paper must add
         that the intervention's specificity was never positively shown, and
         must not say "the decoder ignores it" without that qualification.
    I-b  no direction is DETECTABLE at $\alpha=1$ but some direction is
         DETECTABLE at $\alpha=2$. Licenses: the published intervention sits
         immediately below the decoder's sensitivity threshold. Report the
         ladder; "not at all" becomes "not at a magnitude the decoder is barely
         able to register".
    I-c  some direction is DIRECTION-APPROPRIATE, but only at $\alpha \ge 4$,
         with that cell still interpretable. Licenses: a rank-1 write along a
         probe direction *can* steer this decoder, but only at $\ge 4\times$ the
         magnitude the count transplant supplies. The count null is then a null
         at a magnitude demonstrably below the steering threshold and must be
         demoted toward `underpowered`, quoting the factor.

  If `carrier` is never DIRECTION-APPROPRIATE at any $\alpha$ while `rand` and
  `count` destroy the audio at the same $\alpha$, the honest reading is I-a and
  not "the carrier is not causal": the carrier text is also in the prompt and
  attended to at every step, so a residual-stream write is not the only route to
  it. This is written down now so it cannot be discovered later.

  A verdict is issued **per checkpoint**. Llasa-8B is the one the reviewer's
  claim is about; a disagreement between the two checkpoints is reported as a
  disagreement, not averaged.

**WHAT WOULD MAKE THIS EXERCISE UNINTERPRETABLE**

  V1. any sanity gate failing on a checkpoint;
  V2. the carrier probe's leave-one-item-out sign accuracy below 0.9 at the
      patch layer, which would mean the "control direction" is not decodable and
      cannot be a control;
  V3. the diffseed floor itself being degenerate in more than a quarter of runs,
      which would mean the floor is not a floor.

Usage:
  python analysis/causal_positive_control.py --model qwen17b --gpu 2 --selftest
  python analysis/causal_positive_control.py --model qwen17b --gpu 2
  python analysis/causal_positive_control.py --model llasa8b --gpu 3 --selftest
  python analysis/causal_positive_control.py --model llasa8b --gpu 3
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

from analysis.causal_count import Rank1PatchHook  # noqa: E402
from analysis.causal_count_second import (  # noqa: E402
    CONFIG, RecordDecodeHook, Run, Store, build_directions, item_id,
    load_stimuli, payload_of, same_payload, stem_for,
)

DATA_ROOT = Path(DATA_ROOT)

# ------------------------------------------------------------------ design

PROTOCOL = {"qwen17b": "decode", "llasa8b": "resume", "qwen06b": "decode",
            "llasa1b": "resume"}

ALPHAS = [1.0, 2.0, 4.0, 8.0, 16.0]
DIRECTIONS = ["count", "carrier", "rand"]

PATCH_TEMPLATES = ["t1", "t3", "t5"]
CARRIER_DONOR_TEMPLATE = {"t1": "t3", "t3": "t5", "t5": "t1"}
RECEIVER_KS = [16, 24, 32]
DONOR_KS = {16: [8, 32], 24: [8, 32], 32: [8, 16]}
SEEDS = [0]
DIFFSEED_OFFSET = 100
RAND_SEED_BASE = 20260813


# ------------------------------------------------------------------ the hook


class ScaledRank1PatchHook(Rank1PatchHook):
    """`Rank1PatchHook` with a scalar multiplier on the write.

    The parent adds ``(c_donor - c_recv) * v`` at each patched position. This
    adds ``scale * (c_donor - c_recv) * v``. `scale` is ``alpha * s``, where
    `s` is the magnitude-matching factor (exactly 1.0 for the count direction).

    The multiplication is skipped entirely when ``scale == 1.0`` so that the
    float32 arithmetic is not merely equal but *the same operations* as the
    parent's. That is what makes G6 --- a bitwise comparison of a full
    generation under this hook against one under the parent --- meaningful
    rather than decorative. The parent's own bitwise no-op gate (G3) depends on
    ``delta`` being exactly ``0`` for a self-patch, and ``0 * scale`` is still
    exactly ``0``, so the no-op survives any scale as well.
    """

    def __init__(self, states: torch.Tensor, vec: torch.Tensor, lo: int,
                 scale: float = 1.0) -> None:
        super().__init__(states, vec, lo)
        self.scale = float(scale)

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
        if self.scale == 1.0:
            return super().__call__(module, args, kwargs, output)
        hs = output[0] if isinstance(output, tuple) else output
        cp = kwargs.get("cache_position")
        if cp is None:
            return output
        sel = (cp >= self.lo) & (cp < self.hi)
        if not bool(sel.any()):
            return output
        self.fired += int(sel.sum())
        idx = (cp[sel] - self.lo).to(self.states.device)
        v = self.vec.to(device=hs.device, dtype=hs.dtype)
        v32 = self.vec.to(device=hs.device, dtype=torch.float32)
        cur = self._coeff(hs[0, sel, :], v32)
        don = self._coeff(self.states[idx].to(hs.device), v32)
        delta = (don - cur) * self.scale
        self.max_abs_delta = max(self.max_abs_delta, float(delta.abs().max()))
        hs = hs.clone()
        hs[0, sel, :] = hs[0, sel, :] + (delta.to(hs.dtype)[:, None] * v[None, :])
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


# ------------------------------------------------------------------ directions


def _load_H(model: str, layer: int, stim: dict,
            family: str = "word_rep") -> tuple[np.ndarray, list[str], np.ndarray,
                                               list[str], float]:
    """Terminal-window mean states per item, exactly as `build_directions` does.

    The probe grid differs per checkpoint (`every3` on Qwen, `every4` on
    Llasa-8B), so the layer is looked up in each file's own `probe_layers`
    array rather than used as a column index. Duplicated from
    `build_directions` rather than refactored out of it, because that function
    is load-bearing for the published result and must not be edited here; the
    duplication is checked against it in `fit_directions` below, which asserts
    the ridge vector re-derived from this loader is bitwise the published one.
    """
    act = DATA_ROOT / "activations" / model
    H, ks, iids, tmpls, pos_norm = [], [], [], [], []
    col = None
    for iid, it in stim.items():
        if it["family"] != family:
            continue
        f = act / f"{iid}_s0.npz"
        if not f.exists():
            continue
        try:
            z = np.load(f)
            h, probes = z["hidden"], [int(p) for p in z["probe_layers"]]
        except Exception:  # noqa: BLE001
            continue
        if layer not in probes:
            raise SystemExit(f"[dirs] layer {layer} not on {model}'s grid {probes}")
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
        iids.append(iid)
        tmpls.append(it["template"])
    return (np.stack(H), iids, np.asarray(ks, dtype=float), tmpls,
            float(np.mean(pos_norm)))


def _ridge_dir(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The count probe's own estimator: centred ridge, alpha=1, unit-normalised."""
    Xc = H - H.mean(0, keepdims=True)
    A = Xc.T @ Xc + 1.0 * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - y.mean()))
    return w / np.linalg.norm(w)


def _loto_sign_accuracy(H: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-item-out accuracy of the sign of the centred ridge prediction.

    A control direction has to be a direction the state actually carries. This
    refits the same estimator with each item held out and asks whether the
    held-out item lands on the correct side of the training mean.

    Solved in the dual (n x n rather than d x d, identical solution because
    $(X^\\top X + I)^{-1}X^\\top = X^\\top(XX^\\top + I)^{-1}$), because n is
    at most 60 and d is 2048--4096. Only the diagnostic uses the dual form; the
    direction itself is fit by the primal expression `build_directions` uses,
    and G7 asserts the two agree.
    """
    n, ok = H.shape[0], 0
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ytr = H[tr], y[tr]
        mu, ybar = Xtr.mean(0, keepdims=True), ytr.mean()
        A = Xtr - mu
        dual = np.linalg.solve(A @ A.T + 1.0 * np.eye(A.shape[0]), ytr - ybar)
        pred = float((H[i] - mu[0]) @ (A.T @ dual)) + ybar
        ok += int((pred >= ybar) == (y[i] >= ybar))
    return ok / n


def fit_directions(model: str, layer: int, stim: dict) -> dict:
    """count / carrier(t_recv -> t_donor) / rand at one layer, plus diagnostics."""
    H, iids, ks, tmpls, mean_norm = _load_H(model, layer, stim)
    tmpls = np.asarray(tmpls)

    # G7: the count direction re-derived from this loader must be the published
    # one, bitwise-ish, or the loader has drifted and nothing below is comparable.
    v_count = _ridge_dir(H, np.log2(ks))
    v_pub = build_directions(model, layer, stim)["ridge"]["v"]
    # `build_directions` stores the direction as float32, so the comparison is
    # made at float32 -- which is also the dtype the hook receives. Bitwise
    # equality there is the strongest statement available and is what is
    # required: anything weaker would let the loader drift.
    assert np.array_equal(v_count.astype(np.float32), v_pub), \
        "[G7] the count direction re-derived here is not the published one"
    v_count = v_pub.astype(np.float64)

    rng = np.random.default_rng(RAND_SEED_BASE + layer)
    r = rng.normal(size=H.shape[1])
    v_rand = r / np.linalg.norm(r)

    carriers = {}
    for a, b in CARRIER_DONOR_TEMPLATE.items():
        sel = np.isin(tmpls, [a, b])
        if sel.sum() < 6:
            continue
        y = (tmpls[sel] == b).astype(float)
        v = _ridge_dir(H[sel], y)
        proj = H[sel] @ v
        carriers[f"{a}->{b}"] = dict(
            v=v.astype(np.float32), n_items=int(sel.sum()),
            loto_sign_accuracy=_loto_sign_accuracy(H[sel], y),
            separation=float(proj[y == 1].mean() - proj[y == 0].mean()))

    lo, hi = np.isin(ks, [2, 3, 4]), np.isin(ks, [16, 24, 32])
    proj = H @ v_count
    return dict(
        layer=int(layer), mean_pos_norm=mean_norm, n_items=int(H.shape[0]),
        count=dict(v=v_count.astype(np.float32),
                   loto_sign_accuracy=_loto_sign_accuracy(H, np.log2(ks)),
                   separation=float(proj[hi].mean() - proj[lo].mean())),
        rand=dict(v=v_rand.astype(np.float32)),
        carrier=carriers)


# ------------------------------------------------------------------ backends


class DecodeArm:
    """Qwen3-TTS: patch the first P decode steps of a free generation."""

    protocol = "decode"

    def __init__(self, model: str, gpu: int) -> None:
        from analysis.causal_count_second import QwenBackend
        self.be = QwenBackend(model, gpu)
        self.be.device = f"cuda:{gpu}"
        self.family = self.be.family
        self.device = self.be.device
        cfg = CONFIG[model]
        self.P, self.mx, self.layers = cfg["patch_pos"], cfg["max_new"], cfg["patch_layers"]
        self._base: dict[tuple[str, int], Run] = {}
        self._states: dict[tuple[str, int], dict[int, torch.Tensor]] = {}

    def plen(self, text: str) -> int:
        return self.be.plen(text)

    def baseline(self, text: str, iid: str, seed: int, layer: int) -> Run:
        key = (iid, seed)
        if key not in self._base:
            plen = self.plen(text)
            recs = {layer: RecordDecodeHook(plen, plen + self.P)}
            run = self.be.generate(text, seed, self.mx, hooks=recs)
            self._base[key] = run
            st = recs[layer].stack()
            self._states.setdefault(key, {})[layer] = st
        elif layer not in self._states.get(key, {}):
            plen = self.plen(text)
            recs = {layer: RecordDecodeHook(plen, plen + self.P)}
            self.be.generate(text, seed, self.mx, hooks=recs)
            self._states.setdefault(key, {})[layer] = recs[layer].stack()
        return self._base[key]

    def states(self, iid: str, seed: int, layer: int) -> torch.Tensor | None:
        return self._states.get((iid, seed), {}).get(layer)

    def reference(self, text: str, iid: str, seed: int, layer: int) -> Run:
        return self.baseline(text, iid, seed, layer)

    def patched(self, text: str, seed: int, layer: int, hook) -> Run:
        return self.be.generate(text, seed, self.mx, hooks={layer: hook})

    def payload(self, run: Run) -> np.ndarray:
        return payload_of(self.be, run)


class ResumeArm:
    """Llasa: the published protocol -- teacher-force P tokens, patch, resume."""

    protocol = "resume"
    family = "llasa"

    def __init__(self, model: str, gpu: int) -> None:
        import analysis.causal_count as cc
        spec = BY_KEY[model]
        cc.HF_ID = spec.hf_id
        self.cc = cc
        self.run = cc.Runner(gpu)
        self.device = self.run.device
        cfg = CONFIG[model]
        self.P, self.mx = cfg["patch_pos"], cfg["max_new"]
        self._gen: dict[tuple[str, int], torch.Tensor] = {}
        self._caps: dict[tuple[str, int, int], torch.Tensor] = {}

    def plen(self, text: str) -> int:
        return int(self.run.prompt(text).shape[1])

    def body_len(self, text: str, iid: str, seed: int) -> int:
        return int(self._body(text, iid, seed).shape[0])

    def _free(self, text: str, iid: str, seed: int) -> torch.Tensor:
        key = (iid, seed)
        if key not in self._gen:
            gen, _ = self.run.generate(self.run.prompt(text), seed, self.mx)
            self._gen[key] = gen
        return self._gen[key]

    def baseline(self, text: str, iid: str, seed: int, layer: int) -> Run:
        gen = self._free(text, iid, seed)
        sp = self.run.speech_ids(gen)
        hit = int(gen.shape[0]) >= self.mx
        return Run(len(sp), hit, not hit, units=sp)

    def states(self, iid: str, seed: int, layer: int) -> torch.Tensor | None:
        # Captured lazily against the item's own text, which the caller supplies
        # through `capture`. Kept out of `states` so the interface matches
        # DecodeArm; see `capture` below.
        return self._caps.get((iid, seed, layer))

    def capture(self, text: str, iid: str, seed: int, layer: int) -> torch.Tensor | None:
        key = (iid, seed, layer)
        if key not in self._caps:
            gen = self._free(text, iid, seed)
            caps = self.run.capture(self.run.prompt(text), gen, [layer], self.P)
            self._caps[key] = caps.get(layer)
        return self._caps[key]

    def _body(self, text: str, iid: str, seed: int) -> torch.Tensor:
        return self.run.strip_eos(self._free(text, iid, seed))

    def reference(self, text: str, iid: str, seed: int, layer: int) -> Run:
        body = self._body(text, iid, seed)
        p_eff = int(min(self.P, body.shape[0]))
        prefix = torch.cat([self.run.prompt(text)[0], body[:p_eff]]).unsqueeze(0)
        gen, hit = self.run.generate(prefix, seed, self.mx - p_eff)
        full = torch.cat([body[:p_eff], gen])
        sp = self.run.speech_ids(full)
        return Run(len(sp), bool(hit), not bool(hit), units=sp)

    def patched(self, text: str, iid: str, seed: int, layer: int, hook) -> Run:
        body = self._body(text, iid, seed)
        p_eff = int(min(self.P, body.shape[0]))
        prefix = torch.cat([self.run.prompt(text)[0], body[:p_eff]]).unsqueeze(0)
        gen, hit = self.run.generate(prefix, seed, self.mx - p_eff,
                                     hook_layer=layer, hook=hook)
        full = torch.cat([body[:p_eff], gen])
        sp = self.run.speech_ids(full)
        return Run(len(sp), bool(hit), not bool(hit), units=sp)

    def payload(self, run: Run) -> np.ndarray:
        return np.asarray(run.units, dtype=np.int32)


# ------------------------------------------------------------------ helpers


def coeff(states: torch.Tensor, v: np.ndarray) -> np.ndarray:
    """Per-position coordinate along `v`, in float32, as the hook computes it."""
    h = states.to(torch.float32).numpy()
    return h @ v.astype(np.float32)


def rec_pc(**kw) -> dict:
    r = dict(arm="PC", kind=None, recv_item=None, recv_k=None, family=None,
             template=None, seed=None, donor_item=None, donor_k=None,
             donor_template=None, donor_kind="none", donor_seed=None,
             layer=None, patch_pos=None, patch_pos_eff=None, alpha=None,
             direction=None, scale=None, write_median_abs_delta=None,
             write_rel_magnitude=None, n_gen_tokens=None, n_speech_tokens=None,
             hit_cap=None, stopped=None, noop_identical=None, hook_fired=None,
             max_abs_delta=None)
    r.update(kw)
    return r


# ------------------------------------------------------------------ selftest


def selftest(arm, model: str, stim: dict, dirs: dict) -> bool:
    cfg = CONFIG[model]
    L = cfg["central_layer"]
    rid = item_id("word_rep", "t1", 16, stim)
    did = item_id("word_rep", "t1", 32, stim)
    rtext, dtext = stim[rid]["text"], stim[did]["text"]
    ok = True

    print(f"  [G7] count direction re-derived and asserted equal: PASS")
    for tag, c in sorted(dirs["carrier"].items()):
        print(f"  [V2] carrier probe {tag}: LOTO sign accuracy "
              f"{c['loto_sign_accuracy']:.2f} on n={c['n_items']}, "
              f"separation {c['separation']:.3f}")
        ok &= c["loto_sign_accuracy"] >= 0.9
    print(f"  [--] count probe LOTO sign accuracy "
          f"{dirs['count']['loto_sign_accuracy']:.2f}, separation "
          f"{dirs['count']['separation']:.3f}")

    if arm.protocol == "decode":
        r1 = arm.baseline(rtext, rid, 0, L)
        r2 = arm.be.generate(rtext, 0, cfg["max_new"])
        rep = same_payload(arm.payload(r1), arm.payload(r2))
        print(f"  [G1] free generation reproducible bitwise: {rep}")
        ok &= rep
        st = arm.states(rid, 0, L)
        n_rec = 0 if st is None else int(st.shape[0])
        fired = n_rec in (cfg["patch_pos"], min(cfg["patch_pos"], r1.n_units + 1))
        print(f"  [G2] record hook captured {n_rec}/{cfg['patch_pos']}: {fired}")
        ok &= fired
        ref = r1
        dst = None
        arm.baseline(dtext, did, 0, L)
        dst = arm.states(did, 0, L)
    else:
        b1 = arm.baseline(rtext, rid, 0, L)
        b2 = arm.baseline(rtext, rid, 0, L)
        print(f"  [G1] free generation memoised/reproducible: "
              f"{b1.n_units == b2.n_units}")
        st = arm.capture(rtext, rid, 0, L)
        n_rec = 0 if st is None else int(st.shape[0])
        print(f"  [G2] capture returned {n_rec}/{cfg['patch_pos']} positions")
        ok &= n_rec > 0
        ref = arm.reference(rtext, rid, 0, L)
        arm.baseline(dtext, did, 0, L)
        dst = arm.capture(dtext, did, 0, L)

    v = torch.from_numpy(dirs["count"]["v"]).to(arm.device)
    plen = arm.plen(rtext)

    # G3 -- no-op self patch
    h = ScaledRank1PatchHook(st.to(arm.device), v, plen, 1.0)
    r_noop = (arm.patched(rtext, 0, L, h) if arm.protocol == "decode"
              else arm.patched(rtext, rid, 0, L, h))
    same = same_payload(arm.payload(r_noop), arm.payload(ref))
    print(f"  [G3] no-op self-patch bitwise: {same} "
          f"(max|delta| {h.max_abs_delta:.3g}, fired {h.fired})")
    ok &= same and h.max_abs_delta == 0.0

    # G6 -- scaled hook at scale 1.0 == the unmodified hook, bitwise
    p_eff = int(min(dst.shape[0], st.shape[0]))
    h_ref = Rank1PatchHook(dst[:p_eff].to(arm.device), v, plen)
    r_ref = (arm.patched(rtext, 0, L, h_ref) if arm.protocol == "decode"
             else arm.patched(rtext, rid, 0, L, h_ref))
    h_new = ScaledRank1PatchHook(dst[:p_eff].to(arm.device), v, plen, 1.0)
    r_new = (arm.patched(rtext, 0, L, h_new) if arm.protocol == "decode"
             else arm.patched(rtext, rid, 0, L, h_new))
    g6 = same_payload(arm.payload(r_ref), arm.payload(r_new))
    print(f"  [G6] ScaledRank1PatchHook(scale=1) == Rank1PatchHook, bitwise: {g6}")
    ok &= g6
    moved = not same_payload(arm.payload(r_ref), arm.payload(ref))
    print(f"  [G4] a real donor patch changes the continuation: {moved} "
          f"(max|delta| {h_ref.max_abs_delta:.3g})")
    ok &= moved

    print(f"  SANITY GATE ({model}): {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ------------------------------------------------------------------ the grid


def run_grid(arm, model: str, store: Store, stim: dict, dirs: dict,
             alphas: list[float], seeds: list[int]) -> None:
    cfg = CONFIG[model]
    L, P = cfg["central_layer"], cfg["patch_pos"]
    t0, n_gen = time.time(), 0
    mean_norm = dirs["mean_pos_norm"]

    def emit_baseline(iid: str, seed: int) -> Run:
        nonlocal n_gen
        text = stim[iid]["text"]
        run = arm.baseline(text, iid, seed, L)
        if arm.protocol == "resume":
            arm.capture(text, iid, seed, L)
        cond = f"base|{iid}|s{seed}"
        if not store.has(cond):
            store.put(rec_pc(cond_id=cond, stem=stem_for(cond), kind="baseline",
                             recv_item=iid, recv_k=stim[iid]["k"],
                             family=stim[iid]["family"],
                             template=stim[iid].get("template"), seed=seed,
                             layer=int(L), n_gen_tokens=run.n_units,
                             n_speech_tokens=run.n_units, hit_cap=run.hit_cap,
                             stopped=run.stopped), run)
            n_gen += 1
        return run

    def donor_states(iid: str, seed: int) -> torch.Tensor | None:
        emit_baseline(iid, seed)
        if arm.protocol == "resume":
            return arm.capture(stim[iid]["text"], iid, seed, L)
        return arm.states(iid, seed, L)

    for tmpl in PATCH_TEMPLATES:
        for rk in RECEIVER_KS:
            rid = item_id("word_rep", tmpl, rk, stim)
            if rid is None:
                continue
            rtext = stim[rid]["text"]
            plen = arm.plen(rtext)
            for seed in seeds:
                emit_baseline(rid, seed)
                rst = donor_states(rid, seed)
                if rst is None:
                    continue

                # the reference this checkpoint's own published arm uses
                refcond = (f"ref|{rid}|s{seed}|P{P}")
                if not store.has(refcond):
                    ref = (arm.baseline(rtext, rid, seed, L)
                           if arm.protocol == "decode"
                           else arm.reference(rtext, rid, seed, L))
                    store.put(rec_pc(cond_id=refcond, stem=stem_for(refcond),
                                     kind="reference", recv_item=rid, recv_k=rk,
                                     family="word_rep", template=tmpl, seed=seed,
                                     layer=int(L), patch_pos=int(P),
                                     n_gen_tokens=ref.n_units,
                                     n_speech_tokens=ref.n_units,
                                     hit_cap=ref.hit_cap, stopped=ref.stopped),
                              ref)
                    n_gen += 1

                # ---- donors, and the magnitude the paper's own write has
                count_donors = []
                for dk in DONOR_KS[rk]:
                    d = item_id("word_rep", tmpl, dk, stim)
                    if d:
                        count_donors.append((d, dk, seed))
                ctmpl = CARRIER_DONOR_TEMPLATE[tmpl]
                cdon = item_id("word_rep", ctmpl, rk, stim)
                carrier_donors = [(cdon, rk, seed)] if cdon else []
                diffseed_donor = (rid, rk, seed + DIFFSEED_OFFSET)

                vc = dirs["count"]["v"]
                mags = []
                for did, _dk, ds in count_donors:
                    dst = donor_states(did, ds)
                    if dst is None:
                        continue
                    n = int(min(dst.shape[0], rst.shape[0]))
                    mags.append(float(np.median(np.abs(
                        coeff(dst[:n], vc) - coeff(rst[:n], vc)))))
                if not mags:
                    continue
                M = float(np.median(mags))

                plan = []
                for did, dk, ds in count_donors:
                    plan.append(("count", "crossk", did, dk, ds, vc, None))
                    plan.append(("rand", "crossk", did, dk, ds,
                                 dirs["rand"]["v"], M))
                ckey = f"{tmpl}->{ctmpl}"
                if carrier_donors and ckey in dirs["carrier"]:
                    for did, dk, ds in carrier_donors:
                        plan.append(("carrier", "crosstemplate", did, dk, ds,
                                     dirs["carrier"][ckey]["v"], M))
                # the sampler re-roll floor, and the no-op gate
                plan.append(("count", "diffseed", diffseed_donor[0],
                             diffseed_donor[1], diffseed_donor[2], vc, None))
                plan.append(("count", "self", rid, rk, seed, vc, None))

                for dname, kind, did, dk, ds, v, target in plan:
                    dst = donor_states(did, ds)
                    if dst is None:
                        continue
                    n = int(min(dst.shape[0], rst.shape[0]))
                    nat = float(np.median(np.abs(
                        coeff(dst[:n], v) - coeff(rst[:n], v))))
                    s = 1.0 if target is None else (
                        float(target / nat) if nat > 0 else 0.0)
                    # The published resume protocol bounds the patched window by
                    # the receiver's own body length as well, so that the hook
                    # can never reach a position the prefix does not contain and
                    # start patching *generated* frames. Under the decode-time
                    # protocol there is no prefix and the bound does not apply.
                    p_eff = int(min(P, dst.shape[0]))
                    if arm.protocol == "resume":
                        p_eff = int(min(p_eff, arm.body_len(rtext, rid, seed)))
                    if p_eff <= 0:
                        continue
                    a_list = ([1.0] if kind in ("self", "diffseed") else alphas)
                    for a in a_list:
                        cond = (f"pc|{rid}|s{seed}|{dname}|{kind}|{did}|ds{ds}|"
                                f"a{a:g}|L{L}|P{P}")
                        if store.has(cond):
                            continue
                        hook = ScaledRank1PatchHook(
                            dst[:p_eff].to(arm.device),
                            torch.from_numpy(v).to(arm.device), plen,
                            float(a) * s)
                        run = (arm.patched(rtext, seed, L, hook)
                               if arm.protocol == "decode"
                               else arm.patched(rtext, rid, seed, L, hook))
                        identical = None
                        if kind == "self":
                            ref = (arm.baseline(rtext, rid, seed, L)
                                   if arm.protocol == "decode"
                                   else arm.reference(rtext, rid, seed, L))
                            # Compared in memory, never through the stored wav:
                            # soundfile writes float32 as PCM_16 and the gate
                            # would measure the container.
                            identical = same_payload(arm.payload(run),
                                                     arm.payload(ref))
                            if not identical:
                                print(f"  !! NO-OP GATE FAILED at {cond}",
                                      flush=True)
                        store.put(rec_pc(
                            cond_id=cond, stem=stem_for(cond), kind="patch",
                            recv_item=rid, recv_k=rk, family="word_rep",
                            template=tmpl, seed=seed, donor_item=did,
                            donor_k=int(dk),
                            donor_template=stim[did].get("template"),
                            donor_kind=kind, donor_seed=int(ds), layer=int(L),
                            patch_pos=int(P), patch_pos_eff=p_eff,
                            alpha=float(a), direction=dname, scale=float(a) * s,
                            write_median_abs_delta=float(a) * s * nat,
                            write_rel_magnitude=float(a) * s * nat / mean_norm,
                            n_gen_tokens=run.n_units,
                            n_speech_tokens=run.n_units, hit_cap=run.hit_cap,
                            stopped=run.stopped, noop_identical=identical,
                            max_abs_delta=float(hook.max_abs_delta),
                            hook_fired=int(hook.fired)), run)
                        n_gen += 1
                        del hook
                        torch.cuda.empty_cache()
                print(f"[PC/{model}] {rid} s{seed}: {n_gen} generations, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[PC/{model}] DONE {n_gen} generations in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = ap.parse_args()

    check_gpu(args.gpu)
    torch.cuda.set_device(args.gpu)
    stim = load_stimuli()
    cfg = CONFIG[args.model]
    L = cfg["central_layer"]
    dirs = fit_directions(args.model, L, stim)

    meta = dict(model=args.model, layer=int(L), patch_pos=int(cfg["patch_pos"]),
                protocol=PROTOCOL[args.model], alphas=args.alphas,
                seeds=args.seeds, mean_pos_norm=dirs["mean_pos_norm"],
                count=dict(loto_sign_accuracy=dirs["count"]["loto_sign_accuracy"],
                           separation=dirs["count"]["separation"]),
                carrier={k: {kk: vv for kk, vv in v.items() if kk != "v"}
                         for k, v in dirs["carrier"].items()})
    out = REPO / "data/results" / f"causal_pc_{args.model}_directions.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"[PC/{args.model}] {json.dumps(meta)}", flush=True)

    arm = (DecodeArm(args.model, args.gpu) if PROTOCOL[args.model] == "decode"
           else ResumeArm(args.model, args.gpu))
    if args.selftest:
        raise SystemExit(0 if selftest(arm, args.model, stim, dirs) else 1)
    store = Store(f"pc_{args.model}", arm.family)
    run_grid(arm, args.model, store, stim, dirs, args.alphas, args.seeds)


if __name__ == "__main__":
    main()
