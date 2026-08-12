#!/usr/bin/env python3
"""Is the repetition count *causally* used by the decoder, or merely decodable?

Every mechanistic claim this project has made about the count is correlational.
A ridge probe reads $\\log_2 k$ off Llasa-1B's residual stream at $R^2\\approx0.62$
(`analysis/probe_count.py`), and the probe-past-horizon experiment showed that
readability degrades with $k$ --- but a readable feature is not a used feature.
The rival account the paper has never been able to exclude is exactly this: the
count survives in the representation and only the output policy fails. That is a
*causal* claim about what the stop decision reads, and no correlational probe can
settle it. Nothing causal has ever been run on these states.

This script runs the two standard interventions, on the same decoder, scored by
the same CTC pipeline the behavioural results use.

Arm A --- activation patching. Take a repeated item at $k\\in\\{16,24,32\\}$,
teacher-force the first $P$ tokens it generated on its own, and at one layer $L$
overwrite the residual stream at those $P$ generated positions with the states a
*donor* item had at the same generated positions. Because the KV cache above $L$
is rebuilt from the patched states, everything the model attends to from that
window onward is donor-derived. Then resume free generation and ask the audio how
many repetitions came out. Donors:

  cross-k     the same carrier sentence asking for a different $k$. If the count
              is a causal variable, splicing in the states of a $k{=}32$ run
              should push a $k{=}16$ receiver's rendered count up, and a $k{=}8$
              donor should push it down.
  control     the length-matched distinct-word control at the receiver's own $k$.
  same-k      the *same item at a different sampling seed*. Prompt length is
              identical, so this donor carries no positional offset and no count
              difference: it measures how much the rendered count moves when you
              splice in foreign states that agree about the count. It is the
              control that separates "the count moved" from "the state moved".
  unrelated   a repeated-*sentence* item from a different family, carrying no
              adverb, no matching carrier and no information about the
              receiver's $k$. This is the disruption control: if splicing in
              states that cannot possibly encode the receiver's count moves the
              rendered count as much as an informative donor does, then the
              protocol is perturbing generation rather than transplanting a
              count, and no comparison between donors means anything.
  self        the receiver's own states (no-op). See the sanity gate.

Arm B --- steering. Build a count direction $\\hat v$ at one layer, either as the
difference-in-means of terminal states between high-$k$ and low-$k$ repeated
items or as the ridge probe's weight vector, and add $\\alpha\\Delta\\hat v$ to the
residual stream at every generated position, where $\\Delta$ is the projected
high-minus-low separation, so $\\alpha=1$ means "move the state by one full
low-$k$-to-high-$k$ step". Sweep $\\alpha$, and run the identical sweep on the
length-matched *control* item. A direction that is about the count should move
the repeated arm more than the control arm; a direction that is a generic
"say more speech" knob should move both.

------------------------------------------------------------------------------
PRE-COMMITTED INTERPRETATION.  Written before any result was looked at.
------------------------------------------------------------------------------

**The paired reference.** A patched run is not compared against the item's own
free generation, because the two consume the sampler's random stream at different
offsets --- the patched run is handed $P$ tokens for free and starts drawing at
step $P$. Every patched condition is instead paired with a *resume reference*:
the identical prefix of $P$ own tokens, resumed with no hook, same seed. Then the
only difference between a cell and its reference is the patch itself, and the
no-op patch becomes an exact test rather than an approximate one.

**Sanity gate (both arms).** The no-op self-patch must reproduce its resume
reference *bitwise*, and $\\alpha=0$ steering with the hook attached must
reproduce the free baseline *bitwise*, in 100% of runs. If either fails, the
intervention machinery is wrong and **nothing else in this file is reportable**
--- the verdict is `pipeline inconclusive` and the arms are not interpreted.
(`--selftest` runs the gate alone, before the grid, so a plumbing bug costs
minutes rather than a GPU-hour.)

**The readout.** Rendered count from the CTC judge, the same estimator as
`src/common/score_counts.py` (occurrences of the target unit for a repeated item,
units-rendered for a control). Llasa-1B's rendered count on repeated text is
violently heavy-tailed --- single-seed values of 3 and 476 both occur at $k{=}24$
--- so the effect is measured on $\\log(1+\\text{count})$, paired within receiver
run against that run's own resume reference, and summarised by the median paired
shift with a bootstrap CI and a two-sided sign test. Duration, speech-token
count, cap-hit rate and degenerate-audio rate are carried alongside every cell,
because an intervention that merely makes the model babble longer would move the
count too and must not be allowed to look like a count effect.

**Arm A counts as a causal hit** iff, at some (layer, position) cell:

  A1. the paired shift in $\\log(1+\\text{count})$ is in the donor's direction ---
      up for a higher-$k$ donor, down for a lower-$k$ donor --- with a two-sided
      sign test surviving Holm correction across all cells tested, **and the
      higher-$k$ and lower-$k$ donors moving the count in opposite raw
      directions**; AND
  A2. the same-$k$-different-seed donor at the matched cell does **not** produce a
      shift of comparable magnitude (else the effect is "foreign state", not
      "different count"); AND
  A3. the effect survives restricting to non-degenerate, non-cap-hit runs, and
      the patched arm's degenerate+empty rate does not exceed the reference
      arm's by more than 15 points.

Magnitude is reported as a **transfer ratio**: the paired shift divided by the
baseline gap between donor-$k$ and receiver-$k$ items. 1.0 means the receiver
rendered the donor's count; 0.0 means the patch did nothing.

**Arm B counts as a causal hit** iff:

  B1. rendered count is monotone in $\\alpha$ --- pooled Spearman $\\rho$ over all
      (item, $\\alpha$) pairs of the repeated arm with $|\\rho|>0.5$ and $p<0.05$,
      and consistent sign in at least two thirds of items; AND
  B2. degenerate+empty rate at $|\\alpha|=2$ exceeds baseline by no more than 15
      points; AND
  B3. specificity: $|$effect$|$ on repeated items exceeds $|$effect$|$ on the
      length-matched controls. If the two are within a factor of 1.5 the
      direction is reported as a **generic length knob, not a count direction**,
      whatever B1 says.

**A null is the expected outcome and is reportable as such.** If the sanity gate
passes and neither arm clears its bar, the finding is: *the count is linearly
decodable from these states but is not causally read by the stop decision at any
locus we intervened on*. That is the result this file is designed to be able to
return, and it dovetails with the F5-TTS duration intervention (fixing duration
repairs $k<12$ failures completely and $k\\ge12$ not at all): the stop decision
looks at how much speech has been emitted, not at how many units.

**A third verdict exists and is checked first: "intervention too disruptive to
interpret".** A null and a broken instrument look identical in a table of medians
and they are not the same claim, so the difference is decided by an explicit
control rather than by inspection. The verdict is returned when either

  D1. the **unrelated-donor** cell moves the rendered count by at least half as
      much (median $|$raw shift$|$) as the informative cross-$k$ cell at the same
      layer and position --- states that cannot encode the receiver's count are
      doing most of the work, so the protocol perturbs generation rather than
      transplanting a count; or
  D2. the patched arm's degenerate-plus-empty rate exceeds the reference arm's by
      more than 15 points.

`intervention too disruptive to interpret` outranks both other verdicts. We would
rather report "we could not test this" than a null a reviewer can dismantle.

**The token budget is part of the measurement.** At 2048 new tokens half of these
generations were stopped by our own budget rather than by the model, and this
repo already knows what that does: `src/common/population.py` excludes `hit_cap`
rows because their counts are censored downward (median relative error $-0.44$
against $-0.08$). A conclusion drawn over a half-censored population cannot tell
"the patch did nothing" from "the budget truncated everything", so the budget
here is 3584 tokens (about 72 s of audio), the cap-hit rate is reported per cell
rather than left in a JSON file, and every cell is also summarised on the one
readout a budget cannot censor: **the stop rate**, the fraction of runs that
emitted the end-of-speech token on their own. Whether an intervention makes the
decoder *stop* is, after all, the question the paper is actually about.

**Uninterpretable** if: the sanity gate fails; or more than half the runs in an
arm are degenerate, empty or cap-hit; or the bootstrap CI on the *reference*
median $\\log(1+\\text{count})$ at the receiver's $k$ is wider than the entire
baseline gap between donor-$k$ and receiver-$k$, in which case the instrument
cannot resolve an effect of the size we are looking for and the arm is declared
underpowered rather than null.

The opposite-directions requirement in A1 was added after the first 13 patched
runs had been scored and before the grid finished, for a reason visible in the
design rather than in those numbers: the donor pool is unbalanced two-to-one
toward lower-$k$ donors (a $k{=}32$ receiver has no higher-$k$ partner in the
ladder), so a patch that merely disrupts a runaway loop and shortens the
utterance would register as a signed shift "toward the donor" in two thirds of
pairs while being nothing of the kind. The requirement raises the bar and is
recorded here as a mid-flight strengthening rather than quietly folded in.

Confounds we cannot remove in a pilot, named so they are not silently ignored:
patching across two prompts of different lengths transfers states that were
computed at slightly different absolute positions (the $k{=}8$ and $k{=}32$
prompts differ by 24 tokens); the same-$k$-different-seed donor is the arm that
holds prompt length exactly fixed, which is why it is a control and not an
afterthought.

------------------------------------------------------------------------------
FOLLOW-UP (2026-08-12, after the first pass returned "too disruptive").
PRE-COMMITTED before any follow-up audio existed.
------------------------------------------------------------------------------

The first pass failed for two separable reasons, and each has one fix.

*Arm A failed because the intervention was too coarse.* Splicing a whole
2048-dimensional state transfers the donor's count and also its phase, its
speaker state, its position in its own sentence, and everything else. The
**rank-1 projection patch** transfers only the component along the count
direction: at layer $L$ it adds $(c_{\\text{donor}} - c_{\\text{recv}})\\hat v$,
where $c = h\\cdot\\hat v$ and $\\hat v$ is the ridge probe's weight vector at that
layer. This is exactly "replace the count coordinate, keep everything else", and
because the correction is written as a *difference* rather than as a
subtract-then-re-add, the self-patch adds precisely zero and the no-op gate stays
bitwise exact in bfloat16.

*Arm B failed because the dose was too large.* The difference-in-means direction
has $\\Delta=17.0$ against a mean per-position state norm of 37.8 --- $\\alpha=1$
is a 45% perturbation, and at $|\\alpha|\\ge0.5$ it simply destroys the audio
(100% degenerate at $\\alpha=-1$). The ridge direction is a third of that
($\\Delta=5.84$, 15%) and stayed inside the interpretable range. The follow-up
sweeps **only the ridge direction**, at $\\alpha\\in\\{0,\\pm0.25,\\pm0.5,\\pm0.75,
\\pm1\\}$, over $k\\in\\{16,24,32\\}$ and four carriers --- 12 repeated items and
their 12 length-matched controls, three times the items of the first pass.

**Interpretable band.** For any arm, an $\\alpha$ (or cell) whose degenerate-plus-
empty rate exceeds **25%** is out of band and is excluded from the dose-response
before it is fitted. If no band of at least three $\\alpha$ values survives, the
sweep is *underpowered*, not null.

**Ridge sweep counts as a hit (R)** iff all three hold:

  R1 *dose-response*: pooled Spearman $\\rho$ between $\\alpha$ and
     $\\log(1+\\text{count})$ on within-item z-scores of the repeated arm, computed
     inside the band, with $|\\rho|>0.4$, $p<0.05$, and consistent per-item sign
     in at least two thirds of items; AND
  R2 *not a duration artifact*: the same correlation, partialling out
     $\\log(\\text{duration})$, keeps its sign and $|\\rho_{\\text{partial}}|>0.25$;
     and every in-band $\\alpha$ has degeneracy $\\le 25\\%$; AND
  R3 *specificity*: the repeated arm's effect (median within-item log-count
     difference between the extreme in-band $\\alpha$s) is at least **1.5x** the
     control arm's.

**The lead is declared dead** --- and I will say so in exactly those words --- if
any one of:

  D-a the paired $\\alpha{=}{+}1$ vs $\\alpha{=}{-}1$ effect on repeated items
      shrinks below **0.35** log-count in median magnitude at $n\\ge16$ items
      (the first pass's +0.96 on 8 items is what justified this follow-up; less
      than about 40% of it means the first pass was noise);
  D-b the control arm moves as much as the repeated arm (ratio $<1.5$);
  D-c fewer than two thirds of repeated items agree on the sign;
  D-d no interpretable band of three or more $\\alpha$ values exists.

A hit and a death are both single sentences. There is no third option in which
the lead "still looks promising"; if it clears R1--R3 it is real, if it trips
D-a--D-d it is dead, and anything else is reported as underpowered with the
number of items that would be needed.

**Rank-1 patch counts as readable (P1)** iff the disruption index --- the
unrelated donor's median $|\\text{shift}|$ over the informative cross-$k$ donor's
--- falls **below 0.5**, i.e. a donor that actually differs in $k$ moves the
count at least twice as much as one that cannot know it. P1 is a gate, not a
result: **if P1 fails the arm stays uninterpretable and nothing else in it is
reported**, exactly as in the first pass. Conditional on P1:

  P2 cross-$k$ signed shift positive, sign test surviving Holm across cells, and
     higher-$k$ and lower-$k$ donors moving the count in opposite raw directions;
  P3 patched degeneracy no more than 15 points above the reference arm's, and
     stop-rate no more than 15 points below it.

P1 ∧ P2 ∧ P3 is a causal locus. **P1 ∧ ¬P2 is the outcome I expect and it is a
real result**: the count coordinate can be transplanted without damaging the
utterance, and moving it does not move the rendered count --- which would say the
stop decision does not read this direction, rather than that we cannot tell.

Usage (generation only; scoring is `analysis/causal_count_score.py`):
  python analysis/causal_count.py --arm A --gpu 3 --selftest
  python analysis/causal_count.py --arm A --gpu 3
  python analysis/causal_count.py --arm B --gpu 3
  python analysis/causal_count.py --arm R --gpu 2            # ridge alpha sweep
  python analysis/causal_count.py --arm A --rank1 --gpu 3    # rank-1 patch
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
from src.models.llasa_gen import (  # noqa: E402
    SPEECH_END, build_prompt, extract_speech_ids,
)

DATA_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts")
HF_ID = "HKUSTAudio/Llasa-1B"
MAX_NEW = 3584               # ~72 s of audio; see "token budget" above
TEMPERATURE = 0.8
TOP_P = 1.0

# t2 is excluded by the judge-vocabulary audit (`okay`/`hmm` are never emitted by
# the CTC recogniser), so it cannot measure anything here either.
TEMPLATES = ["t1", "t3", "t5", "t6"]
# Patching is the expensive arm --- every receiver-seed needs its own donor
# captures --- and at a 3584-token budget the card yields about three
# generations a minute. Three carriers rather than four buys back a fifth of
# that, and 36 pairs per cell is already an order of magnitude more than the
# pilot's first pass had. The steering arm keeps all four.
PATCH_TEMPLATES = ["t1", "t3", "t5"]
RECEIVER_KS = [16, 24, 32]
DONOR_KS = {16: [8, 32], 24: [8, 32], 32: [8, 16]}
SEEDS = [0, 1]
DIFFSEED_OFFSET = 100          # donor = same item, this seed offset

PATCH_LAYERS = [3, 7, 11]      # index into model.model.layers, of 16
PATCH_POS = [128]              # how many generated positions get the donor state
CENTRAL = (7, 128)
# (layer, P) cells swept for the cross-k donors. The pilot's first pass spread
# itself over three window lengths and got two pairs per cell, which cannot
# decide anything; this one spends the whole budget on one window length at
# three depths and gets 36. Window length is the axis to sacrifice because the
# question "which layer carries the count" is the one a mech-interp paper has to
# answer, and "how long a window" is a knob we can sweep later if the depth
# answer is interesting.
CELLS = [(3, 128), (7, 128), (11, 128)]
# Unrelated donors: repeated *sentences*, a different family with a different
# carrier, no adverb, and no information about the receiver's k. Chosen from the
# ones that ran to a natural stop rather than into the old token budget, so the
# donor states cover the whole patch window.
UNRELATED = ["sr_s1_k16", "sr_s4_k08", "sr_s1_k08", "sr_s2_k08"]

STEER_LAYER = 13               # the probe's best late layer on word_rep
STEER_LAYER_2 = 7
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
STEER_KS = [16, 32]
# Follow-up: the ridge direction only, finely and symmetrically, over three k
# and four carriers. The first pass had two alphas and eight items, which is
# why its rep-vs-control dissociation was a lead and not a result.
RIDGE_ALPHAS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
RIDGE_KS = [16, 24, 32]
LOW_K = [2, 3, 4]
HIGH_K = [16, 24, 32]


# ---------------------------------------------------------------- hooks


class PatchHook:
    """Overwrite one layer's output at a contiguous span of absolute positions.

    Registered with ``with_kwargs=True`` so the hook can read ``cache_position``,
    which is the model's own statement of which absolute positions this forward
    is computing. That is exact under both the prefill (many positions at once)
    and the incremental decode (one position), so the span never has to be
    inferred from tensor shapes.
    """

    def __init__(self, states: torch.Tensor, lo: int) -> None:
        self.states = states          # [P, d], for absolute positions [lo, lo+P)
        self.lo = lo
        self.hi = lo + states.shape[0]
        self.fired = 0

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
        hs = output[0] if isinstance(output, tuple) else output
        cp = kwargs.get("cache_position")
        if cp is None:
            return output
        sel = (cp >= self.lo) & (cp < self.hi)
        if not bool(sel.any()):
            return output
        self.fired += int(sel.sum())
        hs = hs.clone()
        idx = (cp[sel] - self.lo).to(self.states.device)
        hs[0, sel, :] = self.states[idx].to(device=hs.device, dtype=hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


class Rank1PatchHook:
    """Transfer only the donor's coordinate along one direction.

    At each patched position the hook adds ``(c_donor - c_recv) * v``, where
    ``c = h . v`` for a unit vector ``v``. That is algebraically the same as
    projecting the receiver's component out and writing the donor's in, but it
    is written as a *difference* on purpose: for a self-patch the two
    coefficients are the same number, the correction is exactly ``0 * v``, and
    ``h + 0`` is bitwise ``h``. The subtract-then-re-add form
    ``h - (h.v)v + (h.v)v`` is mathematically identical and numerically is not
    --- in bfloat16 it perturbs every patched position, which would break the
    no-op gate and, worse, would leave a small nonspecific perturbation in every
    real patch as well.

    The two coefficients are computed *inside* the hook, by the same expression,
    on contiguous float32 copies. That also looks like pedantry and also is not:
    computing the donor's coefficient outside with a matmul and the receiver's
    inside with another left them disagreeing by 1.5e-3 on a self-patch --- a
    different reduction order for a different stride pattern --- which is small,
    nonzero, and therefore fatal to a bitwise no-op gate. Same function, same
    layout, bitwise-equal inputs, bitwise-equal outputs, exact zero.
    """

    def __init__(self, states: torch.Tensor, vec: torch.Tensor, lo: int) -> None:
        self.states = states        # [P, d] donor states
        self.vec = vec              # [d] unit direction
        self.lo = lo
        self.hi = lo + states.shape[0]
        self.fired = 0
        self.max_abs_delta = 0.0

    @staticmethod
    def _coeff(h: torch.Tensor, v32: torch.Tensor) -> torch.Tensor:
        return (h.contiguous().to(torch.float32) * v32).sum(-1)

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
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
        delta = don - cur
        self.max_abs_delta = max(self.max_abs_delta, float(delta.abs().max()))
        hs = hs.clone()
        hs[0, sel, :] = hs[0, sel, :] + (delta.to(hs.dtype)[:, None] * v[None, :])
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


class RecordHook:
    """Copy one layer's output at the prefill, for the generated positions only."""

    def __init__(self, lo: int) -> None:
        self.lo = lo
        self.out: torch.Tensor | None = None

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
        hs = output[0] if isinstance(output, tuple) else output
        cp = kwargs.get("cache_position")
        if cp is None or int(cp.shape[0]) <= 1:
            return output
        self.out = hs[0, self.lo:, :].detach().to("cpu").clone()
        return output


class SteerHook:
    """Add a fixed vector to one layer's output at every *generated* position."""

    def __init__(self, vec: torch.Tensor, plen: int) -> None:
        self.vec = vec
        self.plen = plen
        self.fired = 0

    def __call__(self, module, args, kwargs, output):  # noqa: ANN001
        hs = output[0] if isinstance(output, tuple) else output
        cp = kwargs.get("cache_position")
        if cp is None:
            return output
        sel = cp >= self.plen
        if not bool(sel.any()):
            return output
        self.fired += int(sel.sum())
        hs = hs.clone()
        hs[0, sel, :] = hs[0, sel, :] + self.vec.to(device=hs.device, dtype=hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


# ---------------------------------------------------------------- runner


class Runner:
    def __init__(self, gpu: int) -> None:
        check_gpu(gpu)
        self.device = f"cuda:{gpu}"
        torch.cuda.set_device(gpu)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(HF_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(self.device).eval()
        self.eos = self.tok.convert_tokens_to_ids(SPEECH_END)
        self.layers = self.model.model.layers
        self._prompt_cache: dict[str, torch.Tensor] = {}

    def prompt(self, text: str) -> torch.Tensor:
        if text not in self._prompt_cache:
            self._prompt_cache[text] = build_prompt(self.tok, text).to(self.device)
        return self._prompt_cache[text]

    def generate(self, ids: torch.Tensor, seed: int, max_new: int,
                 hook_layer: int | None = None, hook=None) -> tuple[torch.Tensor, bool]:
        handle = None
        if hook is not None:
            handle = self.layers[hook_layer].register_forward_hook(hook, with_kwargs=True)
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                out = self.model.generate(
                    ids, max_new_tokens=max_new, eos_token_id=self.eos,
                    do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
                    pad_token_id=self.tok.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        gen = out[0][ids.shape[1]:]
        hit_cap = int(gen.shape[0]) >= max_new
        return gen, hit_cap

    def strip_eos(self, gen: torch.Tensor) -> torch.Tensor:
        return gen[:-1] if (gen.numel() and gen[-1].item() == self.eos) else gen

    def speech_ids(self, gen: torch.Tensor) -> list[int]:
        toks = self.tok.batch_decode(self.strip_eos(gen), skip_special_tokens=True)
        return extract_speech_ids(toks)

    def capture(self, prompt_ids: torch.Tensor, gen: torch.Tensor,
                layers: list[int], n: int) -> dict[int, torch.Tensor]:
        """Teacher-forced states at the first `n` generated positions.

        The model is causal, so one forward over [prompt + generated] reproduces
        exactly the states that produced each generated token during sampling.
        Kept in bfloat16 on the CPU: bf16 is the model's own dtype, so a donor
        round-trip through this buffer is lossless.

        The capture is taken from `generate`'s own prefill, over exactly
        `prompt + n` tokens, and not from a plain `model(...)` forward. That
        looks like pedantry and is not: a plain forward and a `generate` prefill
        of the same tokens disagree in bfloat16 at up to 1.0 absolute on this
        layer, because `generate` passes an explicit attention mask and a cache
        and lands on a different SDPA kernel. Capturing on the other path made
        the no-op patch differ from its reference by generation step 26 --- a
        pipeline that would have reported a spurious causal effect of about the
        size we are looking for, from nothing but rounding. States must be
        recorded on the same path they will be written back into.
        """
        g = self.strip_eos(gen)
        n = min(n, int(g.shape[0]))
        if n <= 0:
            return {}
        full = torch.cat([prompt_ids[0], g[:n]]).unsqueeze(0)
        plen = prompt_ids.shape[1]
        recs = {l: RecordHook(plen) for l in layers}
        handles = [self.layers[l].register_forward_hook(recs[l], with_kwargs=True)
                   for l in layers]
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                self.model.generate(
                    full, max_new_tokens=1, eos_token_id=self.eos, do_sample=True,
                    top_p=TOP_P, temperature=TEMPERATURE,
                    pad_token_id=self.tok.eos_token_id,
                )
        finally:
            for h in handles:
                h.remove()
        return {l: recs[l].out for l in layers if recs[l].out is not None}


# ---------------------------------------------------------------- bookkeeping


class Store:
    """Token files plus a manifest, in the layout the standard pipeline expects."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.tok_dir = DATA_ROOT / "tokens" / key
        self.tok_dir.mkdir(parents=True, exist_ok=True)
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

    def put(self, rec: dict, sp_ids: list[int]) -> None:
        np.save(self.tok_dir / f"{rec['stem']}.npy", np.asarray(sp_ids, dtype=np.int32))
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.done[rec["cond_id"]] = rec


_STEM: dict[str, str] = {}


def stem_for(cond_id: str) -> str:
    """Filesystem-safe stem; the manifest carries the meaning."""
    s = (cond_id.replace("|", "__").replace("+", "p")
         .replace("-", "m").replace(".", "d"))
    _STEM[cond_id] = s
    return s


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
             n_gen_tokens=None, n_speech_tokens=None, hit_cap=None,
             noop_identical=None, hook_fired=None)
    r.update(kw)
    return r


# ---------------------------------------------------------------- arm A


def arm_a(run: Runner, store: Store, stim: dict, rank1: bool = False) -> None:
    t0 = time.time()
    n_gen = 0
    receivers = [(t, k) for k in RECEIVER_KS for t in PATCH_TEMPLATES]
    # Rank-1 mode needs a direction at every patched layer. The ridge weight
    # vector is the one the probe actually leans on, and the one that stayed in
    # the interpretable range when it was used for steering.
    dirs = ({L: build_directions(L, stim)["ridge"]["v"] for L in PATCH_LAYERS}
            if rank1 else {})
    if rank1:
        print("[A/rank1] ridge directions at layers " + str(sorted(dirs)), flush=True)

    cache: dict[tuple[str, int], torch.Tensor] = {}

    def emit_baseline(iid: str, seed: int) -> torch.Tensor:
        """Free generation for `iid` at `seed`, recorded once, returned always.

        Memoised because the same donor serves several receivers: the k=8 item of
        a template is a donor for all three of that template's receivers, and
        re-sampling it each time would cost a quarter of the arm's GPU budget for
        a sequence we know is deterministic in the seed.
        """
        nonlocal n_gen
        if (iid, seed) in cache:
            return cache[(iid, seed)]
        prompt = run.prompt(stim[iid]["text"])
        gen, hit = run.generate(prompt, seed, MAX_NEW)
        cache[(iid, seed)] = gen
        cond = f"base|{iid}|s{seed}"
        if not store.has(cond):
            sp = run.speech_ids(gen)
            store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="A",
                               kind="baseline", recv_item=iid,
                               recv_k=stim[iid]["k"], family=stim[iid]["family"],
                               template=stim[iid]["template"], seed=seed,
                               n_gen_tokens=int(gen.shape[0]),
                               n_speech_tokens=len(sp), hit_cap=hit), sp)
            n_gen += 1
        return gen

    for tmpl, rk in receivers:
        rid = item_id("word_rep", tmpl, rk, stim)
        if rid is None:
            continue
        rprompt = run.prompt(stim[rid]["text"])

        donor_specs: list[tuple[str, str, int, int | None]] = []
        for dk in DONOR_KS[rk]:
            did = item_id("word_rep", tmpl, dk, stim)
            if did:
                donor_specs.append(("crossk", did, dk, None))
        cid = item_id("control_word", tmpl, rk, stim)
        if cid:
            donor_specs.append(("control", cid, rk, None))
        uid = UNRELATED[PATCH_TEMPLATES.index(tmpl) % len(UNRELATED)]
        if uid in stim:
            donor_specs.append(("unrelated", uid, stim[uid]["k"], None))

        for seed in SEEDS:
            r_gen = emit_baseline(rid, seed)
            r_body = run.strip_eos(r_gen)
            r_len = int(r_body.shape[0])

            # ---- donor states, captured at each patch length in the grid
            donors: dict[str, dict] = {}
            for kind, did, dk, dseed in donor_specs + [("diffseed", rid, rk,
                                                        seed + DIFFSEED_OFFSET)]:
                ds = seed if dseed is None else dseed
                d_gen = emit_baseline(did, ds)
                dprompt = run.prompt(stim[did]["text"])
                caps = {P: run.capture(dprompt, d_gen, PATCH_LAYERS, P)
                        for P in PATCH_POS}
                donors[f"{kind}|{did}|s{ds}"] = dict(caps=caps, k=dk, seed=ds,
                                                     kind=kind, item=did)
                del d_gen
                torch.cuda.empty_cache()

            # the receiver's own states, for the no-op gate
            self_caps = {P: run.capture(rprompt, r_gen, [CENTRAL[0]], P)
                         for P in PATCH_POS}

            # ---- resume references: same prefix, no hook, same seed. Every
            #      patched cell is paired against the reference at its own P, so
            #      the sampler's stream is aligned and the no-op is exact.
            for P in PATCH_POS:
                p_eff = int(min(P, r_len))
                if p_eff <= 0:
                    continue
                cond = f"resume|{rid}|s{seed}|P{P}"
                if store.has(cond):
                    continue
                prefix = torch.cat([rprompt[0], r_body[:p_eff]]).unsqueeze(0)
                gen, hit = run.generate(prefix, seed, MAX_NEW - p_eff)
                full = torch.cat([r_body[:p_eff], gen])
                sp = run.speech_ids(full)
                store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="A",
                                   kind="resume", recv_item=rid, recv_k=rk,
                                   family="word_rep", template=tmpl, seed=seed,
                                   patch_pos=P, patch_pos_eff=p_eff,
                                   n_gen_tokens=int(full.shape[0]),
                                   n_speech_tokens=len(sp), hit_cap=hit), sp)
                n_gen += 1

            # ---- the patched conditions
            plan: list[tuple[str, str | None, int, int, int, int]] = []
            for key, d in donors.items():
                cells = CELLS if d["kind"] == "crossk" else [CENTRAL]
                for (L, P) in cells:
                    plan.append((d["kind"], key, d["k"], d["seed"], L, P))
            for P in PATCH_POS:                      # the no-op gate
                plan.append(("self", None, rk, seed, CENTRAL[0], P))

            for kind, key, dk, ds, L, P in plan:
                dtag = "self" if key is None else key.split("|")[1]
                cond = f"patch|{rid}|s{seed}|{kind}|{dtag}|ds{ds}|L{L}|P{P}"
                if store.has(cond):
                    continue
                src = self_caps[P] if key is None else donors[key]["caps"][P]
                if not src or L not in src:
                    continue
                p_eff = int(min(P, src[L].shape[0], r_len))
                if p_eff <= 0:
                    continue
                prefix = torch.cat([rprompt[0], r_body[:p_eff]]).unsqueeze(0)
                if rank1:
                    v = torch.from_numpy(dirs[L]).to(run.device)
                    hook = Rank1PatchHook(src[L][:p_eff].to(run.device), v,
                                          rprompt.shape[1])
                else:
                    hook = PatchHook(src[L][:p_eff].to(run.device), rprompt.shape[1])
                gen, hit = run.generate(prefix, seed, MAX_NEW - p_eff,
                                        hook_layer=L, hook=hook)
                full = torch.cat([r_body[:p_eff], gen])
                sp = run.speech_ids(full)
                identical = None
                if kind == "self":
                    ref = store.done.get(f"resume|{rid}|s{seed}|P{P}")
                    if ref is not None:
                        rp = np.load(store.tok_dir / f"{ref['stem']}.npy")
                        identical = bool(len(sp) == len(rp)
                                         and bool((np.asarray(sp) == rp).all()))
                store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="A",
                                   kind="patch", recv_item=rid, recv_k=rk,
                                   family="word_rep", template=tmpl, seed=seed,
                                   donor_item=(None if key is None else dtag),
                                   donor_k=int(dk), donor_kind=kind,
                                   donor_seed=int(ds), layer=int(L),
                                   patch_pos=int(P), patch_pos_eff=p_eff,
                                   n_gen_tokens=int(full.shape[0]),
                                   n_speech_tokens=len(sp), hit_cap=bool(hit),
                                   noop_identical=identical,
                                   rank1=bool(rank1),
                                   max_abs_delta=getattr(hook, "max_abs_delta", None),
                                   hook_fired=int(hook.fired)), sp)
                n_gen += 1
                del hook, gen, full
                torch.cuda.empty_cache()

            del donors, self_caps, r_gen, r_body
            torch.cuda.empty_cache()
            print(f"[A] {rid} s{seed}: {n_gen} generations, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    print(f"[A] DONE {n_gen} generations in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- arm B


def build_directions(layer: int, stim: dict) -> dict:
    """Count directions at one layer, from the activations already on disk.

    Two constructions, because either could be the wrong one and the paper should
    not get to pick after the fact. The difference-in-means direction is what the
    state actually does between low and high $k$; the ridge weight vector is what
    a linear readout of $\\log_2 k$ leans on. They are not the same vector and
    need not have the same causal status.
    """
    act = DATA_ROOT / "activations" / "llasa1b"
    H, ks, pos_norm = [], [], []
    for iid, it in stim.items():
        if it["family"] != "word_rep":
            continue
        f = act / f"{iid}_s0.npz"
        if not f.exists():
            continue
        try:
            h = np.load(f)["hidden"]
        except Exception:  # noqa: BLE001
            continue
        if h.shape[0] < 24 or layer >= h.shape[1]:
            continue
        w = max(4, h.shape[0] // 3)
        tail = h[-w:, layer, :].astype(np.float64)
        H.append(tail.mean(axis=0))
        pos_norm.append(float(np.linalg.norm(tail, axis=1).mean()))
        ks.append(it["k"])
    H = np.stack(H)
    ks = np.asarray(ks, dtype=float)

    lo = np.isin(ks, LOW_K)
    hi = np.isin(ks, HIGH_K)
    d_dim = H[hi].mean(0) - H[lo].mean(0)
    v_dim = d_dim / np.linalg.norm(d_dim)

    # Ridge on log2 k, primal (d=2048, n~54): one small solve. This is a
    # direction, not a reported accuracy, so no cross-validation is needed --
    # `probe_count.py` is where the honest generalisation number lives.
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
                         mean_state_norm=float(np.linalg.norm(H, axis=1).mean()),
                         # the steered vector is added to *per-position* states,
                         # so the honest statement of how hard we are pushing is
                         # relative to those, not to the time-averaged summary
                         mean_pos_norm=float(np.mean(pos_norm)),
                         n_items=int(H.shape[0]))
    return out


def arm_b(run: Runner, store: Store, stim: dict, seeds: list[int]) -> None:
    t0 = time.time()
    n_gen = 0

    dirs = {L: build_directions(L, stim) for L in (STEER_LAYER, STEER_LAYER_2)}
    meta = {str(L): {n: {k: v for k, v in d.items() if k != "v"}
                     for n, d in dd.items()} for L, dd in dirs.items()}
    (REPO / "data/results/causal_steer_directions.json").write_text(
        json.dumps(meta, indent=2))
    print("[B] directions:", json.dumps(meta), flush=True)

    plan: list[tuple[str, int, str, float]] = []
    for k in STEER_KS:
        for tmpl in TEMPLATES:
            for fam in ("word_rep", "control_word"):
                iid = item_id(fam, tmpl, k, stim)
                if iid is None:
                    continue
                plan += [(iid, STEER_LAYER, "dim", a) for a in ALPHAS]
                plan += [(iid, STEER_LAYER, "ridge", a) for a in (-1.0, 1.0)]
                plan += [(iid, STEER_LAYER_2, "dim", a) for a in (-1.0, 1.0)]

    for seed, (iid, L, dname, a) in [(s, p) for s in seeds for p in plan]:
        cond = f"steer|{iid}|s{seed}|L{L}|{dname}|a{a:+.1f}"
        if store.has(cond):
            continue
        it = stim[iid]
        prompt = run.prompt(it["text"])
        d = dirs[L][dname]
        # alpha is in units of the low-k -> high-k projection gap, sign-fixed so
        # that positive alpha always means "toward high k" whichever way the raw
        # direction happens to point.
        vec = torch.from_numpy(d["v"] * np.float32(a * d["delta"] * d["sign"]))
        hook = SteerHook(vec.to(run.device), prompt.shape[1])
        gen, hit = run.generate(prompt, seed, MAX_NEW, hook_layer=L, hook=hook)
        sp = run.speech_ids(gen)
        identical = None
        if a == 0.0:
            base, _ = run.generate(prompt, seed, MAX_NEW)
            identical = bool(gen.shape == base.shape and bool((gen == base).all()))
            del base
        store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="B",
                           kind="steer", recv_item=iid, recv_k=it["k"],
                           family=it["family"], template=it["template"],
                           seed=seed, layer=int(L), alpha=float(a),
                           direction=dname,
                           rel_magnitude=float(abs(a) * d["delta"]
                                               / d["mean_pos_norm"]),
                           n_gen_tokens=int(gen.shape[0]),
                           n_speech_tokens=len(sp), hit_cap=bool(hit),
                           noop_identical=identical,
                           hook_fired=int(hook.fired)), sp)
        n_gen += 1
        del hook, gen
        torch.cuda.empty_cache()
        if n_gen % 10 == 0:
            print(f"[B] {n_gen} generations, {(time.time()-t0)/60:.1f} min",
                  flush=True)

    print(f"[B] DONE {n_gen} generations in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- self-test


def selftest(run: Runner, stim: dict) -> bool:
    """The sanity gate, run alone before any grid is spent.

    Four assertions, in the order in which a failure would matter:

      1. free generation is reproducible under a fixed seed;
      2. the hook fires on the positions it claims to;
      3. a no-op patch (the item's own captured states) reproduces the resume
         reference *bitwise*;
      4. a real donor patch actually changes the continuation --- otherwise
         assertion 3 would be satisfied by a hook that does nothing at all, and
         the whole experiment would be a null by construction.
    """
    ok = True
    rid = item_id("word_rep", "t1", 16, stim)
    did = item_id("word_rep", "t1", 32, stim)
    rprompt, dprompt = run.prompt(stim[rid]["text"]), run.prompt(stim[did]["text"])

    g1, _ = run.generate(rprompt, 0, 512)
    g2, _ = run.generate(rprompt, 0, 512)
    rep = bool(g1.shape == g2.shape and bool((g1 == g2).all()))
    print(f"  [1] free generation reproducible: {rep}")
    ok &= rep

    body = run.strip_eos(g1)
    P = min(128, int(body.shape[0]))
    prefix = torch.cat([rprompt[0], body[:P]]).unsqueeze(0)
    ref, _ = run.generate(prefix, 0, 512 - P)

    caps = run.capture(rprompt, g1, [CENTRAL[0]], P)
    hook = PatchHook(caps[CENTRAL[0]].to(run.device), rprompt.shape[1])
    noop, _ = run.generate(prefix, 0, 512 - P, hook_layer=CENTRAL[0], hook=hook)
    fired = hook.fired == P + 0  # prefill only; decode positions are past `hi`
    print(f"  [2] hook fired on {hook.fired} positions (expected {P}): {fired}")
    ok &= fired
    same = bool(noop.shape == ref.shape and bool((noop == ref).all()))
    print(f"  [3] no-op patch reproduces resume reference bitwise: {same}")
    ok &= same

    dg, _ = run.generate(dprompt, 0, 512)
    dcaps = run.capture(dprompt, dg, [CENTRAL[0]], P)
    n = min(P, dcaps[CENTRAL[0]].shape[0])
    hook2 = PatchHook(dcaps[CENTRAL[0]][:n].to(run.device), rprompt.shape[1])
    pg, _ = run.generate(prefix, 0, 512 - P, hook_layer=CENTRAL[0], hook=hook2)
    moved = not (pg.shape == ref.shape and bool((pg == ref).all()))
    print(f"  [4] donor patch changes the continuation: {moved}")
    ok &= moved

    # ---- rank-1 projection variant: the same two questions again, because the
    # difference form is the whole reason its no-op can be exact.
    v = torch.from_numpy(build_directions(CENTRAL[0], stim)["ridge"]["v"]).to(run.device)
    h5 = Rank1PatchHook(caps[CENTRAL[0]].to(run.device), v, rprompt.shape[1])
    n5, _ = run.generate(prefix, 0, 512 - P, hook_layer=CENTRAL[0], hook=h5)
    same5 = bool(n5.shape == ref.shape and bool((n5 == ref).all()))
    print(f"  [5] rank-1 no-op reproduces reference bitwise: {same5} "
          f"(max |delta| {h5.max_abs_delta:.3g})")
    ok &= same5
    h6 = Rank1PatchHook(dcaps[CENTRAL[0]][:n].to(run.device), v, rprompt.shape[1])
    p6, _ = run.generate(prefix, 0, 512 - P, hook_layer=CENTRAL[0], hook=h6)
    moved6 = not (p6.shape == ref.shape and bool((p6 == ref).all()))
    print(f"  [6] rank-1 donor patch changes the continuation: {moved6} "
          f"(max |delta| {h6.max_abs_delta:.3g})")
    ok &= moved6

    print(f"  SANITY GATE: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def arm_r(run: Runner, store: Store, stim: dict, seeds: list[int]) -> None:
    """The follow-up sweep: ridge direction only, fine alphas, three times the items."""
    t0 = time.time()
    n_gen = 0
    d = build_directions(STEER_LAYER, stim)["ridge"]
    meta = {k: v for k, v in d.items() if k != "v"}
    meta["alphas"] = RIDGE_ALPHAS
    meta["rel_magnitude_at_alpha1"] = float(d["delta"] / d["mean_pos_norm"])
    (REPO / "data/results/causal_ridge_direction.json").write_text(json.dumps(meta, indent=2))
    print("[R] ridge@L%d: %s" % (STEER_LAYER, json.dumps(meta)), flush=True)

    items: list[str] = []
    for k in RIDGE_KS:
        for tmpl in TEMPLATES:
            for fam in ("word_rep", "control_word"):
                iid = item_id(fam, tmpl, k, stim)
                if iid:
                    items.append(iid)
    print(f"[R] {len(items)} items x {len(RIDGE_ALPHAS)} alphas x {len(seeds)} seeds",
          flush=True)

    for seed in seeds:
        for iid in items:
            it = stim[iid]
            prompt = run.prompt(it["text"])
            for a in RIDGE_ALPHAS:
                cond = f"ridge|{iid}|s{seed}|L{STEER_LAYER}|a{a:+.2f}"
                if store.has(cond):
                    continue
                vec = torch.from_numpy(d["v"] * np.float32(a * d["delta"] * d["sign"]))
                hook = SteerHook(vec.to(run.device), prompt.shape[1])
                gen, hit = run.generate(prompt, seed, MAX_NEW,
                                        hook_layer=STEER_LAYER, hook=hook)
                sp = run.speech_ids(gen)
                identical = None
                if a == 0.0:
                    base, _ = run.generate(prompt, seed, MAX_NEW)
                    identical = bool(gen.shape == base.shape
                                     and bool((gen == base).all()))
                    del base
                store.put(rec_base(cond_id=cond, stem=stem_for(cond), arm="R",
                                   kind="steer", recv_item=iid, recv_k=it["k"],
                                   family=it["family"], template=it["template"],
                                   seed=seed, layer=STEER_LAYER, alpha=float(a),
                                   direction="ridge",
                                   rel_magnitude=float(abs(a) * d["delta"]
                                                       / d["mean_pos_norm"]),
                                   n_gen_tokens=int(gen.shape[0]),
                                   n_speech_tokens=len(sp), hit_cap=bool(hit),
                                   noop_identical=identical,
                                   hook_fired=int(hook.fired)), sp)
                n_gen += 1
                del hook, gen
                torch.cuda.empty_cache()
            if n_gen % 20 < len(RIDGE_ALPHAS):
                print(f"[R] {n_gen} generations, {(time.time()-t0)/60:.1f} min",
                      flush=True)
    print(f"[R] DONE {n_gen} generations in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "R"], required=True)
    ap.add_argument("--rank1", action="store_true",
                    help="arm A: transfer only the count coordinate, not the state")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--selftest", action="store_true",
                    help="run the sanity gate only, and exit")
    ap.add_argument("--b-seeds", type=int, nargs="+", default=[SEEDS[0]],
                    help="sampling seeds for the steering arm")
    args = ap.parse_args()

    stim = load_stimuli()
    run = Runner(args.gpu)
    if args.selftest:
        raise SystemExit(0 if selftest(run, stim) else 1)
    if args.arm == "A":
        arm_a(run, Store("patchr1" if args.rank1 else "patch1b"), stim,
              rank1=args.rank1)
    elif args.arm == "R":
        arm_r(run, Store("steerr"), stim, args.b_seeds)
    else:
        arm_b(run, Store("steer1b"), stim, args.b_seeds)


if __name__ == "__main__":
    main()
