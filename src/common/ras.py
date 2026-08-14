#!/usr/bin/env python3
"""Repetition Aware Sampling (VALL-E 2, arXiv:2406.05370), our implementation.

=============================================================================
PRE-COMMITTED. Written to disk before a single RAS waveform existed. Nothing
above the POST-HOC line at the bottom of `analysis/rep_aware_sampling.py` may be
edited after looking at a result; a rule that turned out to be wrong is reported
as wrong, not rewritten.
=============================================================================

WHY THIS ARM EXISTS
-------------------
A reviewer found prior art the paper had not engaged with. The paper says the
count deficit survives "the field's standard mitigations", but what was actually
swept is repetition-penalty *magnitude* (five arms on XTTS-v2, four on
Qwen3-TTS-0.6B) and greedy-versus-sampled decoding. Those are one axis:
penalising a token by how often it has already been emitted, uniformly over all
history, with a scalar. VALL-E 2's Repetition Aware Sampling is a different
axis, it is published, and it is engineered around exactly the variable this
paper isolates -- in the authors' words it "refines the original nucleus
sampling process by accounting for token repetition in the decoding history. It
not only stabilizes the decoding but also circumvents the infinite loop issue."
The objection is accepted. This file tests the axis we did not.

WHAT THE PAPER SAYS, VERBATIM
-----------------------------
From the HTML of arXiv:2406.05370v1, section on Repetition Aware Sampling:

  "we first generate the target code c_{t'} by nucleus sampling with a
   pre-defined top-p value v. Then, we calculate the repetition ratio r of token
   c_{t'} in the preceding code sequence with a window size K. If the ratio r
   exceeds a pre-defined repetition threshold ratio t_n, we replace the target
   code c_{t'} by random sampling"

  r <- (1/K) * sum_{k=0}^{K} 1[c_{t'} == c_{t'-k}]

  "we set the hyperparameter K = 10, t_r = 0.1, and select the top-p value v
   from 0.0 to 0.8 with the intervals of 0.1"

  "This technique stabilizes the first codec code generation without increasing
   latency."

That is the whole of the method description. It is a two-branch sampler: draw
from the nucleus; if the drawn token is repetitive, throw that draw away and
draw again from the untruncated model distribution.

WHAT WE IMPLEMENTED, AND EVERY DECISION THE DESCRIPTION LEAVES OPEN
-------------------------------------------------------------------
This is *our* implementation of a described method, not the authors' code. The
authors released no code for RAS. Every choice below is ours and is listed so a
reader can disagree with a specific one rather than with "an approximation".

1. WHERE IT IS APPLIED. Every autoregressive decode step of the Qwen3-TTS
   talker, i.e. the first-codebook (codebook 0) acoustic-token stream, which is
   the exact structural analogue of VALL-E 2's AR stage and of the sentence
   "stabilizes the first codec code generation". It applies to *all* tokens at
   *all* steps, not only to steps that happen to fall inside the target word:
   the rule cannot know where the target is, and neither can the model. The
   sub-talker (codebooks 1-15, Qwen's analogue of VALL-E 2's NAR stage) keeps
   its stock sampling untouched, as in the paper.

2. THE WINDOW. K = 10 previous emitted codebook-0 tokens, the paper's value. At
   step t < K the window is whatever history exists (shorter), because the
   talker's generated sequence starts empty: text and speaker conditioning
   enter as summed input embeddings, not as token positions, so there is no
   prompt-token history to slide the window into. VALL-E 2's window can reach
   back into the enrolled-speech codes; ours cannot. This makes our rule
   slightly *less* likely to fire in the first 10 frames (0.8 s of audio) and
   is irrelevant at the k >= 6 rungs the verdict is read at.

3. THE k = 0 TERM, AND WHAT THE THRESHOLD ACTUALLY MEANS. The printed sum runs
   from k = 0, and 1[c_t == c_{t-0}] is the token compared with itself, which
   is identically 1. Taken literally, r = (1 + #matches in the last K) / K, so
   at K = 10, t_r = 0.1 and a strict ">" the rule fires iff the sampled token
   occurs *at least once* in the preceding 10 tokens. We implement the printed
   formula literally, with strict ">", and therefore that is the operative rule.
   The alternative reading -- sum over k = 1..K only, so the rule fires iff the
   token occurs at least *twice* in the preceding 10 -- is not what is printed,
   but it is what a reader might implement. We count how often it would have
   fired at every step and record it alongside (`ras_fired_alt`), so the
   sensitivity to this ambiguity is measured rather than argued about.

4. WHAT "NUCLEUS SAMPLING" IS HERE. The checkpoint's own shipped decoding
   configuration, read at run time off the kwargs the `qwen_tts` library
   actually passes to the talker: temperature 0.9, top_k 50, top_p 1.0, and the
   shipped repetition_penalty 1.05 (a stock HF logits processor that runs
   *before* us, so we see the distribution the model would really have sampled
   from). We reuse Hugging Face's own `TemperatureLogitsWarper`,
   `TopKLogitsWarper` and `TopPLogitsWarper` objects in Hugging Face's own
   order, so the first branch is bit-for-bit the sampler the unmodified arm
   uses. VALL-E 2 sweeps v from 0.0 to 0.8 and has no temperature or top-k;
   `qwen06brasgr` covers the small-v end (v = 0.0, near-greedy nucleus), which
   is the regime the paper emphasises, since RAS is what makes a tiny top-p
   safe.

5. WHAT "RANDOM SAMPLING" IS. The literal reading: a multinomial draw from the
   model's own distribution p(c_t | ...), untruncated and untempered -- softmax
   of the logits as they reach us (i.e. after the shipped repetition penalty and
   the library's token suppression, which are properties of the checkpoint, not
   of the sampler). It is the flattest of the available readings and therefore
   the most able to escape a loop, which is the reading most favourable to the
   mitigation. `--ras-fallback tempered` implements the other reading (keep
   temperature 0.9, drop only the truncation) and is not used for the verdict.

6. RNG DISCIPLINE. The rule is implemented as a logits processor that does its
   own sampling and hands Hugging Face a degenerate one-hot distribution, so
   the token we chose is the token that gets emitted. Our internal draws are
   taken from the global generator and the generator state is *restored* before
   we return, so the draw Hugging Face then makes consumes exactly the RNG the
   unmodified arm's draw would have consumed. Consequence: with the rule
   disabled our path is bitwise identical to the stock path -- which is the
   sanity gate below -- and with it enabled the two arms stay on the same RNG
   stream, so the comparison is paired rather than merely matched.

THE CHECKPOINT
--------------
Qwen3-TTS-12Hz-0.6B-Base (`qwen06b`). Chosen because (a) the deficit on it is
the largest in the panel that also has a decoding-rule sweep to be compared
against -- 7.9% exact on repeated items against 100.0% on the length-matched
controls at k >= 6, a +92.1 point gap; (b) its decoding path is the cleanest to
modify honestly: the talker is a stock `GenerationMixin`, so a logits processor
is injected into the real sampling loop rather than around it, and the existing
`TalkerGenerateCapture` monkeypatch already owns that call site; (c) the arm
plumbing exists -- `qwen06brp10/15/30` and `qwen06bgreedy` are the precedent for
adding a decoding-rule arm, including keeping it out of the panel via
`population.ABLATIONS`; (d) it is cheap: 8.4 s/item, 180 items per arm.

THE MEASUREMENT
---------------
Exactly the statistic the rest of the paper reports, computed by the same code
path (`analysis/crosslingual_es.exact_rate`, `analysis/exclusion_sensitivity`):
over k >= 6, after `population.panel()`'s exclusions, with the CTC judge
(wav2vec2-large-960h-lv60-self -- NOT Whisper, which is autoregressive and
biased against the phenomenon), the fraction of items whose counted occurrences
equal k, for repeated items and for their length-matched controls, and the
control-minus-repeated difference in points.

---------------------------------------------------------------- SANITY GATE
Decided first; it can end the arm, and it is the only thing in this file that
can make everything else unreportable.

  A. DETERMINISM CONTROL. The stock path, run twice in one process on the same
     items with the same seed, must produce identical codebook-0 token
     sequences. This measures whether bitwise identity is even attainable on
     this hardware and library build before it is demanded of us.
  B. NO-OP EQUIVALENCE. RAS with the threshold set above its maximum possible
     value (t_r = 2.0; r can never exceed (1+K)/K = 1.1) must reproduce the
     stock path exactly: identical codebook-0 token sequences AND identical wav
     bytes, same seed, same process.

  PASS         A and B both hold. The implementation is testing what it claims.
  UNREPORTABLE A holds and B fails. The rule's plumbing changes the generation
               even when the rule never fires, so nothing downstream measures
               RAS and nothing in the run is reportable.
  VOID         A fails. Bitwise identity is unattainable in this environment, so
               B cannot be demanded. The whole arm is then reported as
               inconclusive-by-construction with that as the reason -- not
               quietly downgraded to a weaker gate.

------------------------------------------------------------ ENGAGEMENT GATE
A mitigation that never fires cannot be shown not to work. The processor counts
its own firings, so this is measured, not assumed.

  ENGAGED      RAS fires on >= 1% of decode steps on repeated items at k >= 6,
               AND fires more often on repeated items than on their controls.
  VACUOUS      Otherwise. The arm is reported as not having tested RAS: whatever
               the gap does, it did not do it because of this rule, and the
               paper may claim nothing about this axis from it.

--------------------------------------------------------- INFORMATIVENESS GATE
The gap is a difference of two rates. RAS's escape branch samples from an
untruncated distribution and can damage rendering; if it breaks the *controls*
too, the gap shrinks for a reason that has nothing to do with counting.

  INFORMATIVE   control exact rate under RAS >= 50% (the same bar
                `analysis/crosslingual_es.py` pre-registered for the same
                reason).
  UNINFORMATIVE below that: reported as "RAS degrades rendering on both
                families", which is a finding about RAS's audio quality, not
                about the counting claim, and the arm answers nothing either
                way.

------------------------------------------------------------------- VERDICT
Given the gates, the exact-rate gap decides. The band comes from the arm
structure that already exists on this checkpoint: the published decoding-rule
sweep spans exact-rate gaps of +66.7 to +92.1 points (repetition_penalty 3.0:
+66.7; 1.5: +70.0; greedy: +90.0; 1.0: +90.0; shipped 1.05, 3 seeds, the
published panel number: +92.1). Those five numbers were computed from
already-landed CSVs before this file was written and are hardcoded in
`analysis/rep_aware_sampling.py` so they cannot drift.

  SURVIVES     gap >= +66.7 points. RAS leaves the deficit inside the range the
               penalty/greedy sweep already spans on this checkpoint: the
               repetition-history-aware axis behaves like the repetition-penalty
               axis, and the paper's robustness claim extends to it and says so,
               naming RAS.
  MITIGATES    gap <= +33.4 points, i.e. at most half the smallest gap any arm
               of the existing sweep reaches. A published mitigation that
               actually works on this failure is worth more to the field than
               another robustness check that passes, and it is reported first
               and loudly: the robustness claim is retracted for this axis and
               the paper reports RAS as a working partial fix.
  INCONCLUSIVE +33.4 < gap < +66.7. RAS moves the deficit further than any
               penalty setting does but does not halve it. This licenses
               exactly one sentence: the deficit is reduced but not closed by
               RAS, reported with both numbers. It does NOT license extending
               the robustness claim to this axis, and it does NOT license
               calling RAS a fix.

Standing rule for reading the output, fixed here: where a choice exists, take
the one that *shrinks* the measured gap. Every ambiguity is resolved in the
mitigation's favour, so a surviving deficit is a lower bound on the deficit and
an upper bound on what RAS does about it.

Usage: imported by `src/models/qwen_gen.py` behind `--ras`.
"""
from __future__ import annotations

import torch
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# The paper's hyperparameters.
RAS_WINDOW = 10
RAS_THRESHOLD = 0.1
# Any threshold above (1 + K) / K is unreachable, so the rule can never fire.
RAS_NOOP_THRESHOLD = 2.0


class RepetitionAwareSampling(LogitsProcessor):
    """Two-branch sampler: nucleus draw, then re-draw if the token repeats.

    Implemented as a logits processor rather than a patched sampling loop
    because Hugging Face runs custom processors *before* the temperature/top-k/
    top-p warpers (`generation/utils.py`, `_get_logits_processor`, the
    `_merge_criteria_processor_list` call at line 1248 precedes the warper
    block at 1266). So `scores` reaches us as the model's own distribution with
    only the checkpoint's repetition penalty and token suppression applied,
    which is exactly the distribution the paper's fallback branch samples from,
    and we can construct the nucleus branch ourselves from the same tensor.

    We then return a one-hot score vector (0 for the chosen token, -inf
    elsewhere). Every warper downstream is a no-op on a one-hot vector -- a
    temperature divide maps -inf to -inf and 0 to 0, top-k keeps at least the
    argmax, top-p keeps at least one token -- so Hugging Face's own
    `torch.multinomial` emits our token with probability one. The alternative,
    passing `do_sample=False` and letting the argmax pick our token, would have
    silently deleted the warpers from the stock path and made the no-op gate
    untestable.
    """

    def __init__(self, *, window: int = RAS_WINDOW, threshold: float = RAS_THRESHOLD,
                 temperature: float | None = None, top_k: int | None = None,
                 top_p: float | None = None, fallback: str = "full",
                 self_term: bool = True):
        self.window = int(window)
        self.threshold = float(threshold)
        self.fallback = fallback
        self.self_term = bool(self_term)
        self.temperature = temperature
        # Hugging Face's own warpers, in Hugging Face's own order and under
        # Hugging Face's own construction conditions (utils.py:1266-1276), so
        # the nucleus branch is not a reimplementation of nucleus sampling but
        # the same objects the stock path would have used.
        self.warpers = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            self.warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        self.reset()

    def reset(self) -> None:
        self.n_steps = 0
        self.n_fired = 0
        self.n_fired_alt = 0   # the k=1..K reading of the printed sum
        self.n_hist_short = 0  # steps whose window was shorter than K

    def stats(self) -> dict:
        return dict(ras_steps=self.n_steps, ras_fired=self.n_fired,
                    ras_fired_alt=self.n_fired_alt,
                    ras_short_window_steps=self.n_hist_short,
                    ras_window=self.window, ras_threshold=self.threshold,
                    ras_fallback=self.fallback, ras_self_term=self.self_term)

    def _ratio(self, hist: torch.Tensor, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(literal r, alternative r) for the candidate token.

        `hist` is [B, w] with w <= K, `tok` is [B, 1]. The literal reading adds
        the k = 0 self-match; the alternative drops it. Both are divided by K,
        not by w: the paper divides by the window size, so a short early window
        makes the ratio smaller rather than renormalising it.
        """
        if hist.numel() == 0:
            matches = torch.zeros(tok.shape[0], device=tok.device, dtype=torch.float32)
        else:
            matches = (hist == tok).sum(dim=-1).to(torch.float32)
        r_lit = (matches + 1.0) / float(self.window)
        r_alt = matches / float(self.window)
        return r_lit, r_alt

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.n_steps += 1
        dev = scores.device
        # Save and restore the RNG around our own draws, so the draw Hugging
        # Face makes on the one-hot we return consumes exactly the randomness
        # the unmodified arm's draw would have consumed. This is what makes the
        # no-op case bitwise identical to stock and keeps the two arms on one
        # RNG stream.
        cpu_state = torch.get_rng_state()
        dev_state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None

        warped = self.warpers(input_ids, scores.clone())
        probs = torch.softmax(warped, dim=-1)
        tok = torch.multinomial(probs, num_samples=1)          # [B, 1]

        hist = input_ids[:, -self.window:] if input_ids.shape[1] else input_ids
        if hist.shape[1] < self.window:
            self.n_hist_short += 1
        r_lit, r_alt = self._ratio(hist, tok)
        r = r_lit if self.self_term else r_alt
        fire = r > self.threshold                               # strict, as printed
        self.n_fired += int(fire.sum().item())
        self.n_fired_alt += int((r_alt > self.threshold).sum().item())

        if bool(fire.any()):
            if self.fallback == "tempered" and self.temperature not in (None, 1.0):
                fb = torch.softmax(scores / self.temperature, dim=-1)
            else:
                fb = torch.softmax(scores, dim=-1)
            tok2 = torch.multinomial(fb, num_samples=1)
            tok = torch.where(fire.unsqueeze(-1), tok2, tok)

        torch.set_rng_state(cpu_state)
        if dev_state is not None:
            torch.cuda.set_rng_state(dev_state, dev)

        out = torch.full_like(scores, float("-inf"))
        out.scatter_(1, tok, 0.0)
        return out
