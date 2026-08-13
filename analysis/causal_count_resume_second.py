#!/usr/bin/env python3
"""Run the *published* Llasa-1B causal protocol, unchanged, on another checkpoint.

Why this file exists
--------------------
`analysis/causal_count_second.py` had to change the protocol to reach Qwen3-TTS
at all: Qwen's talker consumes a sum of sixteen RVQ-codebook embeddings plus a
text term at each step, its codec ids never leave `generate()`, and there is no
way to hand it a teacher-forced prefix. So that file patches at the first $P$
*decode* steps of a free generation and pairs against the free baseline, instead
of teacher-forcing a prefix and pairing against a resume reference.

That change was pre-registered together with a control: **re-run Llasa-1B under
the new protocol and check it reproduces the published answer** (U1). The control
was run and **it failed**. Under the decode-time protocol Llasa-1B's cells do not
all sit at the noise floor (disruption index 1.02, largest median |shift| above
25% of a full transfer) and the verdict degrades from `readable null` to
`too disruptive to interpret`, with the equivalence bound widening from 0.089 to
0.236 log-count. The *point estimate* agrees --- the bridge median +0.065 lies
inside the published 90% CI [-0.089, +0.070] --- but the instrument is much
blunter on this checkpoint, so the pre-committed conclusion stands: the two
protocols are not interchangeable on Llasa, and a Qwen-versus-published-Llasa
comparison cannot be presented as one instrument measuring two checkpoints.

Notably the same protocol is *not* blunt on Qwen-0.6B, where every cell sits at
the noise floor, degeneracy and cap-hits are 0.0%, and the bound is 0.066 --- so
the disruption is a property of Llasa's 50 Hz heavy-tailed decoding under a free
window, not of the protocol as such. The Qwen arm remains internally valid on its
own controls. What the failed bridge forbids is the cross-protocol comparison.

This file repairs that on the axis where it *can* be repaired. The published
protocol cannot be carried to Qwen, but it can be carried to another **Llasa**
checkpoint, where the architecture allows it. Doing that answers the reviewers'
single-checkpoint objection with **zero protocol confound**: the same code, the
same window, the same donors, the same resume references, the same scorer --- a
different checkpoint.

How it stays the same code
--------------------------
The published arm is `analysis/causal_count.py`'s `arm_a(..., rank1=True)`. It is
parameterised entirely by module-level constants, so rather than reimplement it
(and risk a silent divergence in exactly the code a causal claim rests on), this
file **rebinds those constants and calls the original function**. The only
substitution is `build_directions`, which hard-codes
`activations/llasa1b` in its body and so cannot be reached by a constant; it is
replaced by `analysis/causal_count_second.py`'s generalised version, which was
verified against the original on Llasa-1B layer 13 and agrees to cosine
1.00000000 with an identical low-to-high separation of 5.840594.

`MAX_NEW` is deliberately *not* changed: every Llasa checkpoint shares the same
X-codec2 tokenizer at 50 Hz, so 3584 tokens is the same 72 s of audio budget on
8B as it was on 1B, and the cap-hit rate stays comparable.

Two checkpoints are run:
  llasa1b  a fresh sample under the published protocol, in the *current*
           environment. This matters because the published manifest is no longer
           re-derivable: instantiating `causal_count.py`'s own `Runner` today and
           asking for the same item at the same seed returns 873 tokens where the
           manifest records 1528 (transformers 4.57.3 / torch 2.11.0+cu130). This
           run separates "the environment moved" from "the checkpoint differs".
  llasa8b  the second checkpoint, and the point of the exercise.

Verdicts, gates and thresholds are unchanged and are the ones pre-committed in
`analysis/causal_count_second.py`; scoring is
`analysis/causal_count_second_score.py --pair resume`.

Usage (env `base`):
  python analysis/causal_count_resume_second.py --model llasa8b --gpu 3
  python analysis/causal_count_resume_second.py --model llasa1b --gpu 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import analysis.causal_count as cc  # noqa: E402
from analysis.causal_count_second import CONFIG, build_directions  # noqa: E402
from common.registry import BY_KEY  # noqa: E402
from src.common.gpus import check_gpu  # noqa: E402


def configure(model: str) -> None:
    """Rebind the published module's constants to another Llasa checkpoint.

    Every name below is looked up as a module global inside `arm_a` at call
    time, so rebinding here is enough and the function body is untouched.
    """
    spec = BY_KEY[model]
    assert spec.family == "llasa", (
        f"{model} is not a Llasa checkpoint; the published protocol needs a "
        "teacher-forceable token sequence, which is exactly what Qwen3-TTS "
        "does not have (see the module docstring)")
    cfg = CONFIG[model]
    P = cfg["patch_pos"]
    cc.HF_ID = spec.hf_id
    cc.PATCH_LAYERS = list(cfg["patch_layers"])
    cc.PATCH_POS = [P]
    cc.CENTRAL = (cfg["central_layer"], P)
    cc.CELLS = [(L, P) for L in cfg["patch_layers"]]
    cc.build_directions = lambda layer, stim: build_directions(model, layer, stim)
    print(f"[resume/{model}] {spec.hf_id}: layers {cc.PATCH_LAYERS}, "
          f"central {cc.CENTRAL}, window {P} tokens "
          f"({P / spec.token_rate_hz:.2f} s), budget {cc.MAX_NEW}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llasa1b", "llasa8b"])
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    check_gpu(args.gpu)
    configure(args.model)
    stim = cc.load_stimuli()
    run = cc.Runner(args.gpu)
    # A distinct key, so the published `patchr1` artifacts are never appended to.
    cc.arm_a(run, cc.Store(f"rs_{args.model}"), stim, rank1=True)


if __name__ == "__main__":
    main()
