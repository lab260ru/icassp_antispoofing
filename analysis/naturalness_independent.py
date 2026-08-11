#!/usr/bin/env python3
"""Naturalness confound, scored by a referee independent of the model panel.

`analysis/text_perplexity.py` answered the "is repeated text just improbable
text" objection using Llasa-1B as the scorer. A reviewer correctly pointed out
that Llasa-1B is not a neutral referee: it is one of the panel's own backbones
(Llama-3.2-1B). A likelihood check run under a member of the thing being
studied cannot rule out the panel's own inductive biases leaking into the
"naturalness" measurement.

This script reruns the same check under text LMs that are architecturally and
organisationally unrelated to every panel backbone:

  panel backbone -> family it rules out
  Llama-3.2 (Llasa 1B/3B/8B)   -> no Llama-derived scorer
  GPT-2-style AR decoder (XTTS-v2) -> no GPT-2-family scorer as primary
  Qwen3 (Qwen3-TTS 0.6B/1.7B)      -> no Qwen-derived scorer

Primary scorer: `microsoft/phi-2` (2.7B). Different lab, different training
recipe, different position-embedding/attention details from all three panel
backbones, and large enough to be a competent general-purpose referee rather
than a noisy one -- a referee that can't model English well is not a referee
whose verdict on "which sentence is more natural" means much.

Secondary scorer: `gpt2-large` (774M). Included only as a cross-check, not as
the primary evidence, because XTTS-v2's AR decoder is itself GPT-2-style
architecture -- gpt2-large is *arguably* not fully independent of one panel
member, even though no panel member's weights derive from it. If phi-2 and
gpt2-large agree, the result does not depend on which non-independence
concern you find more convincing.

Method is unchanged from text_perplexity.py: per-token NLL of the raw
stimulus text (`model(ids, labels=ids).loss`, which HF's causal-LM loss
already averages over tokens), compared between every `control_word` item and
the `word_rep` item named in its `control_of`. New in this script: a 95%
bootstrap CI on the paired gap, resampled over stimulus *templates* (not over
individual pairs) -- pairs sharing a template share a carrier sentence and are
not independent draws, so resampling raw pairs would understate the CI. This
mirrors the template-bootstrap already used for the capacity-gain CIs in
`analysis/capacity.py`.

Usage:
  python analysis/naturalness_independent.py --gpu 0
  python analysis/naturalness_independent.py --gpu 0 --skip-secondary   # phi-2 only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PANEL_BACKBONES = {
    "llasa1b/3b/8b": "Llama-3.2 (Meta)",
    "xtts2/xtts2norp": "GPT-2-style AR latent decoder (Coqui)",
    "qwen06b/17b": "Qwen3 (Alibaba)",
}


def score_stimuli(lm_id: str, gpu: int, stimuli: list[dict]) -> dict[str, float]:
    """Per-token NLL of every stimulus's raw text under `lm_id`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu}"
    tok = AutoTokenizer.from_pretrained(lm_id)
    lm = AutoModelForCausalLM.from_pretrained(lm_id, dtype=torch.bfloat16).to(dev).eval()

    nll: dict[str, float] = {}
    for it in stimuli:
        ids = tok(it["text"], return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = lm(ids, labels=ids)
        nll[it["item_id"]] = float(out.loss)

    del lm
    torch.cuda.empty_cache()
    return nll


def template_bootstrap_ci(pairs: list[tuple[str, int, float, float]],
                           n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI on mean(control - repeated), resampling stimulus templates.

    Pairs sharing a template (same carrier sentence, only the filler/target
    word differs across k) are correlated; resampling templates rather than
    individual pairs is what makes the CI honest about the true number of
    independent units in the design (6 templates, not 54 pairs).
    """
    templates = sorted({p[0] for p in pairs})
    by_template: dict[str, list[float]] = {t: [] for t in templates}
    for t, _k, r, c in pairs:
        by_template[t].append(c - r)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picked = rng.choice(templates, size=len(templates), replace=True)
        vals = [d for t in picked for d in by_template[t]]
        boots.append(float(np.mean(vals)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def analyse(scorer_id: str, nll: dict[str, float], stimuli: list[dict]) -> dict:
    """Build the full pairwise-comparison report for one scorer's NLL table."""
    pairs = []  # (template, k, nll_repeated, nll_control)
    for it in stimuli:
        if it["family"] != "control_word" or not it.get("control_of"):
            continue
        rep_id = it["control_of"]
        if rep_id in nll and it["item_id"] in nll:
            pairs.append((it["template"], it["k"], nll[rep_id], nll[it["item_id"]]))

    if not pairs:
        return dict(scorer=scorer_id, n_pairs=0)

    rep = np.array([p[2] for p in pairs])
    ctl = np.array([p[3] for p in pairs])
    d = ctl - rep
    lo, hi = template_bootstrap_ci(pairs)

    ks = sorted({p[1] for p in pairs})
    by_k_diff = {int(k): float(np.mean([c - r for _, kk, r, c in pairs if kk == k]))
                 for k in ks}
    by_k_frac_ctl_higher = {int(k): float(np.mean([(c - r) > 0 for _, kk, r, c in pairs if kk == k]))
                            for k in ks}

    return dict(
        scorer=scorer_id,
        n_pairs=len(pairs),
        per_item=nll,
        rep_mean=float(rep.mean()),
        ctl_mean=float(ctl.mean()),
        diff_mean=float(d.mean()),
        diff_ci95=[lo, hi],
        frac_ctl_higher=float((d > 0).mean()),
        by_k=by_k_diff,
        by_k_frac_ctl_higher=by_k_frac_ctl_higher,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, choices=[0, 1, 3],
                     help="GPU 2 is reserved for an in-flight generation job.")
    ap.add_argument("--primary-lm", default="microsoft/phi-2",
                     help="Independent referee: not Llama, not GPT-2-family, not Qwen.")
    ap.add_argument("--secondary-lm", default="gpt2-large",
                     help="Cross-check only; GPT-2-family, so not used as primary evidence.")
    ap.add_argument("--skip-secondary", action="store_true")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--out", default="data/results/text_nll_independent.json")
    args = ap.parse_args()

    stimuli = [json.loads(l) for l in open(args.stimuli)]

    print(f"[primary] scoring {len(stimuli)} stimuli under {args.primary_lm} on cuda:{args.gpu}")
    primary_nll = score_stimuli(args.primary_lm, args.gpu, stimuli)
    primary = analyse(args.primary_lm, primary_nll, stimuli)
    print(f"  {primary['n_pairs']} pairs; control-repeated = {primary['diff_mean']:+.3f} "
          f"[{primary['diff_ci95'][0]:+.3f}, {primary['diff_ci95'][1]:+.3f}]; "
          f"control less probable in {100*primary['frac_ctl_higher']:.0f}% of pairs")

    result: dict = dict(
        primary=primary,
        panel_backbones_avoided=PANEL_BACKBONES,
        note=(
            "gpt2-large is reported as a secondary/cross-check scorer only: XTTS-v2's "
            "AR decoder is itself GPT-2-style, so a GPT-2-family LM is not a fully "
            "independent referee for that one panel member. The primary result above "
            "does not depend on it."
        ),
    )

    if not args.skip_secondary:
        print(f"[secondary] scoring under {args.secondary_lm} on cuda:{args.gpu}")
        secondary_nll = score_stimuli(args.secondary_lm, args.gpu, stimuli)
        secondary = analyse(args.secondary_lm, secondary_nll, stimuli)
        print(f"  {secondary['n_pairs']} pairs; control-repeated = {secondary['diff_mean']:+.3f} "
              f"[{secondary['diff_ci95'][0]:+.3f}, {secondary['diff_ci95'][1]:+.3f}]; "
              f"control less probable in {100*secondary['frac_ctl_higher']:.0f}% of pairs")
        result["secondary"] = secondary

    # Top-level convenience aliases mirroring text_nll.json's flat schema, so a
    # reader diffing the two files can compare scorer=Llasa-1B vs scorer=phi-2
    # field-for-field without digging into the "primary" sub-object.
    result["scorer"] = primary.get("scorer")
    result["n_pairs"] = primary.get("n_pairs")
    result["rep_mean"] = primary.get("rep_mean")
    result["ctl_mean"] = primary.get("ctl_mean")
    result["diff_mean"] = primary.get("diff_mean")
    result["diff_ci95"] = primary.get("diff_ci95")
    result["frac_ctl_higher"] = primary.get("frac_ctl_higher")
    result["by_k"] = primary.get("by_k")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
