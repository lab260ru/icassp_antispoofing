#!/usr/bin/env python3
"""Is the repeated/control contrast confounded by text naturalness?

A reviewer's objection worth taking seriously: repeated text is unusual text, so
maybe models fail on it because it is improbable, not because it is periodic. The
objection has real force only if the control items are *more* natural than their
repeated twins. They are constructed not to be --- "quite really truly fairly
rather somewhat extremely notably ..." is word salad too --- but that should be
measured rather than asserted.

We score each stimulus under a general-purpose causal LM and compare the
per-token negative log-likelihood of every repeated item against the control that
shares its template and k. If the two are comparable, or if controls are the
*less* probable of the pair, naturalness cannot explain a gap that runs the other
way.

Usage:
  python analysis/text_perplexity.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    # Llasa's own Llama backbone is already local and is a reasonable
    # general-purpose text LM for this purpose.
    ap.add_argument("--lm", default="HKUSTAudio/Llasa-1B")
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--out", default="data/results/text_nll.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{args.gpu}"
    tok = AutoTokenizer.from_pretrained(args.lm)
    lm = AutoModelForCausalLM.from_pretrained(
        args.lm, dtype=torch.bfloat16).to(dev).eval()

    items = [json.loads(l) for l in open(args.stimuli)]
    nll: dict[str, float] = {}
    for it in items:
        ids = tok(it["text"], return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = lm(ids, labels=ids)
        nll[it["item_id"]] = float(out.loss)

    by = {it["item_id"]: it for it in items}
    pairs = []
    for it in items:
        if it["family"] != "control_word" or not it.get("control_of"):
            continue
        rep_id = it["control_of"]
        if rep_id in nll and it["item_id"] in nll:
            pairs.append((it["template"], it["k"], nll[rep_id], nll[it["item_id"]]))

    res = dict(n_pairs=len(pairs), per_item=nll)
    if pairs:
        rep = np.array([p[2] for p in pairs])
        ctl = np.array([p[3] for p in pairs])
        d = ctl - rep
        res.update(
            rep_mean=float(rep.mean()), ctl_mean=float(ctl.mean()),
            diff_mean=float(d.mean()),
            frac_ctl_higher=float((d > 0).mean()),
            by_k={int(k): float(np.mean([c - r for _, kk, r, c in pairs if kk == k]))
                  for k in sorted({p[1] for p in pairs})},
        )
        print(f"paired stimuli: {len(pairs)}")
        print(f"  repeated  mean NLL/token = {rep.mean():.3f}")
        print(f"  control   mean NLL/token = {ctl.mean():.3f}")
        print(f"  control - repeated       = {d.mean():+.3f} "
              f"({100*(d>0).mean():.0f}% of pairs have the CONTROL less probable)")
        print("  by k:", {k: round(v, 2) for k, v in res["by_k"].items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
