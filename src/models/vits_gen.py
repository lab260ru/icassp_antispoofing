#!/usr/bin/env python3
"""A non-autoregressive baseline on the same repetition ladder.

Round-6 reviewers made the objection that the paper's own title invites: it
claims something about \\emph{autoregressive} TTS without ever running a decoder
that is not autoregressive. If a non-AR synthesiser miscounts repetitions in the
same way, the theorem's framing is the wrong explanation for the phenomenon and
the paper's scope claim is wrong. If it does not, the AR-specific framing earns
its keep.

VITS (`facebook/mms-tts-eng`) is the cleanest available contrast. It has no
autoregressive decoder at all: a text encoder feeds a duration predictor, and a
normalising-flow decoder emits the whole waveform in one shot. There is no
generation-step recurrence for a per-repetition state map to act on, so
Assumption 1 has nothing to apply to and Theorem 1 makes no prediction about it.
Its stopping behaviour is likewise structural --- output length is the sum of
predicted phoneme durations --- rather than a learned stop decision, which is
exactly the mechanism the theorem says gets destroyed.

Two honest caveats about what this can and cannot show. VITS is a smaller and
older model than the panel, so a difference could be about capability rather than
architecture; and its duration predictor consumes a phoneme sequence whose length
already encodes k, which is arguably why it should succeed --- that is the point
of the comparison, not a confound to apologise for. It is a baseline for the
architectural claim, not a matched control for model quality.

Usage:  python src/models/vits_gen.py --model vits --gpu 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402
HDD = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vits")
    ap.add_argument("--checkpoint", default="facebook/mms-tts-eng")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    # The duration predictor is the part of VITS that decides how long the
    # utterance is, and so the part that would have to carry the count. Running
    # it at one stock setting and concluding "VITS is immune" leaves open that
    # the setting, not the architecture, is what is immune -- so both knobs are
    # exposed and the ablation arm varies them.
    ap.add_argument("--noise-scale-duration", type=float, default=None,
                    help="stochastic duration-predictor temperature (stock 0.8)")
    ap.add_argument("--speaking-rate", type=float, default=None,
                    help="divides predicted durations (stock 1.0)")
    args = ap.parse_args()
    check_gpu(args.gpu)

    from transformers import VitsModel, AutoTokenizer
    import soundfile as sf

    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = VitsModel.from_pretrained(args.checkpoint).to(dev).eval()
    if args.noise_scale_duration is not None:
        model.noise_scale_duration = args.noise_scale_duration
    if args.speaking_rate is not None:
        model.speaking_rate = args.speaking_rate
    sr = model.config.sampling_rate
    print(f"[{args.model}] duration predictor: noise_scale_duration="
          f"{model.noise_scale_duration}, speaking_rate={model.speaking_rate}")

    items = [json.loads(l) for l in open(args.stimuli)]
    # Only the families the architectural comparison is about. Numbers and
    # twisters carry their own confounds and are not part of the contrast.
    items = [i for i in items if i["family"] in ("word_rep", "control_word")]
    if args.limit:
        items = items[:args.limit]

    outdir = HDD / "audio" / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = HDD / "tokens" / f"{args.model}_meta.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if meta_path.exists():
        for line in meta_path.open():
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"[{args.model}] {args.checkpoint} on {dev}, sr={sr}, "
          f"{len(items)} items x {len(args.seeds)} seeds")
    t0, n = time.time(), 0
    with meta_path.open("a") as mf:
        for seed in args.seeds:
            for it in items:
                key = (it["item_id"], seed)
                if key in done:
                    continue
                torch.manual_seed(seed)
                enc = tok(it["text"], return_tensors="pt").to(dev)
                with torch.no_grad():
                    wav = model(**enc).waveform[0].float().cpu().numpy()
                stem = f"{it['item_id']}_s{seed}"
                sf.write(outdir / f"{stem}.wav", wav, sr)
                # n_steps has no autoregressive meaning here; hit_cap is always
                # False because there is no generation budget to hit, and the
                # scorer's exclusion rules rely on that field existing.
                mf.write(json.dumps(dict(
                    item_id=it["item_id"], seed=seed, model=args.model,
                    duration_s=float(len(wav) / sr), hit_cap=False,
                    n_steps=None, sr=sr)) + "\n")
                mf.flush()
                n += 1
                if n % 50 == 0:
                    print(f"[{args.model}] {n} items  "
                          f"{(time.time()-t0)/max(n,1):.2f}s/item", flush=True)
    print(f"[{args.model}] done, {n} new generations in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
