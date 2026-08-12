#!/usr/bin/env python3
"""A second, modern non-autoregressive baseline.

Rounds 8--10 all made the same objection to the architectural control: VITS is a
2021 model, small and unscaled, standing alone against six 2024--25 AR
checkpoints. Its null could mean "not autoregressive" or it could mean "not
capable enough to show structure". The k-band analysis argues against the second
reading (`analysis/nonar_baseline.py`), but a second, contemporary non-AR model
settles it better than an argument does.

F5-TTS is the right comparator: 2024, flow-matching over a diffusion transformer,
non-autoregressive by construction --- the whole mel sequence is denoised in
parallel, with duration set by a total-length estimate rather than by a learned
per-step stop decision. It is trained at a scale comparable to the panel, so
"weak old model" cannot explain a null here.

Reference audio is the same clip the XTTS and Qwen runs use, so voice and
recording conditions are held fixed across every model in the study.

Usage:  python src/models/f5_gen.py --gpu 2 --seeds 0
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
# The same reference clip and transcript the Qwen runs use, so voice and
# recording conditions are identical across every model in the study.
DEFAULT_REF_WAV = REPO / "data/ref/qwen_ref.wav"
DEFAULT_REF_TXT = REPO / "data/ref/qwen_ref.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f5tts")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--ref-wav", default=str(DEFAULT_REF_WAV))
    ap.add_argument("--ref-txt", default=str(DEFAULT_REF_TXT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    check_gpu(args.gpu)

    import soundfile as sf
    from f5_tts.api import F5TTS

    ref_txt = Path(args.ref_txt).read_text().strip() if Path(args.ref_txt).exists() else ""
    tts = F5TTS(device=f"cuda:{args.gpu}")

    items = [json.loads(l) for l in open(args.stimuli)]
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

    print(f"[{args.model}] {len(items)} items x {len(args.seeds)} seeds on "
          f"cuda:{args.gpu}", flush=True)
    t0, n, fails = time.time(), 0, 0
    with meta_path.open("a") as mf:
        for seed in args.seeds:
            for it in items:
                if (it["item_id"], seed) in done:
                    continue
                torch.manual_seed(seed)
                stem = f"{it['item_id']}_s{seed}"
                dst = outdir / f"{stem}.wav"
                try:
                    wav, sr, _ = tts.infer(
                        ref_file=args.ref_wav, ref_text=ref_txt,
                        gen_text=it["text"], seed=seed, remove_silence=False)
                except Exception as exc:  # noqa: BLE001 - one bad item must not stop the sweep
                    fails += 1
                    print(f"[{args.model}] FAIL {stem}: {type(exc).__name__} "
                          f"{str(exc)[:80]}", flush=True)
                    continue
                wav = np.asarray(wav, dtype=np.float32)
                sf.write(dst, wav, sr)
                # No generation budget exists here -- length comes from a duration
                # estimate, not a stop decision -- so hit_cap is always False and
                # the shared exclusion rules still find the field they expect.
                mf.write(json.dumps(dict(
                    item_id=it["item_id"], seed=seed, model=args.model,
                    duration_s=float(len(wav) / sr), hit_cap=False,
                    n_steps=None, sr=int(sr))) + "\n")
                mf.flush()
                n += 1
                if n % 25 == 0:
                    print(f"[{args.model}] {n} items  "
                          f"{(time.time()-t0)/max(n,1):.2f}s/item", flush=True)
    print(f"[{args.model}] done, {n} generations, {fails} failures, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
