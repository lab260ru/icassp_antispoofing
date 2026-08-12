#!/usr/bin/env python3
"""Vocode saved Llasa speech-token ids to waveforms.

Runs in the isolated `xcodec2` conda env: xcodec2==0.1.5 hard-pins torch 2.5, so
it cannot share an interpreter with the 2.11 stack the LMs run under. Decoupling
generation from vocoding also means a codec problem never costs us GPU hours of
LM sampling.

Usage (in the xcodec2 env):
  python src/models/xcodec2_decode.py --model llasa1b --gpu 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--codec", default="HKUSTAudio/xcodec2")
    ap.add_argument("--batch-log", type=int, default=50)
    args = ap.parse_args()
    check_gpu(args.gpu)

    tok_dir = Path(DATA_ROOT) / "tokens" / args.model
    out_dir = Path(DATA_ROOT) / "audio" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(tok_dir.glob("*.npy"))
    todo = [f for f in files if not (out_dir / f"{f.stem}.wav").exists()]
    print(f"[{args.model}] {len(files)} token files, {len(todo)} to decode", flush=True)
    if not todo:
        return

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    from xcodec2.modeling_xcodec2 import XCodec2Model
    codec = XCodec2Model.from_pretrained(args.codec).eval().to(device)

    n_fail = 0
    for i, f in enumerate(todo):
        ids = np.load(f)
        if ids.size == 0:
            # a degenerate generation: emit a zero-length marker file so the
            # downstream stage sees "produced nothing" rather than "missing"
            sf.write(out_dir / f"{f.stem}.wav", np.zeros(1, dtype=np.float32), SR)
            continue
        t = torch.from_numpy(ids.astype(np.int64)).to(device).unsqueeze(0).unsqueeze(0)
        try:
            with torch.no_grad():
                wav = codec.decode_code(t)
            sf.write(out_dir / f"{f.stem}.wav",
                     wav[0, 0].float().cpu().numpy(), SR)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL {f.stem}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % args.batch_log == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)

    print(f"[{args.model}] done, {n_fail} failures", flush=True)


if __name__ == "__main__":
    main()
