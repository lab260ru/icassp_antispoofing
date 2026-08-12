#!/usr/bin/env python3
"""Transcribe generated audio with a non-autoregressive CTC recogniser.

Why a second recogniser. Our instrument audit showed Whisper large-v3
systematically undercounts *correct* repeated speech: on concatenative audio with
known ground truth its counted/true ratio falls to 0.27--0.65 for k >= 4, while
matched distinct-word audio of the same length transcribes at ratio 1.00. That is
not a random error, it is a bias pointing in the same direction as the effect we
are trying to measure, and its cause is structural --- Whisper's decoder is itself
autoregressive with a language-model prior, so it is subject to exactly the
repetition-suppression dynamics this paper is about. Using it to judge repetition
counting is measuring the phenomenon with an instrument made of the phenomenon.

A CTC recogniser has no autoregressive decoder and no internal language model. It
emits a per-frame distribution over characters and collapses blanks; the number of
times a word appears in the output is governed by how many times it appears in
the acoustics, not by what a decoder expects to come next. It is a worse
recogniser in absolute terms --- higher WER, no punctuation, no casing --- but it
is the right instrument here, because its errors are not correlated with
repetition.

We keep both transcripts. Agreement between an AR and a non-AR judge is evidence;
disagreement localises the bias.

Usage:
  python src/common/asr_ctc.py --model llasa1b --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
CTC_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    # wav2vec2 large, fine-tuned on 960h LibriSpeech with self-training. Pure
    # CTC: no decoder, no LM, no beam search by default.
    ap.add_argument("--ctc", default="facebook/wav2vec2-large-960h-lv60-self")
    ap.add_argument("--chunk-s", type=float, default=25.0,
                    help="split long audio; CTC attention is quadratic in frames")
    args = ap.parse_args()
    check_gpu(args.gpu)

    aud_dir = Path(DATA_ROOT) / "audio" / args.model
    out_path = Path(DATA_ROOT) / "asr_ctc" / f"{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["stem"])
            except Exception:  # noqa: BLE001
                pass

    files = sorted(aud_dir.glob("*.wav"))
    todo = [f for f in files if f.stem not in done]
    print(f"[{args.model}] {len(files)} wavs, {len(todo)} to transcribe (CTC)", flush=True)
    if not todo:
        return

    import librosa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    device = f"cuda:{args.gpu}"
    proc = Wav2Vec2Processor.from_pretrained(args.ctc)
    model = Wav2Vec2ForCTC.from_pretrained(args.ctc).to(device).eval()

    @torch.no_grad()
    def transcribe(wav: np.ndarray, sr: int) -> str:
        if sr != CTC_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=CTC_SR)
        # Chunk long audio with a small overlap, then join. Repetition items run
        # to 40 s and the conv+attention stack is happier in pieces; the overlap
        # keeps a word from being cut in half at a boundary.
        step = int(args.chunk_s * CTC_SR)
        overlap = int(0.25 * CTC_SR)
        pieces = []
        i = 0
        while i < wav.size:
            seg = wav[max(0, i - (overlap if i else 0)): i + step]
            if seg.size < CTC_SR // 20:
                break
            inp = proc(seg, sampling_rate=CTC_SR, return_tensors="pt")
            logits = model(inp.input_values.to(device)).logits
            ids = torch.argmax(logits, dim=-1)
            pieces.append(proc.batch_decode(ids)[0].strip())
            i += step
        return " ".join(p for p in pieces if p)

    fout = open(out_path, "a")
    for i, f in enumerate(todo):
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        rec = dict(stem=f.stem, sr=int(sr), duration_s=float(wav.size / sr))
        if wav.size < sr // 20 or float(np.sqrt(np.mean(wav ** 2))) < 1e-5:
            rec["text"] = ""
        else:
            try:
                rec["text"] = transcribe(wav, sr)
            except Exception as e:  # noqa: BLE001
                rec.update(text="", asr_error=f"{type(e).__name__}: {e}")
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    fout.close()
    print(f"[{args.model}] CTC transcription done", flush=True)


if __name__ == "__main__":
    main()
