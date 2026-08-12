#!/usr/bin/env python3
"""Transcribe generated audio with Whisper large-v3, keeping word timestamps.

Word timestamps are not a nicety here: they are how a repetition boundary in the
*audio* is mapped back to a decoder step index (step = t_word * token_rate_hz),
which is what the state-space analysis needs.

`condition_on_prev_tokens=False` matters — Whisper's own decoder is itself an AR
model prone to repetition lock-in, and conditioning on previous text lets its
loops contaminate our measurement of the TTS model's loops.

Usage:
  python src/common/asr_transcribe.py --model llasa1b --gpu 3
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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"


def audio_flags(wav: np.ndarray, sr: int) -> dict:
    """Cheap degeneracy detectors that do not depend on the transcript.

    A TTS failure does not always produce wrong *words* — it can produce noise,
    a stuck vowel, or silence, none of which an ASR-only metric would catch.
    """
    import librosa

    out: dict = {}
    if wav.size < sr // 10:
        return dict(too_short=True, rms=float(np.sqrt(np.mean(wav ** 2)) if wav.size else 0.0))
    out["rms"] = float(np.sqrt(np.mean(wav ** 2)))
    # trailing silence: a truncated generation ends abruptly; a stuck one runs on
    frame = max(1, sr // 50)
    env = np.sqrt(np.convolve(wav ** 2, np.ones(frame) / frame, mode="same"))
    thr = 0.02 * (env.max() + 1e-9)
    voiced = env > thr
    out["voiced_frac"] = float(voiced.mean())
    last = np.nonzero(voiced)[0]
    out["trailing_silence_s"] = float((wav.size - last[-1]) / sr) if last.size else float(wav.size / sr)
    # spectral flatness: near 1 means noise/babble rather than speech
    try:
        sfm = librosa.feature.spectral_flatness(y=wav)[0]
        out["spectral_flatness"] = float(np.mean(sfm))
    except Exception:  # noqa: BLE001
        out["spectral_flatness"] = float("nan")
    # exact-loop detector: strong autocorrelation peak of the mel envelope at a
    # lag above 0.4 s indicates the model is re-rendering the same material
    try:
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=32, hop_length=256)
        e = np.log1p(mel).mean(axis=0)
        e = e - e.mean()
        if e.size > 8:
            ac = np.correlate(e, e, mode="full")[e.size - 1:]
            ac = ac / (ac[0] + 1e-9)
            min_lag = int(0.4 * sr / 256)
            if ac.size > min_lag + 2:
                out["loop_ac_peak"] = float(ac[min_lag:].max())
                out["loop_ac_lag_s"] = float((min_lag + int(np.argmax(ac[min_lag:]))) * 256 / sr)
    except Exception:  # noqa: BLE001
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--batch-size", type=int, default=8)
    # Whisper decodes to whatever language it is told. The default is `en` so
    # every existing English run reproduces byte-for-byte; the cross-lingual arm
    # passes `es`. Left on `en` for Spanish audio, Whisper would *translate*
    # rather than transcribe, and the resulting transcript would contain no
    # Spanish target word at all -- a silent zero on every item.
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    check_gpu(args.gpu)

    aud_dir = Path(DATA_ROOT) / "audio" / args.model
    out_path = Path(DATA_ROOT) / "asr" / f"{args.model}.jsonl"
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
    print(f"[{args.model}] {len(files)} wavs, {len(todo)} to transcribe", flush=True)
    if not todo:
        return

    # The `automatic-speech-recognition` pipeline routes audio through
    # torchcodec, which fails to load against the installed FFmpeg. Driving the
    # processor and model directly avoids that path entirely and gives explicit
    # control over long-form decoding.
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa
    device = f"cuda:{args.gpu}"
    proc = WhisperProcessor.from_pretrained(args.asr)
    model = (WhisperForConditionalGeneration
             .from_pretrained(args.asr, dtype=torch.float16).to(device).eval())
    ASR_SR = 16000

    def transcribe(wav: np.ndarray, sr: int) -> tuple[str, list]:
        if sr != ASR_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=ASR_SR)
        inp = proc(wav, sampling_rate=ASR_SR, return_tensors="pt",
                   truncation=False, padding="longest", return_attention_mask=True)
        kw = {k: (v.to(device, torch.float16) if k == "input_features" else v.to(device))
              for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(
                **kw, language=args.language, task="transcribe",
                return_timestamps="word", condition_on_prev_tokens=False,
                return_segments=True,
            )
        seq = out["sequences"] if isinstance(out, dict) else out
        text = proc.batch_decode(seq, skip_special_tokens=True)[0]
        # Word timestamps, best effort: each segment carries its own generate
        # result with per-token times, offset by the segment start.
        words: list = []
        try:
            for seg in (out.get("segments", [[]])[0] if isinstance(out, dict) else []):
                res = seg.get("result", {})
                tt = res.get("token_timestamps")
                tk = seg.get("tokens")
                if tt is None or tk is None:
                    continue
                tt = tt[0] if tt.dim() == 2 else tt
                off = float(seg.get("start", 0.0))
                for j, tid in enumerate(tk.tolist()):
                    piece = proc.tokenizer.decode([tid], skip_special_tokens=True)
                    if piece.strip() and j < len(tt):
                        words.append(dict(w=piece, t0=float(tt[j]) + off,
                                          t1=float(tt[min(j + 1, len(tt) - 1)]) + off))
        except Exception:  # noqa: BLE001
            words = []
        return text, words

    fout = open(out_path, "a")
    for i, f in enumerate(todo):
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        rec = dict(stem=f.stem, sr=int(sr), duration_s=float(wav.size / sr))
        rec.update(audio_flags(wav, sr))
        if wav.size < sr // 20 or float(np.sqrt(np.mean(wav ** 2))) < 1e-5:
            rec.update(text="", words=[])
        else:
            try:
                text, words = transcribe(wav, sr)
                rec.update(text=text, words=words)
            except Exception as e:  # noqa: BLE001
                rec.update(text="", words=[], asr_error=f"{type(e).__name__}: {e}")
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    fout.close()
    print(f"[{args.model}] transcription done", flush=True)


if __name__ == "__main__":
    main()
