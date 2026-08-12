#!/usr/bin/env python3
"""Validate the CTC judge on periodic audio with exact ground truth.

We disqualified Whisper by measuring it against concatenative audio whose true
repetition count is known. Adopting a replacement without putting it through the
same test would be exactly the error we criticised, so this script runs both
recognisers on identical material.

Construction. Take a model's own $k{=}1$ rendering of a carrier sentence --- one
utterance, verified to contain exactly one occurrence of the target word --- and
concatenate it $N$ times with a short silence between copies. The result is
genuinely periodic audio containing exactly $N$ occurrences of every word in the
carrier, with no segmentation or splicing inside a word, so the ground truth is
exact by construction rather than by annotation. A distinct-word control is built
the same way from the matched control item, which contains no repetition, so any
counting failure specific to periodicity shows up as a gap between the two.

The audio is concatenative and therefore not perfectly natural; the quantity of
interest is the direction and magnitude of each recogniser's counting bias, not
its absolute word error rate.

Usage:  python analysis/ctc_validation.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
KS = [2, 4, 8, 16, 32]
GAP_S = 0.12


def build(wav: np.ndarray, sr: int, n: int) -> np.ndarray:
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    return np.concatenate([np.concatenate([wav, gap]) for _ in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--source-model", default="xtts2",
                    help="whose k=1 renderings supply the atoms")
    ap.add_argument("--ctc", default="facebook/wav2vec2-large-960h-lv60-self")
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/ctc_validation.json"))
    # Both judges have to be told which language they are hearing. For the CTC
    # judge that is baked into the checkpoint (`--ctc`); for Whisper it is a
    # decode-time flag, and leaving it at `en` over Spanish audio makes Whisper
    # translate instead of transcribe, which would score zero occurrences of a
    # Spanish target for a reason that has nothing to do with counting.
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    from common.score_counts import count_occurrences, normalise

    stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.stimuli)}
    aud = Path(DATA_ROOT) / "audio" / args.source_model

    # Atoms: k=1 repeated items (exactly one occurrence of the target) and, as
    # the non-periodic control, k=2 control items (two DISTINCT fillers).
    atoms = []
    for iid, it in stim.items():
        if it["family"] == "word_rep" and it["k"] == 1:
            f = aud / f"{iid}_s0.wav"
            if f.exists():
                atoms.append(("periodic", it["target_unit"], f))
    ctl_atoms = []
    for iid, it in stim.items():
        if it["family"] == "control_word" and it["k"] == 2:
            f = aud / f"{iid}_s0.wav"
            if f.exists() and it.get("boundary_units"):
                ctl_atoms.append(("distinct", it["boundary_units"][0], f))
    atoms, ctl_atoms = atoms[:6], ctl_atoms[:6]
    print(f"atoms: {len(atoms)} periodic, {len(ctl_atoms)} distinct-word control")

    dev = f"cuda:{args.gpu}"
    from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor,
                              WhisperForConditionalGeneration, WhisperProcessor)
    cproc = Wav2Vec2Processor.from_pretrained(args.ctc)
    cmod = Wav2Vec2ForCTC.from_pretrained(args.ctc).to(dev).eval()
    wproc = WhisperProcessor.from_pretrained(args.whisper)
    wmod = (WhisperForConditionalGeneration
            .from_pretrained(args.whisper, dtype=torch.float16).to(dev).eval())

    @torch.no_grad()
    def ctc_text(w: np.ndarray, sr: int) -> str:
        step, out = int(25 * sr), []
        for i in range(0, w.size, step):
            seg = w[i:i + step]
            if seg.size < sr // 20:
                break
            lg = cmod(cproc(seg, sampling_rate=sr,
                            return_tensors="pt").input_values.to(dev)).logits
            out.append(cproc.batch_decode(torch.argmax(lg, -1))[0].strip())
        return " ".join(out)

    @torch.no_grad()
    def whisper_text(w: np.ndarray, sr: int) -> str:
        inp = wproc(w, sampling_rate=sr, return_tensors="pt", truncation=False,
                    padding="longest", return_attention_mask=True)
        kw = {k: (v.to(dev, torch.float16) if k == "input_features" else v.to(dev))
              for k, v in inp.items()}
        out = wmod.generate(**kw, language=args.language, task="transcribe",
                            condition_on_prev_tokens=False, return_segments=True)
        seq = out["sequences"] if isinstance(out, dict) else out
        return wproc.batch_decode(seq, skip_special_tokens=True)[0]

    res: dict = {"source_model": args.source_model, "ctc": args.ctc,
                 "whisper": args.whisper, "language": args.language,
                 "stimuli": args.stimuli, "trials": []}
    for kind, pool in (("periodic", atoms), ("distinct", ctl_atoms)):
        for _, unit, f in pool:
            wav, sr = sf.read(f, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                sr = 16000
            for n in KS:
                big = build(wav, sr, n)
                for judge, fn in (("ctc", ctc_text), ("whisper", whisper_text)):
                    try:
                        txt = fn(big, sr)
                        c = count_occurrences(normalise(txt), unit)
                    except Exception as e:  # noqa: BLE001
                        txt, c = f"ERROR {type(e).__name__}", -1
                    res["trials"].append(dict(kind=kind, unit=unit, true=n, judge=judge,
                                              counted=c, dur_s=float(big.size / sr),
                                              text=txt[:200]))
            print(f"  {kind} {unit}: done", flush=True)

    # summarise
    summary: dict = {}
    for kind in ("periodic", "distinct"):
        for judge in ("ctc", "whisper"):
            for n in KS:
                v = [t["counted"] / t["true"] for t in res["trials"]
                     if t["kind"] == kind and t["judge"] == judge
                     and t["true"] == n and t["counted"] >= 0]
                if v:
                    summary[f"{kind}_{judge}_k{n}"] = float(np.median(v))
            v = [t["counted"] / t["true"] for t in res["trials"]
                 if t["kind"] == kind and t["judge"] == judge
                 and t["true"] >= 4 and t["counted"] >= 0]
            if v:
                summary[f"{kind}_{judge}_kge4"] = float(np.median(v))
    res["summary"] = summary

    print("\ncounted/true ratio (median):")
    print(f"{'':22s}" + "".join(f"k={n:<6d}" for n in KS) + "  k>=4")
    for kind in ("periodic", "distinct"):
        for judge in ("ctc", "whisper"):
            row = "".join(f"{summary.get(f'{kind}_{judge}_k{n}', float('nan')):<8.2f}"
                          for n in KS)
            print(f"{kind:10s} {judge:10s} {row}  "
                  f"{summary.get(f'{kind}_{judge}_kge4', float('nan')):.2f}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
