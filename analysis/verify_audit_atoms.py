#!/usr/bin/env python3
"""Which donor atoms actually contain their target word exactly once?

The enlarged audit assumed every k=1 rendering contains one occurrence of the
target. It does not: two checkpoints render the target as a different word
entirely, and the judge -- correctly -- counts zero. This gates atoms on presence
in the UNCONCATENATED clip, which is a transcription question, not a counting
one, so it does not use the counting behaviour under test to license its own
ground truth.
"""
import json, os, sys
from pathlib import Path
REPO = Path("/home/kirill/icassp_antispoofing")
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")
import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from common.score_counts import count_occurrences, normalise

DATA = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
MODELS = ["llasa1b","llasa3b","llasa8b","qwen06b","qwen17b","xtts2"]
stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(REPO/"data/stimuli/stimuli.jsonl")}
dev = "cuda:2"
proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
mod = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(dev).eval()

@torch.no_grad()
def text(w, sr):
    iv = proc(w, sampling_rate=sr, return_tensors="pt").input_values.to(dev)
    return proc.batch_decode(torch.argmax(mod(iv).logits, -1))[0].strip()

out = {}
for m in MODELS:
    aud = Path(DATA)/"audio"/m
    rows = []
    for iid, it in stim.items():
        if it["family"] != "word_rep" or it["k"] != 1: continue
        f = aud/f"{iid}_s0.wav"
        if not f.exists(): continue
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=16000); sr = 16000
        t = text(w, sr)
        n = count_occurrences(normalise(t), it["target_unit"])
        rows.append(dict(item=iid, unit=it["target_unit"], counted=int(n),
                         valid=bool(n == 1), text=t[:120]))
    out[m] = rows
    ok = sum(r["valid"] for r in rows)
    print(f"{m:9s} {ok}/{len(rows)} atoms contain the target exactly once")
    for r in rows:
        if not r["valid"]:
            print(f"    REJECT {r['unit']:8s} n={r['counted']}  {r['text'][:70]!r}")
Path(REPO/"data/results/_audit_parts/atom_validity.json").write_text(json.dumps(out, indent=2))
