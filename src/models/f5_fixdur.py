#!/usr/bin/env python3
"""An intervention on the hypothesis, not another correlation.

Two non-autoregressive baselines disagree about the counting deficit: VITS does
not show it, F5-TTS does. The story we told to explain that --- and labelled
post-hoc --- is that what matters is whether the model must represent "how many"
internally. VITS predicts a duration per input token, so the count rides the
input sequence; F5-TTS estimates one total duration and denoises the whole mel in
parallel, so the count has to live somewhere inside.

That story makes a prediction, and F5-TTS happens to expose the knob that tests
it. `infer(fix_duration=...)` supplies the total duration directly instead of
letting the model estimate it. If mis-estimating "how much speech to make" is
what produces the deficit, handing F5-TTS the right duration should reduce it. If
the deficit survives a correct duration, the story is wrong and the mechanism is
somewhere else.

The duration we supply must not be read off the repeated item's own output, which
would be circular. It is taken from the model's own \emph{control} rendering at
the same k --- items F5-TTS gets right --- so it is the duration a correct
rendering of k units takes, estimated without reference to how the repeated item
came out.

One unit trap, caught by the first run producing silence: `fix_duration` is the
length of the reference *plus* the generated speech, not of the generated speech
alone. Passing the target directly asks for a total shorter than the reference
clip, and the model duly emits almost nothing. The reference duration is measured
here and added.

Either result is worth having. A reduction is the first positive evidence for the
replacement hypothesis; no reduction retires it, and the paper says so rather
than leaving an untested story standing.

Usage:  python src/models/f5_fixdur.py --gpu 2 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402

HDD = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts")
DEFAULT_REF_WAV = REPO / "data/ref/qwen_ref.wav"
DEFAULT_REF_TXT = REPO / "data/ref/qwen_ref.txt"


def control_durations(behavioural: str) -> dict[int, float]:
    """Median duration of F5-TTS's own control renderings, per k.

    Controls are the items it counts correctly, so their duration is what k
    units of speech in this carrier actually takes --- an estimate that never
    looks at the repeated item's output.
    """
    d = pd.read_csv(behavioural)
    d = d[(d.model == "f5tts") & (d.family == "control_word")
          & (~d.outcome.isin(["empty", "degenerate"]))]
    return {int(k): float(g.duration_s.median()) for k, g in d.groupby("k")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f5fix")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--behavioural", default="data/results/behavioural_f5.csv")
    ap.add_argument("--kmin", type=int, default=6,
                    help="only the range where the deficit exists")
    ap.add_argument("--ref-wav", default=str(DEFAULT_REF_WAV))
    ap.add_argument("--ref-txt", default=str(DEFAULT_REF_TXT))
    args = ap.parse_args()
    check_gpu(args.gpu)

    import soundfile as sf
    from f5_tts.api import F5TTS

    # fix_duration covers reference + generated audio, so the reference length
    # has to be added to the target or the model is asked for negative speech.
    ref_dur = float(sf.info(args.ref_wav).duration)
    durs = control_durations(args.behavioural)
    if not durs:
        raise SystemExit("no control durations; run src/models/f5_gen.py first")
    print(f"[{args.model}] reference {ref_dur:.2f}s; control durations by k: "
          f"{ {k: round(v, 1) for k, v in sorted(durs.items())} }")

    ref_txt = Path(args.ref_txt).read_text().strip()
    tts = F5TTS(device=f"cuda:{args.gpu}")

    items = [json.loads(l) for l in open(args.stimuli)]
    # Repeated items only: the control is already rendered correctly, so there is
    # nothing for the intervention to fix there.
    items = [i for i in items if i["family"] == "word_rep" and i["k"] >= args.kmin]

    outdir = HDD / "audio" / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = HDD / "tokens" / f"{args.model}_meta.jsonl"
    done = set()
    if meta_path.exists():
        for line in meta_path.open():
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"[{args.model}] {len(items)} items x {len(args.seeds)} seeds", flush=True)
    t0, n, fails = time.time(), 0, 0
    with meta_path.open("a") as mf:
        for seed in args.seeds:
            for it in items:
                if (it["item_id"], seed) in done:
                    continue
                target = durs.get(int(it["k"]))
                if target is None:
                    continue
                torch.manual_seed(seed)
                try:
                    wav, sr, _ = tts.infer(
                        ref_file=args.ref_wav, ref_text=ref_txt,
                        gen_text=it["text"], seed=seed, remove_silence=False,
                        fix_duration=float(ref_dur + target))
                except Exception as exc:  # noqa: BLE001
                    fails += 1
                    print(f"[{args.model}] FAIL {it['item_id']}: "
                          f"{type(exc).__name__} {str(exc)[:70]}", flush=True)
                    continue
                wav = np.asarray(wav, dtype=np.float32)
                sf.write(outdir / f"{it['item_id']}_s{seed}.wav", wav, sr)
                mf.write(json.dumps(dict(
                    item_id=it["item_id"], seed=seed, model=args.model,
                    duration_s=float(len(wav) / sr),
                    fix_duration=float(ref_dur + target), target_gen_s=float(target),
                    hit_cap=False, n_steps=None, sr=int(sr))) + "\n")
                mf.flush()
                n += 1
    print(f"[{args.model}] done, {n} generations, {fails} failures, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
