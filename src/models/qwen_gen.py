#!/usr/bin/env python3
"""Qwen3-TTS (0.6B / 1.7B Base) generation with hidden-state instrumentation.

Mirrors the interface of llasa_gen.py (CLI, output layout, metadata schema),
but the internals differ because Qwen3-TTS is architecturally very different
from Llasa:

  * Llasa is a single autoregressive LM over a flat token stream (text tokens
    then speech tokens all in one vocabulary) -> a plain HF `AutoModelForCausalLM`
    with `generate()` + a teacher-forced second pass gives us everything.

  * Qwen3-TTS Base is a dual-track model: a "talker" transformer
    (`Qwen3TTSTalkerForConditionalGeneration`, found at
    `model.model.talker` inside the `qwen_tts` package's
    `Qwen3TTSModel` wrapper) autoregressively emits one 12.5 Hz acoustic frame
    per decoder step. Each frame is 16 RVQ codebook ids: the *first* codebook
    id is produced by the talker's own `codec_head` and sampled by the
    standard HF generation loop; the other 15 are filled in by a much smaller
    sub-talker ("code_predictor") conditioned on the talker's hidden state for
    that step. Critically, text is not concatenated as separate token
    positions in the talker's self-attention sequence: in the model's default
    ("streaming") mode, each decode step's *input* embedding is the elementwise
    **sum** of a codec embedding and (for the first few steps) a text-token
    embedding (`trailing_text_hidden`, added inside
    `Qwen3TTSTalkerForConditionalGeneration.forward`). There is therefore no
    column range of the attention matrix that belongs to "the text" the way
    there is for Llasa -- see the Level 3 note below.

One-pass design (no teacher-forced second pass needed)
--------------------------------------------------------
`Qwen3TTSForConditionalGeneration.generate()` (the mid-level API the
`qwen_tts` wrapper calls) *always* calls the talker's own
`.generate(..., output_hidden_states=True, return_dict_in_generate=True)`
internally (see `qwen_tts/core/models/modeling_qwen3_tts.py`, ~line 2064) --
this is not optional, so the full per-step, per-layer hidden-state trajectory
is already being materialized by the library on *every* call, instrumented or
not. It then quietly discards all but the last layer. We monkeypatch the bound
method `model.model.talker.generate` on the loaded instance to *also* stash
the raw `GenerateDecoderOnlyOutput` (and the `trailing_text_hidden` kwarg) in a
closure variable, so we never need a second forward pass: a single call to
`generate_voice_clone()` gives us the audio *and* (when instrumented) the full
hidden-state trajectory for free. This is a read-only interception -- we never
alter what gets passed into the real `generate()` call, so generation/sampling
is byte-for-byte what the public API would have produced anyway.

Capability levels actually achieved (see the task's graceful-degradation
priority order)
------------------------------------------------------------------------
  1. Behavioural (audio + metadata): YES, unconditionally, via
     `Qwen3TTSModel.generate_voice_clone()`.
  2. Hidden states: YES, via the monkeypatch above. We record, per generated
     acoustic frame, the talker's hidden state at `probe_layers` (registry
     `probe_layer_indices`), using the same step<->frame alignment the
     library itself uses internally to build `talker_hidden_states_list`
     (hidden state of decode step j predicted frame j; we simply keep more
     than the last layer). Entropy/top1 of the next-frame (first-codebook)
     distribution are computed from the recorded final-layer state through
     `talker.codec_head`, so no extra forward pass is needed for those either.
  3. Text attention: NOT reachable, and we do not fake it. `output_attentions`
     is never requested. Even if it were, in the model's default streaming
     mode text and codec content are summed into the *same* per-step input
     embedding (see `Qwen3TTSForConditionalGeneration.generate_icl_prompt`
     and the `trailing_text_hidden` addition in
     `Qwen3TTSTalkerForConditionalGeneration.forward`), so an attention
     column cannot be attributed to "the text" the way it can for Llasa's
     concatenated token stream. `create_voice_clone_prompt`/`generate()` does
     expose a `non_streaming_mode=True` flag that would give text its own
     contiguous block of positions (and could in principle be probed), but
     that is not the model's default/tested code path and using it would
     risk exactly the Level-1 audio quality the task marks mandatory, so we
     do not use it. `text_lo`/`text_hi` are therefore written as `null` in
     the metadata (an honest "not applicable", not a fabricated span), plus
     a bonus `trailing_text_len` field recording how many initial decode
     steps did carry a distinct text-token embedding, for anyone who wants a
     (temporal, not spatial) proxy.

Usage:
  python src/models/qwen_gen.py --model qwen06b --gpu 3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.common.gpus import DEFAULT_GPU, check_gpu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices  # noqa: E402

REF_WAV = REPO / "data/ref/qwen_ref.wav"
REF_TXT = REPO / "data/ref/qwen_ref.txt"


class TalkerGenerateCapture:
    """Monkeypatches `talker.generate` to stash its raw return value.

    `Qwen3TTSForConditionalGeneration.generate()` always calls
    `self.talker.generate(..., output_hidden_states=True,
    return_dict_in_generate=True)` and then throws away everything but the
    last layer's hidden state. We intercept the call, forward it unchanged
    (so behaviour/sampling is identical to the stock API), and keep the full
    `GenerateDecoderOnlyOutput` plus the `trailing_text_hidden` kwarg around
    so the caller can pull per-layer states out after the fact.
    """

    def __init__(self, talker):
        self.talker = talker
        self.result = None
        self.trailing_text_len = None
        self._orig = talker.generate

        def capturing_generate(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            self.result = out
            tth = kwargs.get("trailing_text_hidden")
            self.trailing_text_len = int(tth.shape[1]) if tth is not None else None
            return out

        talker.generate = capturing_generate

    def clear(self):
        self.result = None
        self.trailing_text_len = None


def decode_step_count(result) -> int:
    """Number of AR decode forward-calls (excludes the prefill call)."""
    return len(result.hidden_states) - 1


def eos_trim_length(result, eos_id: int) -> int:
    """Index of the first frame whose first codebook == eos, else all frames.

    Mirrors `Qwen3TTSForConditionalGeneration.generate()`'s own trimming
    (modeling_qwen3_tts.py ~line 2284). For batch_size=1 (what we always run)
    HF's generate() loop stops the instant EOS is sampled and never issues a
    forward call that would materialize an EOS *frame*, so this is normally a
    no-op (effective_length == decode_step_count); kept for defensiveness /
    fidelity to the library's own semantics.
    """
    n = decode_step_count(result)
    for j in range(n):
        codec_ids = result.hidden_states[j + 1][-1]  # frame finalized at call j+1
        if codec_ids is not None and int(codec_ids[0, 0].item()) == eos_id:
            return j
    return n


def extract_trajectory(result, probe_layers: list[int], effective_length: int):
    """[T, n_probe, D] hidden states, aligned one row per generated frame.

    hs[j] (j=0..N-1, N=decode_step_count) is the forward call whose hidden
    state predicted frame j (frame j itself is finalized one call later, at
    hs[j+1][-1] -- see module docstring). Layer index l (0-indexed, per
    `probe_layer_indices`) lives at `hs[j][0][l + 1]` (index 0 of that tuple
    is the embedding layer, matching the llasa_gen.py convention).
    """
    hs = result.hidden_states
    layer_idxs = [l + 1 for l in probe_layers]
    rows = []
    for j in range(effective_length):
        layers_tuple = hs[j][0]
        vecs = torch.stack([layers_tuple[li][0, -1, :] for li in layer_idxs], dim=0)
        rows.append(vecs)
    return torch.stack(rows, dim=0)  # [T, n_probe, D]


def build_prompt(tts) -> list:
    ref_text = REF_TXT.read_text().strip()
    return tts.create_voice_clone_prompt(
        ref_audio=str(REF_WAV), ref_text=ref_text, x_vector_only_mode=False
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: checkpoint's own generation_config.json value")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--instrument-seed", type=int, default=0,
                    help="only this seed gets the instrumented capture saved")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language", default="English")
    args = ap.parse_args()
    check_gpu(args.gpu)

    spec = BY_KEY[args.model]
    assert spec.family == "qwen", f"{args.model} is not a qwen spec"
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    if not REF_WAV.exists() or not REF_TXT.exists():
        raise SystemExit(f"missing reference audio/transcript: {REF_WAV} / {REF_TXT}")

    from qwen_tts import Qwen3TTSModel

    print(f"[{spec.key}] loading {spec.hf_id}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        spec.hf_id, device_map=device, dtype=torch.bfloat16, attn_implementation="eager",
    )
    talker = tts.model.talker
    capture = TalkerGenerateCapture(talker)

    n_layers = talker.config.num_hidden_layers
    d_model = talker.config.hidden_size
    eos_id = talker.config.codec_eos_token_id
    probes = probe_layer_indices(n_layers, spec.probe_layers)
    print(f"[{spec.key}] talker_layers={n_layers} d_model={d_model} "
          f"state_probes={probes} text_attention=UNAVAILABLE(fused-embedding architecture)",
          flush=True)

    print(f"[{spec.key}] voice-clone reference: {REF_WAV.name} "
          f"({REF_TXT.read_text().strip()!r})", flush=True)
    prompt_items = build_prompt(tts)

    items = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        items = items[: args.limit]

    audio_dir = Path(DATA_ROOT) / "audio" / spec.key
    act_dir = Path(DATA_ROOT) / "activations" / spec.key
    tok_dir = Path(DATA_ROOT) / "tokens"
    audio_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tok_dir / f"{spec.key}_meta.jsonl"
    meta_f = open(meta_path, "a")

    done = set()
    if meta_path.exists():
        for line in open(meta_path):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["seed"]))
            except Exception:
                pass

    gen_kwargs_base = dict(max_new_tokens=args.max_new_tokens)
    if args.temperature is not None:
        gen_kwargs_base["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs_base["top_p"] = args.top_p

    t_start = time.time()
    n_done = 0
    for seed in args.seeds:
        for it in items:
            key = (it["item_id"], seed)
            if key in done:
                continue
            torch.manual_seed(seed)
            capture.clear()

            wavs, sr = tts.generate_voice_clone(
                text=it["text"], language=args.language,
                voice_clone_prompt=prompt_items, **gen_kwargs_base,
            )
            wav = wavs[0]
            result = capture.result

            n_steps = decode_step_count(result)
            n_speech_tokens = eos_trim_length(result, eos_id) if n_steps else 0
            hit_cap = n_steps >= args.max_new_tokens
            prompt_len = int(result.hidden_states[0][0][0].shape[1])

            audio_path = audio_dir / f"{it['item_id']}_s{seed}.wav"
            import soundfile as sf
            sf.write(str(audio_path), wav, sr)

            rec = dict(
                item_id=it["item_id"], seed=seed, model=spec.key, k=it["k"],
                family=it["family"], n_speech_tokens=n_speech_tokens,
                prompt_len=prompt_len, text_lo=None, text_hi=None,
                trailing_text_len=capture.trailing_text_len,
                hit_cap=hit_cap, est_duration_s=n_speech_tokens / spec.token_rate_hz,
                sr=sr, wav_duration_s=float(len(wav)) / sr,
            )

            # ---- instrumented capture (no second forward pass needed) -----
            if seed == args.instrument_seed and it.get("instrumented") and n_speech_tokens > 0:
                with torch.no_grad():
                    hs_traj = extract_trajectory(result, probes, n_speech_tokens)  # [T, n_probe, D]
                    last_layer_state = hs_traj[:, -1, :].to(talker.codec_head.weight.dtype)
                    logits = talker.codec_head(last_layer_state).float()
                    pr = torch.softmax(logits, dim=-1)
                    ent = (-(pr * torch.log(pr.clamp_min(1e-12))).sum(-1)).cpu().numpy()
                    top1 = pr.max(-1).values.cpu().numpy()
                np.savez_compressed(
                    act_dir / f"{it['item_id']}_s{seed}.npz",
                    hidden=hs_traj.to(torch.float16).cpu().numpy(),
                    probe_layers=np.array(probes),
                    entropy=ent.astype(np.float32), top1=top1.astype(np.float32),
                )
                rec["instrumented"] = True
                del hs_traj, logits, pr

            capture.clear()
            meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t_start
                print(f"[{spec.key}] {n_done} items  {el/n_done:.1f}s/item  "
                      f"last={it['item_id']} k={it['k']} ntok={n_speech_tokens}", flush=True)
            torch.cuda.empty_cache()

    meta_f.close()
    print(f"[{spec.key}] DONE {n_done} items in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
