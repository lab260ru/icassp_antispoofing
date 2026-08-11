#!/usr/bin/env python3
"""XTTS-v2 generation with optional state instrumentation.

Mirrors the interface of `llasa_gen.py`, adapted to XTTS's architecture:

  1. **Generate** — sample mel/acoustic codes autoregressively via the GPT2-style
     AR decoder (`Xtts.gpt`), then vocode them to a waveform with the HiFiGAN
     decoder. Unlike Llasa there is no separate codec-decode stage: XTTS
     produces the waveform directly, in this same script.

  2. **Instrument** (optional) — one teacher-forced forward over
     `[cond_latents][text][generated audio tokens]` through the *raw* GPT2Model
     (`model.gpt.gpt`), built by hand from the exact embedding recipe XTTS's own
     `GPT.forward()`/`get_logits()` use. Because the model is causal, this single
     clean forward recovers hidden states and attention for every generated
     step without touching `generate()` internals — same idea as llasa_gen.py's
     instrumented pass, just re-derived for XTTS's cond+text+mel layout instead
     of Llasa's flat token stream.

Verified against the installed `coqui-tts` 0.27.5 source (see comments below
for the specific files/lines this was checked against):

  - `TTS/tts/layers/xtts/gpt.py` (`GPT` class): `self.gpt` (built by
    `build_hf_gpt_transformer`) is a *plain* `transformers.GPT2Model` with its
    own `wte`/`wpe` deleted (text/mel embeddings are computed externally and
    passed in as `inputs_embeds`). `GPT.get_logits()` builds
    `emb = cat([cond_latents, text_emb, mel_emb], dim=1)` and calls
    `self.gpt(inputs_embeds=emb, output_attentions=..., attention_mask=...)`
    -- that is exactly the recipe the instrumented pass below replicates by
    hand so it can also request `output_hidden_states=True`.
  - `TTS/tts/layers/xtts/gpt_inference.py` (`GPT2InferenceModel`): confirms
    `self.transformer` *is* `self.gpt.gpt` (shared object, shared config), and
    that its `lm_head = Sequential(final_norm, mel_head)` -- i.e. logits =
    `mel_head(final_norm(last_hidden_state))`, which the instrumented pass
    reuses verbatim for the entropy/top1 diagnostic.
  - `TTS/tts/models/xtts.py` (`Xtts.inference()`): the real inference recipe --
    `text_tokens = tokenizer.encode(text.strip().lower(), lang=...)`,
    `gpt_codes = self.gpt.generate(cond_latents=..., text_inputs=text_tokens, ...)`,
    then `gpt_latents = self.gpt(text_tokens, text_len, gpt_codes, ..., cond_latents=...,
    return_latent=True)` for vocoding. This script calls the *same* low-level
    pieces directly (never `Xtts.inference()`/`synthesize()`, which have their
    own `enable_text_splitting` chunker) so there is no sentence-splitting stage
    to defeat: the whole stimulus is always fed to the GPT decoder as one
    contiguous text span, satisfying "split_sentences=False" by construction.
  - `transformers/models/gpt2/modeling_gpt2.py` (4.57.1, installed in the
    `coqui` env): `GPT2Attention.forward` always returns `(attn_output,
    attn_weights)` when `config._attn_implementation == "eager"`, and
    `attn_weights` is computed unconditionally inside `eager_attention_forward`
    regardless of the caller's `output_attentions` flag -- only the *outer*
    `GPT2Model` bothers to collect/return it when `output_attentions=True`. So
    forward hooks on the `.attn` submodules of the probe layers see full
    per-layer attention regardless of what's passed to the top-level call,
    letting us request `output_attentions=False` at the top (avoiding an
    O(n_layers) memory blow-up across all 30 layers) and slice+discard inside
    each hook instead -- the same trick `llasa_gen.py`'s `TextAttentionRecorder`
    uses, just without also paying for the top-level collection. With `sdpa`
    (the default), `output_attentions=True` silently returns `None` weights, so
    `model.gpt.gpt.config._attn_implementation` is forced to `"eager"` once at
    load time.
  - `GPT2Model.forward`: the last entry of `hidden_states` is *post*-`ln_f`
    (same HF convention Llama uses), so indexing `hidden_states[l + 1]` for
    `l == n_layers - 1` gives the normed final state -- consistent with how
    `probe_layer_indices()` always includes the last layer.

Usage:
  python src/models/xtts_gen.py --model xtts2 --gpu 2 --seeds 0 1 2 \
      --stimuli data/stimuli/stimuli.jsonl --instrument-seed 0 --limit 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices  # noqa: E402

DEFAULT_REF_WAV = REPO / "data/ref/xtts_ref.wav"


class TextAttentionRecorder:
    """Hook that keeps only the text-span columns of each attention row.

    Storing full attention is O(T^2 * heads * layers); we only ever need how
    much mass each *text* position receives, so the hook reduces to
    [positions, text_len] (head-averaged) inside the forward pass. See the
    module docstring for why this works even with `output_attentions=False`
    at the top-level call.
    """

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.buf: dict[int, np.ndarray] = {}
        self.handles = []

    def _make(self, layer_idx: int):
        def hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None or w.dim() != 4:
                return
            # [B, H, Q, K] -> mean over heads, slice text columns
            sl = w[0, :, :, self.lo:self.hi].mean(dim=0)
            self.buf[layer_idx] = sl.detach().to(torch.float16).cpu().numpy()
        return hook

    def attach(self, gpt2model, layer_indices):
        blocks = gpt2model.h
        for i in layer_indices:
            self.handles.append(blocks[i].attn.register_forward_hook(self._make(i)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def load_xtts(spec, device: str):
    """Load Xtts from the HF hub snapshot (respects HF_HOME), force eager
    attention on the AR decoder so attention hooks see real weights, and
    return (model, config)."""
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = snapshot_download(spec.hf_id)
    config = XttsConfig()
    config.load_json(f"{model_dir}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    model.to(device)
    model.eval()
    # sdpa (the transformers default) silently drops attention weights when
    # output_attentions=True; force eager so our hooks get real tensors.
    model.gpt.gpt.config._attn_implementation = "eager"
    return model, config


def build_text_inputs(model, text: str, language: str, device: str) -> torch.Tensor:
    """Reproduce Xtts.inference()'s exact text preprocessing (strip+lower before
    tokenizing) so token ids match what a normal inference call would use."""
    sent = text.strip().lower()
    ids = model.tokenizer.encode(sent, lang=language)
    return torch.IntTensor(ids).unsqueeze(0).to(device)


@torch.no_grad()
def instrumented_pass(model, cond_latent, text_tokens, gen_ids, attn_probes, probes):
    """Teacher-forced forward over [cond][text][start_audio, gen_ids...] through
    the raw GPT2Model, replicating GPT.get_logits()'s embedding recipe by hand
    so we can also request output_hidden_states=True.

    Returns dict with hidden [T, n_probes, d], attn (via recorder.buf, sliced
    to the mel rows by the caller), entropy [T], top1 [T], and the column
    offsets used, so the caller can build text_lo/text_hi/mel_offset.
    """
    gpt = model.gpt  # the `GPT` wrapper (embeddings, pos-embeddings, heads, .gpt=GPT2Model)
    device = cond_latent.device

    start_text = gpt.start_text_token
    stop_text = gpt.stop_text_token
    start_audio = gpt.start_audio_token

    text_padded = torch.nn.functional.pad(text_tokens, (1, 0), value=start_text)
    text_padded = torch.nn.functional.pad(text_padded, (0, 1), value=stop_text)
    text_emb = gpt.text_embedding(text_padded) + gpt.text_pos_embedding(text_padded)

    mel_ids = torch.cat(
        [torch.full((1, 1), start_audio, dtype=gen_ids.dtype, device=device), gen_ids], dim=1
    )
    mel_emb = gpt.mel_embedding(mel_ids) + gpt.mel_pos_embedding(mel_ids)

    emb = torch.cat([cond_latent, text_emb, mel_emb], dim=1)

    cond_len = cond_latent.shape[1]
    text_lo = cond_len + 1
    text_hi = cond_len + 1 + text_tokens.shape[1]
    mel_offset = cond_len + text_emb.shape[1]
    T = gen_ids.shape[1]

    recorder = TextAttentionRecorder(text_lo, text_hi)
    recorder.attach(gpt.gpt, attn_probes)
    out = gpt.gpt(inputs_embeds=emb, output_hidden_states=True, output_attentions=False,
                   return_dict=True)
    recorder.detach()

    hs = np.stack(
        [out.hidden_states[l + 1][0, mel_offset: mel_offset + T, :].to(torch.float16).cpu().numpy()
         for l in probes],
        axis=1,
    )  # [T, n_probes, d]

    final_hidden = out.last_hidden_state[0, mel_offset: mel_offset + T, :]
    logits = gpt.mel_head(gpt.final_norm(final_hidden)).float()
    pr = torch.softmax(logits, dim=-1)
    ent = (-(pr * torch.log(pr.clamp_min(1e-12))).sum(-1)).cpu().numpy()
    top1 = pr.max(-1).values.cpu().numpy()

    attn = {f"attn_l{l}": v[mel_offset: mel_offset + T, :] for l, v in recorder.buf.items()}

    return dict(hidden=hs, entropy=ent.astype(np.float32), top1=top1.astype(np.float32),
                attn=attn, text_lo=text_lo, text_hi=text_hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=2, help="use GPU 2 or 3 only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--instrument-seed", type=int, default=0,
                    help="only this seed gets the instrumented second pass")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn-probes", type=int, default=4)
    ap.add_argument("--language", default="en")
    ap.add_argument("--speaker-wav", default=str(DEFAULT_REF_WAV))
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="if >0, shrinks the model's built-in ~602-token cap")
    # sampling knobs; default None -> use the value shipped in config.json
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--length-penalty", type=float, default=None)
    ap.add_argument("--gpt-cond-len", type=int, default=None)
    ap.add_argument("--gpt-cond-chunk-len", type=int, default=None)
    ap.add_argument("--max-ref-len", type=int, default=None)
    args = ap.parse_args()

    if args.gpu not in (2, 3):
        print(f"[warn] GPU {args.gpu} requested; GPUs 0/1 are reserved for Llasa runs.", flush=True)

    spec = BY_KEY[args.model]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    print(f"[{spec.key}] loading {spec.hf_id}", flush=True)
    model, config = load_xtts(spec, device)

    def cfg(name, override):
        return override if override is not None else config[name]

    temperature = cfg("temperature", args.temperature)
    top_p = cfg("top_p", args.top_p)
    top_k = cfg("top_k", args.top_k)
    repetition_penalty = cfg("repetition_penalty", args.repetition_penalty)
    length_penalty = cfg("length_penalty", args.length_penalty)
    gpt_cond_len = cfg("gpt_cond_len", args.gpt_cond_len)
    gpt_cond_chunk_len = cfg("gpt_cond_chunk_len", args.gpt_cond_chunk_len)
    max_ref_len = cfg("max_ref_len", args.max_ref_len)

    if args.max_new_tokens > 0:
        model.gpt.max_gen_mel_tokens = min(model.gpt.max_gen_mel_tokens, args.max_new_tokens)
    effective_cap = model.gpt.max_gen_mel_tokens

    n_layers = model.gpt.layers
    probes = probe_layer_indices(n_layers, spec.probe_layers)
    attn_probes = sorted({0, n_layers // 3, 2 * n_layers // 3, n_layers - 1})[: args.attn_probes]
    print(f"[{spec.key}] layers={n_layers} state_probes={probes} attn_probes={attn_probes} "
          f"gen_cap={effective_cap} temperature={temperature} top_p={top_p} top_k={top_k} "
          f"repetition_penalty={repetition_penalty}", flush=True)

    print(f"[{spec.key}] speaker ref: {args.speaker_wav}", flush=True)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=args.speaker_wav, gpt_cond_len=gpt_cond_len,
        gpt_cond_chunk_len=gpt_cond_chunk_len, max_ref_length=max_ref_len,
    )
    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

    sr = int(config.audio.output_sample_rate)

    items = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        items = items[: args.limit]

    aud_dir = Path(DATA_ROOT) / "audio" / spec.key
    act_dir = Path(DATA_ROOT) / "activations" / spec.key
    aud_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(DATA_ROOT) / "tokens" / f"{spec.key}_meta.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_f = open(meta_path, "a")

    done = set()
    if meta_path.exists():
        for line in open(meta_path):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["seed"]))
            except Exception:
                pass

    t_start = time.time()
    n_done = 0
    for seed in args.seeds:
        for it in items:
            key = (it["item_id"], seed)
            if key in done:
                continue
            torch.manual_seed(seed)

            text_tokens = build_text_inputs(model, it["text"], args.language, device)
            text_len = torch.tensor([text_tokens.shape[-1]], device=device)
            char_len = len(it["text"].strip())
            char_limit = model.tokenizer.char_limits.get(args.language, 250)

            with torch.no_grad():
                gpt_codes = model.gpt.generate(
                    cond_latents=gpt_cond_latent, text_inputs=text_tokens,
                    do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature,
                    length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                    num_beams=1, num_return_sequences=1, output_attentions=False,
                )
            raw_len = int(gpt_codes.shape[-1])
            hit_cap = raw_len >= effective_cap
            ends_with_eos = bool(raw_len and gpt_codes[0, -1].item() == model.gpt.stop_audio_token)
            gen_no_eos = gpt_codes[:, :-1] if ends_with_eos else gpt_codes
            n_speech_tokens = int(gen_no_eos.shape[-1])

            # ---- vocode: official Xtts.inference() recipe, verbatim --------
            wav_path = aud_dir / f"{it['item_id']}_s{seed}.wav"
            if raw_len == 0:
                wav = np.zeros(1, dtype=np.float32)
            else:
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * model.gpt.code_stride_len], device=device
                )
                with torch.no_grad():
                    gpt_latents = model.gpt(
                        text_tokens, text_len, gpt_codes, expected_output_len,
                        cond_latents=gpt_cond_latent, return_attentions=False, return_latent=True,
                    )
                    if gpt_latents.shape[1] == 0:
                        wav = np.zeros(1, dtype=np.float32)
                    else:
                        wav = model.hifigan_decoder(gpt_latents, g=speaker_embedding)
                        wav = wav.cpu().squeeze().float().numpy()
            import soundfile as sf
            sf.write(wav_path, wav, sr)
            wav_duration_s = float(np.asarray(wav).size / sr)

            rec = dict(
                item_id=it["item_id"], seed=seed, model=spec.key, k=it["k"],
                family=it["family"], n_speech_tokens=n_speech_tokens,
                prompt_len=int(gpt_cond_latent.shape[1] + text_tokens.shape[1] + 2 + 1),
                text_lo=int(gpt_cond_latent.shape[1] + 1),
                text_hi=int(gpt_cond_latent.shape[1] + 1 + text_tokens.shape[1]),
                hit_cap=hit_cap, est_duration_s=n_speech_tokens / spec.token_rate_hz,
                sr=sr, wav_duration_s=wav_duration_s,
                text_chars=char_len, text_over_char_limit=char_len > char_limit,
            )

            # ---- instrumented teacher-forced pass -------------------------
            if seed == args.instrument_seed and it.get("instrumented") and n_speech_tokens:
                res = instrumented_pass(
                    model, gpt_cond_latent, text_tokens, gen_no_eos, attn_probes, probes,
                )
                np.savez_compressed(
                    act_dir / f"{it['item_id']}_s{seed}.npz",
                    hidden=res["hidden"], probe_layers=np.array(probes),
                    entropy=res["entropy"], top1=res["top1"], **res["attn"],
                )
                assert res["text_lo"] == rec["text_lo"] and res["text_hi"] == rec["text_hi"]
                rec["instrumented"] = True

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
