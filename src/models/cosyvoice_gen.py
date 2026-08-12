#!/usr/bin/env python3
"""CosyVoice 2 (FunAudioLLM/CosyVoice2-0.5B) generation with state instrumentation.

Why this model is in the panel
------------------------------
Every other autoregressive member of the panel learns the text->speech alignment
implicitly, from the LM objective alone. CosyVoice 2 does not: its speech
tokeniser is trained *supervised*, the FSQ speech tokens are derived from a
supervised ASR encoder, and the flow-matching decoder is conditioned on an
explicitly aligned token stream. If the counting failure tracks periodicity in AR
decoders as such, CosyVoice 2 should show it too; if alignment supervision is the
fix, this is the checkpoint where the gap closes. Either way the answer is a
result, which is why the run exists.

Interface: identical to `xtts_gen.py` / `qwen_gen.py` -- `--model`, `--gpu`,
`--seeds`, `--stimuli`, resumable through the `<key>_meta.jsonl` ledger, audio
to `DATA_ROOT/audio/<key>`, activations to `DATA_ROOT/activations/<key>`.

Three things had to be done differently, each verified against the checked-out
CosyVoice source (`third_party/CosyVoice`, commit recorded in the meta ledger):

  1. **No `inference_zero_shot()`.** That entry point runs
     `frontend.text_normalize(text, split=True)`, which sends English through
     `split_paragraph(..., token_max_n=80, token_min_n=60)` -- i.e. it *chops the
     stimulus into sentence groups and synthesises them independently*. A
     `sentence_rep` item at k=16 is sixteen sentences and would be generated as
     several separate utterances, each one re-conditioned from scratch: the
     periodic conditioning under study would be destroyed by the harness before
     the decoder ever saw it. This script therefore calls the low-level pieces
     (`frontend.frontend_zero_shot` -> `llm.inference` -> `model.token2wav`)
     directly with the whole stimulus as one contiguous text span, exactly as
     `xtts_gen.py` bypasses `Xtts.inference()`'s own sentence splitter.
     Normalisation is also skipped (`text_frontend=False` semantics), which is
     what CosyVoice's own README asks for when reproducing their results, and
     which keeps this model's input byte-identical to what every other panel
     member is given.

  2. **The decode loop is re-implemented, verbatim.** `Qwen2LM.inference_wrapper`
     already asks `Qwen2Encoder.forward_one_step` for `output_hidden_states=True`
     and then keeps only the last layer, so a faithful copy of that loop gets the
     full per-layer trajectory for free -- no second teacher-forced pass, and no
     change to what is sampled. The copy below preserves the original's
     `min_len`/`max_len`, its EOS suppression below `min_len`, its `sampling_ids`
     call and its `lm_input` update, line for line. The one addition is
     bookkeeping.

  3. **`CUDA_VISIBLE_DEVICES`, not `.to(device)`.** `CosyVoice2Model.__init__`
     and `CosyVoiceFrontEnd.__init__` both hardcode
     `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` -- i.e.
     cuda:0, which on this host belongs to someone else. There is no device
     argument to pass. The only safe way to honour `src/common/gpus.py` is to
     hide every other card from the process before torch initialises, so `--gpu 3`
     is implemented as `CUDA_VISIBLE_DEVICES=3` and everything inside sees a
     single device numbered 0.

The model's own generation ceiling
----------------------------------
CosyVoice 2 has no fixed token budget like XTTS-v2's ~602 mel tokens. Its cap is
*proportional* to the text: `Qwen2LM.inference` sets
`max_len = 20 * n_text_tokens` (`max_token_text_ratio`) and
`min_len = 2 * n_text_tokens`, with EOS masked out below `min_len`. At the 25 Hz
token rate that is 0.8 s of speech per text token as a ceiling and 0.08 s as a
floor. Both bounds are recorded per item (`max_len`, `min_len`, `hit_cap`,
`hit_min`) so a truncation on the model's own ceiling can be told apart from a
counting failure, and so the floor -- which can *force* a model to keep talking --
is visible rather than silent.

Usage:
  python src/models/cosyvoice_gen.py --model cosyvoice2 --gpu 3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid as uuidlib
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")
COSY_ROOT = Path(os.environ.get(
    "COSYVOICE_ROOT", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/third_party/CosyVoice"))

DEFAULT_PROMPT_WAV = REPO / "data/ref/qwen_ref.wav"
DEFAULT_PROMPT_TXT = REPO / "data/ref/qwen_ref.txt"


def _pin_gpu(gpu: int) -> None:
    """Hide every card but `gpu` *before* torch initialises.

    CosyVoice hardcodes cuda:0 in three places and exposes no device argument;
    this is the only way to keep it on an allowed card. Called before `import
    torch`, so cuda:0 inside the process is physical GPU `gpu`.
    """
    check_gpu(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _patch_audio_io() -> None:
    """Replace CosyVoice's `load_wav` with a soundfile-based equivalent.

    `torchaudio.load` in 2.11+ dispatches to TorchCodec, which does not load
    against the FFmpeg installed on this host -- the same trap that forced the
    Whisper judge onto processor+model instead of `pipeline` (see
    implementation-notes.md §7). The replacement is semantically identical to
    `cosyvoice/utils/file_utils.py::load_wav`: read, average to mono, resample to
    the requested rate. Patched in both namespaces because `cli/frontend.py`
    imports the symbol directly.
    """
    import soundfile as sf
    import torch
    import torchaudio

    import cosyvoice.cli.frontend as _fe
    import cosyvoice.utils.file_utils as _fu

    def load_wav(wav, target_sr, min_sr=16000):
        data, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, f"wav sample rate {sample_rate} < {min_sr}"
            speech = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    _fu.load_wav = load_wav
    _fe.load_wav = load_wav


def load_cosyvoice(hf_id: str):
    """Snapshot from the HF cache (so the revision is pinnable) and construct."""
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(hf_id)
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice2

    _patch_audio_io()
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    return cosy, model_dir


def decode(llm, lm_input, min_len: int, max_len: int, record: bool, probe_layers):
    """Verbatim copy of `Qwen2LM.inference_wrapper`'s non-vLLM branch, plus
    optional hidden-state capture.

    Reference: cosyvoice/llm/llm.py, `Qwen2LM.inference_wrapper`. The sampling
    call, the EOS suppression below `min_len`, the stop-token test and the
    single-token `lm_input` update are unchanged; `forward_one_step` is inlined
    only so that the hidden states it already computes (it passes
    `output_hidden_states=True` itself and discards all but the last layer) can
    be kept.
    """
    import torch

    out_tokens: list[int] = []
    cache = None
    hidden_rows: list = []
    logits_rows: list = []
    stopped = None
    for i in range(max_len):
        masks = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                      device=lm_input.device)).to(torch.bool)
        outs = llm.llm.model(inputs_embeds=lm_input, attention_mask=masks[:, -1, :],
                             output_hidden_states=True, return_dict=True, use_cache=True,
                             past_key_values=cache)
        cache = outs.past_key_values
        y_pred = outs.hidden_states[-1]
        logp = llm.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        if record:
            hidden_rows.append(torch.stack(
                [outs.hidden_states[l + 1][0, -1, :] for l in probe_layers], dim=0
            ).to(torch.float16).cpu())
            logits_rows.append(logp[0].detach().float().cpu())
        top_ids = llm.sampling_ids(logp.squeeze(dim=0), out_tokens, 25,
                                   ignore_eos=True if i < min_len else False)
        if top_ids in llm.stop_token_ids:
            stopped = int(top_ids)
            break
        out_tokens.append(top_ids)
        lm_input = llm.speech_embedding.weight[top_ids].reshape(1, 1, -1)
    return out_tokens, hidden_rows, logits_rows, stopped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cosyvoice2")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU, help="use GPU 2 or 3 only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--instrument-seed", type=int, default=0,
                    help="only this seed records hidden states")
    ap.add_argument("--no-instrument", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-wav", default=str(DEFAULT_PROMPT_WAV))
    ap.add_argument("--prompt-text", default=None)
    # The model's own ceiling/floor. Defaults are the library's; exposed so the
    # ceiling can be *reported* rather than assumed, and raised if it is ever
    # found to be censoring a cell (the XTTS-v2 trap).
    ap.add_argument("--max-token-text-ratio", type=float, default=20.0)
    ap.add_argument("--min-token-text-ratio", type=float, default=2.0)
    args = ap.parse_args()

    _pin_gpu(args.gpu)

    import numpy as np
    import torch

    from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices

    spec = BY_KEY[args.model]
    device = "cuda:0"  # == physical args.gpu, see _pin_gpu

    print(f"[{spec.key}] loading {spec.hf_id} (physical GPU {args.gpu})", flush=True)
    cosy, model_dir = load_cosyvoice(spec.hf_id)
    llm = cosy.model.llm
    n_layers = llm.llm.model.config.num_hidden_layers
    probes = probe_layer_indices(n_layers, spec.probe_layers)
    print(f"[{spec.key}] llm_layers={n_layers} d={llm.llm.model.config.hidden_size} "
          f"state_probes={probes} sample_rate={cosy.sample_rate}", flush=True)

    prompt_text = (args.prompt_text if args.prompt_text is not None
                   else DEFAULT_PROMPT_TXT.read_text().strip())
    print(f"[{spec.key}] zero-shot prompt: {Path(args.prompt_wav).name} {prompt_text!r}",
          flush=True)
    # Built once and reused: the prompt's speech tokens, mel features and speaker
    # embedding do not depend on the stimulus, and recomputing them per item
    # would run the ONNX tokeniser 540 times for no reason.
    prompt_input = cosy.frontend.frontend_zero_shot(
        "", prompt_text, args.prompt_wav, cosy.sample_rate, "")
    prompt_text_tok = prompt_input["prompt_text"]
    prompt_text_len = int(prompt_text_tok.shape[1])
    prompt_speech_token = prompt_input["llm_prompt_speech_token"]

    items = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        items = items[: args.limit]

    aud_dir = Path(DATA_ROOT) / "audio" / spec.key
    act_dir = Path(DATA_ROOT) / "activations" / spec.key
    aud_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(DATA_ROOT) / "tokens" / f"{spec.key}_meta.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if meta_path.exists():
        for line in open(meta_path):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["seed"]))
            except Exception:  # noqa: BLE001
                pass
    meta_f = open(meta_path, "a")

    import soundfile as sf

    sr = int(cosy.sample_rate)
    t_start = time.time()
    n_done = 0
    for seed in args.seeds:
        for it in items:
            key = (it["item_id"], seed)
            if key in done:
                continue
            torch.manual_seed(seed)

            # ---- text -> tokens, no normalisation, no sentence splitting ----
            text_tok, text_tok_len = cosy.frontend._extract_text_token(it["text"])
            text_tok = text_tok.to(device)
            n_text = int(text_tok.shape[1])

            # ---- lm_input, exactly as Qwen2LM.inference builds it ------------
            full_text = torch.concat([prompt_text_tok.to(device), text_tok], dim=1)
            text_emb = llm.llm.model.model.embed_tokens(full_text)
            sos_emb = llm.llm_embedding.weight[llm.sos].reshape(1, 1, -1)
            task_id_emb = llm.llm_embedding.weight[llm.task_id].reshape(1, 1, -1)
            pst = prompt_speech_token.to(device)
            if pst.shape[1] != 0:
                prompt_speech_emb = llm.speech_embedding(pst)
            else:
                prompt_speech_emb = torch.zeros(1, 0, llm.llm_input_size,
                                                dtype=text_emb.dtype, device=text_emb.device)
            lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)

            min_len = int(n_text * args.min_token_text_ratio)
            max_len = int(n_text * args.max_token_text_ratio)

            record = (seed == args.instrument_seed and bool(it.get("instrumented"))
                      and not args.no_instrument)
            with torch.inference_mode():
                tokens, hidden_rows, logit_rows, stopped = decode(
                    llm, lm_input, min_len, max_len, record, probes)

            n_speech_tokens = len(tokens)
            hit_cap = n_speech_tokens >= max_len          # the model's own ceiling
            # The floor is not a stop condition but a suppression: EOS is masked
            # out for the first `min_len` steps, so an item that ends at or just
            # above the floor may have been *forced* to keep talking.
            hit_min = n_speech_tokens <= min_len

            # ---- token -> waveform, the library's own non-stream path --------
            wav_path = aud_dir / f"{it['item_id']}_s{seed}.wav"
            if n_speech_tokens == 0:
                wav = np.zeros(1, dtype=np.float32)
            else:
                this_uuid = str(uuidlib.uuid1())
                cosy.model.hift_cache_dict[this_uuid] = None
                with torch.inference_mode():
                    speech = cosy.model.token2wav(
                        token=torch.tensor(tokens).unsqueeze(dim=0),
                        prompt_token=prompt_input["flow_prompt_speech_token"],
                        prompt_feat=prompt_input["prompt_speech_feat"],
                        embedding=prompt_input["flow_embedding"],
                        token_offset=0, uuid=this_uuid, stream=False, finalize=True, speed=1.0)
                cosy.model.hift_cache_dict.pop(this_uuid, None)
                wav = speech.cpu().squeeze().float().numpy()
            sf.write(wav_path, wav, sr)

            rec = dict(
                item_id=it["item_id"], seed=seed, model=spec.key, k=it["k"],
                family=it["family"], n_speech_tokens=n_speech_tokens,
                prompt_len=int(lm_input.shape[1]),
                # the stimulus text occupies a contiguous span of LM positions:
                # [sos][prompt_text][stimulus text][task_id][prompt speech tokens]
                text_lo=1 + prompt_text_len,
                text_hi=1 + prompt_text_len + n_text,
                n_text_tokens=n_text, max_len=max_len, min_len=min_len,
                hit_cap=bool(hit_cap), hit_floor=bool(hit_min),
                stop_token=stopped,
                est_duration_s=n_speech_tokens / spec.token_rate_hz,
                sr=sr, wav_duration_s=float(np.asarray(wav).size) / sr,
            )

            if record and n_speech_tokens:
                hs = torch.stack(hidden_rows[:n_speech_tokens], dim=0).numpy()  # [T,P,D]
                lg = torch.stack(logit_rows[:n_speech_tokens], dim=0)
                pr = lg.exp()
                ent = (-(pr * lg.clamp_min(-1e9)).sum(-1)).numpy()
                top1 = pr.max(-1).values.numpy()
                np.savez_compressed(
                    act_dir / f"{it['item_id']}_s{seed}.npz",
                    hidden=hs, probe_layers=np.array(probes),
                    entropy=ent.astype(np.float32), top1=top1.astype(np.float32),
                )
                rec["instrumented"] = True

            meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t_start
                print(f"[{spec.key}] {n_done} items  {el/n_done:.1f}s/item  "
                      f"last={it['item_id']} k={it['k']} ntok={n_speech_tokens} "
                      f"cap={max_len}{' HIT' if hit_cap else ''}", flush=True)
            del lm_input, hidden_rows, logit_rows
            torch.cuda.empty_cache()

    meta_f.close()
    print(f"[{spec.key}] DONE {n_done} items in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
