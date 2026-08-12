#!/usr/bin/env python3
"""Llasa generation with optional state instrumentation.

Two passes per item:

  1. **Generate** — sample speech tokens autoregressively (fast attention).
     Speech tokens are saved as integer ids; waveform synthesis happens later in
     the isolated `xcodec2` env, so the LM pass never depends on the codec's
     torch pin.

  2. **Instrument** (optional) — one teacher-forced forward over
     [prompt + generated tokens] with `output_hidden_states=True`, plus hooks on
     probe attention layers that slice the attention row down to the text span.
     Because the model is causal, a single clean forward recovers the whole
     trajectory without touching `generate()` internals.

     **The saved arrays are offset from each other by one step.** `hidden` and
     `attn_*` are sliced `[plen:]`, so index `t` is the state left *after* token
     `t` was emitted; `entropy` and `top1` come from `logits[plen-1:-1]`, so
     index `t` is the distribution that *produced* token `t`. Our step `t` is
     therefore their step `t+1`. Anything that joins the two — reading a logit
     off a stored hidden state, or aligning attention to a predicted token —
     must shift one of them. Getting this wrong is not loud: reconstructing the
     EOS logit from the last probe layer gives 0.075 nats of error when shifted
     and 3.55 nats when not, which looks like a bad probe rather than a bad
     index. An earlier version of this docstring asserted that position `t`
     carries the state that produced token `t`, which is false for the arrays as
     sliced.

     Note also that the last probe layer is already post-final-RMSNorm
     (`transformers` appends `norm(h)` as the final `hidden_states` element), so
     exact logits are one matmul with `lm_head.weight` — no forward pass needed.
     Llasa-1B ties embeddings; Llasa-8B does not.

Usage:
  python src/models/llasa_gen.py --model llasa1b --gpu 0 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from src.common.gpus import DEFAULT_GPU, check_gpu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.registry import BY_KEY, DATA_ROOT, probe_layer_indices  # noqa: E402

TEXT_START = "<|TEXT_UNDERSTANDING_START|>"
TEXT_END = "<|TEXT_UNDERSTANDING_END|>"
SPEECH_START = "<|SPEECH_GENERATION_START|>"
SPEECH_END = "<|SPEECH_GENERATION_END|>"


def build_prompt(tokenizer, text: str) -> torch.Tensor:
    formatted = f"{TEXT_START}{text}{TEXT_END}"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted},
        {"role": "assistant", "content": SPEECH_START},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", continue_final_message=True
    )


def text_span(tokenizer, ids: torch.Tensor) -> tuple[int, int]:
    """Column range [lo, hi) of the prompt occupied by the user's text."""
    flat = ids[0].tolist()
    s = tokenizer.convert_tokens_to_ids(TEXT_START)
    e = tokenizer.convert_tokens_to_ids(TEXT_END)
    lo = flat.index(s) + 1 if s in flat else 0
    hi = flat.index(e) if e in flat else len(flat)
    return lo, hi


def extract_speech_ids(tokens: list[str]) -> list[int]:
    out = []
    for t in tokens:
        if t.startswith("<|s_") and t.endswith("|>"):
            out.append(int(t[4:-2]))
    return out


class TextAttentionRecorder:
    """Hook that keeps only the text-span columns of each attention row.

    Storing full attention is O(T^2 * heads * layers); we only ever need how
    much mass each *text* position receives, so the hook reduces to
    [positions, text_len] (head-averaged) inside the forward pass.
    """

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.buf: dict[int, np.ndarray] = {}
        self.handles = []

    def _make(self, layer_idx: int):
        def hook(_module, _args, output):
            # eager attention returns (attn_output, attn_weights)
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None or w.dim() != 4:
                return
            # [B, H, Q, K] -> mean over heads, slice text columns
            sl = w[0, :, :, self.lo:self.hi].mean(dim=0)
            self.buf[layer_idx] = sl.detach().to(torch.float16).cpu().numpy()
        return hook

    def attach(self, model, layer_indices):
        layers = model.model.layers
        for i in layer_indices:
            self.handles.append(layers[i].self_attn.register_forward_hook(self._make(i)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--instrument-seed", type=int, default=0,
                    help="only this seed gets the instrumented second pass")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn-probes", type=int, default=4)
    args = ap.parse_args()
    check_gpu(args.gpu)

    spec = BY_KEY[args.model]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{spec.key}] loading {spec.hf_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()

    def set_attn(mode: str) -> None:
        """Swap the attention kernel between passes.

        Sampling is the bulk of the cost and does not need attention weights, so
        it runs under SDPA; only the single teacher-forced instrumentation pass
        needs eager, which is the only kernel that materialises the weights our
        hooks read. Dispatch reads `config._attn_implementation` at forward
        time, so flipping it per pass is sufficient.
        """
        model.config._attn_implementation = mode
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config._attn_implementation = mode
            for lyr in getattr(inner, "layers", []):
                if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "config"):
                    lyr.self_attn.config._attn_implementation = mode

    n_layers = model.config.num_hidden_layers
    probes = probe_layer_indices(n_layers, spec.probe_layers)
    # attention probes: early / mid / late / last, a subset of the state probes
    attn_probes = sorted({0, n_layers // 3, 2 * n_layers // 3, n_layers - 1})[: args.attn_probes]
    print(f"[{spec.key}] layers={n_layers} state_probes={probes} attn_probes={attn_probes}",
          flush=True)

    eos = tok.convert_tokens_to_ids(SPEECH_END)

    items = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        items = items[: args.limit]

    tok_dir = Path(DATA_ROOT) / "tokens" / spec.key
    act_dir = Path(DATA_ROOT) / "activations" / spec.key
    tok_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(DATA_ROOT) / "tokens" / f"{spec.key}_meta.jsonl"
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
            ids = build_prompt(tok, it["text"]).to(device)
            lo, hi = text_span(tok, ids)
            plen = ids.shape[1]

            set_attn("sdpa")
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                    do_sample=True, top_p=args.top_p, temperature=args.temperature,
                    pad_token_id=tok.eos_token_id,
                )
            gen = out[0][plen:]
            hit_cap = int(gen.shape[0]) >= args.max_new_tokens
            gen_no_eos = gen[:-1] if (gen.numel() and gen[-1].item() == eos) else gen
            sp_tokens = tok.batch_decode(gen_no_eos, skip_special_tokens=True)
            sp_ids = extract_speech_ids(sp_tokens)

            rec = dict(
                item_id=it["item_id"], seed=seed, model=spec.key, k=it["k"],
                family=it["family"], n_speech_tokens=len(sp_ids),
                prompt_len=plen, text_lo=lo, text_hi=hi, hit_cap=hit_cap,
                est_duration_s=len(sp_ids) / spec.token_rate_hz,
            )
            np.save(tok_dir / f"{it['item_id']}_s{seed}.npy",
                    np.asarray(sp_ids, dtype=np.int32))

            # ---- instrumented teacher-forced pass -------------------------
            if seed == args.instrument_seed and it.get("instrumented") and sp_ids:
                full = out[0][: plen + gen_no_eos.shape[0]].unsqueeze(0)
                recorder = TextAttentionRecorder(lo, hi)
                recorder.attach(model, attn_probes)
                # Eager attention materialises a T-by-T matrix per layer, which
                # is the only way to read attention weights and also what makes
                # this pass impossible past a few thousand tokens: at k=128 the
                # sequence is ~4000 long and one softmax wants 5.4 GiB. When no
                # attention probes are requested -- the count probe reads hidden
                # states only -- SDPA does the same forward without ever forming
                # that matrix.
                want_attn = bool(attn_probes)
                set_attn("eager" if want_attn else "sdpa")
                with torch.no_grad():
                    fwd = model(full, output_hidden_states=True,
                                output_attentions=want_attn)
                recorder.detach()
                # hidden_states: tuple(n_layers+1) of [1, T, d]; index 0 is the
                # embedding output, so layer l is at index l+1.
                hs = np.stack(
                    [fwd.hidden_states[l + 1][0, plen:, :].to(torch.float16).cpu().numpy()
                     for l in probes],
                    axis=1,
                )  # [T_gen, n_probes, d]
                attn = {f"attn_l{l}": v[plen:, :] for l, v in recorder.buf.items()}
                logits = fwd.logits[0, plen - 1: -1, :].float()
                pr = torch.softmax(logits, dim=-1)
                ent = (-(pr * torch.log(pr.clamp_min(1e-12))).sum(-1)).cpu().numpy()
                top1 = pr.max(-1).values.cpu().numpy()
                # Uncompressed: zlib on a 150 MB fp16 array costs more CPU and
                # peak RSS than the disk it saves, and disk is not the scarce
                # resource here.
                np.savez(
                    act_dir / f"{it['item_id']}_s{seed}.npz",
                    hidden=hs, probe_layers=np.array(probes),
                    entropy=ent.astype(np.float32), top1=top1.astype(np.float32),
                    **attn,
                )
                rec["instrumented"] = True
                del fwd, hs, attn, logits, pr
            meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t_start
                print(f"[{spec.key}] {n_done} items  {el/n_done:.1f}s/item  "
                      f"last={it['item_id']} k={it['k']} ntok={len(sp_ids)}", flush=True)
            torch.cuda.empty_cache()

    meta_f.close()
    print(f"[{spec.key}] DONE {n_done} items in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
