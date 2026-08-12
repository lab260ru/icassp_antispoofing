#!/usr/bin/env python3
"""Recover the teacher-forcing input for Qwen3-TTS, which was never saved.

`analysis/jacobian_q.py` needs one thing from a checkpoint: a *fixed*,
differentiable, teacher-forced forward pass over a trajectory the decoder
itself produced. For Llasa that is free -- `llasa_gen.py` stores the generated
codec integers, and the teacher-forced input is `[prompt ++ speech ids]`.
Qwen3-TTS gives neither half of that for free:

  * `qwen_gen.py` never writes the codec ids. They exist only inside
    `generate()` (`result.hidden_states[j][1]`, a `[1,16]` tensor per frame),
    are read once for EOS trimming, and are discarded. `tokens/qwen06b/` does
    not exist; only `qwen06b_meta.jsonl` and the activation summaries do.
  * Even with the ids, there is no token sequence to feed. The talker's input
    at each decode step is not an embedding lookup but a **sum of seventeen
    vectors**: one embedding per RVQ codebook (codebook 0 through
    `talker.model.codec_embedding`, codebooks 1..15 through
    `talker.code_predictor.get_input_embeddings()[i]`), plus a text term
    (`trailing_text_hidden[:, step]`, or `tts_pad_embed` once the text is
    exhausted). Text and codec content are additively fused, which is also why
    this architecture exposes no text-attention span -- a fact the Jacobian
    estimator does not care about, since it needs no attention read-head.

Rather than reimplement that seventeen-way sum (and the 85-row voice-clone
prefill that precedes it, which is built inside an `@torch.no_grad()` /
`@torch.inference_mode()` code path), this script **captures the input
embeddings the model actually used**, with a `forward_pre_hook` on the talker's
decoder stack. Generation calls that stack once with the `[1, 85, d]` prefill
and once per decode step with a `[1, 1, d]` row; concatenating them in order
reconstructs, exactly and by construction, the sequence a teacher-forced pass
must feed. There is no reconstruction formula to get wrong, and the equivalence
is checked rather than asserted (see below).

Fidelity choices, and why
-------------------------
* **Generation runs in bfloat16 with eager attention**, byte-for-byte the
  settings `qwen_gen.py` used, with the same `torch.manual_seed(seed)` and the
  same voice-clone prompt. The point is to reproduce the trajectory already on
  disk, so the CTC-judged rendered count for that `(item_id, seed)` still
  describes the sequence we measure. We record both our `n_speech_tokens` and
  the ledger's, and `--report` prints the agreement rate; where they disagree,
  the analysis must not use `tau_rendered` for that item.
* **The captured embeddings are stored as float32** and the Jacobian analysis
  loads the talker in float32. This is the same split Llasa already runs under:
  the trajectory is sampled in bf16, the derivative is taken in fp32. The
  embeddings play the role Llasa's integer ids play -- fixed conditioning, not
  something differentiated through.
* **A teacher-forcing equivalence check is written into every dump.** We store
  the final-layer hidden state generation produced at each step; the analysis
  side re-runs the whole captured sequence in one shot and must reproduce it.
  If a hook were attached to the wrong tensor, or the concatenation order were
  wrong, this is what would catch it, and it is reported as a gate rather than
  a diagnostic.

Usage (env `qwen`):
  conda run -n qwen python analysis/qwen_codec_dump.py --model qwen06b --gpu 2
  python analysis/qwen_codec_dump.py --model qwen06b --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402
from common.registry import BY_KEY, DATA_ROOT  # noqa: E402

KS = (16, 24, 32)
# Pass thresholds for the teacher-forcing equivalence gate. The residual is
# dominated by the dtype split -- generation ran in bfloat16 (~3 decimal digits
# of mantissa) and the one-shot re-run is float32 -- so ~1e-2 relative is the
# expected floor, not a defect. Direction agreement is the sharp test: a hook on
# the wrong tensor or a mis-ordered concatenation destroys the cosine, it does
# not nudge the norm.
EQUIV_COS_MIN = 0.999
EQUIV_REL_MAX = 0.05


def out_dir(model: str) -> Path:
    return Path(DATA_ROOT) / "tokens" / model


class StackCapture:
    """Records every `inputs_embeds` the talker's decoder stack is called with.

    One prefill call of width `plen`, then one call of width 1 per decode step.
    Also keeps the stack's own `last_hidden_state` per call, which is what the
    teacher-forcing equivalence gate compares against.
    """

    def __init__(self, stack):
        self.stack = stack
        self.embeds: list[torch.Tensor] = []
        self.last: list[torch.Tensor] = []
        self._h = [
            stack.register_forward_pre_hook(self._pre, with_kwargs=True),
            stack.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, _mod, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is None and args:
            e = args[0]
        if torch.is_tensor(e) and e.dim() == 3:
            self.embeds.append(e[0].detach().float().cpu())
        return None

    def _post(self, _mod, _args, _kwargs, output):
        h = getattr(output, "last_hidden_state", None)
        if torch.is_tensor(h):
            self.last.append(h[0].detach().float().cpu())
        return None

    def clear(self):
        self.embeds.clear()
        self.last.clear()

    def remove(self):
        for h in self._h:
            h.remove()


def select_items(stimuli: str, model: str, ks, seeds) -> list[tuple[str, int]]:
    """The same population `jacobian_q.py` measures: word_rep items at the
    requested k with a matched control, both members cap-free in the *repaired*
    ledger, judge-unmeasurable templates dropped."""
    from src.common.population import excluded_templates
    stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(stimuli)}
    meta = {}
    for line in open(Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"):
        r = json.loads(line)
        meta[(r["item_id"], r["seed"])] = r
    bad = excluded_templates()
    want: list[tuple[str, int]] = []
    for iid, it in stim.items():
        if it["family"] != "word_rep" or it["k"] not in ks or it["template"] in bad:
            continue
        ctrl = next((c for c, v in stim.items() if v.get("control_of") == iid), None)
        if ctrl is None:
            continue
        for s in seeds:
            mw, mc = meta.get((iid, s)), meta.get((ctrl, s))
            if not mw or not mc or mw.get("hit_cap") or mc.get("hit_cap"):
                continue
            want += [(iid, s), (ctrl, s)]
    seen = set()
    return [w for w in want if not (w in seen or seen.add(w))]


def report(model: str) -> None:
    """Agreement between the re-sampled trajectory and the ledger's."""
    meta = {}
    for line in open(Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"):
        r = json.loads(line)
        meta[(r["item_id"], r["seed"])] = r
    rows = []
    for f in sorted(out_dir(model).glob("*.npz")):
        iid, _, s = f.stem.rpartition("_s")
        z = np.load(f)
        m = meta.get((iid, int(s)))
        rows.append(dict(item=iid, seed=int(s), mine=int(z["n_speech_tokens"]),
                         ledger=int(m["n_speech_tokens"]) if m else -1,
                         hit_cap=bool(z["hit_cap"]),
                         equiv_cos=float(z["equiv_cos"]),
                         equiv_rel=float(z["equiv_rel"])))
    if not rows:
        print("nothing dumped yet")
        return
    same = sum(r["mine"] == r["ledger"] for r in rows)
    cos = np.array([r["equiv_cos"] for r in rows])
    rel = np.array([r["equiv_rel"] for r in rows])
    print(f"{len(rows)} dumps; n_speech_tokens identical to the ledger in "
          f"{same}/{len(rows)} ({100*same/len(rows):.0f}%)")
    ok = bool(np.all(cos > EQUIV_COS_MIN) and np.all(rel < EQUIV_REL_MAX))
    print(f"teacher-forcing equivalence: cosine min {cos.min():.6f} "
          f"median {np.median(cos):.6f}; relative error max {rel.max():.3e} "
          f"-> {'PASS' if ok else 'FAIL'} "
          f"(need cos>{EQUIV_COS_MIN}, rel<{EQUIV_REL_MAX})")
    print(f"cap hits in this dump: {sum(r['hit_cap'] for r in rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen06b")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--language", default="English")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="summarise existing dumps and exit (no GPU)")
    args = ap.parse_args()

    if args.report:
        report(args.model)
        return

    check_gpu(args.gpu)
    spec = BY_KEY[args.model]
    assert spec.family == "qwen", f"{args.model} is not a qwen spec"
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    want = select_items(args.stimuli, args.model, tuple(args.ks), tuple(args.seeds))
    if args.limit:
        want = want[: args.limit]
    dest = out_dir(args.model)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [w for w in want if not (dest / f"{w[0]}_s{w[1]}.npz").exists()]
    print(f"[{spec.key}] {len(want)} (item, seed) needed, {len(todo)} to generate",
          flush=True)
    if not todo:
        report(args.model)
        return

    from qwen_tts import Qwen3TTSModel
    from src.models.qwen_gen import build_prompt, decode_step_count, eos_trim_length

    print(f"[{spec.key}] loading {spec.hf_id} bfloat16/eager on {device} "
          f"(identical to qwen_gen.py, so the trajectory reproduces)", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        spec.hf_id, device_map=device, dtype=torch.bfloat16,
        attn_implementation="eager")
    talker = tts.model.talker
    eos_id = talker.config.codec_eos_token_id
    cap = StackCapture(talker.model)

    stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.stimuli)}
    prompt_items = build_prompt(tts)

    t0 = time.time()
    for n, (iid, seed) in enumerate(todo):
        torch.manual_seed(seed)
        cap.clear()
        _wavs, _sr = tts.generate_voice_clone(
            text=stim[iid]["text"], language=args.language,
            voice_clone_prompt=prompt_items, max_new_tokens=args.max_new_tokens)
        result = None
        n_steps = len(cap.embeds) - 1
        # widths: [plen, 1, 1, ...]; concatenating gives the teacher-forced input
        plen = int(cap.embeds[0].shape[0])
        E = torch.cat(cap.embeds, dim=0)              # [plen + n_steps, d]
        H = torch.cat(cap.last, dim=0)                # same rows, stack output
        hit_cap = bool(n_steps >= args.max_new_tokens)
        # trim to the frames that were actually rendered, matching the ledger's
        # definition; generation stops on EOS without materialising an EOS frame,
        # so this is usually a no-op, and never lengthens.
        t_gen = min(n_steps, int(args.max_new_tokens))
        E = E[: plen + t_gen]
        H = H[: plen + t_gen]

        np.savez(dest / f"{iid}_s{seed}.npz",
                 embeds=E.numpy().astype(np.float32),
                 gen_hidden_last=H[plen:].numpy().astype(np.float32),
                 plen=np.int32(plen), n_speech_tokens=np.int32(t_gen),
                 hit_cap=np.bool_(hit_cap),
                 equiv_cos=np.float32(np.nan), equiv_rel=np.float32(np.nan))
        el = time.time() - t0
        print(f"[{n+1}/{len(todo)}] {iid} s{seed} plen={plen} T={t_gen} "
              f"cap={hit_cap} {el/(n+1):.1f}s/item", flush=True)
    cap.remove()
    del tts
    torch.cuda.empty_cache()

    # ---- teacher-forcing equivalence gate ----------------------------------
    # Re-run each captured sequence in ONE shot through the same stack and
    # require it to reproduce what incremental generation produced. This is the
    # check that the hook was on the right tensor and the concatenation order is
    # right; a wrong answer here invalidates every Jacobian downstream.
    print("\nteacher-forcing equivalence gate (fp32, one-shot vs incremental)",
          flush=True)
    from qwen_tts import Qwen3TTSModel as _M
    tts32 = _M.from_pretrained(spec.hf_id, device_map=device, dtype=torch.float32,
                               attn_implementation="sdpa")
    stack = tts32.model.talker.model
    for p in tts32.model.parameters():
        p.requires_grad_(False)
    for f in sorted(dest.glob("*.npz")):
        z = dict(np.load(f))
        E = torch.tensor(z["embeds"], device=device).unsqueeze(0)
        with torch.no_grad():
            out = stack(inputs_embeds=E, use_cache=False)
        got = out.last_hidden_state[0, int(z["plen"]):, :].float().cpu().numpy()
        ref = z["gen_hidden_last"]
        m = min(len(got), len(ref))
        a, b = got[:m].ravel(), ref[:m].ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
        z["equiv_cos"] = np.float32(cos)
        z["equiv_rel"] = np.float32(rel)
        np.savez(f, **z)
    report(args.model)


if __name__ == "__main__":
    main()
