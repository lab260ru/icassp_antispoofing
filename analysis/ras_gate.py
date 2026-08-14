#!/usr/bin/env python3
"""Sanity gate for the repetition-aware-sampling implementation.

The gate is pre-committed in `src/common/ras.py` and is copied here as code so
it cannot drift:

  A. DETERMINISM CONTROL. The stock path, run twice in one process on the same
     items with the same seed, must produce identical codebook-0 token
     sequences. Bitwise identity has to be shown attainable before it is
     demanded of the RAS path.
  B. NO-OP EQUIVALENCE. RAS with t_r = 2.0 -- unreachable, since r <= (1+K)/K =
     1.1 -- must reproduce the stock path exactly: identical token sequences AND
     identical wav bytes, same seed, same process.

  PASS         A and B hold.
  UNREPORTABLE A holds, B fails: the plumbing changes generation even when the
               rule never fires, so nothing downstream measures RAS.
  VOID         A fails: bitwise identity is unattainable here, and the arm is
               reported as inconclusive-by-construction.

Two further things are measured here and reported, neither of them gating:

  * WIRING. That the `input_ids` the processor sees really are the talker's
    emitted codebook-0 history, checked by replaying every step's window
    against the returned sequence. If that were false the repetition ratio
    would be computed against the wrong tensor and the arm would be measuring
    nothing, silently.
  * LIBRARY DRIFT. The freshly generated stock audio against the *stored*
    `qwen06b` audio for the same item and seed. `qwen_gen.eos_trim_length`'s
    docstring already records that this environment's transformers no longer
    issues the trailing forward call it used to, so the panel's stored Qwen
    audio was produced by a build that no longer exists here. If the stored and
    fresh audio differ, that is why, and it is the reason the RAS arms are
    compared against `qwen06brasoff` -- a stock baseline regenerated today --
    rather than against the stored panel rows. The comparison goes through the
    same PCM-16 round trip the stored file went through; comparing the
    in-memory float32 array against a 16-bit file's read-back reports drift on
    every item unconditionally, which is what the first version of this check
    did and what motivated regenerating the baseline in the first place. The
    regenerated baseline is worth having either way -- it is the arm the paired
    flip table is computed against -- so the fix costs nothing but the claim.

Usage:
  python analysis/ras_gate.py --gpu 2 --n 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common.gpus import check_gpu  # noqa: E402
from src.common.ras import RAS_NOOP_THRESHOLD, RepetitionAwareSampling  # noqa: E402
from src.common.registry import BY_KEY, DATA_ROOT  # noqa: E402
from src.models.qwen_gen import TalkerGenerateCapture, build_prompt  # noqa: E402


class TracingRAS(RepetitionAwareSampling):
    """RAS that also records the window it saw at every step."""

    def reset(self):
        super().reset()
        self.trace: list[tuple[int, list[int], int]] = []

    def __call__(self, input_ids, scores):
        hist = input_ids[0, -self.window:].tolist() if input_ids.shape[1] else []
        out = super().__call__(input_ids, scores)
        tok = int(torch.argmax(out[0]).item())
        self.trace.append((int(input_ids.shape[1]), hist, tok))
        return out


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--model", default="qwen06b")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/ras_sanity_gate.json"))
    args = ap.parse_args()
    check_gpu(args.gpu)

    spec = BY_KEY[args.model]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    items = [json.loads(l) for l in open(args.stimuli)]
    # Span the ladder rather than sampling the top of the file: the whole point
    # is that a long, loop-prone generation reproduces, not just a short one.
    want = [("word_rep", k) for k in (1, 8, 16, 24, 32)] + \
           [("control_word", k) for k in (8, 24, 32)]
    picked, seen = [], set()
    for fam, k in want:
        for it in items:
            if it["family"] == fam and it["k"] == k and it["item_id"] not in seen:
                picked.append(it)
                seen.add(it["item_id"])
                break
    picked = picked[: args.n]

    from qwen_tts import Qwen3TTSModel
    print(f"[gate] loading {spec.hf_id} on {device}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        spec.hf_id, device_map=device, dtype=torch.bfloat16, attn_implementation="eager")
    talker = tts.model.talker
    prompt_items = build_prompt(tts)

    def run(item, ras_factory):
        cap = TalkerGenerateCapture(talker, ras_factory=ras_factory)
        try:
            torch.manual_seed(args.seed)
            wavs, sr = tts.generate_voice_clone(
                text=item["text"], language="English",
                voice_clone_prompt=prompt_items, max_new_tokens=args.max_new_tokens)
            seq = cap.result.sequences[0].tolist()
            return dict(wav=wavs[0], sr=sr, seq=seq, ras=cap.ras)
        finally:
            talker.generate = cap._orig   # always un-patch, even on a raise

    noop_factory = (lambda gk: RepetitionAwareSampling(
        threshold=RAS_NOOP_THRESHOLD, temperature=gk.get("temperature"),
        top_k=gk.get("top_k"), top_p=gk.get("top_p")))
    live_factory = (lambda gk: TracingRAS(
        temperature=gk.get("temperature"), top_k=gk.get("top_k"),
        top_p=gk.get("top_p")))

    res: dict = {"model": args.model, "seed": args.seed, "device": device,
                 "n_items": len(picked), "max_new_tokens": args.max_new_tokens,
                 "items": []}
    a_ok, b_ok, wiring_ok, drift_same = [], [], [], []
    for it in picked:
        r1 = run(it, None)                 # stock
        r2 = run(it, None)                 # stock again  -> gate A
        r3 = run(it, noop_factory)         # RAS at its no-op setting -> gate B
        r4 = run(it, live_factory)         # RAS live, for wiring + firing

        row = dict(item_id=it["item_id"], family=it["family"], k=it["k"],
                   n_tok_stock=len(r1["seq"]), n_tok_stock2=len(r2["seq"]),
                   n_tok_noop=len(r3["seq"]), n_tok_ras=len(r4["seq"]),
                   seq_sha_stock=sha(np.array(r1["seq"])),
                   seq_sha_stock2=sha(np.array(r2["seq"])),
                   seq_sha_noop=sha(np.array(r3["seq"])),
                   wav_sha_stock=sha(r1["wav"]), wav_sha_noop=sha(r3["wav"]),
                   A_stock_deterministic=bool(r1["seq"] == r2["seq"]),
                   B_seq_identical=bool(r1["seq"] == r3["seq"]),
                   B_wav_identical=bool(sha(r1["wav"]) == sha(r3["wav"])),
                   noop_fired=int(r3["ras"].n_fired),
                   ras_fired=int(r4["ras"].n_fired),
                   ras_fired_alt=int(r4["ras"].n_fired_alt),
                   ras_steps=int(r4["ras"].n_steps))
        row["ras_fire_rate"] = (row["ras_fired"] / row["ras_steps"]
                                if row["ras_steps"] else float("nan"))

        # WIRING: every step's window must equal the emitted prefix.
        tr = r4["ras"].trace
        seq4 = r4["seq"]
        bad = 0
        for t, (width, hist, tok) in enumerate(tr):
            if width != t or hist != seq4[max(0, t - r4["ras"].window):t]:
                bad += 1
            elif t < len(seq4) and tok != seq4[t]:
                bad += 1
        row["wiring_mismatched_steps"] = bad
        row["wiring_ok"] = bool(bad == 0 and len(tr) > 0)

        # LIBRARY DRIFT against the stored panel audio, reported not gating.
        # The comparison must go through the same PCM-16 round trip the stored
        # file went through: `soundfile.write` on a float32 array writes a
        # 16-bit WAV, so hashing the in-memory float array against the file's
        # read-back can never match and would report drift on every item
        # whether or not any exists. The first version of this check did
        # exactly that; it is fixed here rather than quietly dropped, because
        # its (wrong) answer is what motivated regenerating the baseline arm.
        stored = Path(DATA_ROOT) / "audio" / args.model / f"{it['item_id']}_s{args.seed}.wav"
        if stored.exists():
            import io

            import soundfile as sf
            w0, _ = sf.read(str(stored), dtype="float32")
            buf = io.BytesIO()
            sf.write(buf, r1["wav"], r1["sr"], format="WAV")
            buf.seek(0)
            w_rt, _ = sf.read(buf, dtype="float32")
            row["stored_wav_sha"] = sha(w0)
            row["fresh_wav_roundtrip_sha"] = sha(w_rt)
            row["stored_matches_fresh"] = bool(sha(w0) == sha(w_rt))
            row["stored_n_samples"] = int(len(w0))
            row["fresh_n_samples"] = int(len(w_rt))
            drift_same.append(row["stored_matches_fresh"])

        a_ok.append(row["A_stock_deterministic"])
        b_ok.append(row["B_seq_identical"] and row["B_wav_identical"])
        wiring_ok.append(row["wiring_ok"])
        res["items"].append(row)
        print(f"  {row['item_id']:<20s} k={row['k']:<3d} A={row['A_stock_deterministic']} "
              f"B={b_ok[-1]} wiring={row['wiring_ok']} "
              f"ntok={row['n_tok_stock']}/{row['n_tok_ras']} "
              f"fire={100*row['ras_fire_rate']:.1f}% noop_fired={row['noop_fired']}",
              flush=True)

    res["A_pass"] = bool(all(a_ok)) and len(a_ok) > 0
    res["B_pass"] = bool(all(b_ok)) and len(b_ok) > 0
    res["wiring_pass"] = bool(all(wiring_ok))
    res["noop_never_fired"] = bool(all(r["noop_fired"] == 0 for r in res["items"]))
    res["stored_matches_fresh_all"] = bool(all(drift_same)) if drift_same else None
    res["mean_fire_rate"] = float(np.mean([r["ras_fire_rate"] for r in res["items"]]))

    if not res["A_pass"]:
        res["gate"] = "VOID"
        res["reason"] = ("the stock path is not bitwise reproducible in this environment, "
                         "so no-op equivalence cannot be demanded; the arm is "
                         "inconclusive-by-construction")
    elif not res["B_pass"]:
        res["gate"] = "UNREPORTABLE"
        res["reason"] = ("RAS at its no-op threshold does not reproduce the stock path; "
                         "the plumbing changes generation even when the rule never fires, "
                         "so nothing downstream measures RAS")
    else:
        res["gate"] = "PASS"
        res["reason"] = ("RAS at its no-op threshold reproduces the stock decoder bitwise "
                         "(token sequences and wav bytes), on a path shown reproducible")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nGATE: {res['gate']} -- {res['reason']}")
    print(f"  A determinism {res['A_pass']}  B no-op equivalence {res['B_pass']}  "
          f"wiring {res['wiring_pass']}  no-op never fired {res['noop_never_fired']}")
    print(f"  mean RAS fire rate {100*res['mean_fire_rate']:.1f}% of decode steps")
    print(f"  stored panel audio reproduced today: {res['stored_matches_fresh_all']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
