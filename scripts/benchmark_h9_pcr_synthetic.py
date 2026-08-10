#!/usr/bin/env python3
"""Measure a fresh-init, source-free BF16 H9 Res2TCNGuard training step.

This is an implementation throughput probe only: it generates random
fixed-length tensors, reads no corpus manifest/audio/labels, and never loads a
pretrained checkpoint.  The selected batch size still has to be frozen in the
H9 source-run configuration before ODSS materialization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from src.h9_pcr_training import WINDOW_SAMPLES, build_fresh_res2tcn_guard, seed_to_device


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--res2-bundle", required=True, type=Path)
parser.add_argument("--seed", type=int, default=9101)
parser.add_argument("--device", default="auto")
parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--warmup-steps", type=int, default=1)
parser.add_argument("--timed-steps", type=int, default=2)
parser.add_argument(
    "--method",
    choices=("B1", "P"),
    default="P",
    help="P is the common worst-case memory recipe (BCE plus a half-batch rank term).",
)
args = parser.parse_args()

if args.warmup_steps < 0 or args.timed_steps <= 0 or any(batch < 1 for batch in args.batches):
    raise SystemExit("Batch sizes must be positive, warmup nonnegative, and timed steps positive")
if args.method == "P" and any(batch < 2 or batch % 2 for batch in args.batches):
    raise SystemExit("P synthetic training probe requires even batch sizes of at least two")
device = torch.device(seed_to_device(args.seed, args.device))
if not torch.cuda.is_available():
    raise SystemExit("Synthetic H9 throughput probe requires CUDA BF16")
torch.cuda.set_device(device)


def one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_size: int) -> None:
    waveforms = torch.randn(batch_size, WINDOW_SAMPLES, device=device, dtype=torch.float32)
    labels = torch.arange(batch_size, device=device, dtype=torch.long).remainder(2)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(waveforms)
        logits = outputs[1] if isinstance(outputs, tuple) else outputs
        loss = functional.cross_entropy(logits, labels)
        if args.method == "P":
            rank_count = batch_size // 2
            spoof = torch.randn(rank_count, WINDOW_SAMPLES, device=device, dtype=torch.float32)
            bona = torch.randn(rank_count, WINDOW_SAMPLES, device=device, dtype=torch.float32)
            spoof_output = model(spoof)
            bona_output = model(bona)
            spoof_logits = (spoof_output[1] if isinstance(spoof_output, tuple) else spoof_output)[:, 1]
            bona_logits = (bona_output[1] if isinstance(bona_output, tuple) else bona_output)[:, 1]
            loss = loss + 0.1 * torch.relu(1.0 - (spoof_logits - bona_logits)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Synthetic H9 BF16 benchmark produced a non-finite loss")
    loss.backward()
    optimizer.step()


results: list[dict[str, object]] = []
for batch_size in args.batches:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, provenance = build_fresh_res2tcn_guard(args.res2_bundle, seed=args.seed, device=device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        for _ in range(args.warmup_steps):
            one_step(model, optimizer, batch_size)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.timed_steps):
            one_step(model, optimizer, batch_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = {
            "batch_size": batch_size,
            "status": "ok",
            "steps_per_second": args.timed_steps / elapsed,
            "examples_per_second": batch_size * args.timed_steps / elapsed,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "precision": "cuda_bfloat16_autocast",
            "method_memory_recipe": args.method,
            "initialization": provenance["initialization"],
            "checkpoint_loaded": provenance["checkpoint_loaded"],
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {"batch_size": batch_size, "status": "oom"}
        results.append(result)
        print(json.dumps(result, sort_keys=True))
        break

safe = [row for row in results if row["status"] == "ok"]
if not safe:
    raise SystemExit("No safe synthetic H9 BF16 training batch was found")
best = max(safe, key=lambda row: float(row["examples_per_second"]))
print(json.dumps({"recommendation": "freeze_after_protocol_update", "fastest_safe": best, "all_results": results}, sort_keys=True))
