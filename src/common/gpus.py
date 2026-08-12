#!/usr/bin/env python3
"""Which GPUs this project is allowed to use.

GPUs 0 and 1 on this host are not ours. Everything here runs on 2 and 3 only.

This is enforced rather than documented, because a default buried in a dozen
argparse calls is a constraint nobody notices until it is violated: every
generation and transcription entry point calls `check_gpu()` on its `--gpu`
argument and exits with a clear message rather than quietly grabbing a
neighbour's card. `DEFAULT_GPU` is what those entry points default to, so a
forgotten flag lands somewhere allowed.

If the allocation changes, change `ALLOWED` here and nowhere else.
"""
from __future__ import annotations

import os

ALLOWED: tuple[int, ...] = (2, 3)
DEFAULT_GPU: int = ALLOWED[0]


def check_gpu(gpu: int) -> int:
    """Return `gpu` if this project may use it; otherwise exit with a message.

    An override exists for the case where the allocation genuinely changes
    mid-run, but it has to be set deliberately:

        ICASSP_ALLOW_ANY_GPU=1 python src/models/llasa_gen.py --gpu 0
    """
    if os.environ.get("ICASSP_ALLOW_ANY_GPU") == "1":
        return int(gpu)
    if int(gpu) not in ALLOWED:
        raise SystemExit(
            f"[gpu] refusing to use cuda:{gpu}. This project is restricted to "
            f"GPUs {', '.join(map(str, ALLOWED))} (see src/common/gpus.py). "
            f"Pass --gpu {DEFAULT_GPU}, or set ICASSP_ALLOW_ANY_GPU=1 if the "
            f"allocation has actually changed.")
    return int(gpu)
