#!/usr/bin/env python3
"""Launch `xtts_gen.py` under a transformers that dropped `isin_mps_friendly`.

WHY THIS EXISTS
---------------
The `coqui` env's transformers was upgraded to 5.x on 2026-08-12, after the
period ladder's XTTS-v2 rows were generated. Coqui TTS 0.22 imports
`transformers.pytorch_utils.isin_mps_friendly`, a helper that 5.x removed, so
`xtts_gen.py` now dies on import before it loads a single weight.

This restores the symbol *in the importing process only* -- it does not touch
site-packages, so a concurrent session using the same env sees nothing change,
and it disappears the moment the process exits. The helper's whole content is a
branch on Apple Silicon: on MPS it emulates `torch.isin`, which MPS did not
implement, and everywhere else it calls `torch.isin`. Every generation here runs
on CUDA, so the restored function is `torch.isin` verbatim and is not an
approximation of the original on this hardware.

WHAT IT DOES NOT FIX, AND WHY THAT MATTERS TO THE RESULT
--------------------------------------------------------
It does not put the environment back. transformers 4.x -> 5.x is a major
version bump, and the GPT-2 stack XTTS-v2 samples with lives inside it. XTTS-v2
rows generated through this shim are therefore NOT strictly commensurable with
the published XTTS-v2 rows in `data/results/behavioural_period.csv`, which
predate the upgrade -- which is exactly the "difference between two runs made
months apart" that `make_stimuli_period.py` exists to avoid. Any analysis that
mixes them must say so, and must report the verdict with XTTS-v2 excluded as
well as included. `analysis/period_odd.py` does both.

Usage (in the coqui env), identical to xtts_gen.py:
  python src/models/xtts_gen_compat.py --model xtts2 --gpu 3 --seeds 0 1 2 \
      --stimuli data/stimuli/stimuli_period.jsonl
"""
from __future__ import annotations

import runpy
from pathlib import Path

import torch
import transformers.pytorch_utils as _pu

if not hasattr(_pu, "isin_mps_friendly"):
    def isin_mps_friendly(elements, test_elements):  # noqa: ANN001, ANN201
        """`torch.isin`, with the MPS workaround transformers 4.x carried."""
        if elements.device.type == "mps":
            if not torch.is_tensor(test_elements):
                test_elements = torch.tensor(test_elements, device=elements.device)
            test_elements = test_elements.to(elements.device)
            return (elements.tile(test_elements.shape[0], 1)
                    .eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze())
        return torch.isin(elements, test_elements)

    _pu.isin_mps_friendly = isin_mps_friendly

runpy.run_path(str(Path(__file__).with_name("xtts_gen.py")), run_name="__main__")
