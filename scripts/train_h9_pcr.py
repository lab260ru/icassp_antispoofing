#!/usr/bin/env python3
"""Run one locked source-only H9-PCR condition/seed fit.

This thin wrapper intentionally has no target-dataset options and no
pretrained-checkpoint option.  See ``src.h9_pcr_training`` for the source
manifest contract and fresh-init BF16 implementation.
"""

from src.h9_pcr_training import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
