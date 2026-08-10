#!/usr/bin/env python3
"""Freeze the validated source-only H9-PCR checkpoint handoff ledger."""

from src.h9_pcr_checkpoint_ledger import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
