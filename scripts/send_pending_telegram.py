#!/usr/bin/env python3
"""List queued Telegram milestones or explicitly send exactly one of them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.telegram_delivery import parse_pending_messages, select_message, send_telegram_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pending-file",
        type=Path,
        default=REPO_ROOT / "to_human" / "pending-notifications.md",
        help="Queued Markdown notification file (default: repository pending-notifications.md).",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="List queued headings only (the default, no network call).")
    action.add_argument(
        "--send-latest",
        action="store_true",
        help="Send only the final queued section using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
    )
    action.add_argument(
        "--send-heading",
        metavar="HEADING",
        help="Send only one exact queued heading using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
    )
    args = parser.parse_args()
    messages = parse_pending_messages(args.pending_file.read_text(encoding="utf-8"))
    if args.list or (not args.send_latest and args.send_heading is None):
        for index, message in enumerate(messages, start=1):
            print(f"{index}: {message.heading} ({len(message.text)} chars)")
        return

    message = select_message(messages, args.send_heading)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise SystemExit(
            "Telegram credentials are unavailable: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment."
        )
    receipt = send_telegram_message(token=token, chat_id=chat_id, message=message)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
