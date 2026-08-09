# Telegram delivery playbook

Queued milestone text lives in `pending-notifications.md`. The delivery script
is deliberately non-spamming: by default it only lists sections and it sends
exactly one message only when `--send-latest` or `--send-heading` is supplied.
It never reads credentials from a file and never prints them.

List pending messages without network access:

```bash
cd /home/kirill/icassp_antispoofing
PYTHONPATH=. python3 scripts/send_pending_telegram.py --list
```

After the environment has both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`,
send the newest update:

```bash
PYTHONPATH=. python3 scripts/send_pending_telegram.py --send-latest
```

Or send one exact section shown by `--list`:

```bash
PYTHONPATH=. python3 scripts/send_pending_telegram.py \
  --send-heading "2026-08-09 — author-block requirement discovered"
```

The sender returns only a redacted receipt (heading, text length, message ID).
Do not paste token values into this repository, terminal history, or chat.
