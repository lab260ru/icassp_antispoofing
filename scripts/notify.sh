#!/usr/bin/env bash
# Telegram milestone notifier.
#   scripts/notify.sh "message text"
#   scripts/notify.sh --doc path/to/file.pdf "caption"
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
  echo "[notify] no telegram creds; message was: $*" >&2
  exit 0
fi

API="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}"

if [ "${1:-}" = "--doc" ]; then
  DOC="$2"; shift 2
  CAPTION="${*:-}"
  curl -s -o /dev/null -w "[notify] doc http=%{http_code}\n" \
    -F "chat_id=${TELEGRAM_CHAT_ID}" \
    -F "document=@${DOC}" \
    -F "caption=${CAPTION}" \
    "${API}/sendDocument"
else
  MSG="$*"
  curl -s -o /dev/null -w "[notify] msg http=%{http_code}\n" \
    -X POST "${API}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${MSG}" \
    -d "disable_web_page_preview=true"
fi
