"""Safe, explicit delivery of one queued research update through Telegram.

The module has no repository-side credential store. It accepts a bot token and
chat ID only from its caller, sends one selected message at most, and returns a
minimal delivery receipt that does not expose either secret.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


MAX_TELEGRAM_TEXT_CHARS = 4096
_HEADING = re.compile(r"^## (?P<heading>.+?)\s*$", flags=re.MULTILINE)


@dataclass(frozen=True)
class PendingTelegramMessage:
    """A single level-two heading/body pair from the pending-notification file."""

    heading: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.heading}\n\n{self.body}".strip()


class _Response(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool | None: ...


Urlopen = Callable[..., _Response]


def parse_pending_messages(markdown: str) -> tuple[PendingTelegramMessage, ...]:
    """Parse nonempty second-level Markdown sections in stable file order."""
    matches = list(_HEADING.finditer(markdown))
    messages: list[PendingTelegramMessage] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[match.end() : end].strip()
        heading = match.group("heading").strip()
        if not body:
            raise ValueError(f"Queued Telegram section has no body: {heading!r}")
        messages.append(PendingTelegramMessage(heading=heading, body=body))
    if not messages:
        raise ValueError("No level-two queued Telegram sections were found")
    return tuple(messages)


def select_message(messages: tuple[PendingTelegramMessage, ...], heading: str | None = None) -> PendingTelegramMessage:
    """Return the latest message or one exact heading, refusing ambiguous names."""
    if heading is None:
        return messages[-1]
    matches = [message for message in messages if message.heading == heading]
    if len(matches) != 1:
        raise ValueError(f"Queued Telegram heading must match exactly once: {heading!r}")
    return matches[0]


def send_telegram_message(
    *,
    token: str,
    chat_id: str,
    message: PendingTelegramMessage,
    opener: Urlopen = urlopen,
) -> dict[str, object]:
    """Send exactly one bounded plain-text message and return a redacted receipt."""
    if not token.strip() or not chat_id.strip():
        raise ValueError("Both a Telegram bot token and chat ID are required")
    text = message.text
    if len(text) > MAX_TELEGRAM_TEXT_CHARS:
        raise ValueError(f"Queued Telegram text is {len(text)} characters; maximum is {MAX_TELEGRAM_TEXT_CHARS}")
    endpoint = "https://api.telegram.org/bot" + quote(token, safe=":") + "/sendMessage"
    payload = urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = Request(endpoint, data=payload, method="POST")
    try:
        with opener(request, timeout=20) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - normalize network/API failure without credentials
        raise RuntimeError("Telegram delivery request failed") from error
    if not isinstance(decoded, dict) or decoded.get("ok") is not True:
        raise RuntimeError("Telegram delivery API returned a non-success response")
    result = decoded.get("result")
    message_id = result.get("message_id") if isinstance(result, dict) else None
    return {
        "heading": message.heading,
        "text_length": len(text),
        "message_id": message_id,
        "delivered": True,
    }
