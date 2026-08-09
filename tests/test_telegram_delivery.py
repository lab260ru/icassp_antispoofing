"""Pure tests for queued Telegram parsing and explicit one-message delivery."""

from __future__ import annotations

from urllib.parse import parse_qs

import pytest

from src.telegram_delivery import MAX_TELEGRAM_TEXT_CHARS, parse_pending_messages, select_message, send_telegram_message


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


def test_parse_and_select_preserve_order_and_exact_heading() -> None:
    messages = parse_pending_messages("# Queue\n\n## First\n\nBody one\n\n## Second\n\nBody two\n")

    assert [message.heading for message in messages] == ["First", "Second"]
    assert select_message(messages).text == "Second\n\nBody two"
    assert select_message(messages, "First").text == "First\n\nBody one"
    with pytest.raises(ValueError, match="exactly once"):
        select_message(messages, "missing")


def test_parser_rejects_empty_sections() -> None:
    with pytest.raises(ValueError, match="no body"):
        parse_pending_messages("## Empty\n")


def test_delivery_posts_one_redacted_message() -> None:
    message = parse_pending_messages("## Update\n\nCompleted safely.\n")[0]
    captured: dict[str, object] = {}

    def opener(request: object, timeout: int) -> _Response:
        captured["url"] = getattr(request, "full_url")
        captured["data"] = getattr(request, "data")
        captured["timeout"] = timeout
        return _Response(b'{"ok": true, "result": {"message_id": 17}}')

    receipt = send_telegram_message(token="123:secret", chat_id="42", message=message, opener=opener)

    assert receipt == {"heading": "Update", "text_length": len(message.text), "message_id": 17, "delivered": True}
    assert captured["timeout"] == 20
    assert "123:secret" in str(captured["url"])
    form = parse_qs(bytes(captured["data"]).decode("utf-8"))
    assert form["chat_id"] == ["42"]
    assert form["text"] == [message.text]


def test_delivery_refuses_oversized_or_unsuccessful_messages() -> None:
    oversized = parse_pending_messages("## Update\n\n" + "x" * MAX_TELEGRAM_TEXT_CHARS)[0]
    with pytest.raises(ValueError, match="maximum"):
        send_telegram_message(token="t", chat_id="c", message=oversized)

    message = parse_pending_messages("## Update\n\nNormal\n")[0]
    with pytest.raises(RuntimeError, match="non-success"):
        send_telegram_message(token="t", chat_id="c", message=message, opener=lambda *_args, **_kwargs: _Response(b'{"ok": false}'))
