"""One tokenizer interface across the panel.

Boundary detection needs character offsets so a word can be mapped to the text
column its first token occupies. Llasa and Qwen expose HF fast tokenizers;
XTTS-v2 ships a raw `tokenizers` JSON (`vocab.json`) with no HF wrapper. This
module hides that difference behind `encode_offsets(text) -> list[(start, end)]`.

Without it, XTTS would silently fall back to uniform boundary placement, and its
contraction estimate would not be comparable to the rest of the panel.
"""
from __future__ import annotations

import glob
import os

HF_HOME = os.environ.get("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")


def _xtts_prepare(text: str) -> str:
    """Reproduce `VoiceBpeTokenizer.encode`'s string transform.

    XTTS lowercases, prefixes a language tag, and --- the part that matters ---
    replaces every space with a literal `[SPACE]` token. Without replicating
    that, our column indices are off by the number of spaces and boundary
    detection silently points at the wrong text positions.

    Abbreviation and number expansion are also part of the real preprocessor but
    cannot fire on the instrumented families, which contain neither.
    """
    return "[en]" + text.strip().lower().replace(" ", "[SPACE]")


class OffsetTokenizer:
    """Wraps either backend; `prepare` + `offsets` is all the caller needs."""

    def __init__(self, backend, kind: str, prepare=None):
        self._b = backend
        self.kind = kind
        self._prepare = prepare or (lambda t: t)

    def prepare(self, text: str) -> str:
        """The exact string the model tokenizes. Search for units in *this*."""
        return self._prepare(text)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if self.kind == "hf":
            return list(self._b(text, add_special_tokens=False,
                                return_offsets_mapping=True)["offset_mapping"])
        return list(self._b.encode(text, add_special_tokens=False).offsets)

    def n_tokens(self, text: str) -> int:
        return len(self.offsets(text))


def _xtts_vocab_path() -> str | None:
    pats = [
        os.path.join(HF_HOME, "hub", "models--coqui--XTTS-v2", "snapshots", "*", "vocab.json"),
        os.path.expanduser("~/.local/share/tts/*XTTS*v2*/vocab.json"),
    ]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def load(spec) -> OffsetTokenizer | None:
    """Return an offset tokenizer for a ModelSpec, or None if unavailable."""
    if spec.family == "xtts":
        path = _xtts_vocab_path()
        if path is None:
            return None
        try:
            from tokenizers import Tokenizer
            return OffsetTokenizer(Tokenizer.from_file(path), "tok", _xtts_prepare)
        except Exception:  # noqa: BLE001
            return None
    try:
        from transformers import AutoTokenizer
        return OffsetTokenizer(AutoTokenizer.from_pretrained(spec.hf_id), "hf")
    except Exception:  # noqa: BLE001
        return None
