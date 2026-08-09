"""Pinned, lazy OpenAI Whisper transcription and deterministic pairwise WER.

No model is loaded at import time. The H2 runner may use this module only after
the waveform-quality manifest is fixed; this module neither selects pairs nor
declares any H2 quality gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


OPENAI_WHISPER_VERSION = "20250625"
WHISPER_MODEL_NAME = "small.en"
WHISPER_MODEL_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/models/asr/openai-whisper/20250625")
WHISPER_MODEL_SHA256 = "f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872"
_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class WhisperQualityGateConfig:
    """Immutable OpenAI Whisper inference contract for the H2 WER gate."""

    model_name: str = WHISPER_MODEL_NAME
    model_root: Path = WHISPER_MODEL_ROOT
    model_sha256: str = WHISPER_MODEL_SHA256
    package_version: str = OPENAI_WHISPER_VERSION
    device: str = "cuda:0"
    fp16: bool = True
    language: str = "en"
    task: str = "transcribe"
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    condition_on_previous_text: bool = False
    word_timestamps: bool = False
    verbose: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.model_root / f"{self.model_name}.pt"

    def provenance(self) -> dict[str, str | int | float | bool]:
        """Return serializable runtime/model contract without importing Whisper."""
        result = asdict(self)
        result["model_root"] = str(self.model_root)
        result["checkpoint_path"] = str(self.checkpoint_path)
        return result


@dataclass(frozen=True)
class WordErrorResult:
    """Normalized token sequences and their Levenshtein word-error result."""

    reference_words: tuple[str, ...]
    hypothesis_words: tuple[str, ...]
    errors: int
    word_error_rate: float


def normalize_words(text: str) -> tuple[str, ...]:
    """Unicode-normalize, case-fold, and tokenize text for deterministic WER."""
    if not isinstance(text, str):
        raise TypeError(f"Transcript must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return tuple(_WORD_PATTERN.findall(normalized))


def normalized_word_error(reference: str, hypothesis: str) -> WordErrorResult:
    """Compute WER after fixed normalization; empty references are invalid gates."""
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        raise ValueError("Original transcript has no normalized words; WER is undefined")
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[hypothesis_index - 1] + (reference_word != hypothesis_word)
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    errors = int(previous[-1])
    return WordErrorResult(
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
        errors=errors,
        word_error_rate=float(errors / len(reference_words)),
    )


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint in bounded chunks, without loading it into a model."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LazyWhisperTranscriber:
    """Load the pinned Whisper model only when transcript generation is requested."""

    def __init__(self, config: WhisperQualityGateConfig = WhisperQualityGateConfig()) -> None:
        self.config = config
        self._model: Any | None = None
        self._checkpoint_verified = False

    def verify_runtime(self) -> None:
        """Verify package version, CUDA/fp16 contract, and checkpoint identity."""
        try:
            installed_version = metadata.version("openai-whisper")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError("Missing required package openai-whisper") from error
        if installed_version != self.config.package_version:
            raise RuntimeError(
                f"openai-whisper version mismatch: expected {self.config.package_version}, got {installed_version}"
            )
        if self.config.fp16 and not self.config.device.startswith("cuda"):
            raise RuntimeError("fp16 H2 transcription requires a CUDA device")
        import torch

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is unavailable for H2 transcription")
        checkpoint = self.config.checkpoint_path
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pinned Whisper checkpoint is missing: {checkpoint}")
        observed = checkpoint_sha256(checkpoint)
        if observed != self.config.model_sha256:
            raise RuntimeError(f"Whisper checkpoint hash mismatch: expected {self.config.model_sha256}, got {observed}")
        self._checkpoint_verified = True

    def _load(self) -> Any:
        if self._model is None:
            if not self._checkpoint_verified:
                self.verify_runtime()
            import whisper

            self._model = whisper.load_model(
                self.config.model_name,
                device=self.config.device,
                download_root=str(self.config.model_root),
                in_memory=False,
            )
        return self._model

    def transcribe(self, audio: Any) -> str:
        """Transcribe an audio path or waveform under the fixed H2 decode contract."""
        output = self._load().transcribe(
            audio,
            verbose=self.config.verbose,
            language=self.config.language,
            task=self.config.task,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            fp16=self.config.fp16,
            condition_on_previous_text=self.config.condition_on_previous_text,
            word_timestamps=self.config.word_timestamps,
        )
        text = output.get("text")
        if not isinstance(text, str):
            raise RuntimeError("Whisper transcribe result did not contain text")
        return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinned OpenAI Whisper H2 transcript/WER utility")
    parser.add_argument("--audio", help="Optional audio file to transcribe under the pinned contract")
    parser.add_argument("--reference", help="Optional original transcript; prints normalized pairwise WER")
    parser.add_argument("--print-provenance", action="store_true", help="Print the pinned configuration without loading Whisper")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = WhisperQualityGateConfig()
    if args.print_provenance or not args.audio:
        print(json.dumps(config.provenance(), indent=2, sort_keys=True))
        if not args.audio:
            return
    transcript = LazyWhisperTranscriber(config).transcribe(args.audio)
    output: dict[str, Any] = {"transcript": transcript}
    if args.reference is not None:
        output["wer"] = asdict(normalized_word_error(args.reference, transcript))
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
