# H2 fixed ASR/WER runtime

`src/h2_asr_wer.py` provides a lazy, OpenAI Whisper transcript wrapper and
deterministic normalized word-error calculation for the H2 quality gate. It
does not load a model at import time, score a detector, select pairs, or decide
that a quality gate has passed.

## Pinned provenance

| Item | Value |
|---|---|
| Runtime package | `openai-whisper==20250625` |
| Model | `small.en` |
| Checkpoint root | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/models/asr/openai-whisper/20250625` |
| Checkpoint | `small.en.pt` (461.2 MiB) |
| SHA-256 | `f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872` |
| Device contract | CUDA with FP16, logical `cuda:0` |
| Decode contract | English transcription; temperature 0; beam 5; best-of 5; no previous-text conditioning; no word timestamps |

The wrapper checks the installed distribution version and computes the local
checkpoint SHA-256 before its first model load. A missing/mismatched checkpoint
or package is an error; it does not download a replacement or fall back to CPU.

## GPU mapping caveat

Launch the ASR process with physical GPU 3 exposed as its sole visible device:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/h2_asr_wer.py --print-provenance
```

Inside that process, physical GPU 3 is **logical `cuda:0`**. The pinned utility
therefore uses `cuda:0`; using `cuda:3` after setting `CUDA_VISIBLE_DEVICES=3`
would refer to a non-visible logical index. Keep GPU 3 reserved for waveform
quality/ASR work until its pair manifest is frozen.

The eventual H2 runner should store raw original/transformed transcripts,
normalized token sequences, error count, WER, model provenance, and GPU mapping
per pair. `normalized_word_error` rejects an empty original transcript instead
of treating it as a zero-WER match. The protocol's WER <= 5% threshold remains
unevaluated until it is run jointly with the other quality gates.

## Focused unit coverage

`tests/test_h2_asr_wer.py` exercises normalization, insertion/deletion/
substitution WER, and empty-reference rejection only. It does not load Whisper
or transcribe audio.
