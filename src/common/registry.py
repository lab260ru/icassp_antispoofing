"""Model panel definition — single source of truth for every downstream script.

`token_rate_hz` is the *nominal* acoustic-token rate of each model's codec. It
converts an ASR word timestamp into a decoder step index, which is how
repetition boundaries are located in state space. Nominal values are verified
against measured `n_tokens / duration` at smoke-test time; the measured value is
what analysis actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    key: str                 # short name used in every path and table
    hf_id: str
    family: str              # llasa | xtts | qwen
    params: str
    env: str                 # conda env that can run it
    token_rate_hz: float     # nominal acoustic tokens per second
    sample_rate: int
    probe_layers: str        # "all" | "every2" | "every3" | "every4"
    instrumented: bool       # can we capture hidden states + text attention?
    notes: str = ""
    extra: dict = field(default_factory=dict)


PANEL = [
    ModelSpec("llasa1b", "HKUSTAudio/Llasa-1B", "llasa", "1B", "base",
              50.0, 16000, "all", True,
              "Llama-3.2-1B backbone over X-codec2 tokens"),
    ModelSpec("llasa3b", "HKUSTAudio/Llasa-3B", "llasa", "3B", "base",
              50.0, 16000, "every2", True,
              "Llama-3.2-3B backbone; same tokenizer/codec as 1B"),
    ModelSpec("llasa8b", "HKUSTAudio/Llasa-8B", "llasa", "8B", "base",
              50.0, 16000, "every4", True,
              "Llama-3.1-8B backbone; top of the within-family scale ladder"),
    ModelSpec("xtts2", "coqui/XTTS-v2", "xtts", "0.4B", "coqui",
              21.53, 24000, "every3", True,
              "GPT2-style AR latent decoder + HiFiGAN; cross-family control"),
    ModelSpec("qwen06b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen", "0.6B", "qwen",
              12.5, 24000, "every3", True,
              "dual-track multi-codebook LM; instrumentation TBD"),
    ModelSpec("qwen17b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen", "1.7B", "qwen",
              12.5, 24000, "every3", True,
              "same family as 0.6B; mini scale ladder"),
    # Ablation. XTTS-v2 ships repetition_penalty=5.0 on acoustic tokens, which
    # acts directly against the behaviour under study; run with it disabled to
    # show the dissociation is not an artefact of that decoding-time
    # intervention. Separate key so it never mixes with the main run's outputs.
    ModelSpec("xtts2norp", "coqui/XTTS-v2", "xtts", "0.4B", "coqui",
              21.53, 24000, "every3", True,
              "XTTS-v2 with repetition_penalty=1.0 (ablation)"),
    # Mitigation sweep. The repetition penalty is the standard decoding-rule
    # remedy for this failure, and the paper argues such remedies treat a
    # symptom. Sweeping it on one model turns that from an assertion into a
    # measurement: if the count deficit is roughly flat in the penalty, the
    # lever is not acting on the quantity that governs the horizon.
    ModelSpec("xtts2rp2", "coqui/XTTS-v2", "xtts", "0.4B", "coqui",
              21.53, 24000, "every3", False, "XTTS-v2, repetition_penalty=2.0"),
    ModelSpec("xtts2rp3", "coqui/XTTS-v2", "xtts", "0.4B", "coqui",
              21.53, 24000, "every3", False, "XTTS-v2, repetition_penalty=3.0"),
    ModelSpec("xtts2rp8", "coqui/XTTS-v2", "xtts", "0.4B", "coqui",
              21.53, 24000, "every3", False, "XTTS-v2, repetition_penalty=8.0"),
]

BY_KEY = {m.key: m for m in PANEL}

ASR_MODEL = "openai/whisper-large-v3"
CODEC_MODEL = "HKUSTAudio/xcodec2"

# All heavy artefacts live off-repo.
DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"


def probe_layer_indices(n_layers: int, mode: str) -> list[int]:
    """Which decoder layers to record. Always includes the last layer, which is
    the one the readout (stop head / codec head) actually consumes."""
    if mode == "all":
        idx = list(range(n_layers))
    else:
        step = int(mode.replace("every", ""))
        idx = list(range(0, n_layers, step))
    if n_layers - 1 not in idx:
        idx.append(n_layers - 1)
    return sorted(idx)
