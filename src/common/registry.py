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
    # The alignment-supervised AR system. Every other AR member of the panel
    # learns the text->speech alignment implicitly from the LM objective;
    # CosyVoice 2's speech tokens come from a *supervised* ASR encoder and its
    # flow-matching decoder is conditioned on that aligned stream. It is
    # therefore the checkpoint that can separate "AR decoding fails to count"
    # from "unsupervised alignment fails to count": a gap like the panel's says
    # alignment supervision does not fix it, a gap near zero says it does.
    ModelSpec("cosyvoice2", "FunAudioLLM/CosyVoice2-0.5B", "cosyvoice", "0.5B", "cosyvoice",
              25.0, 24000, "every3", True,
              "Qwen2.5-0.5B LM over 25 Hz FSQ speech tokens + flow-matching decoder; "
              "alignment-supervised AR"),
    # Cross-lingual arm. Same XTTS-v2 weights, same speaker reference, same
    # decoding config as `xtts2`; only the stimulus language and the judge
    # change. It is NOT a panel member and is listed in
    # `population.ABLATIONS` for the same reason `cosyvoice2` is: the pipeline
    # scores every model with transcripts into a shared table, and folding a
    # Spanish arm into an English panel would move every macro in the paper.
    # `env` is `coqui_es`, a clone of `coqui` pinned to transformers 4.57.1 --
    # the live `coqui` env has since been upgraded to transformers 5.x, which
    # coqui-tts 0.27.5 cannot import at all.
    ModelSpec("xtts2es", "coqui/XTTS-v2", "xtts", "0.4B", "coqui_es",
              21.53, 24000, "every3", False,
              "XTTS-v2 rendering Spanish stimuli; cross-lingual arm, non-panel"),
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
    # The same sweep on a second architecture, because two review rounds
    # objected that one model cannot support a claim about the field's standard
    # mitigation. Qwen3-TTS-0.6B shows the deficit and ships a penalty of 1.05,
    # far below XTTS-v2's 5.0, so this covers a different part of the range
    # rather than repeating the same one.
    ModelSpec("qwen06brp10", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen", "0.6B", "qwen",
              12.5, 24000, "every3", False, "Qwen3-TTS-0.6B, repetition_penalty=1.0"),
    ModelSpec("qwen06brp15", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen", "0.6B", "qwen",
              12.5, 24000, "every3", False, "Qwen3-TTS-0.6B, repetition_penalty=1.5"),
    ModelSpec("qwen06brp30", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen", "0.6B", "qwen",
              12.5, 24000, "every3", False, "Qwen3-TTS-0.6B, repetition_penalty=3.0"),
    # Greedy decoding. The theorem bounds a readout, so no property of the
    # sampling rule enters the proof -- it covers greedy, sampled and beam
    # alike. Whether the *deficit* survives greedy decoding is a separate,
    # empirical question, and round 14 was right that we had never asked it:
    # every panel run samples. Greedy is deterministic, so one seed is the
    # whole experiment.
    ModelSpec("qwen06bgreedy", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen", "0.6B", "qwen",
              12.5, 24000, "every3", False, "Qwen3-TTS-0.6B, greedy (do_sample=False)"),
    # Non-autoregressive baselines and the duration intervention on one of them.
    # Contrasts, never panel members.
    ModelSpec("vits", "facebook/mms-tts-eng", "vits", "0.04B", "base",
              0.0, 16000, "none", False, "non-AR baseline (2021)"),
    ModelSpec("f5tts", "SWivid/F5-TTS", "f5", "0.34B", "coqui",
              0.0, 24000, "none", False, "non-AR baseline (2024, flow matching)"),
    ModelSpec("f5fix", "SWivid/F5-TTS", "f5", "0.34B", "coqui",
              0.0, 24000, "none", False,
              "F5-TTS with the total duration supplied rather than estimated"),
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
