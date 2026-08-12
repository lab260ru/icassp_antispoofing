#!/usr/bin/env bash
# Environment bring-up for the AR-TTS counting-collapse study.
# Each block is idempotent and logs to logs/env_<name>.log.
#
# Env map
#   base     (py3.13, torch 2.11+cu130) : Llasa LMs, Whisper ASR, all analysis  [additive install only]
#   qwen     (py3.12)                   : Qwen3-TTS via the `qwen-tts` package
#   coqui    (py3.12)                   : XTTS-v2 via `coqui-tts` (idiap fork)
#   xcodec2  (py3.11, torch 2.5 pinned) : offline vocoding of Llasa speech tokens
#   cosyvoice(py3.10)                   : CosyVoice 2 from a source checkout
#
# CRITICAL: install cu130 torch BEFORE the TTS package in each env, else pip
# silently pulls a default CUDA-12 wheel and the driver/toolkit combo breaks.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"
CONDA_BASE="$(conda info --base)"
. "$CONDA_BASE/etc/profile.d/conda.sh"

CU130="--index-url https://download.pytorch.org/whl/cu130"

setup_base() {
  conda activate base
  pip install -q "transformers==4.57.3" accelerate num2words jiwer 2>&1
  # analysis/hierarchical.py: mixed-effects and cluster-robust models, and NUTS
  # for the hierarchical posterior. numpyro pulls the CPU jax wheel and the
  # script pins JAX_PLATFORMS=cpu, so neither touches a GPU.
  pip install -q statsmodels numpyro 2>&1
}

setup_qwen() {
  conda env list | grep -q "^qwen " || conda create -y -n qwen python=3.12
  conda activate qwen
  pip install -q torch torchaudio $CU130
  pip install -q qwen-tts
}

setup_coqui() {
  conda env list | grep -q "^coqui " || conda create -y -n coqui python=3.12
  conda activate coqui
  pip install -q torch torchaudio $CU130
  pip install -q coqui-tts
}

setup_xcodec2() {
  conda env list | grep -q "^xcodec2 " || conda create -y -n xcodec2 python=3.11
  conda activate xcodec2
  pip install -q xcodec2==0.1.5
  # xcodec2 0.1.5 pins torch 2.5 but leaves transformers/torchao unpinned, and
  # both have since moved past it:
  #   transformers 5.x  -> `Could not import module 'PreTrainedModel'`
  #   torchao >= 0.7    -> `module 'torch' has no attribute 'int1'`
  # These three pins are what actually make `decode_code` importable.
  pip install -q "transformers==4.46.3" "tokenizers<0.21" "torchao==0.6.1" soundfile
}

setup_cosyvoice() {
  # CosyVoice 2 is not a pip package: it is a source tree plus a submodule
  # (Matcha-TTS) that has to be on sys.path. `src/models/cosyvoice_gen.py` adds
  # both from $COSYVOICE_ROOT.
  local ROOT="/home/kirill/mnt/hdd_6tb_1/icassp_tts/third_party"
  mkdir -p "$ROOT"
  [ -d "$ROOT/CosyVoice" ] || git clone --recursive --depth 1 \
      https://github.com/FunAudioLLM/CosyVoice.git "$ROOT/CosyVoice"
  conda env list | grep -q "^cosyvoice " || conda create -y -n cosyvoice python=3.10
  conda activate cosyvoice
  pip install -q torch torchaudio $CU130
  # The upstream requirements.txt pins torch 2.3.1 + cu121 wheels; installing it
  # as-is replaces the cu130 build. These are the same packages with the torch
  # pins dropped, and only the ones an inference run actually imports (no
  # deepspeed/tensorrt/gradio/fastapi/grpcio, all training- or serving-only).
  #   diffusers 0.29.0 is a hard pin: Matcha-TTS imports
  #     `diffusers.models.lora.LoRACompatibleLinear`, removed in 0.30.
  #   setuptools <81 is a hard pin: cosyvoice/dataset/processor.py imports
  #     `pkg_resources`, dropped from setuptools 81.
  #   openai-whisper is installed --no-deps because its dependency block pins
  #     triton, which would downgrade the one torch needs. Only
  #     `whisper.log_mel_spectrogram` and `whisper.tokenizer` are used.
  pip install -q "conformer==0.3.2" "diffusers==0.29.0" "hydra-core==1.3.2" \
      "HyperPyYAML==1.2.3" lightning "inflect==7.3.1" "librosa==0.10.2" modelscope \
      "omegaconf==2.3.0" onnxruntime-gpu rich soundfile tiktoken "transformers==4.51.3" \
      einops scipy tqdm more-itertools numba "numpy<2" wetext gdown matplotlib wget \
      pyarrow pyworld "setuptools<81"
  pip install -q --no-deps openai-whisper
}

WHICH="${1:-all}"
case "$WHICH" in
  cosyvoice) setup_cosyvoice 2>&1 | tee "$REPO_ROOT/logs/env_cosyvoice.log" ;;
  base)    setup_base    2>&1 | tee "$REPO_ROOT/logs/env_base.log" ;;
  qwen)    setup_qwen    2>&1 | tee "$REPO_ROOT/logs/env_qwen.log" ;;
  coqui)   setup_coqui   2>&1 | tee "$REPO_ROOT/logs/env_coqui.log" ;;
  xcodec2) setup_xcodec2 2>&1 | tee "$REPO_ROOT/logs/env_xcodec2.log" ;;
  all)     for e in base qwen coqui xcodec2 cosyvoice; do bash "$0" "$e"; done ;;
  *) echo "usage: $0 [base|qwen|coqui|xcodec2|cosyvoice|all]"; exit 1 ;;
esac
echo "[setup_envs] done: $WHICH"
