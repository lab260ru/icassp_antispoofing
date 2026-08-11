#!/usr/bin/env bash
# Environment bring-up for the AR-TTS counting-collapse study.
# Each block is idempotent and logs to logs/env_<name>.log.
#
# Env map
#   base     (py3.13, torch 2.11+cu130) : Llasa LMs, Whisper ASR, all analysis  [additive install only]
#   qwen     (py3.12)                   : Qwen3-TTS via the `qwen-tts` package
#   coqui    (py3.12)                   : XTTS-v2 via `coqui-tts` (idiap fork)
#   xcodec2  (py3.11, torch 2.5 pinned) : offline vocoding of Llasa speech tokens
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

WHICH="${1:-all}"
case "$WHICH" in
  base)    setup_base    2>&1 | tee "$REPO_ROOT/logs/env_base.log" ;;
  qwen)    setup_qwen    2>&1 | tee "$REPO_ROOT/logs/env_qwen.log" ;;
  coqui)   setup_coqui   2>&1 | tee "$REPO_ROOT/logs/env_coqui.log" ;;
  xcodec2) setup_xcodec2 2>&1 | tee "$REPO_ROOT/logs/env_xcodec2.log" ;;
  all)     for e in base qwen coqui xcodec2; do bash "$0" "$e"; done ;;
  *) echo "usage: $0 [base|qwen|coqui|xcodec2|all]"; exit 1 ;;
esac
echo "[setup_envs] done: $WHICH"
