#!/usr/bin/env bash
# Pre-download every checkpoint to the HDD-backed HF cache so GPU time is never
# spent waiting on network. Safe to re-run; hf skips what is already present.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"

CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate base

dl () {  # dl <repo_id> [allow_patterns...]
  echo "=== $1"
  if [ $# -gt 1 ]; then
    hf download "$@" 2>&1 | tail -2
  else
    hf download "$1" 2>&1 | tail -2
  fi
}

dl HKUSTAudio/Llasa-1B
dl HKUSTAudio/Llasa-3B
dl openai/whisper-large-v3
dl HKUSTAudio/xcodec2
dl Qwen/Qwen3-TTS-12Hz-0.6B-Base
dl Qwen/Qwen3-TTS-12Hz-1.7B-Base
dl HKUSTAudio/Llasa-8B
dl coqui/XTTS-v2
echo "[download_models] done"
du -sh "$HF_HOME"
