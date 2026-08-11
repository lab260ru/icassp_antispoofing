#!/usr/bin/env bash
# Downstream pipeline: vocode (Llasa only) -> transcribe -> score.
# Idempotent and resumable at every stage, so it is safe to run repeatedly
# while generation is still filling in.
#
#   bash scripts/run_pipeline.sh <asr_gpu> [model ...]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"

GPU="${1:-0}"; shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(llasa1b llasa3b llasa8b xtts2 qwen06b qwen17b)

for m in "${MODELS[@]}"; do
  # Llasa emits token ids; every other family writes wavs directly.
  if [[ "$m" == llasa* ]]; then
    echo "=== vocode $m"
    conda activate xcodec2
    python src/models/xcodec2_decode.py --model "$m" --gpu "$GPU" 2>&1 | tail -2
    conda deactivate
  fi
  if [ -d "/home/kirill/mnt/hdd_6tb_1/icassp_tts/audio/$m" ]; then
    echo "=== transcribe $m"
    conda activate base
    python src/common/asr_transcribe.py --model "$m" --gpu "$GPU" 2>&1 \
      | grep -viE "warn|future|^\s*$" | tail -2
    conda deactivate
  fi
done

echo "=== score"
conda activate base
# Score EVERY model that has transcripts, not just the ones this invocation
# generated: behavioural.csv is a single shared table, so scoring a subset would
# silently drop the other models' rows from the paper's inputs.
AVAIL=()
for f in /home/kirill/mnt/hdd_6tb_1/icassp_tts/asr/*.jsonl; do
  [ -s "$f" ] && AVAIL+=("$(basename "$f" .jsonl)")
done
if [ ${#AVAIL[@]} -gt 0 ]; then
  python src/common/score_counts.py --models "${AVAIL[@]}" --out data/results/behavioural.csv
fi
