#!/usr/bin/env bash
# Downstream pipeline for the causal-intervention arms: vocode -> CTC -> score.
#
# Deliberately the *same* stages, the same codec and the same recogniser as
# `scripts/run_pipeline.sh` uses for the behavioural panel. An intervention
# scored by a different instrument than the baseline it is compared against
# would measure the instrument. Only the model keys differ, because the patched
# and steered generations are not panel members and must never mix into
# behavioural.csv.
#
#   bash analysis/causal_pipeline.sh [gpu] [key ...]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"

GPU="${1:-3}"; shift || true
KEYS=("$@")
[ ${#KEYS[@]} -eq 0 ] && KEYS=(patch1b steer1b)

for k in "${KEYS[@]}"; do
  [ -d "/home/kirill/mnt/hdd_6tb_1/icassp_tts/tokens/$k" ] || continue
  echo "=== vocode $k"
  conda activate xcodec2
  python src/models/xcodec2_decode.py --model "$k" --gpu "$GPU" 2>&1 | tail -2
  conda deactivate
  echo "=== transcribe $k (CTC)"
  conda activate base
  python src/common/asr_ctc.py --model "$k" --gpu "$GPU" 2>&1 \
    | grep -viE "warn|future|^\s*$" | tail -2
  conda deactivate
done

echo "=== score"
conda activate base
python analysis/causal_count_score.py
