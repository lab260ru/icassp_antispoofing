#!/usr/bin/env bash
# Vocode (Llasa only) and transcribe the second-checkpoint causal arms, then
# score them. Mirrors analysis/causal_pipeline.sh, with one difference that is
# not cosmetic: Qwen3-TTS returns a waveform from `generate_voice_clone`, so
# there is no codec-decode stage at all and the audio written during generation
# is the audio the judge hears. Llasa still goes through xcodec2 in its own env.
#
#   bash analysis/causal_pipeline_second.sh <gpu> <key> [key ...]
#     keys look like pq_qwen06b (rank-1 patch arm) or rq_qwen06b (ridge sweep)
set -euo pipefail
GPU="${1:-2}"; shift || true
KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(pq_qwen06b pq_llasa1b)
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"

for k in "${KEYS[@]}"; do
  if [[ "$k" == *llasa* ]]; then
    echo "== [$k] xcodec2 decode on cuda:$GPU"
    conda activate xcodec2
    python src/models/xcodec2_decode.py --model "$k" --gpu "$GPU"
    conda deactivate
  else
    echo "== [$k] no codec stage: the waveform came straight out of generation"
  fi
  echo "== [$k] CTC judge on cuda:$GPU"
  conda activate base
  python src/common/asr_ctc.py --model "$k" --gpu "$GPU" --processor wav2vec2
  conda deactivate
done

echo "== scoring"
conda activate base
python analysis/causal_count_second_score.py
