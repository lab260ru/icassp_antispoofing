#!/usr/bin/env bash
# Period ladder (data/stimuli/stimuli_period.jsonl): p = 1, 2, 4, 8, k at
# k = 16, 24, 32 with the carrier, the word count and the seeds held fixed.
#
# Breadth over families beats depth on one checkpoint here, because the claim
# under test is about a class of models, so the order is one checkpoint per
# family first (llasa1b, qwen06b, xtts2) and the scale-ladder siblings after.
#
# GPUs 2 and 3 only, per src/common/gpus.py. Two generations at a time, one per
# card, so a neighbour job on either card still has room.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
STIM=data/stimuli/stimuli_period.jsonl
LOG=experiments/exp11_period/logs
mkdir -p "$LOG"

# --- wave 1: the two cheapest checkpoints, one per family, one per card.
conda activate base
python src/models/llasa_gen.py --model llasa1b --gpu 3 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/llasa1b.log" 2>&1 &
P1=$!
conda deactivate
conda activate qwen
python src/models/qwen_gen.py --model qwen06b --gpu 2 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/qwen06b.log" 2>&1 &
P2=$!
conda deactivate
wait $P1 $P2
echo "=== wave 1 generation done"

# Llasa emits speech-token ids; the waveform is synthesised in the isolated
# xcodec2 env. --glob keeps the pass to this arm's items rather than
# re-decoding every npy the model has ever produced.
conda activate xcodec2
python src/models/xcodec2_decode.py --model llasa1b --gpu 3 --glob 'pd_*.npy' \
  >"$LOG/decode_llasa1b.log" 2>&1
conda deactivate

# --- wave 2: the second checkpoint of two families.
conda activate qwen
python src/models/qwen_gen.py --model qwen17b --gpu 2 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/qwen17b.log" 2>&1 &
P3=$!
conda deactivate
conda activate base
python src/models/llasa_gen.py --model llasa3b --gpu 3 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/llasa3b.log" 2>&1 &
P4=$!
conda deactivate
wait $P3 $P4
echo "=== wave 2 generation done"

conda activate xcodec2
python src/models/xcodec2_decode.py --model llasa3b --gpu 3 --glob 'pd_*.npy' \
  >"$LOG/decode_llasa3b.log" 2>&1
conda deactivate

echo "=== PERIOD_GEN_DONE"
