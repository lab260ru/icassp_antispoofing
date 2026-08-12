#!/usr/bin/env bash
# Repetition-penalty sweep on a second architecture.
#
# The paper predicts that a mitigation helps only insofar as it raises q or
# enlarges the readout margin, and tested that on XTTS-v2 alone. Two review
# rounds objected that one architecture cannot support a claim about the field's
# standard fix. Qwen3-TTS-0.6B is the natural second: it shows the deficit, and
# its shipped penalty (1.05) sits far below XTTS-v2's (5.0), so the sweep covers
# a different part of the range rather than repeating the same one.
#
# Only the word_rep and control_word families are needed for the contrast, and
# only k>=6 carries it, but the scorer wants the whole file; generation is cheap
# enough at this size to just run it.
#
# GPUs 2 and 3 only (src/common/gpus.py).
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp11_qwen_penalty/logs
mkdir -p "$LOG"

# Sequential on one card: the other is usually busy vocoding, and three arms of
# this size finish comfortably either way.
GPU="${1:-3}"
conda activate qwen
# 1.0 disables the penalty; 1.05 is shipped; 1.5 and 3.0 push well past it.
python src/models/qwen_gen.py --model qwen06brp10 --gpu "$GPU" --seeds 0 \
  --repetition-penalty 1.0 >"$LOG/rp10.log" 2>&1
python src/models/qwen_gen.py --model qwen06brp15 --gpu "$GPU" --seeds 0 \
  --repetition-penalty 1.5 >"$LOG/rp15.log" 2>&1
python src/models/qwen_gen.py --model qwen06brp30 --gpu "$GPU" --seeds 0 \
  --repetition-penalty 3.0 >"$LOG/rp30.log" 2>&1
conda deactivate

conda activate base
for m in qwen06brp10 qwen06brp15 qwen06brp30; do
  python src/common/asr_ctc.py --model "$m" --gpu "$GPU" 2>&1 | grep -viE "warn|future" | tail -1
done
python src/common/score_counts.py --models qwen06brp10 qwen06brp15 qwen06brp30 \
  --judge ctc --out data/results/behavioural_qwen_penalty.csv 2>&1 | tail -4
