#!/usr/bin/env bash
# Aperiodic controls for k=12..32 (see data/stimuli/stimuli_aperiodic.jsonl).
# Run on every panel member that can turn them round in the time available;
# Llasa-3B/8B are the slow ones and are added last. GPUs 2 and 3 only, per
# src/common/gpus.py -- 0 and 1 belong to someone else.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
STIM=data/stimuli/stimuli_aperiodic.jsonl
LOG=experiments/exp10_aperiodic/logs
mkdir -p "$LOG"

conda activate qwen
python src/models/qwen_gen.py --model qwen06b --gpu 2 --seeds 0 1 2 --stimuli "$STIM" >"$LOG/qwen06b.log" 2>&1 &
python src/models/qwen_gen.py --model qwen17b --gpu 3 --seeds 0 1 2 --stimuli "$STIM" >"$LOG/qwen17b.log" 2>&1 &
conda deactivate
conda activate coqui
python src/models/xtts_gen.py --model xtts2 --gpu 3 --seeds 0 1 2 --stimuli "$STIM" >"$LOG/xtts2.log" 2>&1 &
conda deactivate
conda activate base
python src/models/llasa_gen.py --model llasa1b --gpu 2 --seeds 0 1 2 --stimuli "$STIM" >"$LOG/llasa1b.log" 2>&1 &
wait
echo "=== generation done"

conda activate xcodec2
python src/models/xcodec2_decode.py --model llasa1b --gpu 2 2>&1 | tail -1
conda deactivate
conda activate base
for m in llasa1b xtts2 qwen06b qwen17b; do
  python src/common/asr_ctc.py --model "$m" --gpu 2 2>&1 | grep -viE "warn|future" | tail -1
done
# Score every checkpoint that has aperiodic transcripts, not just the four this
# script generates: the output file is overwritten, so listing only these four
# would delete the Llasa-3B/8B rows that `run_aperiodic_llasa.sh` adds -- the
# same trap `run_pipeline.sh` shipped once already. Rows are emitted only for
# item ids present in "$STIM", so naming a model with no aperiodic audio costs
# nothing.
python src/common/score_counts.py --models llasa1b llasa3b llasa8b xtts2 qwen06b qwen17b \
  --stimuli "$STIM" --judge ctc --out data/results/behavioural_aperiodic.csv 2>&1 | tail -4
