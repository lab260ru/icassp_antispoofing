#!/usr/bin/env bash
# Extension ladder (k = 48..128): does the rendered count saturate?
#
# The main sweep stops at k=32, and fitting c(k)=N*(1-exp(-k/N*)) there puts the
# horizon at or beyond the top of the ladder, so the main sweep cannot tell a
# not-yet-reached horizon from no horizon. This runs the same three carriers far
# past it. See data/stimuli/make_stimuli_ext.py for the stimulus design.
#
# Two deliberate departures from scripts/run_pipeline.sh:
#
#   * --max-new-tokens 8192, not the sweep's 2048. A k=128 item is ~55 s of
#     speech, well past a 2048-token budget at X-codec2's 50 Hz, and an item cut
#     off by *our* budget would be scored as the model failing to count. The
#     sweep's own cap-hit items show this: their median relative error is -0.44
#     against -0.08 for the rest. hit_cap is recorded either way and cap-hit
#     items are excluded from the fit.
#   * XTTS-v2 is not run. Its decoder has a built-in ~602 mel-token ceiling
#     (~26 s), which every k>=48 item exceeds, so its counts would be censored by
#     architecture rather than by any horizon. Raising that ceiling would run the
#     model outside the range it was trained on. Excluded and reported, not
#     silently dropped.
#
#   bash scripts/run_ext_ladder.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"

STIM=data/stimuli/stimuli_ext.jsonl
LOG=experiments/exp08_ext_ladder/logs
mkdir -p "$LOG"

python data/stimuli/make_stimuli_ext.py

echo "=== generation (GPUs 2 and 3 only; two models per card)"
conda activate base
python src/models/llasa_gen.py --model llasa1b --gpu 2 --seeds 0 \
  --stimuli "$STIM" --max-new-tokens 8192 >"$LOG/gen_llasa1b.log" 2>&1 &
python src/models/llasa_gen.py --model llasa8b --gpu 3 --seeds 0 \
  --stimuli "$STIM" --max-new-tokens 8192 >"$LOG/gen_llasa8b.log" 2>&1 &
conda deactivate
conda activate qwen
python src/models/qwen_gen.py --model qwen06b --gpu 2 --seeds 0 \
  --stimuli "$STIM" --max-new-tokens 8192 >"$LOG/gen_qwen06b.log" 2>&1 &
python src/models/qwen_gen.py --model qwen17b --gpu 3 --seeds 0 \
  --stimuli "$STIM" --max-new-tokens 8192 >"$LOG/gen_qwen17b.log" 2>&1 &
conda deactivate
wait
echo "=== generation done"
tail -2 "$LOG"/gen_*.log

echo "=== vocode Llasa"
conda activate xcodec2
for m in llasa1b llasa8b; do
  python src/models/xcodec2_decode.py --model "$m" --gpu 2 2>&1 | tail -1
done
conda deactivate

echo "=== transcribe (CTC judge)"
conda activate base
for m in llasa1b llasa8b qwen06b qwen17b; do
  python src/common/asr_ctc.py --model "$m" --gpu 2 2>&1 \
    | grep -viE "warn|future|^\s*$" | tail -1
done

echo "=== score (separate table; the main one keeps the k<=32 population)"
python src/common/score_counts.py --models llasa1b llasa8b qwen06b qwen17b \
  --stimuli "$STIM" --judge ctc --out data/results/behavioural_ext.csv

echo "=== fit"
python analysis/horizon_shape.py --behavioural data/results/behavioural_ext.csv \
  --kmin 48 --out data/results/horizon_shape_ext.json
