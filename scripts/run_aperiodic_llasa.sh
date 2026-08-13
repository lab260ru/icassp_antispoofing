#!/usr/bin/env bash
# The two arms `scripts/run_aperiodic.sh` left out: Llasa-3B and Llasa-8B on the
# never-cycled 146-word control pool (data/stimuli/stimuli_aperiodic.jsonl).
#
# Why separately: the four fast checkpoints are already generated and both
# generators resume from their `*_meta.jsonl` ledgers, so re-running the original
# driver would pay two model loads off a saturated spinning disk to do nothing.
# The flags here are identical to the ones the other four ran under, so the
# population stays comparable by construction.
#
# Loads are staggered -- three concurrent loads saturate the HF cache disk at
# ~75 MB/s and Llasa-8B alone takes ~10 min to reach the GPU. GPUs 2 and 3 only,
# per src/common/gpus.py; this run takes 2 because sibling jobs hold 3.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
STIM=data/stimuli/stimuli_aperiodic.jsonl
LOG=experiments/exp10_aperiodic/logs
GPU=${GPU:-2}
mkdir -p "$LOG"

conda activate base
# --attn-probes 0: these items carry no `instrumented` flag so the teacher-forced
# pass never fires, but eager attention is the documented OOM at long sequences
# and the flag costs nothing.
python src/models/llasa_gen.py --model llasa3b --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$STIM" --attn-probes 0 >"$LOG/llasa3b.log" 2>&1 &
P3=$!
sleep 240
python src/models/llasa_gen.py --model llasa8b --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$STIM" --attn-probes 0 >"$LOG/llasa8b.log" 2>&1 &
P8=$!
wait $P3 $P8
echo "=== generation done"

conda deactivate
conda activate xcodec2
# `--glob ap_*`: both checkpoints carry 24 undecoded `*i` token files from the
# instrumented extension ladder, which exist for their activations and were
# never meant to have audio. A bare decode would vocode them and the ASR pass
# would then transcribe them, enlarging a population nobody scored.
for m in llasa3b llasa8b; do
  python src/models/xcodec2_decode.py --model "$m" --gpu "$GPU" --glob 'ap_*.npy' 2>&1 | tail -2
done
conda deactivate
conda activate base
for m in llasa3b llasa8b; do
  python src/common/asr_ctc.py --model "$m" --gpu "$GPU" --glob 'ap_*.wav' 2>&1 | grep -viE "warn|future" | tail -1
done
# Score the whole panel into the aperiodic table, not just the new two: the file
# is overwritten, and `score_counts.py` only emits rows for items present in the
# stimulus file, so listing all six is what keeps the other four's rows.
python src/common/score_counts.py --models llasa1b llasa3b llasa8b xtts2 qwen06b qwen17b \
  --stimuli "$STIM" --judge ctc --out data/results/behavioural_aperiodic.csv 2>&1 | tail -4
echo "=== pipeline done"
