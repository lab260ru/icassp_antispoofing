#!/usr/bin/env bash
# Shuffled twins of the period ladder at k=24: same word multiset, no period.
# The pre-commitment, the recovery-fraction test, the three outcome labels and
# the placebo are in the docstring of
# `data/stimuli/make_stimuli_period_shuffled.py`, written before any of this
# audio existed.
#
# Five things here are deliberate.
#
# 1. The stimulus file passed to every generator is `stimuli_period.jsonl`
#    CONCATENATED with `stimuli_period_shuffled.jsonl`, not the new file alone.
#    Every generator resumes from its own `*_meta.jsonl` on (item_id, seed), so
#    the 195 published ladder items are skipped rather than re-rolled, which is
#    what keeps the new arms comparable to the published rungs instead of being
#    a second run of the pipeline. `analysis/period_shuffled.py` verifies the
#    resume actually held, by checking `count_a` is byte-identical on every
#    pre-existing `pd_` row against a snapshot taken before this ran.
#
# 2. Two arms are generated, not one: the shuffled twins AND a re-rendered
#    periodic control whose text is byte-identical to the published item. The
#    control exists because the `coqui` env's transformers was bumped 4.x -> 5.x
#    on 2026-08-12, after the published XTTS-v2 rows were made, so for that
#    checkpoint a periodic-vs-shuffled contrast against the published rows would
#    confound word order with a major version bump of the sampling stack. It is
#    generated for all four checkpoints because a control that exists on one and
#    not the others cannot be pooled.
#
# 3. The four checkpoints are the four the ladder contains -- llasa1b, qwen06b,
#    qwen17b, xtts2. `data/results/behavioural_period.csv` has no llasa8b rows
#    and never had any, so a shuffled llasa8b arm would have no periodic twin to
#    be compared against. llasa3b is out for the reason it is out of the ladder:
#    it fails its replication gate on the p=1 rung.
#
# 4. GPUs 2 and 3 only, per src/common/gpus.py. This launches on cuda:2 alone,
#    two generators at a time, because another experiment holds cuda:3. If
#    cuda:3 frees, nothing here needs changing -- the generators resume, so a
#    second invocation with --gpu 3 simply picks up whatever is left.
#
# 5. XTTS-v2 goes through `xtts_gen_compat.py`, not `xtts_gen.py`: transformers
#    5.x removed the `isin_mps_friendly` helper Coqui TTS 0.22 imports, and
#    `xtts_gen.py` dies on import. The shim restores the symbol in-process only
#    and does NOT modify the environment, which another session shares.
#
# Scoring is `scripts/score_period_shuffled.sh`, which writes to a SIDE CSV and
# never touches `data/results/behavioural_period.csv`.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp11_period/logs
MERGED=experiments/exp11_period/stimuli_shuffled_gen.jsonl
mkdir -p "$LOG" experiments/exp11_period
GPU="${GPU:-2}"

cat data/stimuli/stimuli_period.jsonl data/stimuli/stimuli_period_shuffled.jsonl > "$MERGED"

# --- wave 1: the two slow ones, sharing cuda:$GPU.
conda activate base
python src/models/llasa_gen.py --model llasa1b --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$MERGED" >"$LOG/sh_llasa1b.log" 2>&1 &
P1=$!
conda deactivate
conda activate qwen
python src/models/qwen_gen.py --model qwen17b --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$MERGED" >"$LOG/sh_qwen17b.log" 2>&1 &
P2=$!
conda deactivate
wait $P1 $P2
echo "=== wave 1 generation done"

# --- wave 2: the two fast ones.
conda activate qwen
python src/models/qwen_gen.py --model qwen06b --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$MERGED" >"$LOG/sh_qwen06b.log" 2>&1 &
P3=$!
conda deactivate
conda activate coqui
python src/models/xtts_gen_compat.py --model xtts2 --gpu "$GPU" --seeds 0 1 2 \
  --stimuli "$MERGED" >"$LOG/sh_xtts2.log" 2>&1 &
P4=$!
conda deactivate
wait $P3 $P4
echo "=== wave 2 generation done"

# Llasa emits speech-token ids; the waveform is synthesised in the isolated
# xcodec2 env. Already-decoded items are skipped by the decoder itself.
conda activate xcodec2
python src/models/xcodec2_decode.py --model llasa1b --gpu "$GPU" --glob 'pd_*.npy' \
  >"$LOG/sh_decode_llasa1b.log" 2>&1
conda deactivate

echo "=== PERIOD_SHUFFLED_GEN_DONE"
