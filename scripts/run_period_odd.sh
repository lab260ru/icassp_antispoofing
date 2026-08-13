#!/usr/bin/env bash
# The odd / non-power-of-two rungs of the period ladder: p = 3 and p = 6 at
# k = 24, added to answer the reviewer objection that the published ladder tests
# only p in {1,2,4,8} and contains no odd period at all. The pre-commitment,
# the two interpolation tests and the three outcome labels are in the docstring
# of data/stimuli/make_stimuli_period.py, written before any of this audio
# existed.
#
# Three things here are deliberate:
#
# 1. The FULL `stimuli_period.jsonl` is passed to every generator, not a
#    file trimmed to the 60 new items. Every generator resumes from its own
#    `*_meta.jsonl` on (item_id, seed), so the 135 published rungs are skipped
#    rather than re-rolled -- which is what keeps the new rungs comparable to
#    the old ones instead of being a second run of the pipeline. Do not
#    "optimise" this into a subset file: a subset file plus a wiped meta would
#    re-render the published arms under new samples and silently move the
#    baseline the new rungs are measured against.
#
# 2. The four checkpoints are the four the ladder already contains --
#    llasa1b, qwen06b, qwen17b, xtts2. `data/results/behavioural_period.csv`
#    has no llasa8b rows and never had any, so adding it here would compare a
#    p=3 rung against a p=2 rung that does not exist for that checkpoint.
#    llasa3b is excluded on purpose: it fails its replication gate on p=1.
#
# 3. GPUs 2 and 3 only, per src/common/gpus.py.
#
# Scoring is scripts/score_period.sh (which merges stimuli_disambig.jsonl --
# check that line is still there before running it), then
# analysis/period_odd.py.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
STIM=data/stimuli/stimuli_period.jsonl
LOG=experiments/exp11_period/logs
mkdir -p "$LOG"

# --- wave 1: the two slow ones, one per card.
conda activate base
python src/models/llasa_gen.py --model llasa1b --gpu 3 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/odd_llasa1b.log" 2>&1 &
P1=$!
conda deactivate
conda activate qwen
python src/models/qwen_gen.py --model qwen17b --gpu 2 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/odd_qwen17b.log" 2>&1 &
P2=$!
conda deactivate
wait $P1 $P2
echo "=== wave 1 generation done"

# --- wave 2: the two fast ones.
conda activate qwen
python src/models/qwen_gen.py --model qwen06b --gpu 2 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/odd_qwen06b.log" 2>&1 &
P3=$!
conda deactivate
# XTTS-v2 goes through `xtts_gen_compat.py`, not `xtts_gen.py`: the coqui env's
# transformers was upgraded to 5.x on 2026-08-12, which removed the
# `isin_mps_friendly` helper Coqui TTS 0.22 imports, and `xtts_gen.py` now dies
# on import. The shim restores the symbol in-process only. It does NOT put the
# environment back -- see that file's docstring -- so the XTTS-v2 rows this
# produces are not strictly commensurable with the published XTTS-v2 rows, and
# `analysis/period_odd.py` reports the verdict with XTTS-v2 both in and out.
conda activate coqui
python src/models/xtts_gen_compat.py --model xtts2 --gpu 3 --seeds 0 1 2 \
  --stimuli "$STIM" >"$LOG/odd_xtts2.log" 2>&1 &
P4=$!
conda deactivate
wait $P3 $P4
echo "=== wave 2 generation done"

# Llasa emits speech-token ids; the waveform is synthesised in the isolated
# xcodec2 env. Already-decoded items are skipped by the decoder itself.
conda activate xcodec2
python src/models/xcodec2_decode.py --model llasa1b --gpu 3 --glob 'pd_*.npy' \
  >"$LOG/odd_decode_llasa1b.log" 2>&1
conda deactivate

echo "=== PERIOD_ODD_GEN_DONE"
