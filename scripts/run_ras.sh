#!/usr/bin/env bash
# Repetition Aware Sampling (VALL-E 2, arXiv:2406.05370) on Qwen3-TTS-0.6B.
#
# A reviewer found prior art the paper had not engaged with. What the paper
# swept is repetition-penalty *magnitude* and greedy-versus-sampled; RAS is a
# different axis -- a sampler that reads the repetition history and re-draws --
# and it is engineered around exactly the variable this paper isolates. The
# pre-commitment, the hyperparameters and every decision the published method
# description leaves open are in `src/common/ras.py`, written before any of
# this audio existed.
#
# THREE ARMS, and why each is here:
#   qwen06brasoff  RAS at its no-op threshold. Bitwise identical to the stock
#                  decoder (see `analysis/ras_gate.py`), so it is simultaneously
#                  the sanity gate at full scale and the arm the RAS arms are
#                  compared against. It is NOT redundant with the stored
#                  `qwen06b` rows: the gate shows the stored panel audio does
#                  not reproduce byte-for-byte in this environment, because
#                  transformers has moved under us since the panel was
#                  generated (see `qwen_gen.eos_trim_length`'s docstring). A
#                  baseline regenerated today is the only honest comparator.
#   qwen06bras     RAS at the paper's K=10, t_r=0.1 over the checkpoint's own
#                  shipped nucleus (temperature 0.9, top_k 50, top_p 1.0).
#   qwen06brasgr   The same rule over a top-p=0.0 nucleus -- the small-v regime
#                  VALL-E 2 emphasises, where RAS is what makes a near-greedy
#                  nucleus safe. Included so the mitigation is tested at its
#                  strongest, not only at its default.
#
# The arms are kept out of the panel by `population.ABLATIONS`; scoring writes
# to `data/results/behavioural_ras.csv`, never to a shared table, and all three
# arms are named in one `score_counts.py` call because that script OVERWRITES
# its output rather than appending.
#
# Judge: CTC (wav2vec2-large-960h-lv60-self, greedy, LM-free). NOT Whisper --
# Whisper is autoregressive and biased against the phenomenon being measured.
#
# GPUs 2 and 3 only (src/common/gpus.py).
#
#   bash scripts/run_ras.sh [gpu_a] [gpu_b]
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
export OMP_NUM_THREADS=8
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp14_ras/logs
mkdir -p "$LOG"
GA="${1:-2}"
GB="${2:-3}"
SEEDS="${SEEDS:-0 1 2}"

conda activate qwen
# Lane A: the sanity gate at small scale first (it can end the arm), then the
# paper-default RAS arm.
(
  python analysis/ras_gate.py --gpu "$GA" --n 8 >"$LOG/gate.log" 2>&1
  python src/models/qwen_gen.py --model qwen06bras --gpu "$GA" --seeds $SEEDS \
    --ras >"$LOG/ras.log" 2>&1
) &
# Lane B: the regenerated stock baseline, then the small-top-p arm.
(
  python src/models/qwen_gen.py --model qwen06brasoff --gpu "$GB" --seeds $SEEDS \
    --ras --ras-threshold 2.0 >"$LOG/rasoff.log" 2>&1
  python src/models/qwen_gen.py --model qwen06brasgr --gpu "$GB" --seeds $SEEDS \
    --ras --ras-top-p 0.0 >"$LOG/rasgr.log" 2>&1
) &
wait
conda deactivate

conda activate base
for m in qwen06brasoff qwen06bras qwen06brasgr; do
  python src/common/asr_ctc.py --model "$m" --gpu "$GA" 2>&1 \
    | grep -viE "warn|future" | tail -1
done
# All three arms in ONE call: score_counts.py overwrites its output file.
python src/common/score_counts.py --models qwen06brasoff qwen06bras qwen06brasgr \
  --judge ctc --out data/results/behavioural_ras.csv 2>&1 | tail -4
python analysis/rep_aware_sampling.py 2>&1 | tee "$LOG/verdict.txt"
