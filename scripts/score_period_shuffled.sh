#!/usr/bin/env bash
# Transcribe and score the shuffled twins, then read the pre-committed test.
#
# This is `scripts/score_period.sh` with one deliberate sameness and two
# deliberate differences.
#
# SAME: the merge line includes `stimuli.jsonl`, `stimuli_aperiodic.jsonl`,
# `stimuli_period.jsonl` and `stimuli_disambig.jsonl`. It must. `score_counts.py`
# emits a row only for item ids present in the `--stimuli` file it is given and
# *overwrites* its output rather than appending, so a merge that omits an arm
# silently deletes that arm's rows. `stimuli.jsonl` in particular carries the
# k<=3 anchors the duration model is fitted on; without them every
# `duration_ratio` is NaN and every `outcome` un-classifiable, which is what
# happened to `behavioural_aperiodic.csv`.
#
# DIFFERENT 1: `--out` is a SIDE CSV, `data/results/behavioural_period_shuffled.csv`,
# not `data/results/behavioural_period.csv`. The period CSV is a landed result
# that `analysis/period_ladder.py`, `analysis/period_odd.py` and
# `analysis/disambiguation.py` all read. Writing 1680 new rows into it is a
# decision for whoever owns those results, not a side effect of scoring a new
# arm. Keeping the published file untouched is also what makes the resume check
# possible: `analysis/period_shuffled.py` diffs the side CSV against it and
# asserts `count_a` is unchanged on every pre-existing row, which is the
# evidence that the generators resumed rather than re-rolled.
#
# DIFFERENT 2: no vocabulary audit is regenerated. `data/results/judge_vocab_audit.json`
# is read by `population.panel()` to decide which templates every analysis in
# the repo may report; regenerating it from one arm's rows would redefine the
# exclusion list for the whole paper from a 140-item subset. The list this arm
# inherits (t2, whose fillers the CTC judge never emits) already excludes the
# only template that could be affected, and t2 is not in this arm anyway.
#
# Judge: CTC (wav2vec2-large-960h-lv60-self, greedy, LM-free). NOT Whisper --
# Whisper is autoregressive and is biased against the very phenomenon being
# measured.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp11_period/logs
MERGED=experiments/exp11_period/stimuli_shuffled_score.jsonl
mkdir -p "$LOG"
GPU="${GPU:-2}"
# Every checkpoint the side table is supposed to hold. Passing a SUBSET rewrites
# it with only those models.
MODELS="${*:-llasa1b qwen06b xtts2 qwen17b}"

conda activate base
for m in $MODELS; do
  python src/common/asr_ctc.py --model "$m" --gpu "$GPU" --glob 'pd_*.wav' \
    2>&1 | grep -viE "warn|future" | tail -2
done

cat data/stimuli/stimuli.jsonl data/stimuli/stimuli_aperiodic.jsonl \
    data/stimuli/stimuli_period.jsonl data/stimuli/stimuli_disambig.jsonl \
    data/stimuli/stimuli_period_shuffled.jsonl > "$MERGED"

python src/common/score_counts.py --models $MODELS --stimuli "$MERGED" \
  --judge ctc --out data/results/behavioural_period_shuffled.csv 2>&1 | tail -4

python analysis/period_shuffled.py 2>&1 | tee "$LOG/period_shuffled.txt"
