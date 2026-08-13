#!/usr/bin/env bash
# Transcribe and score the odd / non-power-of-two rungs (p = 3, 6 at k = 24),
# then read the interpolation tests.
#
# This is `scripts/score_period.sh` with two deliberate differences and one
# deliberate sameness.
#
# SAME: the merge line below includes `stimuli_disambig.jsonl`. It must.
# `score_counts.py` emits a row only for item ids present in the `--stimuli`
# file it is given and *overwrites* `behavioural_period.csv` rather than
# appending, so a merge that omits an arm silently deletes that arm's rows from
# a shared table -- and `analysis/disambiguation.py` reads the same file. The
# same applies to `stimuli.jsonl` and `stimuli_aperiodic.jsonl`. Never let a
# scoring pass write over a CSV holding rows it did not regenerate.
#
# DIFFERENT 1: `analysis/period_ladder.py` is run with `--out` pointing at a
# side file, not at `data/results/period_ladder.json`. Adding p=3 and p=6 rows
# changes three of that file's fields -- `kept`, `drop` and `collapse_onto_m`,
# which gains the new (p, m) cells -- and that file is a landed result another
# analysis and the paper both read. Folding the new rungs into it is a decision
# for whoever owns that result, not a side effect of scoring.
#
# DIFFERENT 2: the vocabulary audit again writes to its own file.
# `data/results/judge_vocab_audit.json` decides which templates every analysis
# in the repo may report; regenerating it from one arm's rows would redefine the
# exclusion list for the whole paper from a subset.
#
# Judge: CTC (wav2vec2-large-960h-lv60-self, greedy, LM-free). NOT Whisper --
# Whisper is autoregressive and is biased against the very phenomenon being
# measured.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp11_period/logs
MERGED=experiments/exp11_period/stimuli_merged.jsonl
mkdir -p "$LOG"
# The four checkpoints the period ladder contains. Passing a SUBSET rewrites the
# shared CSV with only those models, so this must name every checkpoint the
# table is supposed to hold.
MODELS="${*:-llasa1b qwen06b xtts2 qwen17b}"

conda activate base
for m in $MODELS; do
  python src/common/asr_ctc.py --model "$m" --gpu 3 --glob 'pd_*.wav' \
    2>&1 | grep -viE "warn|future" | tail -2
done

cat data/stimuli/stimuli.jsonl data/stimuli/stimuli_aperiodic.jsonl \
    data/stimuli/stimuli_period.jsonl data/stimuli/stimuli_disambig.jsonl > "$MERGED"

python src/common/score_counts.py --models $MODELS --stimuli "$MERGED" \
  --judge ctc --out data/results/behavioural_period.csv 2>&1 | tail -4

python analysis/judge_vocab_audit.py --behavioural data/results/behavioural_period.csv \
  --stimuli data/stimuli/stimuli_period.jsonl \
  --out data/results/judge_vocab_audit_period.json 2>&1 | tail -20

python analysis/period_odd.py 2>&1 | tee "$LOG/period_odd.txt"

# The published ladder re-read with the new rungs present, to a side file, so
# the effect of adding them on `collapse_onto_m` can be seen without
# overwriting the landed result.
python analysis/period_ladder.py --out data/results/period_ladder_with_odd.json \
  2>&1 | tee "$LOG/period_ladder_with_odd.txt" | tail -5
