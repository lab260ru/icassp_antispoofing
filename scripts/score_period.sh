#!/usr/bin/env bash
# Transcribe and score the period ladder, then read it.
#
# Two things here are deliberate and neither is obvious:
#
# 1. `score_counts.py` is given a *merged* stimulus file rather than
#    `stimuli_period.jsonl` alone. Its duration model is fitted on k<=3 items,
#    and the period ladder starts at k=16, so scoring it on its own leaves every
#    `duration_ratio` NaN and every `outcome` un-classifiable -- which is what
#    happened to `behavioural_aperiodic.csv`, whose 432 rows carry no `correct`
#    label at all. Merging restores the anchors. The period rows cannot perturb
#    the fits (they contribute no k<=3 points) and cannot leak into the panel
#    (`population.NON_PANEL_ITEM_PREFIXES`).
#
# 2. The vocabulary audit writes to its OWN file. `data/results/judge_vocab_audit.json`
#    is read by `population.panel()` to decide which templates every analysis in
#    the repo may report; regenerating it from one arm's rows would redefine the
#    exclusion list for the whole paper from a 135-item subset.
set -uo pipefail
cd /home/kirill/icassp_antispoofing
export HF_HOME=/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache
CONDA_BASE="$(conda info --base)"; . "$CONDA_BASE/etc/profile.d/conda.sh"
LOG=experiments/exp11_period/logs
MERGED=experiments/exp11_period/stimuli_merged.jsonl
mkdir -p "$LOG"
MODELS="${*:-llasa1b qwen06b}"

conda activate base
for m in $MODELS; do
  python src/common/asr_ctc.py --model "$m" --gpu 3 --glob 'pd_*.wav' \
    2>&1 | grep -viE "warn|future" | tail -2
done

cat data/stimuli/stimuli.jsonl data/stimuli/stimuli_aperiodic.jsonl \
    data/stimuli/stimuli_period.jsonl > "$MERGED"

python src/common/score_counts.py --models $MODELS --stimuli "$MERGED" \
  --judge ctc --out data/results/behavioural_period.csv 2>&1 | tail -4

python analysis/judge_vocab_audit.py --behavioural data/results/behavioural_period.csv \
  --stimuli data/stimuli/stimuli_period.jsonl \
  --out data/results/judge_vocab_audit_period.json 2>&1 | tail -20

python analysis/period_ladder.py 2>&1 | tee "$LOG/period_ladder.txt"
