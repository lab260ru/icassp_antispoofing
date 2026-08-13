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
# The full ladder as reported: three families, and a second Qwen checkpoint added
# so the ordered-scan mean does not rest on one. Passing a SUBSET here rewrites
# the shared CSV with only those models -- the default must therefore name every
# checkpoint the table is supposed to contain, not the two it started with.
MODELS="${*:-llasa1b qwen06b xtts2 qwen17b}"

conda activate base
for m in $MODELS; do
  python src/common/asr_ctc.py --model "$m" --gpu 3 --glob 'pd_*.wav' \
    2>&1 | grep -viE "warn|future" | tail -2
done

# `stimuli_disambig.jsonl` belongs in this list, and its absence was a live bug.
# `score_counts.py` emits a row only for item ids present in the `--stimuli` file
# it is given, and it *overwrites* `behavioural_period.csv` rather than appending
# -- so a merge without the disambiguator arm silently deletes 180 rows per
# checkpoint from the shared table and leaves `analysis/disambiguation.py`, which
# reads that same file, with "no disambiguator rows scored yet". The committed
# CSV had those rows, so the file on disk and the script that claims to build it
# had already drifted apart; re-running this script as written would have
# destroyed a landed result. Every arm scored into this CSV must be merged here.
cat data/stimuli/stimuli.jsonl data/stimuli/stimuli_aperiodic.jsonl \
    data/stimuli/stimuli_period.jsonl data/stimuli/stimuli_disambig.jsonl > "$MERGED"

python src/common/score_counts.py --models $MODELS --stimuli "$MERGED" \
  --judge ctc --out data/results/behavioural_period.csv 2>&1 | tail -4

python analysis/judge_vocab_audit.py --behavioural data/results/behavioural_period.csv \
  --stimuli data/stimuli/stimuli_period.jsonl \
  --out data/results/judge_vocab_audit_period.json 2>&1 | tail -20

python analysis/period_ladder.py 2>&1 | tee "$LOG/period_ladder.txt"
