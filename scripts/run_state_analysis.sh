#!/usr/bin/env bash
# Run the state analysis one model per process, then concatenate.
#
# A single process over the whole panel holds Llasa-8B's 4096-dim activations
# alongside everything else and gets OOM-killed. Per-model invocations release
# that memory between models and make the run resumable: a model whose shard
# already exists is skipped, so a crash costs one model rather than all seven.
#
#   bash scripts/run_state_analysis.sh [model ...]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p data/results/state_shards logs

# Cap BLAS threading. Unbounded, each numpy SVD grabs all 48 cores, so two of
# these running at once spend their time in scheduler contention rather than
# arithmetic. Bounded, the run coexists with the generation jobs on the GPUs.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8

# Refuse to start if another copy is already working; duplicate drivers writing
# the same shards was a real failure mode, not a hypothetical one.
if pgrep -f "state_dynamics.py --models" >/dev/null; then
  echo "state_dynamics already running; refusing to start a second driver" >&2
  exit 1
fi

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(llasa1b llasa3b llasa8b xtts2 qwen06b qwen17b xtts2norp)

for m in "${MODELS[@]}"; do
  shard="data/results/state_shards/${m}.csv"
  if [ -s "$shard" ]; then
    echo "== $m: shard exists, skipping"
    continue
  fi
  if [ ! -d "/home/kirill/mnt/hdd_6tb_1/icassp_tts/activations/$m" ]; then
    echo "== $m: no activations, skipping"
    continue
  fi
  echo "== $m"
  python3 analysis/state_dynamics.py --models "$m" --out "$shard" \
    >> "logs/state_${m}.log" 2>&1
  if [ -s "$shard" ]; then
    echo "   ok ($(($(wc -l < "$shard") - 1)) rows)"
  else
    echo "   FAILED (see logs/state_${m}.log)"
  fi
done

python3 - <<'EOF'
import glob
import pandas as pd
shards = sorted(glob.glob("data/results/state_shards/*.csv"))
if shards:
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    df.to_csv("data/results/state.csv", index=False)
    print(f"merged {len(shards)} shards -> data/results/state.csv "
          f"({len(df)} rows, {df.model.nunique()} models)")
EOF
