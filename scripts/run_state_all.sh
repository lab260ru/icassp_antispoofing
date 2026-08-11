#!/usr/bin/env bash
# Run the state analysis one model per process, then concatenate.
#
# Running all models in a single process gets OOM-killed: each instrumented item
# loads an 80-90 MB activation array (Llasa-8B), and holding several models'
# worth of transient SVD workspace alongside a concurrently running generation
# job exhausts the box. One process per model bounds peak RSS to a single
# model's working set, and a crash costs one model instead of all of them.
#
#   bash scripts/run_state_all.sh [model ...]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p data/results/state_parts logs

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(llasa1b llasa3b llasa8b xtts2 qwen06b qwen17b xtts2norp)

for m in "${MODELS[@]}"; do
  [ -d "/home/kirill/mnt/hdd_6tb_1/icassp_tts/activations/$m" ] || { echo "skip $m (no activations)"; continue; }
  out="data/results/state_parts/${m}.csv"
  if [ -s "$out" ]; then echo "have $m"; continue; fi
  echo "=== $m"
  python3 analysis/state_dynamics.py --models "$m" --out "$out" \
    >> "logs/state_${m}.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -s "$out" ] && echo "  ok $(wc -l < "$out") rows" || echo "  FAILED rc=$rc"
done

python3 - <<'EOF'
import glob
import pandas as pd
parts = sorted(glob.glob("data/results/state_parts/*.csv"))
if parts:
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv("data/results/state.csv", index=False)
    print(f"merged {len(parts)} parts -> data/results/state.csv "
          f"({len(df)} rows, models: {sorted(df.model.unique())})")
EOF
