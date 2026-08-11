#!/usr/bin/env bash
# Install elan (Lean toolchain manager) and create the mathlib-backed project.
# CPU-only; safe to run in parallel with all GPU work.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.elan/bin:$PATH"

if ! command -v elan >/dev/null 2>&1; then
  echo "[lean] installing elan"
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  export PATH="$HOME/.elan/bin:$PATH"
fi

cd "$REPO_ROOT/lean"
if [ ! -d SpectralTTS ]; then
  echo "[lean] creating mathlib project"
  lake +leanprover-community/mathlib4:lean-toolchain new SpectralTTS math
fi

cd SpectralTTS
echo "[lean] fetching mathlib build cache (this is the slow part)"
lake exe cache get || lake exe cache get!
echo "[lean] building deps"
lake build Mathlib.Topology.MetricSpace.Contracting Mathlib.Analysis.SpecialFunctions.Log.Basic
echo "[lean] ready"
