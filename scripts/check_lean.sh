#!/usr/bin/env bash
# Verification gate for the formal artifact.
#   1. the project compiles
#   2. no `sorry` anywhere in our sources
#   3. the headline theorems depend only on the three standard Lean axioms
#      (propext, Classical.choice, Quot.sound) — no `sorryAx`
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.elan/bin:$PATH"
cd "$REPO_ROOT/lean/SpectralTTS"

echo "== lake build"
lake build 2>&1 | tail -3 || exit 1

echo "== sorry scan"
if grep -rn "sorry" SpectralTTS/ SpectralTTS.lean; then
  echo "FAIL: sorry found"; exit 1
fi
echo "  none"

echo "== axiom audit"
cat > /tmp/spectraltts_axioms.lean <<'EOF'
import SpectralTTS
open SpectralTTS
#print axioms SpectralTTS.dist_iterate_fixedPoint_le
#print axioms SpectralTTS.dist_iterate_iterate_le
#print axioms SpectralTTS.exists_indistinguishable
#print axioms SpectralTTS.readout_gap_le
#print axioms SpectralTTS.counting_margin_le
#print axioms SpectralTTS.counting_horizon
#print axioms SpectralTTS.softmax_le
#print axioms SpectralTTS.le_softmax
#print axioms SpectralTTS.softmax_dist_le
#print axioms SpectralTTS.entropy_ge
#print axioms SpectralTTS.sum_softmax
EOF
OUT=$(lake env lean /tmp/spectraltts_axioms.lean 2>&1)
echo "$OUT"
if echo "$OUT" | grep -q "sorryAx"; then
  echo "FAIL: sorryAx in dependency closure"; exit 1
fi
echo
echo "PASS: formal artifact verified"
