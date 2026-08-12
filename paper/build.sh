#!/usr/bin/env bash
# Build the ICASSP PDF. Regenerates the numbers file from the current results
# first, so the PDF can never drift from the CSVs it is supposed to describe.
#   bash paper/build.sh [--no-numbers]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# make_numbers.py resolves its inputs relative to the repo root, so it must run
# from there -- not from paper/.
if [ "${1:-}" != "--no-numbers" ]; then
  ( cd "$REPO_ROOT" && python3 analysis/make_numbers.py ) \
    || echo "[build] numbers step failed; using existing numbers.tex"
fi

cd "$REPO_ROOT/paper"
mkdir -p build

mkdir -p build
# A hand-typed experimental number goes stale silently and reads correctly
# forever after, so check before compiling rather than after.
( cd "$REPO_ROOT" && python3 scripts/check_numbers.py ) || exit 1

tectonic -X compile main.tex --outdir build --keep-intermediates --synctex=0 2>&1 \
  | grep -viE "^(note|warning: )" | tail -20

if [ -f build/main.pdf ]; then
  PAGES=$(python3 - <<'EOF'
try:
    import pypdf
    print(len(pypdf.PdfReader("build/main.pdf").pages))
except Exception:
    print("?")
EOF
)
  SIZE=$(du -h build/main.pdf | cut -f1)
  echo "[build] OK  build/main.pdf  ${PAGES} pages, ${SIZE}"
  [ "$PAGES" != "?" ] && [ "$PAGES" -gt 5 ] && \
    echo "[build] WARNING: ICASSP allows 4 pages + 1 of references (got $PAGES)"
else
  echo "[build] FAILED"; exit 1
fi
