#!/usr/bin/env bash
# Build the arXiv (main + supplement) PDF. Unlike paper/build.sh this does NOT
# regenerate numbers.tex -- the copy here is a frozen snapshot of the
# submitted numbers, and the rewrite edits prose, not data. If results
# change, run analysis/make_numbers.py at the repo root and copy
# paper/numbers.tex here deliberately.
#   bash paper/ICASSP_paper/build.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build

tectonic -X compile main.tex --outdir build --keep-intermediates --synctex=0 2>&1 \
  | grep -viE "^(note|warning: )" | tail -20

if [ -f build/main.pdf ]; then
  PAGES=$(python3 - <<'EOF'
try:
    import pypdf
    r = pypdf.PdfReader("build/main.pdf")
    print(len(r.pages))
except Exception:
    print("?")
EOF
)
  SIZE=$(du -h build/main.pdf | cut -f1)
  echo "[build] OK  build/main.pdf  ${PAGES} pages, ${SIZE}"
else
  echo "[build] FAILED"; exit 1
fi
