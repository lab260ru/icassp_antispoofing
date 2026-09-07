#!/usr/bin/env bash
# Build the rewrite-workspace supplement. Frozen snapshot of the submitted
# supp.tex (tag v2.2-isolating); rewrite happens here, the original under
# paper/supplementary/ stays untouched.
#   bash paper/ICASSP_paper/supplement/build.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build

tectonic -X compile supp.tex --outdir build --keep-intermediates --synctex=0 2>&1 \
  | grep -viE "^(note|warning: )" | tail -20

if [ -f build/supp.pdf ]; then
  PAGES=$(python3 -c "import pypdf; print(len(pypdf.PdfReader('build/supp.pdf').pages))" 2>/dev/null || echo "?")
  echo "[build] OK  build/supp.pdf  ${PAGES} pages"
else
  echo "[build] FAILED"; exit 1
fi
