#!/usr/bin/env bash
# Everything that must hold before this is called a submission.
#
# Written to be run cold, by someone who was not here: each check prints what it
# checked and why it matters, and the script exits non-zero on the first hard
# failure rather than printing a wall of green and one buried problem.
#
#   bash scripts/verify_submission.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
FAIL=0
note() { printf '\n=== %s\n' "$1"; }
ok()   { printf '  OK   %s\n' "$1"; }
bad()  { printf '  FAIL %s\n' "$1"; FAIL=1; }

note "Formal artifact (zero sorry, standard axioms only)"
if bash scripts/check_lean.sh >/tmp/lean_check.$$ 2>&1; then
  ok "$(tail -1 /tmp/lean_check.$$)"
else
  bad "check_lean.sh failed; see /tmp/lean_check.$$"
fi

note "Every reported number is generated, not typed"
if python3 scripts/check_numbers.py; then :; else bad "check_numbers.py failed"; fi

note "Main paper builds and fits ICASSP's 4+1 pages"
( cd paper && bash build.sh >/tmp/build_main.$$ 2>&1 )
PAGES=$(python3 -c "import pypdf;print(len(pypdf.PdfReader('paper/build/main.pdf').pages))" 2>/dev/null || echo "?")
SPILL=$(python3 -c "
import pypdf
t=pypdf.PdfReader('paper/build/main.pdf').pages[4].extract_text()
print(t.find('REFERENCES'))" 2>/dev/null || echo "?")
if [ "$PAGES" = "5" ]; then
  ok "main.pdf is 5 pages (4 content + 1 references)"
else
  bad "main.pdf is $PAGES pages, must be 5"
fi
# main.tex ends with \vfill\pagebreak, so any body text on page 5 pushes the
# bibliography to a sixth page. A small positive number here is the warning sign.
if [ "$SPILL" != "?" ] && [ "$SPILL" -le 5 ] 2>/dev/null; then
  ok "no body text spills past page 4 (offset $SPILL)"
else
  bad "body text spills onto the references page (offset $SPILL)"
fi

note "Supplement builds"
( cd paper/supplementary && tectonic -X compile supp.tex --outdir build >/tmp/build_supp.$$ 2>&1 )
SPAGES=$(python3 -c "import pypdf;print(len(pypdf.PdfReader('paper/supplementary/build/supp.pdf').pages))" 2>/dev/null || echo "?")
if [ "$SPAGES" != "?" ]; then ok "supp.pdf is $SPAGES pages"; else bad "supp.pdf did not build"; fi

note "Every supplement pointer in the paper resolves"
python3 - <<'PY'
import re, pathlib, sys
supp = pathlib.Path('paper/supplementary/supp.tex').read_text()
n_sections = len(re.findall(r'(?m)^\\section\{', supp))
used = set()
for f in ['paper/main.tex', 'paper/results_body.tex', 'paper/discussion_body.tex']:
    text = re.sub(r'(?m)^\s*%.*$', '', pathlib.Path(f).read_text())
    used |= {int(m) for m in re.findall(r'\(?S(\d+)[\),; ]', text)}
bad = sorted(s for s in used if s > n_sections)
print(f"  {'FAIL' if bad else 'OK  '} {len(used)} pointers used, supplement has "
      f"{n_sections} sections" + (f"; dangling: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && FAIL=1

note "Released audio sample is present and matches its manifest"
python3 - <<'PY'
import csv, pathlib, sys
root = pathlib.Path('data/audio_sample')
man = root / 'manifest.csv'
if not man.exists():
    print("  FAIL manifest.csv missing"); sys.exit(1)
rows = list(csv.DictReader(man.open()))
missing = [r['clip'] for r in rows if not (root / r['clip']).exists()]
mb = sum((root / r['clip']).stat().st_size for r in rows if (root / r['clip']).exists()) / 1e6
print(f"  {'FAIL' if missing else 'OK  '} {len(rows)} clips, {mb:.1f} MB"
      + (f"; {len(missing)} missing" if missing else ""))
sys.exit(1 if missing else 0)
PY
[ $? -ne 0 ] && FAIL=1

note "Working tree is committed"
# Build outputs are excluded: this script rebuilds both PDFs a few lines above,
# so including them would make the check unpassable by construction. What must
# be committed is the source that produced them.
DIRTY=$(git status --porcelain -- . ':(exclude)paper/build' \
        ':(exclude)paper/supplementary/build' | wc -l)
if [ "$DIRTY" -eq 0 ]; then
  ok "sources clean at $(git rev-parse --short HEAD)"
else
  bad "uncommitted source changes: $DIRTY files"
  git status --porcelain -- . ':(exclude)paper/build' \
      ':(exclude)paper/supplementary/build' | sed 's/^/       /'
fi

printf '\n'
if [ $FAIL -eq 0 ]; then
  echo "VERIFIED: submission is consistent."
else
  echo "NOT VERIFIED: fix the failures above."
fi
exit $FAIL
