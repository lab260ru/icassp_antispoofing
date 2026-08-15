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

note "The figure is current with the data it plots"
# build.sh regenerates every number from its result JSONs, but not the figure --
# fig_main.pdf is a committed artifact, so it can silently fall behind. It did:
# it was cut before Qwen3-TTS-1.7B joined the period ladder, so panel (b) showed
# three curves for two days while the text beside it said four checkpoints. A
# reader counting lines in the figure would have caught us; nothing here would.
# PDFs embed timestamps, so bytes cannot be compared -- this compares the drawing
# content streams, which change only when what is drawn changes.
python3 - <<'PY'
import subprocess, sys, tempfile, pathlib, pypdf
fig = pathlib.Path('paper/figs/fig_main.pdf')
def content(p):
    return b"".join(pg.get_contents().get_data() for pg in pypdf.PdfReader(str(p)).pages)
if not fig.exists():
    print("  FAIL paper/figs/fig_main.pdf is missing"); sys.exit(1)
# Redraw into a temporary directory, never over the tree. figures.py writes six
# PDFs and every one gets a fresh timestamp, so running it in place left all six
# modified and made this script fail its own "working tree is committed" check --
# a verification script that mutates the repository is one nobody can trust.
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, 'analysis/figures.py', '--outdir', td],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL analysis/figures.py failed:\n{r.stderr[-400:]}"); sys.exit(1)
    fresh = pathlib.Path(td) / 'fig_main.pdf'
    if not fresh.exists():
        print("  FAIL analysis/figures.py did not write fig_main.pdf"); sys.exit(1)
    if content(fresh) != content(fig):
        print("  FAIL fig_main.pdf is stale: regenerating it changes what is drawn. "
              "Run analysis/figures.py and commit the result.")
        sys.exit(1)
print("  OK   fig_main.pdf redraws identically from current data")
PY
[ $? -ne 0 ] && FAIL=1

note "Pre-committed scoring rules have not moved"
python3 scripts/check_precommit_frozen.py || FAIL=1

note "The exclusion rules are blind to the comparison they feed"
python3 scripts/check_exclusions_blind.py || FAIL=1

note "Supplement tables have not gone stale against their result JSONs"
python3 scripts/check_supp_tables.py || FAIL=1

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

note "No page prints text past the bottom"
python3 scripts/check_no_overrun.py || FAIL=1

note "No unresolved citations in either document"
python3 scripts/check_citations_resolve.py || FAIL=1

note "Supplement pointers land on the section they mean"
python3 scripts/check_pointer_targets.py || FAIL=1

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

note "Every back-pointer from the supplement into the paper resolves"
# The check above runs one way only: paper -> supplement. The other direction
# went unchecked and rotted. The body was restructured when the theorem moved
# out of it and periodicity became the headline, and eleven pointers in the
# supplement still named sections from the old numbering -- Section 4.2 for the
# dissociation, 4.3 for the capacity measurement, 4.4 for the negative result,
# none of which exist. A reader following any of them lands somewhere else, and
# nothing in the build complains, because LaTeX never sees these: they are
# hand-typed numerals in one document naming sections in another.
python3 - <<'PY'
import re, pathlib, sys
# \input must be expanded in place: results_body.tex is pulled in *inside*
# Section 3, so concatenating the files in file order would attribute its
# subsections to whatever section came last in main.tex.
def expand(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.match(r'\s*\\input\{([^}]+)\}', line)
        if m:
            out.append(expand(f"paper/{m.group(1)}.tex"))
        else:
            out.append(line)
    return "\n".join(out)
body = expand('paper/main.tex')
# Section numbers the built paper actually has: \section order, and \subsection
# order within the section that contains them.
sections, subs, sec_i, sub_i = [], set(), 0, 0
for line in body.splitlines():
    if line.startswith(r'\section{'):
        sec_i += 1; sub_i = 0; sections.append(sec_i)
    elif line.startswith(r'\subsection{'):
        sub_i += 1; subs.add(f"{sec_i}.{sub_i}")
have = {str(s) for s in sections} | subs
supp = pathlib.Path('paper/supplementary/supp.tex').read_text()
# Every LITERAL `Section~N` in the supplement is by construction a pointer into
# the main paper: the supplement's own cross-references are all \ref{sec:...},
# which LaTeX resolves and would error on. So no context heuristic is needed,
# and none should be used -- an earlier version filtered on the phrase "main
# paper" appearing nearby and silently checked 3 of the 14 pointers.
cited = set(re.findall(r'Section~(\d+(?:\.\d+)?)', supp))
bad = sorted(cited - have, key=lambda s: [int(x) for x in s.split('.')])
print(f"  {'FAIL' if bad else 'OK  '} {len(cited)} back-pointers, paper has "
      f"{sorted(have)}" + (f"; dangling: {bad}" if bad else ""))
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

note "The availability tag in the paper points at what the paper reports"
# main.tex names a git tag as the artifact readers will fetch. Nothing else in
# this suite looks at it, so the tag can silently go stale: it was cut before
# the period ladder and the causal replication landed, and a submission built
# today would have pointed a reviewer at a tree missing both experiments it
# cites. Staleness here is worse than a broken build, because it fails only for
# the reader. This is a WARNING and not a hard failure so the check is usable
# mid-iteration; the last action before submission is to re-cut the tag.
TAG=$(grep -o 'texttt{v[0-9][^}]*}' paper/main.tex | head -1 | sed 's/texttt{//;s/}//')
if [ -z "$TAG" ]; then
  bad "no availability tag found in paper/main.tex"
elif ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  # A warning, not a failure, and deliberately so. The tag is renamed in
  # main.tex *before* it is cut -- the paper's title changed from "locating" to
  # "isolating", so v1.2-locating had to be replaced rather than moved, since a
  # published tag cannot be repointed without breaking whoever fetched it. That
  # leaves a window, which is this one, where the paper names a tag that does
  # not exist yet. scripts/cut_release_tag.sh closes it and refuses to run
  # unless this whole suite is green first.
  printf '  WARN paper/main.tex cites tag %s, not cut yet -- run scripts/cut_release_tag.sh --push\n' "$TAG"
elif [ "$(git rev-list -n1 "$TAG")" = "$(git rev-parse HEAD)" ]; then
  ok "tag $TAG is at HEAD"
else
  BEHIND=$(git rev-list --count "$TAG..HEAD")
  printf '  WARN tag %s is %s commits behind HEAD -- re-cut it before submitting\n' \
         "$TAG" "$BEHIND"
fi

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
