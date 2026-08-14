#!/usr/bin/env bash
# The last action before submission, scripted so it cannot be fumbled at 3am.
#
# main.tex names a git tag as the artifact a reader will fetch. Nothing in the
# build looks at it, so it has gone stale before -- it was cut before the period
# ladder and before the causal replication, and a submission built at that
# moment would have pointed a reviewer at a tree missing both experiments it
# cites. verify_submission.sh warns about staleness; this script fixes it.
#
# It is deliberately not part of verify_submission.sh: cutting and pushing a tag
# is not a check, and a verification script that mutates the repository is a
# verification script nobody can trust to run.
#
# What it does, in order, stopping at the first failure:
#   1. reads the tag name out of paper/main.tex rather than taking it as an
#      argument, so the tag and the paper cannot disagree by construction;
#   2. refuses to run on a dirty tree, since a tag pointing at a commit that
#      does not contain the built PDF's sources is the exact failure being
#      prevented;
#   3. runs the full verification suite and refuses if anything fails;
#   4. cuts an annotated tag at HEAD and pushes it.
#
# Usage:
#   bash scripts/cut_release_tag.sh            # dry run: says what it would do
#   bash scripts/cut_release_tag.sh --push     # actually cut and push
#
# To RENAME the tag (the current one is v1.2-locating, and the paper's title no
# longer says "locating"), edit the availability line in paper/main.tex first,
# commit that, then run this. The old tag is left in place: deleting a published
# tag breaks anyone who already fetched it, and an extra tag costs nothing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

fail() { printf '  FAIL %s\n' "$1"; exit 1; }
ok()   { printf '  OK   %s\n' "$1"; }

printf '\n=== The tag the paper names\n'
TAG=$(grep -o 'texttt{v[0-9][^}]*}' paper/main.tex | head -1 | sed 's/texttt{//;s/}//')
[ -n "$TAG" ] || fail "no availability tag found in paper/main.tex"
ok "paper/main.tex names $TAG"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  if [ "$(git rev-list -n1 "$TAG")" = "$(git rev-parse HEAD)" ]; then
    ok "$TAG already points at HEAD; nothing to do"
    exit 0
  fi
  BEHIND=$(git rev-list --count "$TAG..HEAD")
  printf '  NOTE %s exists and is %s commits behind HEAD.\n' "$TAG" "$BEHIND"
  printf '       A tag that has been published cannot be moved without breaking\n'
  printf '       whoever already fetched it. Rename it in paper/main.tex and\n'
  printf '       commit that change first, then re-run.\n'
  exit 1
fi

printf '\n=== Working tree\n'
DIRTY=$(git status --porcelain -- . ':(exclude)paper/build' \
        ':(exclude)paper/supplementary/build' | wc -l)
[ "$DIRTY" -eq 0 ] || fail "$DIRTY uncommitted source changes; commit them first"
ok "clean at $(git rev-parse --short HEAD)"

printf '\n=== Full verification\n'
if bash scripts/verify_submission.sh >/tmp/vs_tag.$$ 2>&1; then
  ok "verify_submission.sh passed"
else
  sed 's/^/       /' /tmp/vs_tag.$$ | grep -E 'FAIL|WARN' || true
  fail "verify_submission.sh failed; see /tmp/vs_tag.$$"
fi

printf '\n=== Cut\n'
MSG="$TAG: the tree the paper cites

Cut from $(git rev-parse --short HEAD) with verify_submission.sh green: Lean
artifact verified, every reported number generated rather than typed, supplement
tables checked against their result JSONs, main paper 5 pages with no spill,
every supplement pointer and back-pointer resolving, and the released audio
sample matching its manifest."

if [ "$PUSH" -eq 1 ]; then
  git tag -a "$TAG" -m "$MSG" || fail "git tag failed"
  git push origin "$TAG" || fail "git push failed"
  ok "cut and pushed $TAG at $(git rev-parse --short HEAD)"
else
  ok "dry run: would cut annotated tag $TAG at $(git rev-parse --short HEAD)"
  printf '       re-run with --push to do it\n'
fi
