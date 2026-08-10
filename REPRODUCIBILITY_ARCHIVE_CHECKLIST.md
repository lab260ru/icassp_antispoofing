# Pre-submission reproducibility archive checklist

## Status

This is a release-candidate checklist, not a public release, DOI, or
authorization to disclose restricted assets. It exists to turn the paper's
current hash-bound internal record into a reviewable, citable artifact only
after the owner approves the disclosure scope.

## Include in a versioned release

- This repository's tracked source, including `CITATION.cff`, `AGENTS.md`,
  `research-state.yaml`, protocols, result notes, compact CSV/JSON artifacts,
  paper source, compiled `paper/build/main.pdf`, and Codex-only internal
  reviews.
- `ARTIFACT_INDEX.md`, `verification/FINAL_VERIFICATION_20260810.md`, the
  recorded command lines, and all pinned input/result hashes.
- The user-provided ICASSP-2026 template provenance and its extracted pinned
  `spconf`/`IEEEbib` build inputs. Keep the original template ZIP external to
  Git unless its distribution terms are confirmed.

## Keep external or exclude

- Raw datasets, audio, model weights/checkpoints, full logs, cached downloads,
  and local feature Parquets on the HDD. Preserve their referenced paths,
  revisions, schemas, sizes, and SHA-256 values instead.
- `.env`, Telegram/GitHub credentials, chat IDs, keys, and any unredacted
  delivery payloads.
- Any artifact not explicitly vetted for its dataset/model license and the
  owner's disclosure policy.

## Before creating a release

1. Reconcile the supplied ICASSP-2026 `spconf`/`IEEEbib` inputs with the
   official ICASSP-2027 kit, if released; rebuild and run the official checker.
2. Run `PYTHONPATH=. python3 -m pytest -q`, compile `paper/main.tex`, and run
   `scripts/check_paper_pdf.py` in single-anonymous-submission mode.
3. Record the chosen immutable Git commit/tag, the PDF SHA-256, and hashes of
   every compact result artifact in a release manifest.
4. Choose an approved permanent archive (for example, a GitHub release plus a
   DOI-minting archive) and update the paper's availability statement with its
   immutable locator.
5. Verify that the release contains no large restricted asset or credential.

The current paper must not imply that such a public archival locator already
exists.
