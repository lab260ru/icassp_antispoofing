# Constraints

- Never repair a stopped loop by threshold relaxation, target replacement,
  model change, or post-result selection.
- Keep large datasets, model weights, checkpoints, predictions, and logs on
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/`; commit only compact
  manifests, hashes, code, notes, and selected paper artifacts.
- Use BF16 and qualify the fastest safe batch/worker setting before a training
  run. Preserve source-only selection evidence and every terminal prediction.
- Every change to `paper/main.tex` must rebuild `paper/build/main.pdf` in the
  same change.
- The user-provided `ICASSP2026_Paper_Templates.zip` is protected and must not
  be staged.
