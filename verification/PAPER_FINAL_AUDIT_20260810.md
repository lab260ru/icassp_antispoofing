# Final Codex-audit paper verification — 2026-08-10

## Scope

This verifies the working paper after the Codex-only final audit. It is a local
reproducibility/readability check, not an official ICASSP-2027 template or IEEE
PDF eXpress certification.

## Build

`main.tex` was rebuilt after the evidence-boundary and methods-reporting edits:

```bash
cd paper && tectonic --outdir build main.tex
cd paper && tectonic --outdir build main.tex
```

The generated readable PDF is `paper/build/main.pdf`.

## Static preflight

```bash
PYTHONPATH=. python3 scripts/check_paper_pdf.py \
  --pdf paper/build/main.pdf \
  --review-stage single-anonymous-submission
```

Result: pass. The PDF has five US-letter pages, Table 1 on page 3, References
on page 5, and ten embedded font resources. Its named-author strings are
expected and allowed in `single-anonymous-submission` mode.

## Identity

- PDF SHA-256: `5077102e4467782f2c39c41ff29312acef4ad4c270bd9b1798eca93d0f22d094`
- Template/BST source: pinned user-provided ICASSP-2026 `spconf`/`IEEEbib`
  inputs under `paper/template/ICASSP2026/`.
- Internal reviews and resolution:
  `paper/reviews/codex_final_audit_20260810/`.

The paper remains a conservative H1 portability/H2 quality-gate result. H5,
H6, and S1 are sealed supplementary artifacts only; none was cited or used to
broaden the main-paper claim in this revision.
