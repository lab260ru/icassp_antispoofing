# Codex template and provenance review

Scope: read-only review of the current paper source/PDF, vendored ICASSP-2026
template inputs, and submission documentation.

Passes: the pinned spconf.sty and IEEEbib.bst files byte-match the supplied ZIP;
the ZIP hash matches its HDD archival copy; main.tex uses the spconf API and
authorized dual affiliation; and the final PDF embeds all detected fonts.

The reviewer found that an earlier build had stale IEEEtran bibliography
intermediates. The exact rebuildable aux/bbl files were removed, the paper was
rebuilt, and the generated auxiliary record now identifies
template/ICASSP2026/IEEEbib. Intermediate files were then removed again. The
review also found a false local landmark match caused by the phrase stable IDs.
The checker now requires an exact Table 1/I label on page 3 and References on
page 5. The review additionally requested a fully capitalized title, affiliation
spacing, updated paper README, and committed template provenance; all are now
present.
