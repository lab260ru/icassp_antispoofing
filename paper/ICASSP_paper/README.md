# ICASSP_paper — the rewrite workspace

The complete rewrite of the paper happens here, in two folders:

- `main/` — the rewritten paper. `main.tex` is single-file (the old
  results/discussion bodies are inlined); `bash main/build.sh` compiles it
  with tectonic into `main/build/`.
- `supplement/` — a snapshot of the submitted supplement (`supp.tex`),
  compiling standalone via `bash supplement/build.sh`. Not yet rewritten.

**The rule:** everything under `paper/` *outside* this folder —
`paper/main.tex`, `paper/results_body.tex`, `paper/discussion_body.tex`,
`paper/table1.tex`, `paper/supplementary/` — is the submitted record (tag
`v2.2-isolating`, commit `1ab4d9f`) and is frozen. Do not edit it.

What the rewrite changed (2026-08-16):

- **Anonymous** (double-blind): no authors, no repository URL; the
  companion-paper framing is now a third-person citation of Viakhirev et al.
- **No theorem, no Lean.** The mechanism appears once, in the discussion, as
  a tried-and-rejected approach attributed to Viakhirev et al., with the
  measured q range.
- **No supplement pointers** (S3, S14, …) in the body; no "stimuli"
  (now "benchmark"/"items").
- **Shorter abstract**, compact Method with a self-contained Statistics
  paragraph (what the brackets are, what is resampled), $\hat c$ defined at
  first use, split paragraphs throughout.
- **Two more tables**: every non-panel system (Table 2), the period ladder
  (Table 3), and the gap under decoding/analysis variations (Table 4).
- **Figure redrawn** (`main/make_fig.py`, run from the repo root): shared
  checkpoint legend above both panels, CVD-validated family-hue palette
  (blues = Llasa, green = XTTS-v2, warm = Qwen; darker = larger), and panel
  (a) now draws the same population every reported number uses (the
  submitted figure drew unfiltered rows, which contradicted its own
  caption at low k).

Discipline carried over: every measured number is a macro from
`main/numbers.tex` (a frozen copy of the generated file — to refresh, run
`python3 analysis/make_numbers.py` at the repo root and copy
`paper/numbers.tex` here deliberately). `main/refs.bib` is deduplicated
(two conflicting duplicate entries the submitted bib carried are removed).

The verification suite (`scripts/verify_submission.sh`) still guards the
frozen originals only; it says nothing about drafts here.

## ICASSP 2027 paper-kit conformance (2026-09-07)

Checked against the official kit
(`cmsworkshops.com/ICASSP2027/papers/paper_kit.php`), the SPS policy page, and
the shipped `Template.tex` / `Template.pdf`. Our `spconf.sty` is byte-identical
to the official one. `main/build.sh` emits **`main/build/borodin.pdf`** — the
kit wants the file named after the first author's last name — and that is the
file to upload.

Five mismatches, all fixed:

- **Review is non-blind.** "ICASSP does not perform blind reviews, so be sure
  to include the author list in your submitted paper." The r2 rewrite was
  anonymous; the author block is back (Borodin, Kudryavtsev, Mkrtchian /
  BitmanagerAI), and the code link is no longer withheld. *New for 2027: every
  author needs an ORCiD ID — entered on the submission site, not in the PDF.*
- **Compliance with Ethical Standards** is required "irrespective of whether
  ethical approval was needed". Added as an unnumbered section before the
  bibliography. spconf redefines `\@sect` but not `\@ssect`, so `\section*`
  came out `\Large` and left-aligned; the preamble patches `\@ssect` to match.
- **9pt floor** ("no smaller than 9 points throughout the paper, including
  figure captions"). Violated three ways: nine tables at `\scriptsize`
  (7.3pt), `\footnotesize` footnotes (8pt), and figure text at 5-6pt.
- **Abstract** was 178 words against the kit's 100-150. Now 145.
- **Filename** was `main.pdf`.

### Type sizes: 9pt is the floor, not the target

The first pass fixed the 9pt violations with `\ninept` (spconf's 9pt mode),
which fits easily but sets the *whole body* at the minimum. That was wrong:
the official `Template.tex` ships `\ninept` **commented out** (line 42,
`%\ninept`), and `Template.pdf` measures a 10pt body. 10pt is ICASSP's
default; 9pt is only the floor.

So the body stays at the template's 10pt, and the space comes from four
changes that touch no content:

- `\vfill\pagebreak` before the references removed — it breaks the *column*,
  and at 10pt it was discarding the bottom 45% of page 5's left column.
- References at 9pt (`\tabsize`), the usual IEEE setting.
- Captions at 9pt. spconf's `\@makecaption` hardcodes `\vskip 10pt` above
  every caption, which is why `\abovecaptionskip` had no effect; it is
  redefined to honour it. Ten floats paid that twice over.
- `\baselinestretch{.95}`, the same stretch `\ninept` itself uses, applied
  *after* `\maketitle` so the title block keeps the template's exact geometry
  (title-page ink top measures 34.4mm, identical to `Template.pdf`). The kit
  asks for 7-8 lines/inch; article's 10/12 leading gives 6.1, so this moves
  toward the stated spec rather than away from it.

Result: 10pt body, 9pt tables/captions/references, 5 pages, no overfull boxes,
body ends on page 4, page 5 carries only the ethics statement and references.
Every table and the figure are intact. The only sub-9pt glyphs left are 12
mathematical sub/superscripts and footnote reference marks.

Figure changes that came with the 9pt rule: canvas grew from 246x92pt to
243.78x170pt (243.78pt is spconf's exact `\columnwidth`, so `width=\columnwidth`
scales by 1.0 and 9pt stays 9pt on the page); `FS = 9.0` throughout
`make_fig.py`; panel (a)'s arm key moved out of the plot area into its own
legend row; the two italic verdict labels came out, since the caption already
states both.

`table1.tex` lost its `size/type` column (`size` remains) — at 9pt it ran
40.8pt past the column, and the non-AR mechanism split it carried is already
stated in the §3.4 prose.

Deliberately **not** changed: the figure separates its six checkpoints by hue,
which collapses in greyscale (`rgb(176,69,0)` and `rgb(0,151,109)` are 8 grey
levels apart) — the kit asks that colour illustrations print clearly in black
and white. The six pointers to "supplementary material" also stay, though the
kit describes no supplementary channel.
