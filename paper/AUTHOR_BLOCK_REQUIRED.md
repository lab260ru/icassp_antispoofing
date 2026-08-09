# Author block required for ICASSP 2027 submission

The official ICASSP 2027 editorial policy states that papers receive
**single-anonymous review**: reviewers know the authors' names, while reviewers
remain anonymous to authors. The anonymous block in `main.tex` is therefore an
internal working-draft placeholder, not a submission-ready author block.

Before submission, an authorized author must replace it with the confirmed
author names, affiliations, and ordering. Do not invent, infer, or commit
personal author information without authorization. A minimal IEEEtran
conference-mode structure is:

```tex
\author{\IEEEauthorblockN{First Author, Second Author}
\IEEEauthorblockA{Department or organization\\
City, Country\\
email@example.org}}
```

After insertion:

1. Compile `main.tex` and commit the refreshed `paper/build/main.pdf` in the
   same change.
2. Check the author names/order against the submission-system metadata; ICASSP
   lists those fields among initial compliance checks.
3. Run the local layout/font preflight in submission mode:

   ```bash
   PYTHONPATH=. python3 scripts/check_paper_pdf.py \
     --pdf paper/build/main.pdf \
     --review-stage single-anonymous-submission
   ```

4. Run the official template and conference/PDF validation once the approved
   archive and checker are accessible.

This document records a format requirement only; it does not identify the
paper's authors.
