# Citation verification ledger

This ledger is the paper-local record required before an entry is used in
`references.bib`. Each BibTeX record was fetched programmatically through the
DOI content-negotiation endpoint on 2026-08-09. The second source confirms
existence, metadata, and the limited claim attributed in `main.tex`.

| Citation key | DOI and BibTeX source | Independent primary/authoritative source | Claim used in the draft |
|---|---|---|---|
| `jung2022aasist` | Crossref metadata and `https://doi.org/10.1109/ICASSP43922.2022.9747766` with `Accept: application/x-bibtex` | arXiv API record `2110.01200` | AASIST uses integrated spectro-temporal graph attention. |
| `muller2021silence` | Crossref metadata and `https://doi.org/10.21437/ASVSPOOF.2021-9` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.html) | Silence duration can correlate with labels in ASVspoof data and can strongly discriminate. |
| `shim2023shortcut` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2023-1901` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2023/shim23b_interspeech.html) | Controlled shortcut settings can induce near-perfect or worse-than-chance countermeasures. |
| `todisco2019asvspoof` | Crossref metadata and `https://doi.org/10.21437/interspeech.2019-2249` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2019/todisco19_interspeech.html) | ASVspoof 2019 is a common logical-access spoofing research setting. |
| `yamagishi2021asvspoof` | Crossref metadata and `https://doi.org/10.21437/ASVSPOOF.2021-8` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/asvspoof_2021/yamagishi21_asvspoof.html) | ASVspoof 2021 adds channel/compression variability and unmatched evaluation conditions. |
| `muller2024harder` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2024-247` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.html) | The reported cross-domain gap is primarily distributional difference rather than only hardness. |
| `pascu2024generalizable` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2024-1302` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html) | Frozen SSL representations with a simple classifier can generalize and calibrate strongly in the authors' benchmark. |

No unverified citation keys occur in `main.tex`. The model-card and Arena
artifact provenance are tracked in the repository data manifest rather than
represented as an academic bibliographic entry.
