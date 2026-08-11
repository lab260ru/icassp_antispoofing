# Citation verification ledger

This ledger is the paper-local record required before an entry is used in
`references.bib`. Each BibTeX record was fetched programmatically through its
DOI content-negotiation endpoint. The second source confirms existence,
metadata, and the limited claim attributed in `main.tex`.

| Citation key | DOI and BibTeX source | Independent primary/authoritative source | Claim used in the draft |
|---|---|---|---|
| `jung2022aasist` | Crossref metadata and `https://doi.org/10.1109/ICASSP43922.2022.9747766` with `Accept: application/x-bibtex` | arXiv API record `2110.01200` | AASIST uses integrated spectro-temporal graph attention. |
| `muller2021silence` | Crossref metadata and `https://doi.org/10.21437/ASVSPOOF.2021-9` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.html) | Silence duration can correlate with labels in ASVspoof data and can strongly discriminate. |
| `shim2023shortcut` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2023-1901` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2023/shim23b_interspeech.html) | Controlled shortcut settings can induce near-perfect or worse-than-chance countermeasures. |
| `todisco2019asvspoof` | Crossref metadata and `https://doi.org/10.21437/interspeech.2019-2249` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2019/todisco19_interspeech.html) | ASVspoof 2019 is a common logical-access spoofing research setting. |
| `yamagishi2021asvspoof` | Crossref metadata and `https://doi.org/10.21437/ASVSPOOF.2021-8` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/asvspoof_2021/yamagishi21_asvspoof.html) | ASVspoof 2021 adds channel/compression variability and unmatched evaluation conditions. |
| `muller2024harder` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2024-247` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.html) | The reported cross-domain gap is primarily distributional difference rather than only hardness. |
| `pascu2024generalizable` | Crossref metadata and `https://doi.org/10.21437/Interspeech.2024-1302` with `Accept: application/x-bibtex` | [ISCA Archive record](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html) | Frozen SSL representations with a simple classifier can generalize and calibrate strongly in the authors' benchmark. |
| `wang2024asvspoof5` | `https://doi.org/10.21437/ASVSPOOF.2024-1` with `Accept: application/x-bibtex` (retrieved 2026-08-10) | [ISCA Archive record](https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html) | Identifies the ASVspoof 5 held-out corpus used in the audit; the cited paper describes its dataset and challenge setup. |
| `yaroshchuk2023odss` | `https://doi.org/10.1109/WIFS58808.2023.10374863` with `Accept: application/x-bibtex` (retrieved 2026-08-11) | [ODSS dataset record](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ODSS) pinned in H9 | Establishes the public synthetic-speech source dataset used for H9; the paper uses only its documented pairing structure and independently hashes/excludes source rows. |
| `xie2021siamese` | `https://doi.org/10.21437/Interspeech.2021-847` with `Accept: application/x-bibtex` (retrieved 2026-08-11) | [ISCA Archive record](https://www.isca-archive.org/interspeech_2021/xie21_interspeech.html) | Prior pair-based anti-spoof representation learning; used only to distinguish H9 from a generic claim that pairwise learning is new. |
| `tran2024contrastive` | `https://doi.org/10.21437/Interspeech.2024-481` with `Accept: application/x-bibtex` (retrieved 2026-08-11) | [ISCA Archive record](https://www.isca-archive.org/interspeech_2024/tran24_interspeech.html) | Prior cross-entropy plus supervised-contrastive anti-spoofing; motivates the random-pair control rather than a novelty claim for ranking alone. |
| `dowerah2026arena` | `https://doi.org/10.1109/OJSP.2026.3652496` with `Accept: application/x-bibtex` (retrieved 2026-08-11) | [IDIAP publication record](https://publications.idiap.ch/publications/show/5849) | Identifies the released Speech DF Arena context from which H9 pins SONAR and ArAD revisions; no Arena score is used in H9. |
| `kim2025bpl` | `https://doi.org/10.21437/Interspeech.2025-422` with primary ISCA BibTeX metadata (verified 2026-08-11) | [ISCA Archive record](https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.html) | Closest bonafide positive-pair precedent: it aligns two augmentations of one bonafide waveform, rather than a natural--synthetic counterpart. |
| `borodin2024res2tcn` | `https://doi.org/10.48084/etasr.8906` with `Accept: application/x-bibtex` (retrieved 2026-08-11) | [publisher PDF](https://etasr.com/index.php/ETASR/article/download/8906/4310/37727) | Identifies the published Res2TCNGuard architecture. H9 fresh-initializes it and prohibits the released pretrained checkpoint. |

No unverified citation keys occur in `main.tex`. The model-card and Arena
artifact provenance are tracked in the repository data manifest rather than
represented as an academic bibliographic entry.
