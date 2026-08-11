# H9-PCR source-only lambda selection 001

**Status:** complete source-only selection; target audio, labels, scores, and
metrics remain unread.

The sealed 12-fit P grid selected `lambda_rank=1.00` by the predeclared mean
ODSS development EER over seeds 9101--9104. It applies unchanged to the final
P and B2 conditions; B1 retains zero rank weight.

| Rank weight | Mean source-dev EER |
| --- | --- |
| 0.10 | 33.5954% |
| 0.30 | 25.9222% |
| **1.00** | **22.0512%** |

HDD artifact: `runs/h9_paired_counterfactual/h9pcr_lambda_selection_001/h9_pcr_source_only_lambda_selection.json`.
Its SHA-256 is `4a23365af4a43a678987fbd95e9609b3a4b9e4d89fc698addd1d30bf92e2cee3`.
It pins all 12 source-only sidecars, their source-artifact triple, the full
candidate table, and target-firewall booleans. This is source model selection,
not cross-corpus evidence or a paper claim.
