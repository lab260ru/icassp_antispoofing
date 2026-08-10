# H6 supplementary figure protocol — locked compact display

Status: **implementation protocol, to be committed before any real H6 compact
artifact is opened or rendered.** This protocol is downstream of the completed
H6 analysis and creates no new statistical result.

## Purpose and interpretation boundary

The figure is a supplementary display of the existing H6 within-class Spearman
rank-agreement matrix. It shows all registered corpus × class × unordered
model-pair cells. It is descriptive only: neither a high nor a low cell is a
model-quality ranking, model selection criterion, statement of architectural
independence, cue reliance, intervention sensitivity, or causal effect.

No number in the figure is used to reopen H1/H2/H2B/H3/H4/H5/H6, choose a
feature, or choose a training experiment. The implementation refuses an
incomplete H6 matrix rather than creating a plausible partial display.

## Exact sealed input contract

The renderer accepts all and only these literal absolute paths, located under
one `h6_analysis_001` directory:

| Role | Filename | Required SHA-256 |
| --- | --- | --- |
| Matrix | `h6_within_class_score_agreement_matrix.csv` | `105e99fca829af8bf1600fa73374ddcde0f4121b6716a8beff0b1fd503eec2e5` |
| Aggregation | `h6_model_pair_agreement_summary.csv` | `294faa27e8e9e7cce4292c0b5f9e05bbfb2667e88e1bb719113f81914a0326e8` |
| Analysis provenance | `h6_analysis_provenance.json` | `3e9d4f628fdffea5430859db589c32dce3ce5e9ce7c9b1b2448928ae4bd433d8` |

The fixed absolute parent is
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h6_score_agreement/h6_analysis_001`.
The code rejects relative, symlinked, substituted, missing, wrong-name, or
wrong-hash inputs. It requires the exact matrix and summary schemas and rejects
raw-response fields (for example `raw_score`, `score_a`, `score_b`, or
`logit`), so neither raw model responses nor a response-derived substitute can
enter the plotting path.

Before rendering, the contract requires all of the following:

- exactly 280 unique cells: five fixed corpora × two fixed classes × 28 fixed
  unordered model pairs;
- every cell has `cell_status=ok`, exactly 5,000 selected/available/joined
  samples, 200 registered bootstrap attempts, a finite agreement and interval
  in `[-1, 1]`, and the deterministic per-cell bootstrap seed;
- exactly 28 pair summaries, each recomputed from the ten fixed compact matrix
  cells and checked field-for-field;
- the analysis provenance is itself hash-locked and declares a completed
  all-ok 280-cell matrix, zero unavailable cells, compact-output identities,
  `raw_score_only=true`, and `score_values_emitted=false`.

The renderer never opens raw score artifacts, freeze manifests, dataset labels,
audio/waveforms, features, ASR, model weights/code, detector runners, or any
other result table.

## Registered visual design

- **Chart:** one 28-row × 10-column numerical heatmap, selected because the
  completed artifact is a fixed rectangular grid.
- **Rows:** the registered `MODEL_PAIRS` order from H6: combinations of the
  fixed eight-model registry in its existing order. No value-based sort,
  grouping, filtering, marker, or highlight is permitted.
- **Columns:** the fixed five-corpus order, with `bona fide` (label 0) then
  `spoof` (label 1) for each corpus. Thin pair boundaries distinguish corpora;
  there is no value-based rearrangement.
- **Values:** one completed `spearman_agreement` per cell. There are no cell
  annotations because 280 decimal labels would not be legible; the immutable
  compact CSV remains the value-level record.
- **Scale:** fixed `[-1, 1]` blue–neutral–vermillion diverging colour map
  (`#0072B2`, `#F7F7F7`, `#D55E00`) with a labelled horizontal colour bar. It
  must not be rescaled to observed extrema.
- **Typography and outputs:** serif font family, vector PDF plus 300-DPI PNG,
  and metadata that records the input/output hashes, fixed orders, scale, and
  the no-selection interpretation.

## Render and inspection plan

After this implementation is committed, render into a new non-overwritable HDD
directory. Check the command’s three outputs, inspect both PDF and PNG for
legible axis labels, complete 28 × 10 cell geometry, visible colour bar endpoints,
and readable descriptive caveat. Record hashes and this visual inspection in a
new supplementary-figure note. Do not alter values, row order, column order,
scale, colour meaning, or interpretation after inspection.

The intended command is documented in `H6_SUPPLEMENTARY_FIGURE_CLI.md` and
has no inputs beyond the three literal sealed compact paths.
