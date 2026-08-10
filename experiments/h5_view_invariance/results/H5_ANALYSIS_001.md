# H5 analysis 001 — score-free paired view-invariance atlas

Status: complete registered matrix with explicit degenerate cells retained.
This is a terminal descriptive measurement-agreement result. It reads neither
labels nor detector scores, audio, models, ASR, or prior-result tables.

The frozen-manifest-only analyzer revalidated all five H5 source hashes and
the independent seed-2610 identity selection before projecting the registered
feature columns. It materialized all 420 planned
`5 corpora x 3 view pairs x 28 features` cells and all 84 registered
view-pair/feature aggregations, using the locked 200 paired-bootstrap attempts
per valid cell.

## Registered result

- 414/420 cells completed with finite paired values and estimates.
- 6/420 cells are explicit `failed_zero_or_nonfinite_pooled_iqr` cells, all in
  InTheWild: `silence_fraction` and `clipping_fraction` for each of the three
  view pairs. They have zero finite paired samples and were neither imputed nor
  rerun. The matrix remains complete because every registered cell is present.
- 17/84 aggregations meet the locked terminal descriptive view-stability rule;
  67/84 do not. Among the 78 aggregations with five successful dataset cells,
  19 meet the point concordance condition, 32 meet the concordance-CI
  condition, and 30 meet the normalized-shift-CI condition; all three are
  required jointly.

The 17 terminal entries comprise 14 `deterministic_crop`--`full_waveform`
measurements (`rms_dbfs`, `integrated_lufs`, eight spectral measures,
`f0_median_hz`, `hnr_db`, `cpp_proxy_db`, and `group_delay_var`), two
`deterministic_crop`--`preemphasized_crop` entries
(`group_delay_var`, `inst_freq_dispersion`), and one
`full_waveform`--`preemphasized_crop` entry (`group_delay_var`). For example,
the median rank concordance for the stable
`deterministic_crop`--`full_waveform` `integrated_lufs` entry is 0.980952,
with median absolute normalized shift 0.000000. This describes the registered
feature measurements across these precise views only; it is not a feature
ranking, label result, detector result, preprocessing recommendation, or
causal claim.

## HDD ledger

All large results remain under
`/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h5_view_invariance/h5_analysis_001/`.

| Artifact | SHA-256 |
|---|---|
| `h5_view_invariance_matrix.csv` | `a6ea23155943aa38327b5c65da0422c6677741c8d678581ae1b99e93d1fd8438` |
| `h5_view_invariance_aggregation.csv` | `f1cb7c85ad291ab6fcaac0fc5d25c91787b6accc9ca8e33ed6f6fd613216bdb6` |
| input-freeze provenance | `8b6ef2a45acf5f2cda047c11989ac65d18acf0d37aa6c9104bb6dec68b7fe960` |
| selection manifest | `cb1dbd986677dd2ef02f3f0c92661569afa061f2b3acab5f08d29f6f9df7ab51` |

The analysis provenance records `matrix_complete: true`,
`analysis_status: failed_cells_present`, `failed_cell_count: 6`, and no
score/label access. H5 is sealed: do not change its views, feature list, seed,
sample cap, bootstrap count, or descriptive rule after this result. Any
visualization must use a separately locked, result-preserving protocol.
