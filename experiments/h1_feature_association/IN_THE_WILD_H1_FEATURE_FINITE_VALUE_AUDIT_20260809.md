# InTheWild H1 feature finite-value audit

## Scope

This immutable diagnostic records the finite-value state of the completed
InTheWild H1 feature artifact before association analysis. It is a data-quality
inventory only: no score artifact was read, no association was computed, and
the source parquet was not modified.

- Source: `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/features/v1_28/InTheWild/features_wide.parquet`
- SHA-256: `23c33c474d086e96af84122730766157cd2ee7ea3e95766fd71fbc69204cd8ee`
- Registry: `src.audio_features.FEATURE_NAMES` (`v1_28`, 28 H1 features)
- Rows: 95,337 total; 31,779 each for `deterministic_crop`,
  `full_waveform`, and `preemphasized_crop`

For every feature and view, the audit converted its feature column to
`float64` and counted `NaN`, `+Inf`, and `-Inf`; `nonfinite` is their sum.

## Affected H1 features

No `+Inf` values occurred. The entries below are every feature/view pair with
at least one non-finite value. Counts are `NaN / +Inf / -Inf / nonfinite`.

| Feature | Deterministic crop | Full waveform | Pre-emphasized crop | Total non-finite |
|---|---:|---:|---:|---:|
| `integrated_lufs` | 0 / 0 / 1 / 1 | 0 / 0 / 1 / 1 | 0 / 0 / 38 / 38 | 40 |
| `f0_median_hz` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `f0_iqr_hz` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `hnr_db` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `cpp_proxy_db` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `jitter_proxy` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `shimmer_proxy_db` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |
| `f0_delta_hz` | 554 / 0 / 0 / 554 | 577 / 0 / 0 / 577 | 6,097 / 0 / 0 / 6,097 | 7,228 |

The other 20 H1 registry features were finite in all 95,337 rows. This note
does not infer causes, data validity, or any scientific effect from these
counts; it merely fixes the observed input state for reproducible downstream
handling.
