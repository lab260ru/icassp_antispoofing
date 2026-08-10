# H5 input freeze 001 — independent label-free view panel

Status: complete identity/view-only freeze. No feature value, label, score,
model, audio, ASR, detector, or prior-result table was read by this stage.

The H5 freeze used the locked five literal v1_28 feature products, exact
sample_id/view projection, seed 2610, and a 10,000-sample cap per corpus. All
five corpora selected 10,000 sample IDs, each with exactly one row for all three
registered views; no integrity failure occurred. This selection is independent
of all H1/H2/H2B/H4 manifests.

| Dataset | Selected IDs | Input bytes | Input SHA-256 |
|---|---:|---:|---|
| ASVspoof2019_LA | 10,000 | 7,112,074 | `880bd61d77256bbed76991de3ac95e522c9b2e6df9c8d211cbcc25ca93031200` |
| ASVspoof2021_LA | 10,000 | 6,976,663 | `d08be3c0c2b8cd6f473ba149dae07c07600b8d9cfec3480b67b86523bdfe94b7` |
| ASVspoof2021_DF | 10,000 | 6,986,681 | `d344f7361ebe58bccfe06d666e5e3fb5064731116db251a7f564df456f9f4c5c` |
| InTheWild | 10,000 | 22,064,025 | `23c33c474d086e96af84122730766157cd2ee7ea3e95766fd71fbc69204cd8ee` |
| ASVspoof5 | 10,000 | 7,230,837 | `d7dcbb1e117f996eb1d22f156a25e6df1dc45e1f5130dabd33a881350bbb94aa` |

## HDD ledger

| Artifact | SHA-256 |
|---|---|
| `runs/h5_view_invariance/h5_input_freeze_001/h5_input_selection_manifest.parquet` | `cb1dbd986677dd2ef02f3f0c92661569afa061f2b3acab5f08d29f6f9df7ab51` |
| `runs/h5_view_invariance/h5_input_freeze_001/h5_input_freeze_provenance.json` | `8b6ef2a45acf5f2cda047c11989ac65d18acf0d37aa6c9104bb6dec68b7fe960` |

Only a committed, hash-revalidated invocation of the locked H5 analyzer may
read the feature projection next. The final 420-cell result is not yet known.
