# H4 input freeze 001 — score-free five-corpus feature panel

**Status:** completed metadata/identity freeze. This step read only the five
protocol-allowed feature Parquet inputs and their identity, label, and view
fields. It did not compute a feature statistic; inspect a score artifact;
decode audio; initialize ASR; import a detector; or train a model.

## Frozen scope

The non-overwritable HDD freeze is
`runs/h4_label_transportability/h4_input_freeze_001`, created at
`2026-08-09T23:14:49Z`. The strict H4 runner projected only the permitted
identity/label/view and `v1_28` feature schema, rejected response-like paths
and columns, and recorded zero validation failures in every source table.

| Dataset | Input SHA-256 | Rows | Selected source IDs per label |
|---|---|---:|---:|
| ASVspoof2019_LA | `880bd61d77256bbed76991de3ac95e522c9b2e6df9c8d211cbcc25ca93031200` | 30,000 | 5,000 / 5,000 |
| ASVspoof2021_LA | `d08be3c0c2b8cd6f473ba149dae07c07600b8d9cfec3480b67b86523bdfe94b7` | 30,000 | 5,000 / 5,000 |
| ASVspoof2021_DF | `d344f7361ebe58bccfe06d666e5e3fb5064731116db251a7f564df456f9f4c5c` | 30,000 | 5,000 / 5,000 |
| InTheWild | `23c33c474d086e96af84122730766157cd2ee7ea3e95766fd71fbc69204cd8ee` | 95,337 | 5,000 / 5,000 |
| ASVspoof5 | `d7dcbb1e117f996eb1d22f156a25e6df1dc45e1f5130dabd33a881350bbb94aa` | 30,000 | 5,000 / 5,000 |

Selection is a deterministic SHA-256 ranking with seed 2609 and cap 5,000
distinct `source_id` values per `(dataset, label)`. The sealed selection has
50,000 source-cluster rows and retains all three registered waveform views for
each source.

## Artifact ledger

| Artifact | Location | Bytes | SHA-256 |
|---|---|---:|---|
| Selection manifest | HDD `runs/h4_label_transportability/h4_input_freeze_001/h4_input_selection_manifest.parquet` | 3,643,444 | `fe11423b4db7afe4c77790f0259fe9c68e587d3f61279c0139fd5509296ddfa7` |
| Freeze provenance | HDD `runs/h4_label_transportability/h4_input_freeze_001/h4_input_freeze_provenance.json` | — | `52526eaa68d7c6d6d39c8223b4f3fc7b4302696e7d8c409ba7fa32297f66ad93` |

The provenance binds the protocol SHA-256
`b1ad8d3b9551ac76e7524cf1c9ab7bbb0b9c602d4d54deaa03bc1bcd91506049`, the
allowed-column schema, original Parquet schema, byte sizes, label/view counts,
and selection-manifest hash. This committed hash record authorizes only the
frozen-manifest H4 analyzer; it does not authorize H1/H2/H2B/H3 selection or
detector access.
