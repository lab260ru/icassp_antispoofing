# H2 detector-free quality run 002 — gate failure

## Scope

`h2_quality_full_002` is the complete rerun of the frozen ASVspoof2019 LA
crest-factor panel after the transform-induced clipping correction. It applies
only the four committed arms to 1,000 score-independent clips and runs waveform
quality metrics plus pinned Whisper WER. No detector was imported, loaded, or
run.

## Completion and immutable HDD artifacts

- Run root:
  `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/runs/h2_causal_interventions/h2_quality_full_002/`
- Full panel: 4,000/4,000 pair checkpoints, with 1,000 per arm.
- Quality table SHA-256: `f261a1c918d481505aeedd91f904ce7a9b9dc5f5b3fcb9996a17d35475fe9b69`
- Quality provenance SHA-256: `60476acff9e903b29a659b97a7e0a321d20aa64fb0f922f25671bb9ccf1d13f1`
- Quality summary SHA-256: `4c5adad8165e97efe44ca7d8b1e8737eb34e0847a4ee08b114c0c0fd61ea7c8e`
- The compact committed `h2_quality_full_002.summary.json` preserves the
  detector-free run summary; the table, transcripts, and checkpoints remain on
  HDD.

## Predeclared arm gate outcome

The separate quality-freeze command was run against the complete table. It
correctly refused before creating a score-eligible manifest because `drc_cf3`
retained only 181/1,000 pairs (18.1%), below the locked 90% per-arm threshold.
The other arm outcomes are shown below.

| Arm | Retained pairs | Retained fraction | Freeze disposition |
|---|---:|---:|---|
| `drc_cf3` | 181/1,000 | 18.1% | fails 90% gate |
| `drc_cf6` | 6/1,000 | 0.6% | fails 90% gate |
| `small_gain_plus_0p1db` | 646/1,000 | 64.6% | fails 90% gate |
| `polarity` | 999/1,000 | 99.9% | individually passes, panel still blocked |

For transparency, the principal individual pass counts for `drc_cf3` are
STOI 309, WER 338, loudness 322, clipping 962, and target direction 761; for
`drc_cf6` they are 154, 101, 68, 974, and 761, respectively. The gain control
passes STOI/WER/loudness/target direction for 999/997/1,000/1,000 pairs but
clipping for only 650. These diagnostics identify quality-gate incompatibility
of this frozen arm set; they are not detector responses.

## Consequence

No `score_eligible_pairs` manifest exists. The paired scorer was not launched,
and no H2 score delta, causal sensitivity estimate, model ranking, or H3
training decision may be derived from this run. Do not loosen a threshold or
alter an arm under the same freeze. Any future transformation study needs a new
outer-loop protocol, candidate/family decision, score-independent input freeze,
and quality run.
