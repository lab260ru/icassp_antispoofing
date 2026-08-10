# Portable Git-bundle handoff — 2026-08-10

A complete, verified Git bundle of the pushed research branch was created on
the designated HDD after the final H6 supplementary-display delivery
checkpoint. It is an offline recovery artifact, not a substitute for the
normal GitHub push.

| Field | Value |
| --- | --- |
| Bundle path | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/icassp_antispoofing_research_icassp_signal_audit_20260810T143520Z.bundle` |
| SHA-256 | `1e24dd935ef951684044564389dc57ec158b8e00902c5ca23275631eea523df6` |
| Bytes | `2,850,849` |
| Bundle head | `f343c5afb0b563551f2ffb5da27c4c2fb08a9d2b` (`research/icassp-signal-audit`) |
| Verification | `git bundle verify` passed; Git reports complete history. |

To inspect or restore it without changing the current workspace:

```bash
git clone /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/icassp_antispoofing_research_icassp_signal_audit_20260810T143520Z.bundle /tmp/icassp_bundle_restore
git -C /tmp/icassp_bundle_restore switch research/icassp-signal-audit
```

The normal remote remains the primary collaboration path. When continuing,
first inspect `git status --short --branch`, then push the ordinary local branch
without force.
