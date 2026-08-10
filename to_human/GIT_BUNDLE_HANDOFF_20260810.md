# Portable Git-bundle handoff — 2026-08-10

A complete, verified Git bundle of the pushed research branch was created on
the designated HDD during the final 20:00-UTC handoff window. It is an
offline recovery artifact, not a substitute for the normal GitHub push.

| Field | Value |
| --- | --- |
| Bundle path | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/icassp_antispoofing_research_icassp_signal_audit_20260810T193538Z.bundle` |
| SHA-256 | `94635fada2433a7208973bbc2318371c788984e6550562ba6d22a40e53b7caec` |
| Bytes | `9,272,188` |
| Bundle head | `8892b7931344b79351b8b1e22d006be92b574456` (`research/icassp-signal-audit`) |
| Verification | `git bundle verify` passed; Git reports complete history. |

To inspect or restore it without changing the current workspace:

```bash
git clone /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/icassp_antispoofing_research_icassp_signal_audit_20260810T193538Z.bundle /tmp/icassp_bundle_restore
git -C /tmp/icassp_bundle_restore switch research/icassp-signal-audit
```

The normal remote remains the primary collaboration path. When continuing,
first inspect `git status --short --branch`, then push the ordinary local branch
without force.
