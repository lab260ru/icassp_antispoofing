# Portable Git-bundle handoff — 2026-08-09

Because the configured remote cannot authenticate from this environment, a
complete, verified Git bundle of `research/icassp-signal-audit` was created as
a portable backup. It is an offline recovery/delivery artifact, not evidence
of a successful GitHub push.

| Field | Value |
|---|---|
| Bundle path | `/home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/git-bundles/icassp_antispoofing_20260809T235303Z.bundle` |
| SHA-256 | `80ce7fc0954c83202b4c5a1ac493c19e3894d68b180201686238328e8dbe9777` |
| Bytes | `2,420,954` |
| Bundle head | `55e50e8427a7bb845bfc8ecda0d2194878d76bd6` (`research/icassp-signal-audit`) |
| Verification | `git bundle verify` passed; the bundle reports complete history. |

To inspect or restore it without changing the current workspace:

```bash
git clone /home/kirill/mnt/hdd_6tb_1/icassp_antispoofing/handoffs/git-bundles/icassp_antispoofing_20260809T235303Z.bundle /tmp/icassp_bundle_restore
git -C /tmp/icassp_bundle_restore switch research/icassp-signal-audit
```

When an authorized GitHub credential becomes available, push the normal local
branch rather than treating this bundle as a substitute:

```bash
git push origin research/icassp-signal-audit
```
