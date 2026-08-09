# Post-author-policy paper-review status

Attempted: 2026-08-09, against the compiled four-page
`paper/build/main.pdf` after correcting the draft to reflect ICASSP 2027's
single-anonymous author policy.

The required `paper-review` launcher extracted 16,288 characters from the PDF
and started three independent Sonnet reviewers through the supported launcher:
`alfa` at t+0 s, `bravo` at t+10 s, and `charlie` at t+20 s. This satisfies the
required parallel-launch check.

All three exited with code 1 before producing a structured review. Their
35-byte outputs are identical:

```text
Not logged in · Please run /login
```

Consequently this bundle contains no verdict, concerns table, meta-review, or
peer-review recommendation. It is an authentication failure of the local Claude
reviewer runtime, not a positive or negative external review. Do not represent
the paper as peer reviewed. Once `claude /login` is available, create a fresh
timestamped review directory and rerun the three-reviewer workflow.
