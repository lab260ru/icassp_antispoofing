
## 2026-08-12 (final) — third probe checkpoint: 1 of 3, and a threshold that nearly lied

Llasa-3B, run because two checkpoints is not a test, does not lose the count
past the horizon either.

    checkpoint  arm                    MAE   constant   ratio    R²
    Llasa-1B    repeated k=48..128    0.505    0.500     1.01   −0.02
    Llasa-3B    repeated k=48..128    0.494    0.500     0.99   −0.02
    Llasa-8B    repeated k=48..128    0.432    0.500     0.86   +0.19
    Llasa-1B    control  k=48..128    0.431    0.500     0.86   +0.26

**The threshold nearly told a lie in our favour.** "Beats the constant predictor"
put Llasa-3B on the *keeps the count* side by 0.006 MAE on twelve items. Read
naively that is "2 of 3 checkpoints keep the count", which would have made the
theorem look worse — but read the other way ("2 of 3 show nothing recoverable")
it makes the theorem look *better*, and both readings were available from the
same number. The script now calls anything within 2% of the constant predictor
indistinguishable and prints both counts.

**We quote 1 of 3**, the less favourable one. S12 records that the more
favourable reading (2 of 3 null by effect size) exists, so a reader weighs it
rather than discovers it.

The chronology (S16) dates the third checkpoint at 14:50, launched with the
1-of-2 split already in the paper — so its result could only confirm or further
weaken a claim already reported at its weakest. It weakened it.

*If you add a fourth checkpoint:* every macro and verdict string derives from the
lost/kept/tied lists, so the paper re-words itself. Check `PhKeptRatios` reads
sensibly and that S12's hand-written table gets the new row —
`scripts/check_supp_tables.py` will fail if you forget.
