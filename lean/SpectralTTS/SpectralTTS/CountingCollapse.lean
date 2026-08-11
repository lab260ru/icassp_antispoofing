/-
Counting Collapse in Autoregressive Decoders
============================================

Formal core of the paper "Counting Collapse: A Formally Verified Attractor
Theory of Repetition Hallucination in Autoregressive TTS".

Setting.  An autoregressive decoder generating speech for a text prompt that
contains a phrase repeated `k` times passes through a sequence of *repetition
boundary* states `s₀, F s₀, F² s₀, …`, where `F : S → S` is the composite update
the decoder applies over one full repetition of the phrase.  `S` is the decoder
state space (in practice ℝ^d with the Euclidean metric, which is a nonempty
complete metric space).

The modelling assumption is that `F` is the *same* map at every boundary.  This
is what Lemma B (`AttentionDilution.lean`) buys us: softmax attention over `k`
near-identical repeated text spans cannot tell occurrence `j` from occurrence
`j'`, so the conditioning the decoder receives at each boundary is the same up
to `O(δ + 1/k)`.

What is proved here.  *If* `F` is a contraction (`ContractingWith q F`, i.e.
`q < 1` and `F` is `q`-Lipschitz) then the number of repetitions that any
Lipschitz readout can distinguish is finite and bounded explicitly by
`log(2LC/margin) / log(1/q)`.

What is NOT proved here — and must not be claimed.  That a real transformer
decoder satisfies the contraction hypothesis.  `q` is an empirical quantity: the
accompanying experiments estimate it as `q̂` per model from the observed decay of
boundary-state distances.  The Lean artifact certifies the *implication*, not
the premise.
-/
import Mathlib

namespace SpectralTTS

open Filter Function

variable {S : Type*} [MetricSpace S] [CompleteSpace S] [Nonempty S]
variable {F : S → S} {q : NNReal}

/-- The a-priori radius of the boundary-state orbit: every iterate of `F`
starting at `s` stays within `orbitRadius F s q * q ^ n` of the fixed point.
Writing it as a named quantity keeps the later bounds readable; it is the `C` of
the paper. -/
noncomputable def orbitRadius (F : S → S) (s : S) (q : NNReal) : ℝ :=
  dist s (F s) / (1 - q)

theorem orbitRadius_nonneg (hF : ContractingWith q F) (s : S) :
    0 ≤ orbitRadius F s q :=
  div_nonneg dist_nonneg hF.one_sub_K_pos.le

/-- **Theorem A(i) — geometric convergence at repetition boundaries.**
The state after `n` repetitions is within `C q^n` of a single fixed point: the
decoder's record of how many repetitions it has produced decays geometrically.
This is `ContractingWith.apriori_dist_iterate_fixedPoint_le` restated in terms of
`orbitRadius`. -/
theorem dist_iterate_fixedPoint_le (hF : ContractingWith q F) (s : S) (n : ℕ) :
    dist (F^[n] s) (ContractingWith.fixedPoint F hF) ≤ orbitRadius F s q * (q : ℝ) ^ n := by
  have h := hF.apriori_dist_iterate_fixedPoint_le s n
  calc dist (F^[n] s) (ContractingWith.fixedPoint F hF)
      ≤ dist s (F s) * (q : ℝ) ^ n / (1 - q) := h
    _ = orbitRadius F s q * (q : ℝ) ^ n := by unfold orbitRadius; ring

/-- Two repetition counts are separated by at most `C (q^m + q^n)` in state
space.  Both orbits are pinned to the same fixed point, so their mutual distance
inherits the geometric decay. -/
theorem dist_iterate_iterate_le (hF : ContractingWith q F) (s : S) (m n : ℕ) :
    dist (F^[m] s) (F^[n] s) ≤ orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n) := by
  have hm := dist_iterate_fixedPoint_le hF s m
  have hn := dist_iterate_fixedPoint_le hF s n
  calc dist (F^[m] s) (F^[n] s)
      ≤ dist (F^[m] s) (ContractingWith.fixedPoint F hF)
        + dist (ContractingWith.fixedPoint F hF) (F^[n] s) := dist_triangle _ _ _
    _ = dist (F^[m] s) (ContractingWith.fixedPoint F hF)
        + dist (F^[n] s) (ContractingWith.fixedPoint F hF) := by rw [dist_comm _ (F^[n] s)]
    _ ≤ orbitRadius F s q * (q : ℝ) ^ m + orbitRadius F s q * (q : ℝ) ^ n := by gcongr
    _ = orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n) := by ring

/-- **Theorem A(ii) — eventual indistinguishability.**
For any tolerance `ε` there is a horizon `N` beyond which *all* repetition counts
produce states within `ε` of one another.  No amount of additional repetition
creates new state-space structure. -/
theorem exists_indistinguishable (hF : ContractingWith q F) (s : S) {ε : ℝ} (hε : 0 < ε) :
    ∃ N : ℕ, ∀ m n : ℕ, N ≤ m → N ≤ n → dist (F^[m] s) (F^[n] s) < ε := by
  set C := orbitRadius F s q with hC
  have hC0 : 0 ≤ C := orbitRadius_nonneg hF s
  have hden : 0 < 2 * C + 1 := by linarith
  obtain ⟨N, hN⟩ : ∃ N : ℕ, (q : ℝ) ^ N < ε / (2 * C + 1) :=
    exists_pow_lt_of_lt_one (by positivity) (by exact_mod_cast hF.1)
  refine ⟨N, fun m n hm hn => ?_⟩
  have hq0 : (0 : ℝ) ≤ (q : ℝ) := q.coe_nonneg
  have hq1 : (q : ℝ) ≤ 1 := le_of_lt (by exact_mod_cast hF.1)
  have hpm : (q : ℝ) ^ m ≤ (q : ℝ) ^ N := pow_le_pow_of_le_one hq0 hq1 hm
  have hpn : (q : ℝ) ^ n ≤ (q : ℝ) ^ N := pow_le_pow_of_le_one hq0 hq1 hn
  have hbound := dist_iterate_iterate_le hF s m n
  -- C (q^m + q^n)  ≤  2C q^N  ≤  2C ε/(2C+1)  <  ε
  have h1 : C * ((q : ℝ) ^ m + (q : ℝ) ^ n) ≤ 2 * C * (q : ℝ) ^ N := by
    have hsum : (q : ℝ) ^ m + (q : ℝ) ^ n ≤ 2 * (q : ℝ) ^ N := by linarith
    calc C * ((q : ℝ) ^ m + (q : ℝ) ^ n)
        ≤ C * (2 * (q : ℝ) ^ N) := mul_le_mul_of_nonneg_left hsum hC0
      _ = 2 * C * (q : ℝ) ^ N := by ring
  have h2 : 2 * C * (q : ℝ) ^ N ≤ 2 * C * (ε / (2 * C + 1)) :=
    mul_le_mul_of_nonneg_left hN.le (by linarith)
  have h3 : 2 * C * (ε / (2 * C + 1)) < ε := by
    have ht : (0 : ℝ) < ε / (2 * C + 1) := div_pos hε hden
    have heq : ε / (2 * C + 1) * (2 * C + 1) = ε := div_mul_cancel₀ ε hden.ne'
    nlinarith [ht, heq]
  linarith [hbound, h1, h2, h3]

/-- **Theorem A(iii) — no Lipschitz readout can count.**
Any `L`-Lipschitz map `g : S → ℝ` — the model's own stop-token logit, a duration
predictor, or any linear probe — sees repetition counts `m` and `n` collapsed to
within `L C (q^m + q^n)`. -/
theorem readout_gap_le (hF : ContractingWith q F) {L : NNReal} {g : S → ℝ}
    (hg : LipschitzWith L g) (s : S) (m n : ℕ) :
    |g (F^[m] s) - g (F^[n] s)| ≤ (L : ℝ) * (orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n)) := by
  have h1 : |g (F^[m] s) - g (F^[n] s)| = dist (g (F^[m] s)) (g (F^[n] s)) :=
    (Real.dist_eq _ _).symm
  rw [h1]
  calc dist (g (F^[m] s)) (g (F^[n] s))
      ≤ (L : ℝ) * dist (F^[m] s) (F^[n] s) := hg.dist_le_mul _ _
    _ ≤ (L : ℝ) * (orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n)) := by
        exact mul_le_mul_of_nonneg_left (dist_iterate_iterate_le hF s m n) L.coe_nonneg

/-- **Theorem A(iv) — the counting horizon is finite (algebraic form).**
If a readout still resolves two repetition counts `m ≤ n` with decision margin
`margin`, then `q ^ m` cannot have decayed below `margin / (2 L C)`.  Since
`q < 1`, this bounds `m`: only finitely many repetition counts admit a decision
margin at all. -/
theorem counting_margin_le (hF : ContractingWith q F) {L : NNReal} {g : S → ℝ}
    (hg : LipschitzWith L g) (s : S) {margin : ℝ} {m n : ℕ} (hmn : m ≤ n)
    (hsep : margin ≤ |g (F^[m] s) - g (F^[n] s)|) :
    margin ≤ 2 * (L : ℝ) * orbitRadius F s q * (q : ℝ) ^ m := by
  have hq0 : (0 : ℝ) ≤ (q : ℝ) := q.coe_nonneg
  have hq1 : (q : ℝ) ≤ 1 := le_of_lt (by exact_mod_cast hF.1)
  have hpn : (q : ℝ) ^ n ≤ (q : ℝ) ^ m := pow_le_pow_of_le_one hq0 hq1 hmn
  have hgap := readout_gap_le hF hg s m n
  have hC0 : 0 ≤ orbitRadius F s q := orbitRadius_nonneg hF s
  have hstep : (L : ℝ) * (orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n))
      ≤ 2 * (L : ℝ) * orbitRadius F s q * (q : ℝ) ^ m := by
    have hsum : (q : ℝ) ^ m + (q : ℝ) ^ n ≤ 2 * (q : ℝ) ^ m := by linarith
    have hinner : orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n)
        ≤ orbitRadius F s q * (2 * (q : ℝ) ^ m) := mul_le_mul_of_nonneg_left hsum hC0
    calc (L : ℝ) * (orbitRadius F s q * ((q : ℝ) ^ m + (q : ℝ) ^ n))
        ≤ (L : ℝ) * (orbitRadius F s q * (2 * (q : ℝ) ^ m)) :=
          mul_le_mul_of_nonneg_left hinner L.coe_nonneg
      _ = 2 * (L : ℝ) * orbitRadius F s q * (q : ℝ) ^ m := by ring
  linarith

/-- Real-arithmetic bridge: a geometric quantity bounded below is bounded in its
exponent.  Used to turn `counting_margin_le` into an explicit horizon. -/
theorem le_log_div_log_of_le_pow {r c : ℝ} (hr0 : 0 < r) (hr1 : r < 1) (hc : 0 < c)
    {m : ℕ} (h : c ≤ r ^ m) : (m : ℝ) ≤ Real.log c / Real.log r := by
  have hlogr : Real.log r < 0 := Real.log_neg hr0 hr1
  have hlog : Real.log c ≤ Real.log (r ^ m) := Real.log_le_log hc h
  rw [Real.log_pow] at hlog
  rw [le_div_iff_of_neg hlogr]
  linarith [hlog]

/-- **Theorem A — Counting Collapse (explicit horizon).**
The headline statement.  Under contraction, a Lipschitz readout with decision
margin `margin > 0` can separate the repetition count `m` from any larger count
only while

    m ≤ log(margin / (2 L C)) / log q  =  log(2 L C / margin) / log(1 / q).

Beyond that horizon the decoder is provably unable to tell how many repetitions
it has already produced, so its continue/stop decision is decoupled from the
count — the failure mode observed empirically as looping or truncation. -/
theorem counting_horizon (hF : ContractingWith q F) {L : NNReal} {g : S → ℝ}
    (hg : LipschitzWith L g) (s : S) {margin : ℝ} {m n : ℕ}
    (hq0 : (0 : ℝ) < (q : ℝ)) (hmargin : 0 < margin) (hmn : m ≤ n)
    (hsep : margin ≤ |g (F^[m] s) - g (F^[n] s)|) :
    (m : ℝ) ≤ Real.log (margin / (2 * (L : ℝ) * orbitRadius F s q)) / Real.log (q : ℝ) := by
  have hq1 : (q : ℝ) < 1 := by exact_mod_cast hF.1
  have hbound := counting_margin_le hF hg s hmn hsep
  have hC0 : 0 ≤ orbitRadius F s q := orbitRadius_nonneg hF s
  -- a nonzero decision margin forces the prefactor to be strictly positive
  have hprefix : (0 : ℝ) ≤ 2 * (L : ℝ) * orbitRadius F s q :=
    mul_nonneg (mul_nonneg (by norm_num) L.coe_nonneg) hC0
  have hpos : 0 < 2 * (L : ℝ) * orbitRadius F s q := by
    rcases lt_or_eq_of_le hprefix with h | h
    · exact h
    · exfalso
      rw [← h, zero_mul] at hbound
      linarith
  have hratio : margin / (2 * (L : ℝ) * orbitRadius F s q) ≤ (q : ℝ) ^ m := by
    rw [div_le_iff₀ hpos]
    calc margin ≤ 2 * (L : ℝ) * orbitRadius F s q * (q : ℝ) ^ m := hbound
      _ = (q : ℝ) ^ m * (2 * (L : ℝ) * orbitRadius F s q) := by ring
  exact le_log_div_log_of_le_pow hq0 hq1 (by positivity) hratio

end SpectralTTS
