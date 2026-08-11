/-
Attention Dilution
==================

Lemma B of "Counting Collapse: A Formally Verified Attractor Theory of
Repetition Hallucination in Autoregressive TTS".

Setting.  A decoder step attends over the `k` text positions occupied by `k`
repetitions of the same phrase.  Because those positions carry (near-)identical
content, their attention logits are near-identical too: we assume only that the
logits lie within a spread `δ` of one another, which is a directly measurable
quantity.

What is proved.  Every occurrence receives attention weight in
`[e^{-δ}/k, e^{δ}/k]`; consequently the attention profile over the repeated
block is within `(e^{δ} - e^{-δ})/k` of uniform, and its entropy is at least
`log k - δ`.  As `k` grows the profile flattens: the decoder cannot use
attention weight to identify *which* occurrence it is currently rendering.

Why it matters.  This licenses the modelling assumption of
`CountingCollapse.lean` — that the conditioning at every repetition boundary is
the same map `F`.  The link is quantitative but informal (an `O(δ + 1/k)`
perturbation of the conditioning yields an `O(δ + 1/k)` perturbation of `F`); the
paper states it as a remark, not as a formalized theorem.
-/
import Mathlib

namespace SpectralTTS

open Finset Real

variable {k : ℕ}

/-- Softmax over the `k` attention logits of a repeated text block. -/
noncomputable def softmax (z : Fin k → ℝ) (i : Fin k) : ℝ :=
  Real.exp (z i) / ∑ j, Real.exp (z j)

theorem sum_exp_pos [NeZero k] (z : Fin k → ℝ) : 0 < ∑ j, Real.exp (z j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (z j))
    ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne k)⟩, mem_univ _⟩

theorem softmax_nonneg [NeZero k] (z : Fin k → ℝ) (i : Fin k) : 0 ≤ softmax z i :=
  div_nonneg (Real.exp_pos _).le (sum_exp_pos z).le

/-- The softmax profile is a probability distribution. -/
theorem sum_softmax [NeZero k] (z : Fin k → ℝ) : ∑ i, softmax z i = 1 := by
  unfold softmax
  rw [← Finset.sum_div, div_self (sum_exp_pos z).ne']

/-- **Lemma B (upper bound).**  If all logits in the repeated block lie within
`δ` of one another, no single occurrence can capture more than `e^{δ}/k` of the
attention mass. -/
theorem softmax_le [NeZero k] (z : Fin k → ℝ) {δ : ℝ} (hz : ∀ a b, z a - z b ≤ δ) (i : Fin k) :
    softmax z i ≤ Real.exp δ / k := by
  have hk : (0 : ℝ) < k := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne k))
  -- every term of the denominator is at least `exp (z i - δ)`
  have hlow : (k : ℝ) * Real.exp (z i - δ) ≤ ∑ j, Real.exp (z j) := by
    have hterm : ∀ j ∈ (univ : Finset (Fin k)), Real.exp (z i - δ) ≤ Real.exp (z j) := by
      intro j _
      exact Real.exp_le_exp.mpr (by linarith [hz i j])
    calc (k : ℝ) * Real.exp (z i - δ)
        = ∑ _j : Fin k, Real.exp (z i - δ) := by
          rw [Finset.sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ ≤ ∑ j, Real.exp (z j) := Finset.sum_le_sum hterm
  unfold softmax
  rw [div_le_div_iff₀ (sum_exp_pos z) hk]
  have hkey : Real.exp (z i) * (k : ℝ) = Real.exp δ * ((k : ℝ) * Real.exp (z i - δ)) := by
    rw [Real.exp_sub]
    field_simp
  rw [hkey]
  exact mul_le_mul_of_nonneg_left hlow (Real.exp_pos δ).le

/-- **Lemma B (lower bound).**  Symmetrically, no occurrence can be starved
below `e^{-δ}/k`. -/
theorem le_softmax [NeZero k] (z : Fin k → ℝ) {δ : ℝ} (hz : ∀ a b, z a - z b ≤ δ) (i : Fin k) :
    Real.exp (-δ) / k ≤ softmax z i := by
  have hk : (0 : ℝ) < k := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne k))
  have hhigh : ∑ j, Real.exp (z j) ≤ (k : ℝ) * Real.exp (z i + δ) := by
    have hterm : ∀ j ∈ (univ : Finset (Fin k)), Real.exp (z j) ≤ Real.exp (z i + δ) := by
      intro j _
      exact Real.exp_le_exp.mpr (by linarith [hz j i])
    calc ∑ j, Real.exp (z j)
        ≤ ∑ _j : Fin k, Real.exp (z i + δ) := Finset.sum_le_sum hterm
      _ = (k : ℝ) * Real.exp (z i + δ) := by
          rw [Finset.sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold softmax
  rw [div_le_div_iff₀ hk (sum_exp_pos z)]
  have hkey : Real.exp (-δ) * ((k : ℝ) * Real.exp (z i + δ)) = Real.exp (z i) * (k : ℝ) := by
    rw [Real.exp_add, Real.exp_neg]
    field_simp
  calc Real.exp (-δ) * ∑ j, Real.exp (z j)
      ≤ Real.exp (-δ) * ((k : ℝ) * Real.exp (z i + δ)) :=
        mul_le_mul_of_nonneg_left hhigh (Real.exp_pos _).le
    _ = Real.exp (z i) * (k : ℝ) := hkey

/-- **Occurrence indistinguishability.**  The attention profile over a block of
`k` repeated spans is within `(e^{δ} - e^{-δ})/k` of uniform.  For fixed `δ`
this vanishes as `1/k`: attention weight carries no information about *which*
repetition is being attended to. -/
theorem softmax_dist_le [NeZero k] (z : Fin k → ℝ) {δ : ℝ} (hz : ∀ a b, z a - z b ≤ δ)
    (i j : Fin k) :
    |softmax z i - softmax z j| ≤ (Real.exp δ - Real.exp (-δ)) / k := by
  have hsplit : (Real.exp δ - Real.exp (-δ)) / (k : ℝ)
      = Real.exp δ / (k : ℝ) - Real.exp (-δ) / (k : ℝ) := by ring
  have hi_up := softmax_le z hz i
  have hj_up := softmax_le z hz j
  have hi_lo := le_softmax z hz i
  have hj_lo := le_softmax z hz j
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-- **Attention entropy grows like `log k`.**  With `p i = softmax z i`, the
Shannon entropy of the profile over the repeated block satisfies
`H(p) ≥ log k - δ`: near-uniform attention over `k` identical spans is maximally
uninformative, so the decoder's effective conditioning becomes the block mean and
loses occurrence identity. -/
theorem entropy_ge [NeZero k] (z : Fin k → ℝ) {δ : ℝ} (hz : ∀ a b, z a - z b ≤ δ) :
    Real.log k - δ ≤ ∑ i, softmax z i * (-Real.log (softmax z i)) := by
  have hk : (0 : ℝ) < k := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne k))
  have key : ∀ i : Fin k, (Real.log k - δ) * softmax z i
      ≤ softmax z i * (-Real.log (softmax z i)) := by
    intro i
    have hup := softmax_le z hz i
    have hlo : 0 < softmax z i := by
      have h1 : Real.exp (-δ) / (k : ℝ) ≤ softmax z i := le_softmax z hz i
      have h2 : (0 : ℝ) < Real.exp (-δ) / (k : ℝ) := by positivity
      linarith
    have hlog : Real.log (softmax z i) ≤ δ - Real.log k := by
      have h1 : Real.log (softmax z i) ≤ Real.log (Real.exp δ / k) := Real.log_le_log hlo hup
      rwa [Real.log_div (Real.exp_pos δ).ne' hk.ne', Real.log_exp] at h1
    have hneg : Real.log k - δ ≤ -Real.log (softmax z i) := by linarith
    calc (Real.log k - δ) * softmax z i
        ≤ (-Real.log (softmax z i)) * softmax z i :=
          mul_le_mul_of_nonneg_right hneg hlo.le
      _ = softmax z i * (-Real.log (softmax z i)) := mul_comm _ _
  calc Real.log k - δ
      = (Real.log k - δ) * ∑ i, softmax z i := by rw [sum_softmax z]; ring
    _ = ∑ i, (Real.log k - δ) * softmax z i := by rw [Finset.mul_sum]
    _ ≤ ∑ i, softmax z i * (-Real.log (softmax z i)) := Finset.sum_le_sum (fun i _ => key i)

end SpectralTTS
