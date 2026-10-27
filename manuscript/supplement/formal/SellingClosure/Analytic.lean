import Mathlib.Topology.MetricSpace.IsometricSMul
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.LinearCombination
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Ring.Finset
import SellingClosure.Finite

/-!
# Completeness of 𝒰 for every lattice

A lattice with a reduced superbase `(d, a, b, c)` is given by its six conorms
`p : Fin 6 → ℝ`, all non-negative because the superbase is obtuse.  By Selling's identity
the dot product of lattice vectors with coordinates `x, y` in `(a, b, c)` is
`dot p x y = ∑ₛ pₛ (diff x s) (diff y s)`.

The main theorem `arrow_mem_U`: if the support of `p` is connected (equivalently, the
metric is positive definite), every unimodular integer matrix that carries the reduced
superbase to another obtuse superbase of the same lattice belongs to the deposited table
𝒰.  The proof is Lemma A1 (class excess identity) and the Voronoi characterisation of
obtuse superbases, both proved here, followed by the finite check of `Finite.lean`.
-/

namespace SellingClosure

open Finset

/-- The dot product of two lattice vectors, by Selling's identity. -/
def dot (p : Fin 6 → ℝ) (x y : V3) : ℝ := ∑ s, p s * (diff x s : ℝ) * (diff y s : ℝ)

/-- The superbase that the operator `W` produces is obtuse for the metric `p`. -/
def Obtuse (p : Fin 6 → ℝ) (W : Op) : Prop :=
  ∀ a b : Fin 4, a ≠ b → dot p (sb W a) (sb W b) ≤ 0

/-! ## The reduced superbase itself -/

/-- The conorms of the reduced superbase are the `pₛ`: e.g. `-a·b = p_{ab}`. -/
theorem conorm_ab (p : Fin 6 → ℝ) : -dot p (1, 0, 0) (0, 1, 0) = p 3 := by
  simp [dot, Fin.sum_univ_six, diff]

theorem conorm_da (p : Fin 6 → ℝ) : -dot p (-1, -1, -1) (1, 0, 0) = p 0 := by
  simp [dot, Fin.sum_univ_six, diff]

/-! ## Linearity of the coordinate differences -/

theorem diff_vadd (x y : V3) (s : Fin 6) : diff (vadd x y) s = diff x s + diff y s := by
  obtain ⟨a, b, c⟩ := x; obtain ⟨a', b', c'⟩ := y
  fin_cases s <;> simp [diff, vadd] <;> ring

theorem diff_vsmul (k : Int) (x : V3) (s : Fin 6) : diff (vsmul k x) s = k * diff x s := by
  obtain ⟨a, b, c⟩ := x
  fin_cases s <;> simp [diff, vsmul] <;> ring

theorem diff_vneg (x : V3) (s : Fin 6) : diff (vneg x) s = -diff x s := by
  obtain ⟨a, b, c⟩ := x
  fin_cases s <;> simp [diff, vneg] <;> ring

/-- `x = s + 2k` slot by slot. -/
theorem diff_par_half (x : V3) (s : Fin 6) :
    diff x s = diff (par x) s + 2 * diff (half x) s := by
  obtain ⟨a, b, c⟩ := x
  fin_cases s <;> simp [diff, par, half] <;> omega

theorem diff_par_bound (x : V3) (s : Fin 6) : -1 ≤ diff (par x) s ∧ diff (par x) s ≤ 1 := by
  obtain ⟨a, b, c⟩ := x
  fin_cases s <;> simp [diff, par] <;> omega

/-- `t (t + a) ≥ 0` for integers `t` and `a ∈ {-1, 0, 1}`. -/
theorem int_gam_nonneg (t a : Int) (h1 : -1 ≤ a) (h2 : a ≤ 1) : 0 ≤ t * (t + a) := by
  rcases lt_trichotomy t 0 with h | h | h
  · have : t + a ≤ 0 := by omega
    nlinarith
  · simp [h]
  · have : 0 ≤ t + a := by omega
    nlinarith

theorem gam_nonneg (x : V3) (s : Fin 6) : 0 ≤ gam x s :=
  int_gam_nonneg _ _ (diff_par_bound x s).1 (diff_par_bound x s).2

/-! ## Lemma A1: the class excess identity in the reduced frame -/

theorem excess_identity (p : Fin 6 → ℝ) (x : V3) :
    dot p x x - dot p (par x) (par x) = 4 * ∑ s, p s * (gam x s : ℝ) := by
  simp only [dot, ← sum_sub_distrib, mul_sum]
  refine sum_congr rfl fun s _ => ?_
  rw [diff_par_half x s]
  simp only [gam]
  push_cast
  ring

/-! ## Selling's identity in the frame of another superbase -/

/-- A lattice vector given by its coordinates `e` in the basis `(u₁, u₂, u₃)`. -/
def lc (e₁ e₂ e₃ : Int) (u₁ u₂ u₃ : V3) : V3 :=
  vadd (vadd (vsmul e₁ u₁) (vsmul e₂ u₂)) (vsmul e₃ u₃)

theorem diff_lc (e₁ e₂ e₃ : Int) (u₁ u₂ u₃ : V3) (s : Fin 6) :
    diff (lc e₁ e₂ e₃ u₁ u₂ u₃) s = e₁ * diff u₁ s + e₂ * diff u₂ s + e₃ * diff u₃ s := by
  simp [lc, diff_vadd, diff_vsmul]

/-- One slot of Selling's identity for the superbase `(-(u₁+u₂+u₃), u₁, u₂, u₃)`. -/
theorem selling_slot (D₁ D₂ D₃ e₁ e₂ e₃ : ℝ) :
    (e₁ * D₁ + e₂ * D₂ + e₃ * D₃) ^ 2 =
      -((-(D₁ + D₂ + D₃)) * D₁ * (0 - e₁) ^ 2 + (-(D₁ + D₂ + D₃)) * D₂ * (0 - e₂) ^ 2
        + (-(D₁ + D₂ + D₃)) * D₃ * (0 - e₃) ^ 2 + D₁ * D₂ * (e₁ - e₂) ^ 2
        + D₁ * D₃ * (e₁ - e₃) ^ 2 + D₂ * D₃ * (e₂ - e₃) ^ 2) := by
  ring

/-- Selling's identity in the frame of the superbase of `W`: the squared length of a vector
with coordinates `e` is `∑ q_{ab} (e_a - e_b)²` over the six pairs, `q_{ab} = -u_a · u_b`,
`e₀ = 0`. -/
theorem selling_identity (p : Fin 6 → ℝ) (W : Op) (e₁ e₂ e₃ : Int) :
    let x := lc e₁ e₂ e₃ W.1 W.2.1 W.2.2
    dot p x x =
      (-dot p (sb W 0) (sb W 1)) * (0 - e₁) ^ 2 + (-dot p (sb W 0) (sb W 2)) * (0 - e₂) ^ 2
      + (-dot p (sb W 0) (sb W 3)) * (0 - e₃) ^ 2 + (-dot p (sb W 1) (sb W 2)) * (e₁ - e₂) ^ 2
      + (-dot p (sb W 1) (sb W 3)) * (e₁ - e₃) ^ 2
      + (-dot p (sb W 2) (sb W 3)) * (e₂ - e₃) ^ 2 := by
  intro x
  simp only [x, dot, Fin.sum_univ_six, sb, diff_vneg, diff_vadd, diff_lc]
  push_cast
  ring

/-- In the frame of an obtuse superbase, a vector `e' + 2f` with `e' ∈ {0,1}³` is at least
as long as `e'`: a partial sum is shortest in its parity class. -/
theorem partial_shortest (p : Fin 6 → ℝ) (W : Op) (hob : Obtuse p W)
    (e₁ e₂ e₃ f₁ f₂ f₃ : Int) (h₁ : 0 ≤ e₁ ∧ e₁ ≤ 1) (h₂ : 0 ≤ e₂ ∧ e₂ ≤ 1)
    (h₃ : 0 ≤ e₃ ∧ e₃ ≤ 1) :
    dot p (lc e₁ e₂ e₃ W.1 W.2.1 W.2.2) (lc e₁ e₂ e₃ W.1 W.2.1 W.2.2) ≤
      dot p (lc (e₁ + 2 * f₁) (e₂ + 2 * f₂) (e₃ + 2 * f₃) W.1 W.2.1 W.2.2)
        (lc (e₁ + 2 * f₁) (e₂ + 2 * f₂) (e₃ + 2 * f₃) W.1 W.2.1 W.2.2) := by
  rw [selling_identity, selling_identity]
  have q01 := hob 0 1 (by decide); have q02 := hob 0 2 (by decide)
  have q03 := hob 0 3 (by decide); have q12 := hob 1 2 (by decide)
  have q13 := hob 1 3 (by decide); have q23 := hob 2 3 (by decide)
  -- each squared difference grows: (a + 2t)² - a² = 4 t (t + a) ≥ 0 for |a| ≤ 1
  have g : ∀ a t : Int, -1 ≤ a → a ≤ 1 → ((a : ℝ)) ^ 2 ≤ ((a + 2 * t : Int) : ℝ) ^ 2 := by
    intro a t ha hb
    have := int_gam_nonneg t a ha hb
    have h : ((a + 2 * t : Int) : ℝ) ^ 2 - (a : ℝ) ^ 2 = 4 * ((t * (t + a) : Int) : ℝ) := by
      push_cast; ring
    have : (0 : ℝ) ≤ ((t * (t + a) : Int) : ℝ) := by exact_mod_cast this
    linarith
  have t01 := g (0 - e₁) (0 - f₁) (by omega) (by omega)
  have t02 := g (0 - e₂) (0 - f₂) (by omega) (by omega)
  have t03 := g (0 - e₃) (0 - f₃) (by omega) (by omega)
  have t12 := g (e₁ - e₂) (f₁ - f₂) (by omega) (by omega)
  have t13 := g (e₁ - e₃) (f₁ - f₃) (by omega) (by omega)
  have t23 := g (e₂ - e₃) (f₂ - f₃) (by omega) (by omega)
  push_cast at t01 t02 t03 t12 t13 t23 ⊢
  have r : ∀ {q a b : ℝ}, q ≤ 0 → a ≤ b → -q * a ≤ -q * b := fun hq hab =>
    mul_le_mul_of_nonneg_left hab (by linarith)
  have := r q01 (show ((0:ℝ) - e₁) ^ 2 ≤ (0 - (e₁ + 2 * f₁)) ^ 2 by nlinarith [t01])
  have := r q02 (show ((0:ℝ) - e₂) ^ 2 ≤ (0 - (e₂ + 2 * f₂)) ^ 2 by nlinarith [t02])
  have := r q03 (show ((0:ℝ) - e₃) ^ 2 ≤ (0 - (e₃ + 2 * f₃)) ^ 2 by nlinarith [t03])
  have := r q12 (show ((e₁:ℝ) - e₂) ^ 2 ≤ (e₁ + 2 * f₁ - (e₂ + 2 * f₂)) ^ 2 by nlinarith [t12])
  have := r q13 (show ((e₁:ℝ) - e₃) ^ 2 ≤ (e₁ + 2 * f₁ - (e₃ + 2 * f₃)) ^ 2 by nlinarith [t13])
  have := r q23 (show ((e₂:ℝ) - e₃) ^ 2 ≤ (e₂ + 2 * f₂ - (e₃ + 2 * f₃)) ^ 2 by nlinarith [t23])
  linarith

/-! ## Unimodular matrices are invertible over ℤ (Cramer's rule) -/

theorem cramer (W : Op) (z : V3) :
    lc (det z W.2.1 W.2.2) (det W.1 z W.2.2) (det W.1 W.2.1 z) W.1 W.2.1 W.2.2 =
      vsmul (det W.1 W.2.1 W.2.2) z := by
  obtain ⟨⟨a₁, a₂, a₃⟩, ⟨b₁, b₂, b₃⟩, ⟨c₁, c₂, c₃⟩⟩ := W
  obtain ⟨z₁, z₂, z₃⟩ := z
  simp only [lc, vadd, vsmul, det, Prod.mk.injEq]
  refine ⟨by ring, by ring, by ring⟩

theorem unimodular_coords (W : Op) (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (z : V3) : ∃ f₁ f₂ f₃ : Int, z = lc f₁ f₂ f₃ W.1 W.2.1 W.2.2 := by
  set d := det W.1 W.2.1 W.2.2 with hd
  refine ⟨d * det z W.2.1 W.2.2, d * det W.1 z W.2.2, d * det W.1 W.2.1 z, ?_⟩
  have h := cramer W z
  have hsq : d * d = 1 := by rcases hdet with h | h <;> simp [h]
  obtain ⟨⟨a₁, a₂, a₃⟩, ⟨b₁, b₂, b₃⟩, ⟨c₁, c₂, c₃⟩⟩ := W
  obtain ⟨z₁, z₂, z₃⟩ := z
  simp only [lc, vadd, vsmul, Prod.mk.injEq] at h ⊢
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨?_, ?_, ?_⟩
  · linear_combination (-d) * h1 - z₁ * hsq
  · linear_combination (-d) * h2 - z₂ * hsq
  · linear_combination (-d) * h3 - z₃ * hsq

/-! ## The excess vanishes on the support -/

theorem lc_sub_two (e₁ e₂ e₃ g₁ g₂ g₃ : Int) (W : Op) (w : V3)
    (hw : w = lc e₁ e₂ e₃ W.1 W.2.1 W.2.2) (hh : half w = lc g₁ g₂ g₃ W.1 W.2.1 W.2.2) :
    par w = lc (e₁ + 2 * -g₁) (e₂ + 2 * -g₂) (e₃ + 2 * -g₃) W.1 W.2.1 W.2.2 := by
  have hx : ∀ s, diff (par w) s = diff w s - 2 * diff (half w) s := fun s => by
    have := diff_par_half w s; omega
  obtain ⟨⟨a₁, a₂, a₃⟩, ⟨b₁, b₂, b₃⟩, ⟨c₁, c₂, c₃⟩⟩ := W
  obtain ⟨w₁, w₂, w₃⟩ := w
  simp only [par, half, lc, vadd, vsmul, Prod.mk.injEq] at hw hh ⊢
  obtain ⟨hw1, hw2, hw3⟩ := hw
  obtain ⟨hh1, hh2, hh3⟩ := hh
  refine ⟨?_, ?_, ?_⟩
  · have := Int.emod_add_mul_ediv w₁ 2; linear_combination this - 2 * hh1 + hw1
  · have := Int.emod_add_mul_ediv w₂ 2; linear_combination this - 2 * hh2 + hw2
  · have := Int.emod_add_mul_ediv w₃ 2; linear_combination this - 2 * hh3 + hw3

/-- For each partial sum `w'` of an obtuse superbase of the same lattice, `γₛ(w') = 0` on
every slot where the reduced superbase's conorm is non-zero (Lemma A2's step). -/
theorem gam_zero_of_partial (p : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s) (W : Op) (hob : Obtuse p W)
    (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (e₁ e₂ e₃ : Int) (h₁ : 0 ≤ e₁ ∧ e₁ ≤ 1) (h₂ : 0 ≤ e₂ ∧ e₂ ≤ 1) (h₃ : 0 ≤ e₃ ∧ e₃ ≤ 1)
    (s : Fin 6) (hs : p s ≠ 0) : gam (lc e₁ e₂ e₃ W.1 W.2.1 W.2.2) s = 0 := by
  set w := lc e₁ e₂ e₃ W.1 W.2.1 W.2.2 with hw
  obtain ⟨g₁, g₂, g₃, hg⟩ := unimodular_coords W hdet (half w)
  have hpar := lc_sub_two e₁ e₂ e₃ g₁ g₂ g₃ W w hw hg
  -- the partial sum of the reduced superbase in w's class is no shorter than w ...
  have h1 : dot p w w ≤ dot p (par w) (par w) := by
    rw [hpar, hw]; exact partial_shortest p W hob e₁ e₂ e₃ (-g₁) (-g₂) (-g₃) h₁ h₂ h₃
  -- ... and no longer, by Lemma A1 with non-negative conorms
  have hid := excess_identity p w
  have hterm : ∀ t ∈ (univ : Finset (Fin 6)), 0 ≤ p t * (gam w t : ℝ) := fun t _ =>
    mul_nonneg (hp t) (by exact_mod_cast gam_nonneg w t)
  have hsum0 : ∑ t, p t * (gam w t : ℝ) = 0 := by
    have := sum_nonneg hterm; linarith
  have := (sum_eq_zero_iff_of_nonneg hterm).mp hsum0 s (mem_univ s)
  have : (gam w s : ℝ) = 0 := by
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h hs
    · exact h
  exact_mod_cast this

/-! ## The main theorem -/

/-- The support of the conorms, as a Boolean pattern. -/
noncomputable def suppOf (p : Fin 6 → ℝ) : Supp := fun s => decide (p s ≠ 0)

/-- Every partial sum of an arrow has `γ = 0` on the support of the conorms. -/
theorem partials_gok (p : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s) (W : Op)
    (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1) (hob : Obtuse p W) :
    ∀ w ∈ partials W.1 W.2.1 W.2.2, gokB (suppOf p) w = true := by
  obtain ⟨u₁, u₂, u₃⟩ := W
  intro w hw
  rw [gokB_iff]
  intro s hs
  have hs' : p s ≠ 0 := by simpa [suppOf] using hs
  have key := fun e₁ e₂ e₃ h₁ h₂ h₃ =>
    gam_zero_of_partial p hp (u₁, u₂, u₃) hob hdet e₁ e₂ e₃ h₁ h₂ h₃ s hs'
  obtain ⟨a₁, a₂, a₃⟩ := u₁; obtain ⟨b₁, b₂, b₃⟩ := u₂; obtain ⟨c₁, c₂, c₃⟩ := u₃
  simp only [partials, List.mem_cons, List.not_mem_nil, or_false] at hw
  rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · simpa [lc, vadd, vsmul] using key 1 0 0 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul] using key 0 1 0 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul] using key 0 0 1 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul] using key 1 1 0 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul] using key 1 0 1 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul] using key 0 1 1 (by omega) (by omega) (by omega)
  · simpa [lc, vadd, vsmul, add_assoc] using key 1 1 1 (by omega) (by omega) (by omega)

/-- **Completeness of 𝒰.** Let a lattice have a reduced (obtuse) superbase with conorms
`p ≥ 0` whose support is connected.  Every unimodular integer matrix `W` that carries the
reduced superbase to an obtuse superbase of the same lattice lies in the deposited table
𝒰 of 6096 operators. -/
theorem arrow_mem_U (p : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s) (hconn : isConn (suppOf p) = true)
    (W : Op) (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (hob : Obtuse p W) : W ∈ U := by
  obtain ⟨u₁, u₂, u₃⟩ := W
  exact mem_U_of_gok (suppOf p) hconn u₁ u₂ u₃ (partials_gok p hp _ hdet hob) hdet

/-- The sign test on a face makes the superbase obtuse for every metric on that face. -/
theorem obtuse_of_signOK (S : Supp) (W : Op) (hsign : signOK S W = true) (p : Fin 6 → ℝ)
    (hp : ∀ s, 0 ≤ p s) (hoff : ∀ s, S s = false → p s = 0) : Obtuse p W := by
  intro a b hab
  simp only [signOK, List.all_eq_true, List.mem_finRange, true_implies] at hsign
  have hs := hsign a b
  have hab' : (a == b) = false := by simpa using hab
  simp only [hab', Bool.false_or, List.all_eq_true, List.mem_finRange, true_implies,
    Bool.or_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at hs
  apply sum_nonpos
  intro s _
  rcases hs s with h0 | hneg
  · simp [hoff s h0]
  · have : ((diff (sb W a) s * diff (sb W b) s : Int) : ℝ) ≤ 0 := by exact_mod_cast hneg
    push_cast at this
    rw [mul_assoc]
    exact mul_nonpos_of_nonneg_of_nonpos (hp s) this

/-- **Every operator of 𝒰 is an arrow.** For each `W ∈ 𝒰` there is a face (a connected
support) such that `W` carries the reduced superbase of every lattice on that face to an
obtuse superbase. -/
theorem U_arrow (W : Op) (hW : W ∈ U) :
    ∃ S : Supp, isConn S = true ∧ (det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1) ∧
      ∀ p : Fin 6 → ℝ, (∀ s, 0 ≤ p s) → (∀ s, S s = false → p s = 0) → Obtuse p W := by
  have h := List.all_eq_true.mp U_realised W hW
  have hd := List.all_eq_true.mp U_unimodular W hW
  simp only [List.any_eq_true, List.mem_filter] at h
  obtain ⟨S, ⟨_, hS⟩, hsign⟩ := h
  exact ⟨S, hS, by simpa using hd, fun p hp hoff => obtuse_of_signOK S W hsign p hp hoff⟩

/-- **Arrows depend only on the zero pattern.** If `W` carries the reduced superbase to an
obtuse superbase for one metric, it does so for every metric with the same zero conorms. -/
theorem arrow_zero_pattern (p p' : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s) (hp' : ∀ s, 0 ≤ p' s)
    (hsupp : ∀ s, (p s = 0 ↔ p' s = 0)) (hconn : isConn (suppOf p) = true)
    (W : Op) (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (hob : Obtuse p W) : Obtuse p' W := by
  have hW := arrow_mem_U p hp hconn W hdet hob
  have hc := zero_pattern_check (suppOf p 0) (suppOf p 1) (suppOf p 2) (suppOf p 3)
    (suppOf p 4) (suppOf p 5)
  simp only [ofBits_eta, hconn, Bool.not_true, Bool.false_or, List.all_eq_true] at hc
  have := hc W hW
  have hall : (partials W.1 W.2.1 W.2.2).all (gokB (suppOf p)) = true :=
    List.all_eq_true.mpr (partials_gok p hp W hdet hob)
  rw [hall] at this
  simp only [Bool.not_true, Bool.false_or] at this
  refine obtuse_of_signOK _ W this p' hp' fun s hs => ?_
  have : p s = 0 := by simpa [suppOf] using hs
  exact (hsupp s).mp this

/-! ## A lattice's conorms have connected support -/

/-- The seven partial sums of the reduced superbase, in coordinates. -/
def unitPartials : List V3 := partials (1, 0, 0) (0, 1, 0) (0, 0, 1)

/-- If a support is not connected, some partial sum of the reduced superbase has zero
coordinate difference on every slot of the support (a finite check over the 64 supports). -/
theorem disconn_witness : ∀ b₀ b₁ b₂ b₃ b₄ b₅ : Bool,
    let S := ofBits b₀ b₁ b₂ b₃ b₄ b₅
    (isConn S || unitPartials.any fun x => (List.finRange 6).all fun s => !S s || diff x s == 0)
      = true := by
  decide

/-- **The conorms of a lattice have connected support.** If the metric is positive definite on
the lattice (every non-zero integer vector has positive squared length), the support of its
conorms connects the four vectors of the reduced superbase. -/
theorem conn_of_posdef (p : Fin 6 → ℝ) (hpos : ∀ x : V3, x ≠ (0, 0, 0) → 0 < dot p x x) :
    isConn (suppOf p) = true := by
  by_contra hc
  have h := disconn_witness (suppOf p 0) (suppOf p 1) (suppOf p 2) (suppOf p 3) (suppOf p 4)
    (suppOf p 5)
  simp only [ofBits_eta, Bool.eq_false_iff.mpr hc, Bool.false_or, List.any_eq_true] at h
  obtain ⟨x, hx, hall⟩ := h
  have hx0 : x ≠ (0, 0, 0) := by
    simp only [unitPartials, partials, vadd, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hzero : dot p x x = 0 := by
    unfold dot
    apply Finset.sum_eq_zero
    intro s _
    have hs := List.all_eq_true.mp hall s (List.mem_finRange s)
    by_cases hps : p s = 0
    · simp [hps]
    · have hS : suppOf p s = true := by simp [suppOf, hps]
      simp only [hS, Bool.not_true, Bool.false_or, beq_iff_eq] at hs
      simp [hs]
  linarith [hpos x hx0]

/-- **Completeness of 𝒰 for every lattice.** `arrow_mem_U` with positive definiteness in place
of the connected support. -/
theorem arrow_mem_U_of_posdef (p : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s)
    (hpos : ∀ x : V3, x ≠ (0, 0, 0) → 0 < dot p x x)
    (W : Op) (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (hob : Obtuse p W) : W ∈ U :=
  arrow_mem_U p hp (conn_of_posdef p hpos) W hdet hob

/-- **Arrows depend only on the zero pattern, for every lattice.** -/
theorem arrow_zero_pattern_of_posdef (p p' : Fin 6 → ℝ) (hp : ∀ s, 0 ≤ p s)
    (hp' : ∀ s, 0 ≤ p' s) (hsupp : ∀ s, (p s = 0 ↔ p' s = 0))
    (hpos : ∀ x : V3, x ≠ (0, 0, 0) → 0 < dot p x x)
    (W : Op) (hdet : det W.1 W.2.1 W.2.2 = 1 ∨ det W.1 W.2.1 W.2.2 = -1)
    (hob : Obtuse p W) : Obtuse p' W :=
  arrow_zero_pattern p p' hp hp' hsupp (conn_of_posdef p hpos) W hdet hob

end SellingClosure
