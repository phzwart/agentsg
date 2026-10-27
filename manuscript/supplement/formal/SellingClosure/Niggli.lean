import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# Proposition A11 (i) and (ii): Niggli type and obtuse superbases

A cell `(a, b, c)` is described by its Gram entries: `A = a·a`, `B = b·b`, `C = c·c`,
`ab = a·b`, `ac = a·c`, `bc = b·c`.  With `d = -a-b-c`, the six conorms are
`p_ab = -ab` etc. and `p_da = -d·a = A + ab + ac`, `p_db = B + ab + bc`, `p_dc = C + ac + bc`.
-/

namespace SellingClosure.Niggli

/-- Niggli's conditions for a reduced cell, with `a·a ≤ b·b ≤ c·c`. -/
structure NiggliCell (A B C ab ac bc : ℝ) : Prop where
  hAB : A ≤ B
  hBC : B ≤ C
  hab : |2 * ab| ≤ A
  hac : |2 * ac| ≤ A
  hbc : |2 * bc| ≤ B

/-- A11(i), first part: a Niggli cell of type II (no acute angle) gives an obtuse superbase
`(d, a, b, c)`: all six conorms are non-negative. -/
theorem typeII_obtuse {A B C ab ac bc : ℝ} (h : NiggliCell A B C ab ac bc)
    (h1 : ab ≤ 0) (h2 : ac ≤ 0) (h3 : bc ≤ 0) :
    0 ≤ -ab ∧ 0 ≤ -ac ∧ 0 ≤ -bc ∧ 0 ≤ A + ab + ac ∧ 0 ≤ B + ab + bc ∧ 0 ≤ C + ac + bc := by
  have := h.hab; have := h.hac; have := h.hbc; have := h.hAB; have := h.hBC
  rw [abs_le] at *
  refine ⟨by linarith, by linarith, by linarith, by linarith, by linarith, by linarith⟩

/-- A11(i), the bound on the conorms on `d`: `p_dx ≥ ½ x·x − min(|x·y|, |x·z|)`. -/
theorem typeII_d_conorms {A B C ab ac bc : ℝ} (h : NiggliCell A B C ab ac bc)
    (h1 : ab ≤ 0) (h2 : ac ≤ 0) (h3 : bc ≤ 0) :
    A / 2 - min |ab| |ac| ≤ A + ab + ac ∧
    B / 2 - min |ab| |bc| ≤ B + ab + bc ∧
    C / 2 - min |ac| |bc| ≤ C + ac + bc := by
  have := h.hab; have := h.hac; have := h.hbc; have := h.hAB; have := h.hBC
  rw [abs_le] at *
  rw [abs_of_nonpos h1, abs_of_nonpos h2, abs_of_nonpos h3]
  refine ⟨?_, ?_, ?_⟩ <;>
  · rcases min_cases (-ab) (-ac) with ⟨hm, _⟩ | ⟨hm, _⟩ <;>
    rcases min_cases (-ab) (-bc) with ⟨hm', _⟩ | ⟨hm', _⟩ <;>
    rcases min_cases (-ac) (-bc) with ⟨hm'', _⟩ | ⟨hm'', _⟩ <;>
    simp only [hm, hm', hm''] <;> linarith

/-- A11(i), consequence: if at most one of the three products exceeds `τ` in size and
`τ < ¼ a·a`, every conorm on `d` exceeds every conorm off `d` that is at most `τ`. -/
theorem typeII_separation {A B C ab ac bc τ : ℝ} (h : NiggliCell A B C ab ac bc)
    (h1 : ab ≤ 0) (h2 : ac ≤ 0) (h3 : bc ≤ 0) (hτ : τ < A / 4)
    (hone : (|ab| ≤ τ ∧ |ac| ≤ τ) ∨ (|ab| ≤ τ ∧ |bc| ≤ τ) ∨ (|ac| ≤ τ ∧ |bc| ≤ τ)) :
    τ < A + ab + ac ∧ τ < B + ab + bc ∧ τ < C + ac + bc := by
  obtain ⟨hda, hdb, hdc⟩ := typeII_d_conorms h h1 h2 h3
  have := h.hAB; have := h.hBC
  have m1 : min |ab| |ac| ≤ τ := by
    rcases hone with ⟨x, y⟩ | ⟨x, y⟩ | ⟨x, y⟩
    · exact le_trans (min_le_left _ _) x
    · exact le_trans (min_le_left _ _) x
    · exact le_trans (min_le_right _ _) x
  have m2 : min |ab| |bc| ≤ τ := by
    rcases hone with ⟨x, y⟩ | ⟨x, y⟩ | ⟨x, y⟩
    · exact le_trans (min_le_left _ _) x
    · exact le_trans (min_le_left _ _) x
    · exact le_trans (min_le_right _ _) y
  have m3 : min |ac| |bc| ≤ τ := by
    rcases hone with ⟨x, y⟩ | ⟨x, y⟩ | ⟨x, y⟩
    · exact le_trans (min_le_left _ _) y
    · exact le_trans (min_le_right _ _) y
    · exact le_trans (min_le_left _ _) x
  refine ⟨by linarith, by linarith, by linarith⟩

/-- A11(ii): for a cell of type I (all three products positive), none of the orthogonal
superbases `{σ₁a, σ₂b, σ₃c, −σ₁a−σ₂b−σ₃c}` is obtuse: its three basis products
`σ₁σ₂ ab`, `σ₁σ₃ ac`, `σ₂σ₃ bc` cannot all be non-positive. -/
theorem typeI_no_orthogonal_obtuse {ab ac bc : ℝ} (h1 : 0 < ab) (h2 : 0 < ac) (h3 : 0 < bc)
    (σ₁ σ₂ σ₃ : ℝ) (hσ₁ : σ₁ = 1 ∨ σ₁ = -1) (hσ₂ : σ₂ = 1 ∨ σ₂ = -1) (hσ₃ : σ₃ = 1 ∨ σ₃ = -1) :
    ¬ (σ₁ * σ₂ * ab ≤ 0 ∧ σ₁ * σ₃ * ac ≤ 0 ∧ σ₂ * σ₃ * bc ≤ 0) := by
  rintro ⟨x, y, z⟩
  rcases hσ₁ with rfl | rfl <;> rcases hσ₂ with rfl | rfl <;> rcases hσ₃ with rfl | rfl <;>
    linarith

end SellingClosure.Niggli
