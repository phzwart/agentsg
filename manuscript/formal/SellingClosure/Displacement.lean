import Mathlib.Topology.MetricSpace.IsometricSMul
import Mathlib.Tactic.Linarith

/-!
# Prices as displacements (Propositions A9(i) and A10)

The price of a self-map `h` of a metric `x` is its displacement `dist x (h • x)` under an
isometric action of a group `G` on a metric space `X`.  In the paper `G = GL₃(ℤ)` acts on
metric tensors of unit determinant by `𝐆 ↦ 𝐏ᵀ𝐆𝐏` (a right action; take `g • x` with
`g = 𝐏⁻¹` to make it a left action) and `dist` is the affine-invariant distance, for
which every change of basis is an isometry.  Nothing below uses more than that.
-/

namespace SellingClosure

variable {G X : Type*} [Group G] [PseudoMetricSpace X] [MulAction G X] [IsIsometricSMul G X]

/-- The price (displacement) of `g` at `x`. -/
def price (x : X) (g : G) : ℝ := dist x (g • x)

/-- The self-maps of `x` of price at most `ε`. -/
def H (x : X) (ε : ℝ) : Set G := {g | price x g ≤ ε}

/-- The candidates from `x` to `y` within `ε`: operators `P` with `dist y (P • x) ≤ ε`. -/
def C (x y : X) (ε : ℝ) : Set G := {P | dist y (P • x) ≤ ε}

theorem price_one (x : X) : price x (1 : G) = 0 := by simp [price]

/-- A10(i): prices are subadditive. -/
theorem price_mul (x : X) (g h : G) : price x (g * h) ≤ price x g + price x h := by
  unfold price
  calc dist x ((g * h) • x) ≤ dist x (g • x) + dist (g • x) ((g * h) • x) := dist_triangle _ _ _
    _ = dist x (g • x) + dist x (h • x) := by rw [mul_smul, dist_smul]

theorem price_inv (x : X) (g : G) : price x g⁻¹ = price x g := by
  unfold price
  calc dist x (g⁻¹ • x) = dist (g • x) (g • g⁻¹ • x) := (dist_smul g _ _).symm
    _ = dist x (g • x) := by rw [smul_inv_smul, dist_comm]

theorem H_mul {x : X} {ε ε' : ℝ} {g h : G} (hg : g ∈ H x ε) (hh : h ∈ H x ε') :
    g * h ∈ H x (ε + ε') :=
  le_trans (price_mul x g h) (add_le_add hg hh)

/-- A10(ii): with `P₀` the true operator of price `p₀`, the candidates lie between two
cosets of the self-maps: `P₀ H_{ε-p₀} ⊆ C_ε ⊆ P₀ H_{ε+p₀}`. -/
theorem coset_lower (x y : X) (P₀ : G) (ε : ℝ) (h : G) (hh : h ∈ H x (ε - dist y (P₀ • x))) :
    P₀ * h ∈ C x y ε := by
  unfold H price at hh; unfold C
  calc dist y ((P₀ * h) • x) ≤ dist y (P₀ • x) + dist (P₀ • x) (P₀ • h • x) := by
        rw [mul_smul]; exact dist_triangle _ _ _
    _ = dist y (P₀ • x) + dist x (h • x) := by rw [dist_smul]
    _ ≤ ε := by linarith

theorem coset_upper (x y : X) (P₀ : G) (ε : ℝ) (P : G) (hP : P ∈ C x y ε) :
    P₀⁻¹ * P ∈ H x (ε + dist y (P₀ • x)) := by
  unfold C at hP; unfold H price
  calc dist x ((P₀⁻¹ * P) • x) = dist (P₀ • x) (P • x) := by
        rw [← dist_smul P₀, mul_smul, smul_inv_smul]
    _ ≤ dist (P₀ • x) y + dist y (P • x) := dist_triangle _ _ _
    _ ≤ ε + dist y (P₀ • x) := by rw [dist_comm]; linarith

/-- A10(iii): if no self-map has a price in `(ε, 2ε]`, the self-maps of price at most `ε`
form a group. -/
def Hgroup (x : X) (ε : ℝ) (hgap : ∀ g : G, price x g ≤ 2 * ε → price x g ≤ ε)
    (hε : 0 ≤ ε) : Subgroup G where
  carrier := H x ε
  one_mem' := by simp [H, price_one, hε]
  mul_mem' := fun {a b} ha hb => hgap _ (by have := H_mul ha hb; simp only [H] at this ⊢; linarith)
  inv_mem' := fun {a} ha => by simpa [H, price_inv] using ha

/-- A10(iv): if moreover no self-map has a price in `(ε - p₀, ε + p₀]`, the candidates are
exactly one coset. -/
theorem candidates_eq_coset (x y : X) (P₀ : G) (ε : ℝ)
    (hgap : ∀ g : G, price x g ≤ ε + dist y (P₀ • x) → price x g ≤ ε - dist y (P₀ • x))
    (P : G) : P ∈ C x y ε ↔ P₀⁻¹ * P ∈ H x ε := by
  constructor
  · intro hP
    have := hgap _ (coset_upper x y P₀ ε P hP)
    simp only [H]
    have : 0 ≤ dist y (P₀ • x) := dist_nonneg
    linarith
  · intro hh
    have hh' : P₀⁻¹ * P ∈ H x (ε + dist y (P₀ • x)) := by
      simp only [H] at hh ⊢; have := dist_nonneg (x := y) (y := P₀ • x); linarith
    have := coset_lower x y P₀ ε (P₀⁻¹ * P) (hgap _ hh')
    simpa using this

/-- A10(v): the exact closure of a nearby symmetric metric bounds the near-symmetries from
both sides.  If `dist x x₀ = δ`, the symmetries of `x₀` are self-maps of `x` of price at
most `2δ`, and a self-map of `x` of price at most `ε` is a self-map of `x₀` of price at most
`ε + 2δ`. -/
theorem stab_sub_H (x x₀ : X) (g : G) (hg : g • x₀ = x₀) : price x g ≤ 2 * dist x x₀ := by
  unfold price
  calc dist x (g • x) ≤ dist x x₀ + dist x₀ (g • x) := dist_triangle _ _ _
    _ = dist x x₀ + dist (g • x₀) (g • x) := by rw [hg]
    _ = 2 * dist x x₀ := by rw [dist_smul, dist_comm x₀ x]; ring

theorem H_sub_H (x x₀ : X) (ε : ℝ) (g : G) (hg : g ∈ H x ε) :
    g ∈ H x₀ (ε + 2 * dist x x₀) := by
  unfold H price at *
  calc dist x₀ (g • x₀) ≤ dist x₀ x + dist x (g • x) + dist (g • x) (g • x₀) :=
        dist_triangle4 _ _ _ _
    _ = dist x (g • x) + 2 * dist x x₀ := by rw [dist_smul, dist_comm x₀ x, dist_comm x x₀]; ring
    _ ≤ ε + 2 * dist x x₀ := by linarith

/-- A9(i), upper bound: a symmetry `g` of `x₀` displaces `x` by at most twice the distance
from `x` to `x₀`. -/
theorem price_le_two_dist_fixed (x x₀ : X) (g : G) (hg : g • x₀ = x₀) :
    price x g ≤ 2 * dist x x₀ := stab_sub_H x x₀ g hg

/-- A9(i), equality for a twofold: if the midpoint of `x` and `g • x` is unique (as in a
space of non-positive curvature such as the positive definite matrices with the
affine-invariant distance) and `g² = 1`, the midpoint is fixed by `g`, so the price is
exactly twice the distance to the fixed set. -/
theorem twofold_midpoint_fixed (x m : X) (g : G) (hg2 : g * g = 1)
    (hm : dist x m = dist x (g • x) / 2 ∧ dist m (g • x) = dist x (g • x) / 2)
    (huniq : ∀ m' : X, dist x m' = dist x (g • x) / 2 → dist m' (g • x) = dist x (g • x) / 2 →
      m' = m) :
    g • m = m ∧ price x g = 2 * dist x m := by
  have hgx : g • g • x = x := by rw [← mul_smul, hg2, one_smul]
  refine ⟨huniq _ ?_ ?_, ?_⟩
  · calc dist x (g • m) = dist (g • g • x) (g • m) := by rw [hgx]
      _ = dist (g • x) m := dist_smul _ _ _
      _ = dist x (g • x) / 2 := by rw [dist_comm]; exact hm.2
  · calc dist (g • m) (g • x) = dist m x := dist_smul _ _ _
      _ = dist x (g • x) / 2 := by rw [dist_comm]; exact hm.1
  · unfold price; rw [hm.1]; ring

end SellingClosure
