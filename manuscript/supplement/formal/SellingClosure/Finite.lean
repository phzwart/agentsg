import SellingClosure.Defs

/-!
# The finite part of the completeness of 𝒰

Everything here is a finite computation, checked by `native_decide` (which trusts the
Lean compiler in addition to the kernel), together with the elementary lemmas that turn
those computations into statements about all integer vectors.
-/

namespace SellingClosure

/-! ## Facts about the table itself -/

theorem U_length : U.length = 6096 := by native_decide

theorem U_nodup : U.Nodup := by native_decide

/-- Every operator in 𝒰 is unimodular. -/
theorem U_unimodular : U.all (fun W => det W.1 W.2.1 W.2.2 == 1 || det W.1 W.2.1 W.2.2 == -1)
    = true := by native_decide

/-- Every entry of every operator in 𝒰 lies in `{-3, …, 3}`, and some entry is `±3`. -/
theorem U_entries : U.all (fun W => maxAbs W ≤ 3) = true ∧ U.any (fun W => maxAbs W == 3) = true := by
  constructor <;> native_decide

/-- 𝒰 is closed under inversion. -/
theorem U_inv : U.all (fun W => U.any fun W' => mul W W' == idOp) = true := by native_decide

/-- The 48 swaps: relabellings of the reduced superbase, as column operators. -/
def swaps : List Op := [
  ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
  ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
  ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
  ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
  ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
  ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
  ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
  ((0, -1, 0), (0, 0, -1), (-1, 0, 0)),
  ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
  ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
  ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
  ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
  ((-1, -1, -1), (0, 1, 0), (0, 0, 1)),
  ((1, 1, 1), (0, -1, 0), (0, 0, -1)),
  ((-1, -1, -1), (0, 0, 1), (0, 1, 0)),
  ((1, 1, 1), (0, 0, -1), (0, -1, 0)),
  ((0, 1, 0), (-1, -1, -1), (0, 0, 1)),
  ((0, -1, 0), (1, 1, 1), (0, 0, -1)),
  ((0, 1, 0), (0, 0, 1), (-1, -1, -1)),
  ((0, -1, 0), (0, 0, -1), (1, 1, 1)),
  ((0, 0, 1), (-1, -1, -1), (0, 1, 0)),
  ((0, 0, -1), (1, 1, 1), (0, -1, 0)),
  ((0, 0, 1), (0, 1, 0), (-1, -1, -1)),
  ((0, 0, -1), (0, -1, 0), (1, 1, 1)),
  ((-1, -1, -1), (1, 0, 0), (0, 0, 1)),
  ((1, 1, 1), (-1, 0, 0), (0, 0, -1)),
  ((-1, -1, -1), (0, 0, 1), (1, 0, 0)),
  ((1, 1, 1), (0, 0, -1), (-1, 0, 0)),
  ((1, 0, 0), (-1, -1, -1), (0, 0, 1)),
  ((-1, 0, 0), (1, 1, 1), (0, 0, -1)),
  ((1, 0, 0), (0, 0, 1), (-1, -1, -1)),
  ((-1, 0, 0), (0, 0, -1), (1, 1, 1)),
  ((0, 0, 1), (-1, -1, -1), (1, 0, 0)),
  ((0, 0, -1), (1, 1, 1), (-1, 0, 0)),
  ((0, 0, 1), (1, 0, 0), (-1, -1, -1)),
  ((0, 0, -1), (-1, 0, 0), (1, 1, 1)),
  ((-1, -1, -1), (1, 0, 0), (0, 1, 0)),
  ((1, 1, 1), (-1, 0, 0), (0, -1, 0)),
  ((-1, -1, -1), (0, 1, 0), (1, 0, 0)),
  ((1, 1, 1), (0, -1, 0), (-1, 0, 0)),
  ((1, 0, 0), (-1, -1, -1), (0, 1, 0)),
  ((-1, 0, 0), (1, 1, 1), (0, -1, 0)),
  ((1, 0, 0), (0, 1, 0), (-1, -1, -1)),
  ((-1, 0, 0), (0, -1, 0), (1, 1, 1)),
  ((0, 1, 0), (-1, -1, -1), (1, 0, 0)),
  ((0, -1, 0), (1, 1, 1), (-1, 0, 0)),
  ((0, 1, 0), (1, 0, 0), (-1, -1, -1)),
  ((0, -1, 0), (-1, 0, 0), (1, 1, 1))]

theorem swaps_length : swaps.length = 48 := by native_decide

/-- 𝒰 is closed under right multiplication by every swap (relabelling the target cell). -/
theorem U_swaps : U.all (fun W => swaps.all fun Q => U.contains (mul W Q)) = true := by native_decide

/-! ## Supports -/

theorem trees_conn : trees.all (fun T => isConn (fun s => T.contains s)) = true := by native_decide

/-- There are 38 connected supports, the 38 faces of the paper. -/
theorem faces_count : (allSupp.filter isConn).length = 38 := by native_decide

/-- Every connected support contains a spanning tree. -/
theorem conn_has_tree : ∀ b₀ b₁ b₂ b₃ b₄ b₅ : Bool,
    let S := ofBits b₀ b₁ b₂ b₃ b₄ b₅
    (!isConn S || trees.any fun T => T.all fun s => S s) = true := by
  decide

/-! ## The finite check

For every connected support `S`, every unimodular triple of vectors from the box whose
seven partial sums all have `γ = 0` on `S` is an operator of 𝒰. -/

def checkSupp (S : Supp) : Bool :=
  let V := boxList.filter (gokB S)
  V.all fun u₁ => V.all fun u₂ => V.all fun u₃ =>
    !((partials u₁ u₂ u₃).all (gokB S) &&
      (det u₁ u₂ u₃ == 1 || det u₁ u₂ u₃ == -1)) || U.contains (u₁, u₂, u₃)

theorem finite_check : ∀ b₀ b₁ b₂ b₃ b₄ b₅ : Bool,
    let S := ofBits b₀ b₁ b₂ b₃ b₄ b₅
    (!isConn S || checkSupp S) = true := by
  native_decide

/-- Conversely, each operator of 𝒰 maps the reduced superbase to a superbase that is
obtuse for every metric on some face: on a connected support every pairwise product of
the new superbase has non-positive coefficient. -/
def signOK (S : Supp) (W : Op) : Bool :=
  (List.finRange 4).all fun a => (List.finRange 4).all fun b =>
    a == b || (List.finRange 6).all fun s =>
      !S s || decide (diff (sb W a) s * diff (sb W b) s ≤ 0)

theorem U_realised : U.all (fun W => (allSupp.filter isConn).any fun S => signOK S W) = true := by
  native_decide

/-- On every face, every operator of 𝒰 whose seven partial sums have `γ = 0` on the face
passes the sign test there: with Lemma A1 this is the converse of the Voronoi
characterisation used for `arrow_mem_U`, and it makes the arrows depend only on the zero
pattern. -/
theorem zero_pattern_check : ∀ b₀ b₁ b₂ b₃ b₄ b₅ : Bool,
    let S := ofBits b₀ b₁ b₂ b₃ b₄ b₅
    (!isConn S || U.all fun W => !((partials W.1 W.2.1 W.2.2).all (gokB S)) || signOK S W)
      = true := by
  native_decide

/-! ## From the computation to all vectors -/

theorem rng_mem {a : Int} (h1 : -7 ≤ a) (h2 : a ≤ 7) : a ∈ rng := by
  simp only [rng, List.mem_map, List.mem_range]
  exact ⟨(a + 7).toNat, by omega, by omega⟩

theorem boxList_mem {x : V3} (h : inBox x) : x ∈ boxList := by
  obtain ⟨a, b, c⟩ := x
  simp only [inBox] at h
  simp only [boxList, List.mem_flatMap, List.mem_map]
  exact ⟨a, rng_mem (by omega) (by omega), b, rng_mem (by omega) (by omega),
    c, rng_mem (by omega) (by omega), rfl⟩

theorem gokB_iff (S : Supp) (x : V3) : gokB S x = true ↔ ∀ s, S s = true → gam x s = 0 := by
  simp only [gokB, List.all_eq_true, List.mem_finRange, true_implies, Bool.or_eq_true,
    Bool.not_eq_eq_eq_not, Bool.not_true, beq_iff_eq]
  exact forall_congr' fun s => by cases S s <;> simp

/-- Lemma A2 (window), combinatorial form: `γ = 0` along a spanning tree puts a vector in
the box `[-7, 7]³`. -/
theorem box_of_tree (T : List (Fin 6)) (hT : T ∈ trees) (x : V3)
    (h : ∀ s ∈ T, gam x s = 0) : inBox x := by
  obtain ⟨a, b, c⟩ := x
  simp only [trees, List.mem_cons, List.not_mem_nil, or_false] at hT
  simp only [gam, diff, half, par, Int.mul_eq_zero] at h
  simp only [inBox]
  rcases hT with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;>
  · simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at h
    omega

theorem box_of_conn (S : Supp) (hS : isConn S = true) (x : V3) (h : gokB S x = true) :
    inBox x := by
  have := conn_has_tree (S 0) (S 1) (S 2) (S 3) (S 4) (S 5)
  simp only [ofBits_eta, hS, Bool.not_true, Bool.false_or, List.any_eq_true,
    List.all_eq_true] at this
  obtain ⟨T, hT, hTS⟩ := this
  exact box_of_tree T hT x fun s hs => (gokB_iff S x).mp h s (hTS s hs)

/-- **The finite part of completeness.** On a connected support `S`, every unimodular
triple whose seven partial sums have `γ = 0` on `S` is an operator of 𝒰. -/
theorem mem_U_of_gok (S : Supp) (hS : isConn S = true) (u₁ u₂ u₃ : V3)
    (hg : ∀ w ∈ partials u₁ u₂ u₃, gokB S w = true)
    (hdet : det u₁ u₂ u₃ = 1 ∨ det u₁ u₂ u₃ = -1) : (u₁, u₂, u₃) ∈ U := by
  have hc := finite_check (S 0) (S 1) (S 2) (S 3) (S 4) (S 5)
  simp only [ofBits_eta, hS, Bool.not_true, Bool.false_or] at hc
  have hmem : ∀ u ∈ partials u₁ u₂ u₃, u ∈ boxList.filter (gokB S) := fun u hu =>
    List.mem_filter.mpr ⟨boxList_mem (box_of_conn S hS u (hg u hu)), hg u hu⟩
  have h1 := hmem u₁ (by simp [partials])
  have h2 := hmem u₂ (by simp [partials])
  have h3 := hmem u₃ (by simp [partials])
  simp only [checkSupp, List.all_eq_true] at hc
  have := hc u₁ h1 u₂ h2 u₃ h3
  have hall : (partials u₁ u₂ u₃).all (gokB S) = true := List.all_eq_true.mpr hg
  have hd : (det u₁ u₂ u₃ == 1 || det u₁ u₂ u₃ == -1) = true := by
    rcases hdet with h | h <;> simp [h]
  rw [hall, hd] at this
  simpa using this

end SellingClosure
