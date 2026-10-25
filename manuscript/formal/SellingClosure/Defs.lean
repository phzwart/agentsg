import SellingClosure.Data

/-!
# Combinatorial definitions

A lattice is described by a reduced (obtuse) superbase `(d, a, b, c)` with
`d = -a - b - c`.  A lattice vector `x = x₁ a + x₂ b + x₃ c` is stored as the triple
`(x₁, x₂, x₃)`; its superbase coordinates are `(0, x₁, x₂, x₃)` on `(d, a, b, c)`.

The six conorm slots are numbered `0 … 5` in the order `da, db, dc, ab, ac, bc`.
`diff x s` is the difference `cᵢ - cⱼ` of superbase coordinates on slot `s = (i, j)`;
by Selling's identity the dot product is `x · y = ∑ₛ pₛ (diff x s) (diff y s)`.
-/

namespace SellingClosure

def vadd (x y : V3) : V3 := (x.1 + y.1, x.2.1 + y.2.1, x.2.2 + y.2.2)
def vneg (x : V3) : V3 := (-x.1, -x.2.1, -x.2.2)
def vsmul (k : Int) (x : V3) : V3 := (k * x.1, k * x.2.1, k * x.2.2)

/-- Coordinate differences on the six slots `da, db, dc, ab, ac, bc`. -/
def diff (x : V3) : Fin 6 → Int
  | 0 => -x.1
  | 1 => -x.2.1
  | 2 => -x.2.2
  | 3 => x.1 - x.2.1
  | 4 => x.1 - x.2.2
  | 5 => x.2.1 - x.2.2

/-- The parity pattern `s ∈ {0,1}³` of a vector: the coordinates of the partial sum of
the reduced superbase in the same parity class. -/
def par (x : V3) : V3 := (x.1 % 2, x.2.1 % 2, x.2.2 % 2)

/-- The integer half `k` with `x = s + 2k`. -/
def half (x : V3) : V3 := (x.1 / 2, x.2.1 / 2, x.2.2 / 2)

/-- `γₛ = Δk (Δk + Δs)` of Lemma A2. -/
def gam (x : V3) (s : Fin 6) : Int :=
  diff (half x) s * (diff (half x) s + diff (par x) s)

/-- A support pattern: which of the six conorms are non-zero. -/
abbrev Supp := Fin 6 → Bool

/-- `γₛ(x) = 0` on every slot of the support. -/
def gokB (S : Supp) (x : V3) : Bool :=
  (List.finRange 6).all fun s => !S s || gam x s == 0

def det (u₁ u₂ u₃ : V3) : Int :=
  u₁.1 * (u₂.2.1 * u₃.2.2 - u₂.2.2 * u₃.2.1)
  - u₂.1 * (u₁.2.1 * u₃.2.2 - u₁.2.2 * u₃.2.1)
  + u₃.1 * (u₁.2.1 * u₂.2.2 - u₁.2.2 * u₂.2.1)

/-- All seven partial sums of the superbase `(-(u₁+u₂+u₃), u₁, u₂, u₃)`, one per non-zero
parity class (up to sign). -/
def partials (u₁ u₂ u₃ : V3) : List V3 :=
  [u₁, u₂, u₃, vadd u₁ u₂, vadd u₁ u₃, vadd u₂ u₃, vadd (vadd u₁ u₂) u₃]

/-- The endpoints of each slot, vertices `0 = d, 1 = a, 2 = b, 3 = c`. -/
def ends : Fin 6 → Fin 4 × Fin 4
  | 0 => (0, 1) | 1 => (0, 2) | 2 => (0, 3) | 3 => (1, 2) | 4 => (1, 3) | 5 => (2, 3)

/-- One step of reachability from a set of vertices along the support. -/
def grow (S : Supp) (R : Fin 4 → Bool) : Fin 4 → Bool := fun v =>
  R v || (List.finRange 6).any fun s =>
    S s && ((ends s).1 == v && R (ends s).2 || (ends s).2 == v && R (ends s).1)

/-- The support graph on the four vectors of the superbase is connected. -/
def isConn (S : Supp) : Bool :=
  let R := grow S (grow S (grow S fun v => v == 0))
  (List.finRange 4).all R

/-- A support pattern from six bits, in slot order. -/
def ofBits (b₀ b₁ b₂ b₃ b₄ b₅ : Bool) : Supp
  | 0 => b₀ | 1 => b₁ | 2 => b₂ | 3 => b₃ | 4 => b₄ | 5 => b₅

theorem ofBits_eta (S : Supp) : ofBits (S 0) (S 1) (S 2) (S 3) (S 4) (S 5) = S := by
  funext ⟨n, hn⟩
  match n, hn with
  | 0, _ => rfl | 1, _ => rfl | 2, _ => rfl | 3, _ => rfl | 4, _ => rfl | 5, _ => rfl

/-- All 64 support patterns. -/
def allSupp : List Supp :=
  let B := [false, true]
  B.flatMap fun b₀ => B.flatMap fun b₁ => B.flatMap fun b₂ =>
  B.flatMap fun b₃ => B.flatMap fun b₄ => B.map fun b₅ => ofBits b₀ b₁ b₂ b₃ b₄ b₅

/-- The 16 spanning trees of the tetrahedron, as lists of slots. -/
def trees : List (List (Fin 6)) :=
  [[0,1,2],[0,1,4],[0,1,5],[0,2,3],[0,2,5],[0,3,4],[0,3,5],[0,4,5],
   [1,2,3],[1,2,4],[1,3,4],[1,3,5],[1,4,5],[2,3,4],[2,3,5],[2,4,5]]

def inBox (x : V3) : Prop :=
  -7 ≤ x.1 ∧ x.1 ≤ 7 ∧ -7 ≤ x.2.1 ∧ x.2.1 ≤ 7 ∧ -7 ≤ x.2.2 ∧ x.2.2 ≤ 7

def rng : List Int := (List.range 15).map fun (n : Nat) => (n : Int) - 7

def boxList : List V3 :=
  rng.flatMap fun a => rng.flatMap fun b => rng.map fun c => (a, b, c)

/-- Matrix product of two operators given by columns: `(W W')`'s columns are `W u'ᵢ`. -/
def apply (W : Op) (x : V3) : V3 :=
  vadd (vadd (vsmul x.1 W.1) (vsmul x.2.1 W.2.1)) (vsmul x.2.2 W.2.2)

def mul (W W' : Op) : Op := (apply W W'.1, apply W W'.2.1, apply W W'.2.2)

def idOp : Op := ((1, 0, 0), (0, 1, 0), (0, 0, 1))

/-- The four vectors `(d, a, b, c)` of the superbase that an operator produces:
`u₀ = -(u₁ + u₂ + u₃)`, then the three columns. -/
def sb (W : Op) : Fin 4 → V3
  | 0 => vneg (vadd (vadd W.1 W.2.1) W.2.2)
  | 1 => W.1
  | 2 => W.2.1
  | 3 => W.2.2

def maxAbs (W : Op) : Int :=
  [W.1.1, W.1.2.1, W.1.2.2, W.2.1.1, W.2.1.2.1, W.2.1.2.2,
   W.2.2.1, W.2.2.2.1, W.2.2.2.2].foldl (fun m a => max m a.natAbs) 0

end SellingClosure
