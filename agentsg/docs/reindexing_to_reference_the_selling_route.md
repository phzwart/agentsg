# Reindexing to a reference: the Selling route

*How agentsg reindexes an arbitrary unit cell onto a reference setting using the
Selling (Delaunay) reduction and the change-of-basis group it induces.*

---

## 1. The problem

In serial and time-resolved crystallography you index thousands of frames
independently. Each frame's cell is reported in whatever basis the indexer
happened to choose. Before you can merge, compare trajectories, or match a frame
to a known reference form, you must bring every cell into **one common setting** —
that is, find the integer change of basis **P** (columns = new basis vectors in
old coordinates, det = ±1) that maps the frame's cell onto the reference.

The difficulty is that *the same lattice has many equally-valid cells.* A
monoclinic P2 lattice, for example, can be written with its 2-fold along **a**,
**b**, **c**, or an oblique direction, and with several choices of the two
in-plane vectors. These are not small numerical perturbations — they are
genuinely different six-parameter tuples describing the identical lattice.

### Why a naive numerical match fails

Take one P2 lattice and sweep the full Selling change-of-basis group over it
(see `agentsg.cell.selling_settings`). It produces **48 settings**, collapsing to
**24 numerically distinct cells** (the inversion −I is invisible to a metric
tensor, so cells come in identical ± pairs). A few of them:

| a | b | c | α | β | γ | ‖G‖ (Frobenius) |
|---|---|---|---|---|---|---|
| 6.000 | 8.000 | 11.172 | 112.29 | 90.00 | 90.00 | 152.5 |
| 6.000 | 8.000 | 12.530 | 107.47 | 118.61 | 90.00 | 185.6 |
| 6.000 | 11.172 | 8.000 | 112.29 | 90.00 | 90.00 | 152.5 |
| 6.000 | 11.172 | 12.530 | 130.50 | 118.61 | 90.00 | 246.3 |

All of these are **the same lattice**. Yet:

- the Frobenius norm of the metric tensor spans **152.5 – 254.9** across the 24
  settings (a 67 % range);
- a plain Euclidean (L2) distance on the raw six-vector reaches **58.0** between
  two settings of this one lattice.

So a "reduce and compare parameters with an L2 tolerance" approach would scatter
one crystal across two dozen well-separated points and declare them different
lattices. **L2 on cell parameters is not a lattice invariant.** This is the
reindexing ambiguity, made numerical.

### Why the Niggli / Le Page route fails

The classical alternative is: Niggli-reduce both cells, then recover the
relating operator from the lattice holohedry — typically the Le Page / Lebedev
set of candidate two-folds, gated by an angular tolerance (`max_delta`), or the
tolerance-widened metric-automorphism coset used by dials.cosym and by
`agentsg.cell.ambiguity`. That works when both frames land on the *same* side of
every reduction boundary. It fails when they do not.

Niggli (and Buerger) reduction is **discontinuous on the boundary of the reduced
cone**. Near an edge degeneracy — `a ≈ b`, a monoclinic angle near 90°, etc. — a
hair of noise on one frame flips the reduced cell to a different but equivalent
setting (edges swap; the change-of-basis jumps). Two frames of the *same*
lattice can therefore Niggli-reduce to two different canonical cells. The
operator that relates those two reduced cells is a **cell-choice / reduction-flip
transform**, not an element of the crystal's true holohedry.

The usual patch is to enlarge the symmetry group with a tolerance so that the
flip falls inside the Le Page gate or the tolerance metric group. That is a
**bet**: the flip operator is recovered only if it happens to sit inside the
enlarged group. Past the tolerance — a looser near-degeneracy, or measurement
noise larger than `max_delta` — the operator drops out and the flip is **silently
missed**. Tighten the gate and you miss genuine flips; loosen it and you admit
spurious automorphisms. There is no single delta that is both complete and
safe, because the object you need is not a symmetry of either reduced cell: it
is the discontinuity of the reduction itself.

So the Niggli / Le Page route does not fail because Le Page is a bad symmetry
test. It fails because **reindexing across a reduction flip is not a symmetry
problem**. It is a reduction-domain problem, and no amount of angular tolerance
on a single Niggli cell closes the domain.

---

## 2. The solution

The fix has two ingredients, both built on the **Selling / Delaunay reduction**.
Selling is continuous across the same boundaries that flip Niggli: both settings
reduce to the **same obtuse superbase**, so the relating operator is recovered by
a fixed, finite group search — no tolerance-thresholded symmetry group to fall
out of.

### 2a. Reduce, and match on a Selling-invariant, not on the cell

The Selling reduction represents a lattice by its **superbase**
{v₀, v₁, v₂, v₃} with v₀ = −(v₁+v₂+v₃), reduced so every pairwise product
vᵢ·vⱼ ≤ 0 (obtuse). The six **conorms** pᵢⱼ = −vᵢ·vⱼ are a signature of the
lattice that does not depend on which setting you started from. Matching two
lattices means matching their conorm signatures, not their cell parameters.

Sweeping this signature over the **Selling group** — the order-48 group
S₄ × {±I} of superbase relabellings and their negations
(`agentsg.cell.selling_group`) — enumerates every candidate change of basis
that could relate your cell to the reference. Each is integer and unimodular by
construction, so it can only *reindex* the lattice, never distort it.

### 2b. Close the Delaunay boundary (the reduction-flip trap)

The obtuse superbase is unique only in the *interior* of a Delaunay type. When a
conorm passes through zero — e.g. a monoclinic angle near 90°, where one product
vᵢ·vⱼ ≈ 0 — two settings of the same lattice reduce to genuinely **different**
superbases. Matching only within the 24 relabellings of your single superbase
then **misses** the operator relating them. This is the notorious reduction-flip
problem, reappearing inside the superbase.

agentsg closes this by enumerating the finite set of boundary-flip variants of
*both* cells (`superbase_variants`, controlled by `boundary_rel`) and matching
across the closure. That restores completeness: if the two cells are the same
lattice, the operator relating them is guaranteed to be found.

### 2c. The verification residual

A candidate operator P is accepted if P maps cell A's metric onto cell B's within
a **relative residual** on the metric tensor. The residual is 0 when A and B are
exactly the same lattice and grows smoothly with any real deformation between
them, so it doubles as a same-lattice / different-lattice discriminant. The
threshold is calibrated from a baseline of known pairs (`calibrate_verify_tol`),
not guessed.

---

## 3. The API

You do not sweep the group by hand. The whole procedure — reduce both cells,
enumerate the boundary-complete Selling group, match on conorms, verify on the
metric residual — is packaged:

```python
from agentsg.cell.canonical import (
    reindexing_via_canonical,           # -> all valid operators (the coset)
    reindexing_operator_via_canonical,  # -> one operator, or None
    best_reindex_with_residual,         # -> best operator + metric residual
)

ops = reindexing_via_canonical(my_cell, reference_cell)   # list of 3x3 int tuples
P,  r = best_reindex_with_residual(my_cell, reference_cell)
```

Everything is exact integer / rational arithmetic; no runtime dependencies.

---

## 4. Results

### 4a. The invariant collapses what L2 scatters

Reduce the 24 distinct settings above and compute the **Kurlin root distance**
(a provably setting-blind lattice invariant, `root_distance`) from each to the
reference:

| metric | spread across the 24 settings of ONE lattice |
|---|---|
| raw six-vector L2 | up to **58.0** |
| metric tensor ‖G‖ | **152.5 – 254.9** (range 102) |
| **Kurlin root distance** | **≤ 0.017 Å** (numerical noise) |

The root invariant maps all 24 settings to essentially one point; the naive
measures do not.

### 4b. Reindexing returns a coset, not one operator

Reindex a P2 cell `(8, 6, 11, 90, 70, 90)` onto its reduced reference
`(8, 11.172, 6, 90, 90, 112.29)`. `reindexing_via_canonical` returns **four**
operators — all of which reproduce the reference cell exactly:

| op | det | P (rows) |
|---|---|---|
| 0 | +1 | (−1, 1, 0), (0, 0, −1), (0, −1, 0) |
| 1 | −1 | (−1, 1, 0), (0, 0, 1), (0, −1, 0) |
| 2 | +1 | (1, −1, 0), (0, 0, −1), (0, 1, 0) |
| 3 | −1 | (1, −1, 0), (0, 0, 1), (0, 1, 0) |

A reindexing operator is **never unique**: the valid operators form a *coset* —
the geometric solution multiplied by the lattice symmetry (here the 2-fold's
holohedry). Geometry alone cannot choose among them; if the crystal has a
merohedral ambiguity, an intensity correlation over exactly these operators is
what breaks the tie.

### 4c. The residual tracks real deformation

Deform the cell's a-axis away from the reference and reindex. The metric residual
stays ~0 while the lattice is unchanged and rises monotonically with the
deformation, exactly mirroring the root distance:

| Δa (Å) | metric residual | root distance (Å) |
|---|---|---|
| 0.00 | 1.4 × 10⁻¹⁴ | 0.000 |
| 0.05 | 0.80 | 0.056 |
| 0.20 | 3.24 | 0.197 |
| 0.50 | 8.25 | 0.435 |
| 1.00 | 17.0 | 0.965 |
| 2.00 | 36.0 | 2.031 |

At Δa = 0 the residual is machine zero — the operator is *exact*. As the cell
deforms the residual grows smoothly, so a calibrated threshold cleanly separates
"same lattice, reindex it" from "genuinely different lattice."

---

## 5. Summary

To reindex a cell onto a reference the Selling way:

1. **Selling-reduce** both cells to their obtuse superbases, tracking the integer
   change of basis.
2. **Sweep the boundary-complete Selling group** (order 48, plus Delaunay
   boundary variants) to enumerate every candidate operator.
3. **Match on the conorm signature** — a Selling invariant — not on the raw cell
   parameters.
4. **Verify on the metric residual**, accepting operators below a calibrated
   relative threshold.

The result is the **complete coset** of valid reindexing operators together with
their residuals. Geometry surfaces every choice it cannot itself decide; an
intensity tie-break, when needed, acts over that same coset. This is why the
Selling route is robust where a reduce-and-L2 comparison, and where a
Niggli-reduce-then-Le-Page-tolerance bet, are not: it matches lattices across
reduction domains, not settings of a single discontinuous reduced cell.

*See `agentsg.cell.canonical`, `agentsg.cell.selling_group`,
`agentsg.cell.selling_settings`, and `agentsg.cell.rootform`; the numbers above
are reproduced by `examples/selling_extended_hm.py` and the
`test_selling_*` suites. On the Niggli reduction-flip literature and why
tolerance groups only partially close it, see `docs/REDUCTION_FLIP_LITERATURE.md`.*
