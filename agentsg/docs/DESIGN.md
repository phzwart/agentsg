# agentsg — design notes

Phase 1 (symmetry): exact-rational space-group algebra from generators —
**complete**, all 230 groups, exhaustively validated.
Phase 2 (unit cell): numeric metric math, Niggli reduction, and the metric ↔
point-group bridge — **implemented and validated**.

## Why this exists

cctbx's sgtbx does the same job but (a) represents translations with a fixed
denominator (`tr_vec.sg_t_den = 12`) baked in as a C++ `int`, and (b) is
entangled — `uctbx.h` forward-declares `sgtbx::rot_mx` and
`sgtbx::change_of_basis_op` before it can even describe a bare metric tensor.
This package fixes both: exact rationals everywhere (no fixed base), and a hard
package boundary between symmetry (exact) and cell (numeric), crossed at exactly
one point.

**Zero runtime dependencies.** The package imports nothing outside the Python
standard library. `gemmi` and `spglib` appear only in the test suite, as
independent oracles.

## Phase 1 architecture

- `linalg.py` — `Vector3`/`Matrix3` over `fractions.Fraction`. Exact
  add/matmul/inverse (via adjugate/determinant), `mod1()` reduction. No fixed
  denominator anywhere; a 1/2, a 1/3, and a 1/24 all just work.
- `symmetry_op.py` — `SymmetryOp(W, w)`: `x' = Wx + w`. `W` integer (det = ±1)
  in a conventional basis; `w` exact rational, reduced into [0, 1).
  Composition, inverse, and an xyz-triplet parser/printer.
- `group.py` — closure-by-composition: BFS from generators plus centring,
  capped at `max_order=192`. `point_group()`, `centering_translations()`,
  `is_systematically_absent()` — all derived from the operator list.
- `hall.py` — **Hall-symbol parser** (Hall 1981 / ITA). Turns a compact Hall
  symbol into `SymmetryOp` generators + centring vectors that feed
  `close_group()`. Handles lattice symbols (with the `-` centrosymmetric
  prefix), rotation/screw/glide generators, the implied-default-axis rules, and
  the parenthesised origin shift. This is the "verified generator source" the
  original design called for.
- `space_groups.py` — the **230 standard-setting groups** as embedded literal
  data (number, Hermann-Mauguin, Hall, crystal system) with an O(1) lookup API
  (`space_group(key)` by number / HM / Hall). No operation lists are stored:
  generators are produced on demand by parsing the Hall symbol and closing.
- `wyckoff.py` — site symmetry / orbits / multiplicities, plus an exact
  fixed-locus solver (rational RREF of `(W−I)x = −w` giving point/line/plane
  fixed sets).
- `reflections.py` — reflection-condition reporter: classifies the derived
  systematic absences into the familiar integral/zonal/serial families.
- `change_of_basis.py` — `ChangeOfBasis(P, p)` for reindexing, convention
  derived and validated (see below).
- `generators.py` — the original 6-entry seed table, retained for
  back-compatibility (the full table now lives in `space_groups.py`).
- `identify.py` — ops → IT number / Hall by matching the 230 tabulated
  closed sets; origin-shift recovery via exact `(W−I)p = Δw` over
  ``Fraction`` (no fixed twelfths grid). Returns tabulated Hall, not a
  from-scratch Hall encoder (`BuildHSym`). Full ``FixAxes`` search for
  arbitrary basis permutations is deferred.
- `semi_invariants.py` — structure semi-invariants from allowed origins
  (`(W−I)o ∈` centring). Floating origins (nullspace of stacked ``W−I``:
  P1 / unique-axis P2·P3·P4·P6 / …) are exposed via ``floating_origin_basis``,
  pinned to zero in recovered CoBs, and become modulus-0 s.i. constraints;
  only the Cheshire torsion is discretised (denominators from rotation
  orders + translations, so P3 finds `(1/3,2/3,0)`). Trial bases match
  SgInfo on non-absent reflections.
- Reciprocal-space helpers in `group.py` / `reflections.py` —
  ``phase_restriction``, ``equivalent_reflections`` (SgInfo ``sghkl``).

## Validation methodology (test-only oracles)

Every claim in phase 1 is checked against an independent implementation:

- **All 230 operation sets** (`tests/test_all_230.py`): for each space group
  1..230, agentsg closes the group from its parsed Hall symbol and the result
  is required to equal gemmi's operation set *exactly* — same rotation part,
  same translation mod 1. 230/230 pass. Group order and point-group order are
  additionally cross-checked, with spglib spot-checks.
- **Systematic absences** (`tests/test_reflections.py`): agentsg's absence flag
  matches gemmi's for every reflection in a full hkl block across all 230
  groups (167 440 reflections, zero disagreements).
- **Wyckoff multiplicities** (`tests/test_wyckoff.py`): general-position
  multiplicity equals the group order for all 230; the orbit-stabiliser
  identity `mult × |site symmetry| = |G|` holds for every probed point; special
  positions match a spglib-orbit oracle.
- **Unit-cell math** (`tests/test_cell_*.py`): volumes, reciprocal cells,
  d-spacings, orthogonalisation, and Niggli reduction all match gemmi to ~1e-14
  across all seven crystal systems and hundreds of random cells.

The oracles are a **test dependency only** — never imported at runtime, as the
"import agentsg with the oracles hidden" check demonstrates.

## The change-of-basis convention, and how it was checked

Convention adopted (derived from first principles, in `change_of_basis.py`'s
docstring):

- columns of `P` = new basis vectors in terms of the old ones
- `x' = P⁻¹(x − p)` for fractional coordinates (contravariant to the basis)
- `W' = P⁻¹WP`, `w' = P⁻¹(Wp + w − p)` for symmetry operators
- Miller indices transform *with* `P`: `h' = h·P`

Validated (not just derived) against the hexagonal → rhombohedral (obverse)
transform (ITA Table 5.1.3.1): the 9-element hexagonal-setting `R3` group
collapses to exactly the 3-element rhombohedral point group, and both
R-centring vectors reduce to lattice points. See
`tests/test_change_of_basis.py`.

**Subtlety kept from the original notes:** round-tripping a `SymmetryOp` through
a change of basis whose `P` has non-unit determinant is only guaranteed up to a
coset of the *original* centring — the forward `mod1()` reduction erases the
distinction between operators differing by a centring translation. The test
encodes the correct (weaker) invariant.

## The one interface point (exact ↔ numeric)

`cell/constraints.py` is the *only* file in `cell/` allowed to import from the
symmetry side. It carries the identity

    W^T G W = G   for every point-group operation W,

and three functions built on it: `metric_is_invariant` (check a numeric metric
against an exact point group), `symmetrize_metric` (Reynolds-average a noisy
metric onto the invariant subspace), and `free_metric_parameters` (the
dimension of the invariant metric space). The last *derives* the crystal-system
cell restrictions — 6 (triclinic), 4 (monoclinic), 3 (orthorhombic), 2
(tetragonal/trigonal/hexagonal), 1 (cubic) — from the point group alone, for
all 230 groups. No crystal-system table is consulted.

## Phase 2 (unit cell) — implemented

- `cell/metric.py` — `UnitCell(a,b,c,α,β,γ)`: metric tensor `G`, volume,
  reciprocal cell + reciprocal metric `G⁻¹`, orthogonalisation /
  fractionalisation matrices, d-spacing, `d*²`, Bragg 2θ. Real-valued; imports
  nothing from the symmetry side.
- `cell/reduction.py` — Niggli reduction via the stabilised
  Grosse-Kunstleve/Sauter/Adams algorithm (*Acta Cryst.* A60, 1–6, 2004), with
  a relative epsilon for all comparisons (the naive Křivý–Gruber version cycles
  on near-degenerate cells). Returns the reduced cell *and* the integer change
  of basis `M`; `M^T G M` reproduces the reduced metric exactly, and the result
  is lattice-invariant (any skewed superbasis of the same lattice reduces to the
  identical Niggli cell) and idempotent.

## Settings, change of basis, and metric symmetry

Following Zwart, Grosse-Kunstleve & Adams, *Exploring Metric Symmetry* (IUCr
Comp. Comm. Newsletter 7, 2006), agentsg represents **any** setting as a base
space group with an **attached change of basis**, and uses the same machinery to
compare cells and find metric (pseudo-)symmetry.

- **Notation convention** (`setting.py`). A setting is written
  `<base> (<cob>)`, e.g. `Hall: I 4 2 3 (y+z,x+z,x+y)` or `P21 (2a,a+b,c-a)`.
  The three comma-separated fields are the **columns** of the change-of-basis
  matrix `P` — the new basis vectors expressed in the old basis — matching the
  convention already fixed in `change_of_basis.py`
  (`(a',b',c') = (a,b,c)·P`, `x' = P⁻¹(x−p)`). Letters `x,y,z` and `a,b,c` are
  interchangeable; coefficients accept `2a`, `2*x`, `x-y`, `a/2`, and an optional
  constant term becomes the origin shift `p`. When `det(P) ≠ 1` the transform
  rescales the lattice: integer lattice translations of the base map to
  fractional vectors under `P⁻¹`, and closing the group surfaces them as
  **centring** operators. That is precisely how the notation expresses added
  lattice symmetry — verified on the paper's insulin example, where
  `I 4 2 3 (y+z,x+z,x+y)` (det 2) takes order 48 → 96 by acquiring the I-centring
  translation, and the metric transforms to the conventional cubic cell.

- **Lattice symmetry** (`lattice_symmetry.py`). The holohedry of a reduced cell
  is found by the Le Page (1982) / Lebedev *et al.* (2006) method. The 480
  integer matrices with entries in {−1,0,1}, det +1, whose powers stay in
  {−1,0,1} are **computed on import** (not tabulated), as are the 81 two-folds
  among them. For each candidate two-fold the "Le Page delta" — the angle
  between its direct-space axis (measured with `G`) and reciprocal-space axis
  (with `G*`) — is tested against an angular tolerance; accepted two-folds are
  closed together with the inversion centre (every lattice is centrosymmetric)
  to give the holohedry. Validated against spglib on random cells and against
  the seven ideal crystal systems.

- **Sublattices** (`cell/sublattice.py`). Index-*d* sublattices are enumerated in
  **Hermite normal form**, avoiding brute force. Because agentsg's basis-vector
  convention is columns-as-new-vectors, the canonical form reduces each
  above-diagonal entry modulo its **row** pivot (`0≤b<a, 0≤c<a, 0≤e<d`); this
  gives exactly one matrix per sublattice, and the counts reproduce OEIS
  A001001 (1, 7, 13, 35, 31, 91, …).

- **Cell comparison** (`cell/compare.py`). Two cells are related by the
  "lego / target" construction: Niggli-reduce both, take the smaller as a
  building block, enumerate its index-*d* sublattices for `d = round(V_target /
  V_lego)`, transform (`G' = MᵀGM`) and Niggli-reduce each candidate, and accept
  those matching the target within a length (%) and angle (°) tolerance. This
  reproduces the paper's §3.2 result exactly: Native P2₁2₁2₁ vs SeMet1 P2₁ →
  `M = [[2,1,0],[0,1,0],[0,0,1]]`, resulting cell 115.6 115.6 148.9 90 90 115.4.

This layer respects the exact/numeric boundary: the lattice-symmetry and
comparison code is numeric (it works on a real metric tensor and a tolerance),
but every accepted operation is an **exact integer matrix**, so results feed
straight back into the exact symmetry algebra (e.g. characterised as a Hall
symbol + change of basis via `setting.py`).

## Remaining / possible extensions (not blocking)

- **Wyckoff letters** (a, b, c…): the numeric content (orbit, multiplicity,
  site symmetry, fixed locus) is fully computed; only the historical
  letter-to-orbit labelling is not reproduced. Adding the ITA labels would be a
  lookup table.
- **Asymmetric units**: reciprocal CCP4/gemmi conditions and conventional
  real-space ASU bricks for all 230 are implemented (`asu.py` / `asu_data.py`).
  A metric Dirichlet / Voronoi ASU with a sphericity optimiser over Cheshire /
  floating-origin gauges is available as an experimental path. The full
  cctbx/ITA polyhedral real-space inequality gallery (hundreds of half-spaces
  per group) remains a follow-up.
- **Harker sections**: derived algebraically from each Seitz op via the left
  nullspace of `(I−W)` (`harker.py`); no editorial tables.
- **Dual origin choices** (e.g. Fd-3m origins 1 vs 2): modelled as an
  origin-shift `ChangeOfBasis` between generator sets; the table currently
  carries the standard setting per group. `identify_space_group` recovers
  origin shifts; non-origin setting changes still need an explicit CoB or a
  future ``FixAxes`` pass.
- **Hall symbol generation** (`BuildHSym`): identification returns the
  tabulated Hall for the matched standard setting; encoding an arbitrary
  operator list as a fresh Hall string is not implemented.
- **Twin-law search**: the numeric machinery (metric symmetry, tolerance
  checks) and the exact `W`s are now both present — `lattice_symmetry` gives the
  holohedry and `compare_cells` the sublattice relations. A twin-law enumerator
  (cosets of the crystal point group in the lattice point group) is a short
  routine on top of these.
- **Alternate settings** (the ~531 non-standard settings): available now via
  `SpaceGroupSetting` — a base group from the 230 plus an attached change of
  basis — rather than as separate table entries.

## The lattice manifold (G6 / S6 embedding)

The `cell/g6.py` module reframes lattice space as a **continuous manifold**
rather than a discrete catalogue of Bravais types. This is the substrate for
treating crystallography as inference over continuous structural state spaces
(static, serial, time-resolved, operando, and heterogeneous ensembles on one
footing) rather than as a set of isolated, independently-refined models.

Two embeddings, both due to Andrews & Bernstein:

- **G6** (Niggli): a lattice is the point `g = (a·a, b·b, c·c, 2b·c, 2a·c, 2a·b)`
  — the metric tensor as a vector. Andrews & Bernstein (1988; *Geometry of
  Niggli Reduction II: BGAOL*, J. Appl. Cryst. 47, 2014).
- **S6** (Selling/Delaunay): `s = (b·c, a·c, a·b, a·d, b·d, c·d)`, `d = -(a+b+c)`
  — cleaner boundary structure, faster distances (Andrews, Bernstein & Sauter,
  *Selling reduction versus Niggli reduction*, Acta Cryst. A75, 2019).

**Why a naive distance fails, and the fix.** The reduced-cell cone has boundary
polytopes (e.g. `b = c`, an angle through 90°) where two nearly-identical
lattices reduce to representations that are *far apart* in raw G6 — the
reduction flip. The boundary transformations that relate them are the reduction
operations, and they are **not isometric**, so Euclidean G6 distance jumps at a
boundary. `g6_distance`/`s6_distance` remove the discontinuity by minimising over
a bounded orbit of {−1,0,1} unimodular transforms (which contains the
reduction-flip / cell-choice transforms) — the exact-arithmetic core of the
Andrews–Bernstein NCDist idea. (NCDist's full "Follower" additionally chases
successive boundaries for arbitrarily-separated database cells; the bounded orbit
is exact for the reduction flip and for local manifold geometry.)

**Symmetry as a continuous field.** `distance_to_symmetry(cell, point_group)`
returns the G6 distance from a lattice to the subspace of metrics exactly
invariant under that point group — the Reynolds-averaged (symmetrised) metric.
Zero means the lattice has the symmetry; the value grows smoothly as the lattice
distorts away from it. This is BGAOL's "distance to a Bravais-lattice subspace",
and it is the operational form of *symmetry is continuous rather than binary*:
instead of a tolerance-gated yes/no (which is exactly what produces the cliffs
elsewhere in cell reduction), each candidate holohedry has a smooth deficiency
field over the manifold, and `symmetry_deficiency_spectrum` reports the whole
set at once.

**What this enables (and what it is not).** With a boundary-aware metric and a
continuous symmetry field, lattice trajectories (temperature, pressure, time)
become smooth paths on the manifold, cell clustering across a serial dataset is a
nearest-neighbour problem in G6/S6 (as in Andrews–Bernstein NCDist and its use by
Zeldin et al. 2015), and "which Bravais type" becomes "how far to each subspace".
agentsg provides the metric primitives exactly and dependency-free; it does not
(yet) provide the neartree database index or the full multi-boundary Follower —
those are engineering layers on top of these primitives for large-scale cell
database search.

## The root-invariant database, and why roots are computed on the primitive cell

The root-invariant layer (`rootform.py`, `neartree.py`, `celldb.py`,
`pdb_app.py`) indexes lattices by Kurlin's (2022) complete continuous root
invariant: obtuse superbase → six conorms → root products → sorted six-tuple.
Equality of root invariants tests lattice isometry directly, with no orbit
minimisation, and Euclidean distance between them is a true metric on lattice
space — the substrate for nearest-neighbour cell search over the whole PDB.

A correctness invariant governs the whole layer: **the root invariant is an
invariant of the lattice, i.e. of the full translation group, which is the
*primitive* lattice.** A deposited unit cell is the *conventional* cell; for a
centred Bravais type (A, B, C, I, F, R, and the "H" hexagonal-axes label for
rhombohedral groups) the conventional corner lattice is only a sublattice — it
omits the centring nodes. Feeding the conventional basis to Selling reduction
describes the wrong lattice and yields a root that is wrong by tens of Ångström
in root-product units (45–130 Å for common centred groups). `primitive.py`
therefore reduces every centred conventional cell to its primitive cell (IT Vol.
A Table 5.1.3.1 matrices, det = 1/multiplicity) before the root is taken, both on
ingestion (`CellDatabase.add_cell`, keyed on the space-group symbol) and on query
(`nearest`/`RootIndex`, via an optional `sg_hm`). The primitive matrices are
self-validated at import against the package's own `hall.LATTICE_CENTERING`
table — each centring vector must be integral in the primitive basis — and the
whole transform is cross-checked in the test suite against spglib's
`find_primitive` (test-only oracle, as everywhere else). Stored cell parameters
and volume remain the deposited conventional values; only the roots use the
primitive lattice. This is what makes two crystals with the same lattice in
different centred settings land at root distance 0, as they must.
