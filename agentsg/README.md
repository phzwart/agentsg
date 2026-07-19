# agentsg

Self-contained crystallographic **space-group algebra** (exact rational
arithmetic) and **unit-cell math** (numeric), with a hard boundary between the
two crossed at exactly one point.

Unlike cctbx's `sgtbx` — which bakes a fixed translation denominator of 12 into
a C++ `int` (`tr_vec.sg_t_den = 12`) and entangles its unit-cell and symmetry
headers — agentsg uses `fractions.Fraction` everywhere (a 1/2, a 1/3, and a
1/24 all coexist with no common base) and keeps symmetry (exact) and cell
(numeric) in separate packages.

**No runtime dependencies.** `gemmi` and `spglib` are used *only* in the test
suite, as independent oracles — never imported by the package itself.

## Status

- **Phase 1 (symmetry): complete and exhaustively validated.**
  All **230 space groups** are reproduced *exactly* from their Hall symbols —
  the operation set (rotation + translation mod 1) is identical to gemmi's for
  every group (`tests/test_all_230.py`). Systematic absences match gemmi over
  a full hkl grid for all 230 groups.
- **Phase 2 (unit cell): implemented and validated.**
  Metric tensor, reciprocal cell, d-spacings, orthogonalisation, Niggli
  reduction, and the exact↔numeric bridge — all cross-checked against gemmi to
  machine precision.

**1264 tests pass.**

## Quick look

```python
from agentsg import space_group, ops_from_hall
from agentsg.reflections import reflection_conditions
from agentsg.wyckoff import multiplicity, site_symmetry_order
from agentsg.linalg import Vector3
from fractions import Fraction as Fr

# --- any of the 230 groups, by number / Hermann-Mauguin / Hall symbol ---
sg = space_group(225)                       # or "Fm-3m", or "F 4 2 3", or 225
print(sg, "order", sg.order())              # 192
ops = sg.operations()                       # closed, exact operation set

# --- reflection conditions, DERIVED from the operators (no table) ---
print(reflection_conditions(list(ops)))
#   {'hkl': 'h+l, k+l = 2n', 'h00': 'h = 2n', ...}   (F-centring + ...)

# --- Wyckoff multiplicity & site symmetry, computed exactly ---
ops = list(space_group(225).operations())
x = Vector3((Fr(1,4), Fr(1,4), Fr(1,4)))    # the 8c site of Fm-3m
print(multiplicity(x, ops), site_symmetry_order(x, ops))   # 8  24

# --- build a group straight from a Hall symbol ---
print(len(ops_from_hall("-P 2ybc")))        # P2_1/c, order 4
```

```python
# --- phase 2: unit-cell math (numeric) ---
from agentsg.cell import UnitCell, niggli_reduce, free_metric_parameters
from agentsg.group import point_group

uc = UnitCell(6.2, 7.8, 9.1, 78, 82.5, 66.3)   # triclinic
print(uc.volume())                              # cell volume
print(uc.d_spacing((1, 2, 3)))                  # interplanar spacing
print(uc.reciprocal())                          # reciprocal cell

# Niggli reduction (stabilised Grosse-Kunstleve/Sauter/Adams 2004)
reduced_params, change_of_basis = niggli_reduce(9, 5, 7, 80, 100, 95)

# the ONE exact<->numeric bridge: crystal-system cell restrictions DERIVED
# from the point group via W^T G W = G
pg = point_group(space_group(225).operations())
print(free_metric_parameters(pg))              # 1  (cubic: a=b=c, all 90 deg)
```

## What's derived vs. what's a table

**Derived at runtime from the operator list** (no lookup): general/special
positions, multiplicities, site symmetry, point group, Bravais centring,
systematic absences / reflection conditions, and — via the one bridge —
crystal-system cell restrictions.

**Table-shaped** (embedded as verified literal data, `space_groups.py`): the
230 standard-setting Hall/Hermann-Mauguin symbols. Everything numeric follows
from parsing + closing these; no operation lists are stored.

Wyckoff *letters* (a, b, c…) are deliberately **not** reproduced — the
letter-to-orbit assignment is an ITA historical convention; all the numeric
content of a Wyckoff position (orbit, multiplicity, site-symmetry group) is
computed.

## Settings & change of basis

Any setting — including non-standard ones that add lattice symmetry — is
written as a base Hall/Hermann-Mauguin symbol with an **attached change of
basis** (Zwart, Grosse-Kunstleve & Adams, *Exploring Metric Symmetry*, 2006).
The parenthesised part lists the three new basis vectors as combinations of the
old ones (the columns of the change-of-basis matrix `P`); letters may be spelled
`x,y,z` or `a,b,c`:

```python
from agentsg import SpaceGroupSetting

s = SpaceGroupSetting.parse("Hall: I 4 2 3 (y+z,x+z,x+y)")
s.change_of_basis_matrix().det()   # 2  -> the transform doubles the cell
s.order()                          # 96 -> base I432 (48) gains I-centring
```

When `det(P) != 1` the transform rescales the lattice, and lattice translations
that were integral in the base setting appear as fractional **centring**
translations in the new setting — this is how the notation expresses "added
lattice symmetry". `parse_setting`, `parse_cob`, and `format_cob` give the
lower-level parse/print, and `SpaceGroupSetting.operations()` returns the closed
operation set in the new setting.

## Identify, semi-invariants, reciprocal space

From a closed operator set, recover the IT space group (and Hall symbol) —
including when the input differs from the standard setting by an origin shift
only. Origin recovery solves `(W − I) p = Δw` exactly over the rationals:

```python
from agentsg import identify_space_group, hall_from_ops, semi_invariants
from agentsg import phase_restriction, equivalent_reflections
from agentsg.linalg import Vector3

hit = identify_space_group(ops)          # IdentifyResult | None
hit.number, hit.hall, hit.change_of_basis
hit.floating_origin                      # e.g. (0,1,0) for P2 — gauge-free axis
hall_from_ops(ops)                       # tabulated Hall, or ValueError

semi_invariants(ops)                     # e.g. [SemiInvariant((1,1,1), 2)]
# modulus 0  <=>  continuous (floating) origin freedom along that direction
pr = phase_restriction(Vector3((1, 2, 3)), ops)
pr.absent, pr.centric, pr.phase          # phase in turns, or None
eq = equivalent_reflections(Vector3((1, 2, 3)), ops)
eq.multiplicity, eq.epsilon, eq.laue_multiplicity, eq.hkls
# |G| = multiplicity × epsilon; laue_multiplicity folds in Friedel mates
```

Floating origins (P1, P2/P21, P3, P4, P6, …) are handled explicitly:
`floating_origin_basis` gives the continuous nullspace of stacked `(W−I)`,
recovered origin shifts are pinned so free components are zero, and
semi-invariants use modulus-0 constraints on those directions (never
discretising the continuum).

Harker sections / lines are the left-nullspace constraints of `(I−W)`:

```python
from agentsg import harker_sections, space_group
harker_sections(space_group(19).operations())
# sections u=1/2, v=1/2, w=1/2  (P212121)
```

## Asymmetric units (real and reciprocal)

CCP4 / cctbx / gemmi conventions for reciprocal space and conventional
real-space bricks (all 230), plus an experimental metric Dirichlet ASU:

```python
from agentsg import (
    ReciprocalAsu, DirectAsuBrick, optimize_asu, laue_class, space_group,
)
from agentsg.cell.metric import UnitCell

rasu = ReciprocalAsu.from_space_group(19)
rasu.condition_str, rasu.is_in((1, 2, 3))
hkl, isym = rasu.to_asu((3, 2, 1), space_group(19).operations())

brick = DirectAsuBrick.from_space_group(19)
str(brick), brick.contains((0.1, 0.2, 0.3))

laue_class(225)   # 'm-3m'

# Metric Voronoi ASU + sphericity search over allowed origin gauges
opt = optimize_asu(space_group(4).operations(), UnitCell(5, 5, 40, 90, 90, 90))
opt.score, opt.origin_shift
```

Intensity merging in `cell.ambiguity` uses the CCP4 reciprocal ASU by default
(`style='lexmax'` keeps the old lexicographic key). Full ITA polyhedral
real-space inequality galleries are out of scope; bricks cover the practical
conventional ASU.

## Comparing cells

Given a unit cell, `lattice_symmetry` finds its metric symmetry (holohedry) by
the Le Page / Lebedev method — enumerating the 480 integer rotation matrices
(81 two-folds) that a reduced cell can carry, and accepting those whose direct-
and reciprocal-space axes are parallel within an angular tolerance:

```python
from agentsg import lattice_symmetry, evaluate_two_folds
cell = (68.4, 68.4, 68.3, 109.5, 109.4, 109.5)
ls = lattice_symmetry(cell, max_delta=3.0)
ls.order, ls.crystal_system      # (48, 'cubic')  -- a pseudo-cubic reduced cell
# Acceptance is Le Page–gated; each accepted two-fold also carries Kurlin distance:
for s in ls.two_fold_scores[:3]:
    s.le_page_delta, s.kurlin_distance   # degrees, Å (root-invariant)
# Full 81-candidate spectrum (no tolerance gate):
evaluate_two_folds(cell, sort_by="kurlin")
```

See `examples/perturb_metric_symmetry.py` for progressive random perturbations
across crystal systems with markdown tables of Le Page–gated assignment plus
per-holohedry Le Page δ / Kurlin spectra (`examples/perturb_metric_symmetry.md`).

`compare_cells` relates two cells by an exact integer transform (the "lego /
target" construction): it Niggli-reduces both, treats the smaller as a building
block, enumerates its index-*d* sublattices (Hermite normal form), and reports
every transform that reproduces the larger cell within tolerance:

```python
from agentsg.cell import compare_cells
res = compare_cells((61.8, 97.7, 148.9, 90, 90, 90),      # Native  P2_1 2_1 2_1
                    (115.5, 149.0, 115.6, 90, 115, 90))   # SeMet1  P2_1
best = res["solutions"][0]
best.M                # ((2, 1, 0), (0, 1, 0), (0, 0, 1))  -- matches the paper's M exactly
best.resulting_cell   # (115.6, 115.6, 148.9, 90, 90, 115.4)  -- Niggli-reduced matched cell
```

The recovered transform `M = ((2,1,0),(0,1,0),(0,0,1))` reproduces the paper's
result exactly (Zwart, Grosse-Kunstleve & Adams, *Exploring Metric Symmetry*,
2006, §3.2). Note the reported `resulting_cell` is the **Niggli-reduced** matched
cell `(115.6, 115.6, 148.9, 90, 90, 115.4)`, so it sits close to the target Niggli
cell; the paper instead prints the *directly transformed* (un-reduced) cell
`118.3 118.3 148.9 90 90 120` with deviations `-2.5 -2.4 0.1 0 0 -5.0` from the
target. Same `M`, two different but equivalent settings of the same lattice —
agentsg reduces before reporting, the paper does not.

## Reindexing ambiguity (serial crystallography)

In a serial dataset every frame indexes to the same cell and space group, so the
set of reindexing operators is a *dataset-level constant*. The minimal, exact
object is the coset decomposition of the crystal Laue group in the (tolerance)
metric-automorphism group of the cell — computed once and memoised, not
regenerated per frame:

```python
from agentsg.cell import reindexing_ambiguity_operators, ReindexingReference

# monoclinic C2 with beta near 90 -> pseudo-orthorhombic ambiguity (n = 2)
ops = reindexing_ambiguity_operators(5, (40, 50, 60, 90, 90.5, 90),
                                     length_tol_pct=2.0, angle_tol_deg=2.0)

# anchor to one reference basis for the whole dataset; resolve each frame in O(1)
ref = ReindexingReference(5, (40, 50, 60, 90, 90.5, 90))
op, residual = ref.resolve(frame_cell)   # stable across Niggli reduction flips
```

Using the *tolerance* metric-automorphism group (not the exact holohedry) is what
makes this robust: the same finite set contains the exact reindexings, the
pseudo-symmetry branches (β near 90), and the cell-choice / reduction-flip
transforms that make per-frame Niggli reduction discontinuous.

### Two layers: geometry surfaces the choices, intensities decide

The package deliberately separates the two halves of the problem:

1. **Geometry (exact, deterministic)** — `reindexing_ambiguity_operators` and
   `ReindexingReference.resolve` *enumerate and surface* the candidate branches
   (the coset operators) and the metric residual of each. This is exact integer
   algebra with no physics in it.
2. **Intensities (physics, tie-break)** — `set_reference_intensities` +
   `resolve_intensities` pick the correct branch by Pearson-correlating each
   branch's reindexed, ASU-merged intensities against a reference. Every branch's
   CC is returned (`AmbiguityResolution.scores`) so the decision is inspectable.

```python
ref = ReindexingReference(75, (50, 50, 80, 90, 90, 90))   # P4, a==b: TRUE merohedry
ref.resolve((50, 50, 80, 90, 90, 90))          # geometry: residual 0.0 on BOTH branches
ref.set_reference_intensities(reference_I)      # {(h,k,l): I}
res = ref.resolve_intensities(frame_I)          # intensities break the tie
res.best        # the chosen operator
res.scores      # [(op, cc, n_common), ...]  every branch, surfaced
res.margin      # best_cc - second_cc; small => data did not discriminate
```

For cell comparison and reindexing, `surface_geometric_operators` returns the
**complete** set of geometrically-allowed operators (the coset of the Laue group
in the tolerance metric group) — reduction flips and cell-choice transforms
included — each annotated with its metric residual and an `is_metric_symmetry`
flag marking the branches geometry alone cannot decide (residual 0, the
merohedral case that needs intensities):

```python
from agentsg.cell import surface_geometric_operators
for g in surface_geometric_operators(3, (40, 40, 60, 90, 91, 90)):
    print(g)   # identity | cell-change (residual>0) | metric-sym (residual 0)
```

This is the authoritative list the geometry layer surfaces; it decides nothing,
it only enumerates and annotates. `compare_cells` handles the complementary,
volume-*changing* sublattice relations (index > 1); `surface_geometric_operators`
covers the same-volume reindexing coset.

Why the split matters: for **pseudo-merohedral / cell-choice** ambiguity the
branches have slightly different metrics, so `resolve` (geometry) already
discriminates cheaply. For **true merohedral / polar** ambiguity (P4 with a==b
exactly, P3₁ vs P3₂) every branch has an *identical* metric — geometry reports a
tie (residual 0), and only `resolve_intensities` can decide. Note that even the
"canonicalize to a primitive setting" step other tools use is itself
tolerance/tie-break dependent and non-unique at a reduction boundary, so the
intensity layer is what makes resolution robust there too.

**Attribution.** This is a dependency-free, exact-arithmetic re-implementation of
the standard method used by `dials.cosym` (Gildea & Winter, 2018) and
`reindex_to_reference`, built on Brehm & Diederichs (2014) — not a new algorithm.
The value here is the clean two-layer separation and exact rational operators. For
the reduction-flip / cell-choice degeneracy specifically — how it is documented and
how DIALS/cctbx handle it (combinatorial candidate enumeration; the
Andrews–Bernstein G6 boundary embedding) — see
[`docs/REDUCTION_FLIP_LITERATURE.md`](docs/REDUCTION_FLIP_LITERATURE.md).

## The lattice manifold (G6 / S6)

Lattices embed as points in a 6-D space (G6 = metric tensor as a vector; S6 =
Selling scalars), turning lattice space into a continuous manifold. Two
primitives:

```python
from agentsg.cell import g6_distance, distance_to_symmetry
from agentsg import space_group
from agentsg.group import point_group

# boundary-aware distance: continuous across the Niggli reduction-flip
g6_distance((40, 40.001, 60, 90, 91, 90), (40, 39.999, 60, 90, 91, 90))  # ~ small
# raw Euclidean would JUMP by >100 across the a=b boundary:
g6_distance(a, b, boundary_aware=False)

# symmetry as a continuous field, not a yes/no:
cubic = point_group(space_group(225).operations())
distance_to_symmetry((50, 50, 50,   90, 90, 90), cubic)   # 0.0  (is cubic)
distance_to_symmetry((50, 50, 51.5, 90, 90, 90), cubic)   # grows smoothly
```

The distance is robust to cell choice and continuous across reduction boundaries
(Andrews–Bernstein NCDist idea: minimise over the boundary-transform orbit,
which is non-isometric so Euclidean G6 is wrong near a boundary).
`distance_to_symmetry` measures the G6 distance to the metric subspace fixed by a
point group (BGAOL's distance-to-Bravais-subspace) — symmetry becomes a smooth
deficiency field over the manifold rather than a tolerance-gated binary test.
This is the substrate for treating lattice trajectories (T, p, time) as smooth
paths and cell clustering as nearest-neighbour search in G6/S6. See
[`docs/DESIGN.md`](docs/DESIGN.md) for the full framing and references.

## Full-PDB unit-cell database (root-invariant search)

Download every crystallographic unit cell in the PDB and precompute its Kurlin
root invariant for fast lattice-similarity search. Needs the optional DuckDB
extra (`pip install -e ".[db]"`); the core package stays dependency-free.

```bash
# build (or resume) the whole PDB into a single DuckDB file (~206k cells)
python -m agentsg.cell.pdb_app build pdb_cells.duckdb

# query the k nearest lattices to a cell
python -m agentsg.cell.pdb_app query pdb_cells.duckdb \
    --cell 79 79 38 90 90 90 -k 5
```

```python
from agentsg.cell import CellDatabase

db = CellDatabase("pdb_cells.duckdb")
idx = db.build_index()                     # one-time NearTree over precomputed roots
idx.k_nearest((79, 79, 38, 90, 90, 90), k=10)     # ~5 ms median over 206k cells (p99 ~11 ms)
# centred query lattice? pass its symbol so the query is reduced to primitive too:
idx.k_nearest((80, 90, 100, 90, 90, 90), k=10, sg_hm="C 1 2 1")
```

The build is **resumable** (rerun to fill in only missing ids), retries transient
network errors with backoff, and commits incrementally. Roots are computed on the
**primitive** cell: Kurlin's invariant is a *lattice* invariant, and a centred
group's deposited conventional cell (C, I, F, R, …) describes only a sublattice —
so the conventional cell is reduced to primitive before the root is taken. Two
crystals with the same lattice in different centred settings therefore land on the
same point (root distance 0), which a conventional-cell root would miss by tens of
Ångström.

## Volume vs shape: decomposing root distance

The root invariant scales *linearly* with the length scale factor
`s = (V'/V)**(1/3)` (conorms carry units of length², roots their square root), so
a pure isotropic volume change contributes an exactly predictable amount to the
root distance: `d = |s - 1| * ||RI(cell)||`. This lets you separate "how much of
a lattice difference is just volume" from genuine shape change.

```python
from agentsg.cell import (root_volume_decomposition,
                          similarity_distance, similarity_invariant)

# split a root distance into volume and shape legs
dec = root_volume_decomposition(cell_A, cell_B)
dec["total"]             # == root_distance(A, B)
dec["volume_component"]  # |s-1|*||RI(A)||  -- forced by the volume change alone
dec["shape_residual"]    # shape change at matched volume
dec["coupling_angle_deg"]# 90 deg = shape independent of volume; smaller = coupled

# shape-only (scale-invariant) comparison -- the manuscript's similarity relation
similarity_distance(cell_A, cell_B)   # 0 iff lattices are similar (isotropic copies)
similarity_invariant(cell)            # RI(cell) / V**(1/3), a dimensionless key
```

**Choosing a search radius from an edge tolerance.** The root invariant carries
units of length, and for an orthogonal cell it is exactly `sorted(0,0,0,a,b,c)`,
so a single edge change of Δ moves one root component by exactly Δ (root distance
= Δ) and changing all three edges gives `sqrt(3)*Δ`. The single-edge slope has
median 1.0 for near-orthogonal lattices, but is larger (median ~1.3, tail to
~4.5) for cells whose *primitive* basis is strongly non-orthogonal — notably
cubic groups, whose primitive cells are rhombohedra. Pass the cell for an exact
per-cell cutoff whenever the lattice may be non-orthogonal; the analytic form is
a guide:

```python
from agentsg.cell import root_cutoff_for_edge_tolerance as cutoff

cutoff(10)                      # 10.0  -- a single edge may move 10 Å
cutoff(10, n_edges=3)           # 17.32 -- all three edges (sqrt(3)*10), upper envelope
cutoff(10, cell=my_cell)        # exact conservative cutoff for a specific cell
# then: idx.within(my_cell, cutoff(10, cell=my_cell))
```

For a 10 Å edge tolerance: **~10 Å captures a single-edge change, ~17 Å covers a
simultaneous all-edge change**, and calibrating on a specific cell gives an exact
per-cell radius (every lattice within ±10 Å per edge is guaranteed inside it).

### Kurlin distance ↔ volume ratio, and choosing a symmetry cutoff

For a *pure isotropic* volume change the root invariant scales linearly with the
length scale factor, giving an exact, invertible relation between root distance
and volume ratio:

```python
from agentsg.cell import (root_distance_to_volume_ratio,
                          volume_ratio_to_root_distance, symmetry_cutoff)

root_distance_to_volume_ratio(d, cell)      # V'/V = (1 + d/||RI||)**3
volume_ratio_to_root_distance(1.05, cell)   # distance for a 5% volume change
```

This is the key to a **transferable cutoff** when scoring symmetrised cells (the
distance from a cell to its Reynolds-symmetrised metric). That deficiency has
units of length and grows with cell size, so an absolute Å cutoff does not
transfer. Reference it to the cell's own `||RI||` instead — either as a volume
tolerance or as a multiple of measurement noise:

```python
symmetry_cutoff(cell, volume_tol=0.05)   # accept if within a 5% volume equivalent
symmetry_cutoff(cell, noise_frac=0.01)   # accept if within 1% cell noise (z=11 ≈ p95)
```

Both return `(dimensionless) * ||RI(cell)||`, which automatically tracks cell
scale and the per-system spread without a separate per-system table. This pairs
directly with the metric symmetry-deficiency work: use it to gate
`kurlin_distance_to_symmetry` in a scale-correct way (see
`examples/perturb_metric_symmetry.py`).

On a dehydration series (e.g. the 1,113 HEWL tetragonal cells) volume change
explains much of the root distance (r ≈ 0.75) but not all. The total, volume and
shape components form a *triangle* in root space (total ≤ volume + shape), so the
volume component is not a floor — depending on the angle between the legs the
total can land above or below it. Here the shape residual sits at a nearly
constant ≈ 53° from the scaling axis (just past the 45° midpoint toward
orthogonal), i.e. it is a single, well-defined deformation mode that is largely
*independent* of the pure-volume direction — the anisotropic a-vs-c change of the
dehydration series, which the decomposition surfaces directly.

## Install / test

```bash
pip install -e .            # zero dependencies
pip install -e ".[test]"    # adds pytest + gemmi + spglib (test-only oracles)
pytest
```

## Package layout

```
agentsg/
  linalg.py         exact rational Vector3 / Matrix3
  symmetry_op.py    SymmetryOp (W,w); xyz-triplet parse/print
  group.py          closure, point group, centring, absences
  hall.py           Hall-symbol parser  -> generators + centring
  space_groups.py   the 230 groups (verified literal data) + lookup API
  setting.py        SG + attached change-of-basis notation  P21 (2a,a+b,c-a)
  lattice_symmetry.py  Le Page / Lebedev holohedry determination
  wyckoff.py        stabilisers, orbits, multiplicities, fixed loci
  reflections.py    reflection-condition reporter (integral/zonal/serial)
  change_of_basis.py  reindexing / basis changes
  cell/
    metric.py       UnitCell: metric tensor, reciprocal, d-spacing, ...
    reduction.py    Niggli reduction (GKSA 2004) + change of basis
    constraints.py  the single exact<->numeric bridge (W^T G W = G)
    sublattice.py   Hermite-normal-form index-d sublattice enumeration
    compare.py      compare_cells: metric-symmetry / cell comparison
    ambiguity.py    reindexing operators for serial crystallography
                    (coset of Laue in the tolerance lattice group, cached)
    rootform.py     Kurlin (2022) root invariant: obtuse superbase -> conorms
                    -> root products -> sorted six-tuple (complete, continuous)
    canonical.py    flip-free reindexing via canonical superbase matching
    neartree.py     exact metric nearest-neighbour index on the root invariant
    primitive.py    conventional -> primitive cell for centred lattices
                    (A/B/C/I/F/R/H); the root invariant is a lattice invariant
    celldb.py       DuckDB-backed cell database + RootIndex (needs agentsg[db])
    pdb_app.py      resumable full-PDB downloader/builder + query CLI
    crystfel_stream.py  parse CrystFEL .stream per-crystal cells + orientation
    manifold.py     deformation graph / landmarks / symmetry junctions
    g6.py           G6/S6 metric-cone coordinates + boundary-aware distance
```

See `docs/DESIGN.md` for the full rationale and validation methodology.
