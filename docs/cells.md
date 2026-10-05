# Unit cells

Cell math is numeric. Symmetry stays exact. The one bridge is
`agentsg.cell.constraints`: a point-group operator `W` leaves the metric
invariant when `Wᵀ G W = G`, and that equation supplies the crystal-system
restrictions.

```python
from agentsg import space_group
from agentsg.cell import UnitCell, free_metric_parameters, niggli_reduce
from agentsg.group import point_group

uc = UnitCell(6.2, 7.8, 9.1, 78, 82.5, 66.3)
uc.volume()
uc.d_spacing((1, 2, 3))
uc.reciprocal()

reduced_params, change_of_basis = niggli_reduce(9, 5, 7, 80, 100, 95)

pg = point_group(space_group(225).operations())
free_metric_parameters(pg)   # 1 for a cubic metric
```

Niggli reduction follows the stabilised Grosse-Kunstleve, Sauter, and Adams
(2004) algorithm and returns the integer change of basis.

## Lattice symmetry

`lattice_symmetry` enumerates the integer two-fold matrices a reduced cell can
carry and accepts those whose direct and reciprocal axes are parallel within
an angular tolerance (Le Page / Lebedev).

```python
from agentsg import evaluate_two_folds, lattice_symmetry

cell = (68.4, 68.4, 68.3, 109.5, 109.4, 109.5)
ls = lattice_symmetry(cell, max_delta=3.0)
ls.order, ls.crystal_system
evaluate_two_folds(cell, sort_by="kurlin")
```

Each accepted two-fold carries a Le Page angle and a Kurlin distance. The
default angular gate used by the library is 3°.

## Comparing two cells

`compare_cells` Niggli-reduces both cells, treats the smaller as a building
block, enumerates its finite-index sublattices, and reports every integer
transform that reproduces the larger cell within tolerance.

```python
from agentsg.cell import compare_cells

res = compare_cells(
    (61.8, 97.7, 148.9, 90, 90, 90),
    (115.5, 149.0, 115.6, 90, 115, 90),
)
best = res["solutions"][0]
best.M
best.resulting_cell
```

`resulting_cell` is the Niggli-reduced matched cell.

## Search key

The archive search key is `sort(p_ij / sqrt(Σ p))`, the six Selling conorms of
an obtuse superbase divided by the square root of their sum and sorted. It is
in ångström. `sorted_linear_key` and `sorted_linear_distance` are that key and
its Euclidean distance. `pdb_search` distances are in the same metric.

Four properties:

- One key per lattice. The total `T = Σ p` is unchanged by a Selling move, so
  every member of the Selling closure shares the sorted key.
- Rearrangement lower bound. The sorted distance is at most the aligned
  distance of any superbase pairing, so a radius search cannot miss a nearer
  labelled match.
- Lipschitz in the metric tensor.
- Ångström scaling: `key(λ · cell) = λ · key(cell)`.

A cube of edge `a` has key `(0, 0, 0, a/√3, a/√3, a/√3)`, and stretching every
edge by `Δa` moves the key by exactly `Δa`.

Kurlin's square-root products `sort(√p)` are a different coordinate, requested
with `stabilize="sqrt"` (`root_invariant`, `root_distance`). At a vanishing
conorm that map is only Hölder-½. On the Scotty pairs the square-root distance
is 16.7 Å and 16.4 Å for 1fe5→1u4j and 1fe5→1g2x, against linear distances
2.7 Å and 2.9 Å (NCDist 2.1 Å and 2.1 Å), and 13.8 Å against 5.4 Å for
4nl7→2yht (NCDist 2.3 Å). Equality of either key is not identity: certify that
with the Selling-superbase closure.

```python
from agentsg.cell import (
    root_cutoff_for_edge_tolerance,
    root_volume_decomposition,
    similarity_distance,
    sorted_linear_distance,
    sorted_linear_key,
    symmetry_cutoff,
)

a = (78, 78, 37, 90, 90, 90)
b = (79, 79, 38, 90, 90, 90)
sorted_linear_key(a)
sorted_linear_distance(a, b)

dec = root_volume_decomposition(a, b)
dec["total"]              # sorted linear distance
dec["volume_component"]   # the part forced by the volume ratio
dec["shape_residual"]

similarity_distance(a, b)                    # 0 when the lattices are similar
root_cutoff_for_edge_tolerance(10, cell=a)
symmetry_cutoff(a, volume_tol=0.05)
```

Centred conventional cells are reduced to the primitive lattice before the key
is taken. G6 and S6 distances remain available as diagnostics.

## Noise models for serial data

`stabilize="floored"` and `stabilize="soft_threshold"` are noise models for
frame-level serial data. They are not the archive search key. The square-root
key remains available as `stabilize="sqrt"`.
