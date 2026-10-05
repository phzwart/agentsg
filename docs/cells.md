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

The sorted six root products (`root_invariant`, `root_distance`) are the
retrieval coordinate. The key is continuous and Euclidean, and a radius in
that space is a lower bound on the true lattice distance, so a neighbour
search cannot miss a match that lies inside the radius. Equality of the key
is not identity: certify that with the Selling-superbase closure.

```python
from agentsg.cell import (
    root_cutoff_for_edge_tolerance,
    root_distance,
    root_volume_decomposition,
    similarity_distance,
    symmetry_cutoff,
)

a = (78, 78, 37, 90, 90, 90)
b = (79, 79, 38, 90, 90, 90)
root_distance(a, b)

dec = root_volume_decomposition(a, b)
dec["total"]              # root_distance
dec["volume_component"]   # the part forced by the volume ratio
dec["shape_residual"]

similarity_distance(a, b)                    # 0 when the lattices are similar
root_cutoff_for_edge_tolerance(10, cell=a)
symmetry_cutoff(a, volume_tol=0.05)
```

For an orthogonal cell the sorted key is `(0, 0, 0, a, b, c)` in some order,
so one edge moving by Δ moves the key by Δ. Centred conventional cells are
reduced to the primitive lattice before the key is taken. G6 and S6 distances
remain available as diagnostics; prefer the root key for search and for
symmetry deficiency (`kurlin_distance_to_symmetry`).
