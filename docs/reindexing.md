# Reindexing

In a serial dataset every frame indexes to the same cell and space group, so
the set of reindexing operators is a dataset-level constant. It is the coset
of the crystal Laue group in the tolerance metric-automorphism group of the
cell.

```python
from agentsg.cell import ReindexingReference, reindexing_ambiguity_operators

ops = reindexing_ambiguity_operators(
    5, (40, 50, 60, 90, 90.5, 90),
    length_tol_pct=2.0, angle_tol_deg=2.0,
)

ref = ReindexingReference(5, (40, 50, 60, 90, 90.5, 90))
op, residual = ref.resolve(frame_cell)
```

The tolerance group, rather than the exact holohedry, is what contains the
exact reindexings, the pseudo-symmetry branches, and the cell-choice
transforms that jump when Niggli reduction crosses a boundary.

## Geometry and intensities

Geometry enumerates the branches and the metric residual of each.
`surface_geometric_operators` returns that set. A branch with residual 0 is
metric symmetry: geometry cannot choose it. Intensities break the tie.

```python
from agentsg.cell import ReindexingReference, surface_geometric_operators

for g in surface_geometric_operators(3, (40, 40, 60, 90, 91, 90)):
    print(g)

ref = ReindexingReference(75, (50, 50, 80, 90, 90, 90))
ref.set_reference_intensities(reference_I)
res = ref.resolve_intensities(frame_I)
res.best
res.scores     # (operator, correlation, n_common) for every branch
res.margin     # best correlation minus the second
```

`compare_cells` is the complementary tool: it looks for volume-changing
sublattice relations. `surface_geometric_operators` stays at the same volume.

The same finite set is what `dials.cosym` and `reindex_to_reference` compute
(Gildea & Winter, 2018; Brehm & Diederichs, 2014). The notes on reduction
flips and on the Selling-superbase route are linked from the reference section.
