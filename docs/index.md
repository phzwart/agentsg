# agentsg

Exact-rational space-group algebra for all 230 groups, and numeric unit-cell
math, with no required third-party libraries.

Each lattice is given one search key, `sort(p_ij / sqrt(Σ p))`, in ångström.
The total of the six Selling conorms is the same for every obtuse superbase of
the lattice, the sorted distance is a rearrangement lower bound on the labelled
distance, the key is Lipschitz in the metric, and it scales with the cell
edges. A cube of edge `a` occupies `(0, 0, 0, a/√3, a/√3, a/√3)`. Identity and
the reindexing operator come from the finite Selling closure, not from equality
of the key. The same calls are a local HTTP API and a set of MCP tools.

```{toctree}
:maxdepth: 2
:caption: Using agentsg

installation
symmetry
cells
reindexing
database
diagrams
server
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
notes/design
notes/reduction-flip
notes/selling-route
notes/sublattice
```

```python
from agentsg import space_group

sg = space_group(96)
sg.hm, sg.order()
```

`space_group` accepts an International Tables number, a Hermann–Mauguin symbol,
or a Hall symbol. Operation lists, reflection conditions, site multiplicities,
and subgroup edges are derived from those operations. Wyckoff letters are not
assigned: the lettering is an International Tables convention, and the numeric
content of a position (orbit, multiplicity, site-symmetry group) is computed.

The package imports nothing outside the Python standard library. DuckDB,
Matplotlib, and the test oracles (gemmi, spglib) are optional extras.
