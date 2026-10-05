# agentsg

Exact-rational space-group algebra for all 230 groups, and numeric unit-cell
math, with no required third-party libraries.

Each lattice is given a sorted six-vector Selling key: continuous, Euclidean,
and a lower bound on the true distance, so a radius search cannot miss a match.
Identity and the reindexing operator come from the finite Selling closure, not
from equality of the key. The same calls are a local HTTP API and a set of MCP
tools.

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
