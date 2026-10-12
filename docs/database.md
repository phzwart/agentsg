# PDB cell database

A snapshot of 206,214 crystallographic PDB cells is on Zenodo:
[10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222).
Download it as described in [](installation) and install the `db` extra
(`duckdb`, `numpy`, `scipy`).

Each row stores the deposited conventional cell, the space group, the PDB id,
the primitive-lattice root key, one Selling-reduced cell, and the exact
rational change of basis from the deposited cell to that reduced cell.
Each row may also carry two independent annotations: a list of UniProt
accessions, short names, and residue counts, and a PDB deposition-group id
listed as ``DEPOSITION_GROUP_<id>``. Similarity between UniProt lists is not
defined. An entry can have either annotation, both, or neither.
Roots are taken on the primitive lattice, so two centred settings of one
lattice land on the same point.

```python
from agentsg.cell import CellDatabase, root_cutoff_for_edge_tolerance

db = CellDatabase("data/pdb_cells.duckdb")
idx = db.build_index()
hits = idx.k_nearest((100, 100, 100, 90, 90, 90), k=20)

r = root_cutoff_for_edge_tolerance(10, cell=(100, 100, 100, 90, 90, 90))
near = idx.within((100, 100, 100, 90, 90, 90), r)

idx.k_nearest((80, 90, 100, 90, 90, 90), k=10, sg_hm="C 1 2 1")
```

Pass `sg_hm` when the query cell is a centred conventional cell, so the query
is reduced to the primitive lattice as well.

From the command line, after the snapshot is in place:

```bash
python -m agentsg.cell.pdb_app query data/pdb_cells.duckdb \
    --cell 79 79 38 90 90 90 -k 5
```

Rebuilding from RCSB instead of Zenodo. One command downloads the cells, writes the primitive-lattice keys and one Selling-reduced cell, then fills UniProt lists and deposition-group ids as separate columns:

```bash
python -m agentsg.cell.pdb_app build data/pdb_cells.duckdb
```

The build is resumable. Rerun it to fill in only the missing ids and columns. `--sample 100` runs the same steps on 100 random entries.
