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

## Crystal-form clusters

`agentsg.cell.clusters` finds every group of entries of one protein that crystallised in the same lattice. Two entries are joined when

- they carry the same label: the exact set of UniProt accessions, or the deposition group when the entry has no UniProt list;
- their strain price is at most 7 %: the affine-invariant distance between the two primitive metrics after the best common scale, minimised over the widened Selling closure, so the deposited settings do not matter;
- their primitive volumes agree within 7 %. The price is scale-fitted and does not see volume, so this gate keeps a cell of the same shape at another size apart.

A cluster is a connected component of these joins, and only clusters with more than 10 entries are reported. The result does not depend on the order in which entries are visited.

```bash
python -m agentsg.cell.pdb_app clusters data/pdb_cells.duckdb data/clusters.duckdb \
    [--band 0.07] [--vol-tol 0.07] [--min-size 11]
```

```python
from agentsg.cell.clusters import cluster_cells, write_clusters

res = cluster_cells("data/pdb_cells.duckdb")
write_clusters(res, "data/clusters.duckdb")
```

The output holds four tables: `clusters` (label, size, medoid, space groups, price from the medoid), `members` (each entry with its cluster and price from the medoid), `spanning_edges` (the joins that built each cluster, with their prices) and `run_info`.

Clustering needs a snapshot with the UniProt and deposition-group columns. The first Zenodo version has neither, so use the latest version of the record or rebuild with `pdb_app build`. On the October 2026 snapshot (208,509 cells) the run takes about 5 minutes on two cores and finds 1,808 clusters covering 62,216 entries.

Joins are made in three steps per label, none of which changes the components: candidate pairs within a radius of the volume-normalised sorted linear key; a cheap price through the 24 relabellings of the reduced superbase, which is an upper bound on the closure price and accepts most joins; and the exact closure price for the remaining pairs that would still join two components. The key radius is 1.7 times the band. On every pair of 40 sampled labels, the largest ratio of key distance to price on a join was 1.30; `calibrate_key_ratio` repeats this measurement.

Connectivity can chain: a run of intermediate cells can carry one cluster from a crystal form to a neighbouring one. `n_beyond_band` counts the members further than the band from the medoid, and `max_price_to_medoid` gives the furthest. Prices above the band in these columns are upper bounds.
