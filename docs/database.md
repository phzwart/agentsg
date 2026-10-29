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

## One setting and origin per cluster

`agentsg.cell.cluster_frames` puts every entry of a cluster on the setting and origin of one reference: the highest-resolution entry among those whose space group has the cluster's most common IT number. Entries of another IT number (a P1 or P2₁ entry in a P2₁2₁2₁ cluster) are listed and left alone.

Each entry is placed in three steps:

- **Reindexing.** The coset of reindexing operators from the reference to the entry within the band, grouped into classes of the reference's point group (proper operators only). Each class is one indexing the cell cannot decide; all are tried. The band is 7 %, widened to 15 % and then 20 % when nothing is found.
- **Operators and origin.** The entry's space-group operators are written in the reference basis, and the origin shift that turns them into the reference's operators is solved exactly. A class with no such shift does not describe the reference's group and is dropped.
- **Allowed origins.** Among the allowed origin shifts, the proper elements of the Euclidean normalizer and the operators of the group, the placement with the lowest Cα RMSD is kept. Cα atoms are paired by UniProt accession and residue number, so renumbered constructs and different chain labels pair correctly. Floating origin components are fitted, not searched.

Entries more than 20 % from the reference (the far end of a chained cluster) are placed on an already-placed entry close in cell shape, and the two frames are composed (`via`).

The coordinates come from the deposited mmCIF files:

```bash
pip install -e ".[coords]"
python -m agentsg.cell.pdb_coords fetch data/clusters.duckdb data/ca.duckdb --workers 16
python -m agentsg.cell.cluster_frames data/clusters.duckdb data/ca.duckdb data/frames.duckdb --workers 2
```

The `frames` table holds, for each entry, the affine map `x -> L x + t` from its deposited fractional coordinates to the reference's (`L` exact, `t` with its floating part fitted), the probe-chain RMSD, `margin` (the RMSD of the next distinct placement minus the best; below about 1 Å the coordinates barely decide), and `asu_rmsd` (every chain against its nearest reference chain under the group's operators and lattice translations only). `unp_offset` records a residue-numbering shift the pairing needed, and `acc_fallback` an entry whose chains map to another UniProt accession of the same protein.

## Records per protein identity

`agentsg.cell.cluster_records` turns the frames into one directory per protein identity (the exact set of UniProt accessions, or the deposition group of entries without one). No coordinates are written: an entry in the reference frame is rebuilt from its deposited model and its row of `frames.tsv`.

```bash
python -m agentsg.cell.cluster_records data/clusters.duckdb data/ca.duckdb data/frames.duckdb crystal_forms
```

```
crystal_forms/index.tsv                          pdb_id -> identity directory, cluster, status, reference
crystal_forms/<identity>/record.json             name, accessions, and per cluster: reference and its
                                                 resolution, space group, cell, status counts, RMSD summaries
crystal_forms/<identity>/cluster_<id>/frames.tsv one row per entry: resolution, deposited cell and space
                                                 group, status, L, t, RMSD to the reference, margin, ...
crystal_forms/<identity>/cluster_<id>/pairwise.tsv.gz   pdb_a, pdb_b, frame_rmsd, fit_rmsd, n_common
```

The frame maps deposited fractional coordinates to the reference's fractional coordinates:

$$
x_{\mathrm{ref}} = L\,x + t
$$

`L` is exact (rationals, rows separated by `;`); `t` is three decimals separated by `,`, its floating components fitted. `frame_rmsd` compares the two entries after their frames with no fitting, each orthogonalised with its own cell written in the reference setting; `fit_rmsd` is the same residue pairs after a Kabsch superposition. Both use the Cα atoms the two entries share by UniProt residue number; pairs sharing fewer than 20 are omitted.
