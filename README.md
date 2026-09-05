# agentsg — lattice symmetry, root-form cell comparison, and the crystallographic manifold

A dependency-free Python package for unit-cell and space-group operations, built
around Kurlin's (2022) complete **root invariant** for fast, orbit-free lattice
comparison. Includes a full-PDB unit-cell database, a serial-crystallography
reindexing layer, and the manuscript describing the method.

```
.
├── agentsg/            the installable package (src/, tests/, docs/, pyproject.toml)
│   ├── src/agentsg/    zero-dependency runtime
│   ├── tests/          ~3600 tests (pytest)
│   └── docs/           DESIGN.md, reduction-flip literature note, etc.
├── manuscript/         IUCrJ communication (LaTeX + figures; see main_v11.tex)
├── analysis/           database-build & calibration figures + data
│   ├── figures/
│   └── data/           CSV summaries + .npz calibration arrays
└── data/
    └── pdb_cells.duckdb  the built database (206,214 crystallographic PDB cells,
                          roots precomputed on primitive lattices)
```

## Quick start

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -e ".[db,test]"     # or: cd agentsg && pip install -e ".[db,test]"
pytest -q                        # ~3600 tests
```

```python
from agentsg.cell import root_invariant, root_distance, root_cutoff_for_edge_tolerance
from agentsg.cell import CellDatabase

# compare two cells by their root invariant (orbit-free, continuous)
d = root_distance((78,78,37,90,90,90), (79,79,38,90,90,90))

# open the prebuilt PDB database and do a fast nearest-neighbour search
db  = CellDatabase("../data/pdb_cells.duckdb")
idx = db.build_index()                       # cKDTree over stored roots
hits = idx.k_nearest((100,100,100,90,90,90), k=20)

# choose a search radius from an edge tolerance you're willing to accept
r = root_cutoff_for_edge_tolerance(10, cell=(100,100,100,90,90,90))
near = idx.within((100,100,100,90,90,90), r)
```

## What's here

- **Root-form cell comparison** (`agentsg.cell.rootform`) — Kurlin's complete
  root invariant, `root_distance`, the volume/shape decomposition
  (`root_volume_decomposition`, `similarity_invariant`/`similarity_distance`), and
  the edge-tolerance → cutoff calibration (`root_cutoff_for_edge_tolerance`).
- **Full-PDB database** (`agentsg.cell.celldb`, `pdb_app.py`) — resumable
  downloader/builder + DuckDB store with precomputed primitive-lattice roots,
  and a persistent NearTree metric index.
- **Space-group machinery** (`agentsg.hall`, `setting`, `wyckoff`, `reflections`,
  `lattice_symmetry`) — all 230 groups, change-of-basis setting notation, Le Page
  lattice symmetry, derived (not tabulated) from gemmi-verified data.
- **Serial-crystallography reindexing** (`agentsg.cell.ambiguity`, `canonical`,
  `reindex`) — the reindexing coset, reduction-flip handling, and the
  canonical-superbase operator recovery.
- **Manifold layer** (`agentsg.cell.manifold`, `crystfel_stream`) — deformation
  trajectories, spectral landmarks, and CrystFEL stream parsing.

See `agentsg/README.md` for the full API and `agentsg/docs/DESIGN.md` for the
design principles (derive-don't-tabulate; zero runtime dependencies; oracles as
tests only).

## Manuscript

`manuscript/main_v11.tex` — the IUCrJ communication (latest versioned source).
Compile with `pdflatex` (figures `figure1.png` … `figure4.png` ship alongside).
See `manuscript/SUBMISSION_NOTES.txt`.

## Provenance note

The `pdb_cells.duckdb` database was built from RCSB holdings (cell + space group +
PDB ID only). Roots are computed on the **primitive** lattice of each deposited
(conventional) cell; stored cell parameters and volumes remain the deposited
conventional values.
