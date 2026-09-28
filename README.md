# agentsg — lattice symmetry, conservative lattice search, and exact reindexing

A dependency-free Python package for unit-cell and space-group operations.
Each lattice is given a sorted six-vector Selling key: continuous, Euclidean,
and a lower bound on the true distance, so a radius search cannot miss a match.
Identity and the reindexing operator come from the finite Selling closure, not
from equality of the key. The same calls are a local REST API for the Muse
connector and MCP tools for clients such as Claude.

```
.
├── agentsg/            the installable package (src/, tests/, docs/, pyproject.toml)
│   ├── src/agentsg/    zero-dependency runtime
│   ├── tests/          ~3600 tests (pytest)
│   └── docs/           DESIGN.md, reduction-flip literature note, etc.
├── manuscript/         IUCrJ manuscript (see main_v9.tex)
├── analysis/           database-build & calibration figures + data
│   ├── figures/
│   └── data/           CSV summaries + .npz calibration arrays
└── data/
    └── pdb_cells.duckdb  optional download (not in git): 206,214 PDB cells,
                          primitive-lattice roots, one Selling-reduced cell,
                          and the deposited-to-reduced change of basis
                          https://doi.org/10.5281/zenodo.22986222
```

## Muse / local agent

Facebook Muse (or any agent in a VM) should **not** need a public URL. Follow
[`AGENTS.md`](AGENTS.md): install, download the Zenodo cell database, start
`python -m agentsg.serve` on `http://127.0.0.1:8765`, then use `GET /skill.md`.

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

# compare two cells by the sorted search key (continuous lower bound)
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

- **Lattice search key** (`agentsg.cell.rootform`) — the sorted six-vector
  key and its lower-bound distance. Kurlin's structured root form remains the
  complete classification; the sorted key is the lossy filter used for search.
  Also the volume/shape decomposition and the edge-tolerance cutoff.
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

`manuscript/main_v9.tex` — the IUCrJ manuscript (latest versioned source).
Compile with `pdflatex` twice. See `manuscript/SUBMISSION_NOTES.txt`.

## PDB cell database

Download the prebuilt DuckDB snapshot (206,214 crystallographic PDB cells) from
Zenodo: [10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222).
Place it at
`data/pdb_cells.duckdb` (or set `AGENTSG_DB`).

The table was built from RCSB holdings (cell + space group + PDB ID only). Roots
are computed on the **primitive** lattice of each deposited conventional cell;
stored cell parameters and volumes remain the deposited conventional values.
Each row also stores one Selling-reduced cell (`red_a` … `red_gamma`) and the
rational change of basis from the deposited cell to that reduced cell
(`cob00` … `cob22`, exact `numerator/denominator`). You can also rebuild it
with `python -m agentsg.cell.pdb_app build`.
