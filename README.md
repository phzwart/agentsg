# agentsg — lattice symmetry, conservative lattice search, and exact reindexing

A dependency-free Python package for unit-cell and space-group operations.
Each lattice is given a sorted six-vector Selling key: continuous, Euclidean,
and a lower bound on the true distance, so a radius search cannot miss a match.
Identity and the reindexing operator come from the finite Selling closure, not
from equality of the key. The same calls are a local REST API for the Muse
connector and MCP tools for clients such as Claude.
The shareable documentation is the Sphinx site in `docs/`
([build it locally](docs/installation.md), or import this repository on
[Read the Docs](https://readthedocs.org/), which reads `.readthedocs.yaml`).

```
.
├── agentsg/            the installable package (src/, tests/, docs/, pyproject.toml)
│   ├── src/agentsg/    zero-dependency runtime
│   ├── tests/          ~3600 tests (pytest)
│   └── docs/           DESIGN.md, reduction-flip literature note, etc.
├── analysis/           database-build & calibration figures + data
│   ├── figures/
│   └── data/           CSV summaries + .npz calibration arrays
└── data/
    └── pdb_cells.duckdb  optional download (not in git): 206,214 PDB cells,
                          primitive-lattice roots, one Selling-reduced cell,
                          and the deposited-to-reduced change of basis
                          https://doi.org/10.5281/zenodo.22986222
```

## Installation

Python 3.10 or newer, `pip`, and `git`. The installed package has no required
third-party libraries. DuckDB, NumPy, SciPy, Matplotlib, pytest, gemmi, and
spglib are optional extras.

### Clone and create a virtual environment

From the repository root (the directory that contains this file):

```bash
git clone https://github.com/phzwart/agentsg.git
cd agentsg
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

Stay in this environment for the commands below. Install from this directory.
A second `pyproject.toml` lives in `agentsg/` for that subtree alone; the
root file is the one whose tests are `agentsg/tests`.

### What to install

| Extra | Adds | Use it for |
|---|---|---|
| *(none)* | nothing | space groups, cells, Niggli, reindexing |
| `db` | duckdb, numpy, scipy | the PDB cell index |
| `plot` | matplotlib, numpy | ITA plate PNGs |
| `test` | pytest, gemmi, spglib, matplotlib, numpy, duckdb, scipy | the test suite |
| `mcp` | fastmcp ≥ 4 | the MCP server |
| `full` | `db`, `plot`, and `mcp` | PDB search, plates, and the MCP server |

A local checkout that can search PDB cells, draw plates, and serve MCP:

```bash
python -m pip install -e ".[full]"
```

PDB search and plates only:

```bash
python -m pip install -e ".[db,plot]"
```

The library alone, still with no third-party packages:

```bash
python -m pip install -e .
```

The test suite as well:

```bash
python -m pip install -e ".[db,plot,test]"
```

Editable (`-e`) installs the working tree, so later edits are picked up
without reinstalling. A released install from GitHub is
`pip install "git+https://github.com/phzwart/agentsg.git"`.

Check that the package imports and that space group 96 resolves:

```bash
python -c "from agentsg import space_group; print(space_group(96))"
```

That prints the Hermann–Mauguin symbol `P 43 21 2`.

### PDB cell database

Nearest-neighbour search over deposited cells needs the Zenodo snapshot:
206,214 cells, the primitive-lattice roots, one Selling-reduced cell, and the
deposited-to-reduced change of basis.
[10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222).

```bash
mkdir -p data
curl -L --fail -o data/pdb_cells.duckdb \
  "https://zenodo.org/records/22986222/files/pdb_cells.duckdb?download=1"
```

The file is not in git. Space-group and cell calls work without it. PDB
search then returns HTTP 503. Rebuilding from RCSB is
`python -m agentsg.cell.pdb_app build data/pdb_cells.duckdb` and is slow;
use it only if the download fails.

### Tests

From the repository root, after the `test` extra:

```bash
pytest -q
pytest -q -m "not slow"    # skip the long calibration and all-230 stress tests
```

About 3600 tests are collected. gemmi and spglib are oracles in the tests.
They are not imported by the package.

### Local HTTP API

This is the server an agent calls. It listens on localhost and does not use
a token unless you set `AGENTSG_TOKEN`.

```bash
export AGENTSG_DB="$PWD/data/pdb_cells.duckdb"
export AGENTSG_HOST=127.0.0.1
export AGENTSG_PORT=8765
export AGENTSG_PUBLIC_URL="http://127.0.0.1:8765"
export AGENTSG_API_NAME=agentsg

python -m agentsg.serve
```

Leave that process running. Stdout should say
`agentsg serve on http://127.0.0.1:8765`. Then:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/skill.md
curl -sS "http://127.0.0.1:8765/v1/space-group?sg=96"
```

`/health` returns `"status": "ok"` and a `cells` count near 206214 when the
database is in place. `GET /skill.md` is the scientific playbook.
`GET /api` is the endpoint catalog. If port 8765 is taken, change
`AGENTSG_PORT` and `AGENTSG_PUBLIC_URL` together.

[`AGENTS.md`](AGENTS.md) is the same setup written for an agent in a VM.
Plates need the `plot` extra. The database needs the `db` extra and the
Zenodo file.

### MCP

```bash
python -m pip install -e ".[mcp,db]"
python -m agentsg.serve.mcp_app --db "$PWD/data/pdb_cells.duckdb"
```

That listens on `127.0.0.1:9877`. The streamable HTTP endpoint is `/mcp`.
Override the bind address with `--host`, `--port`, or `AGENTSG_MCP_HOST` and
`AGENTSG_MCP_PORT`.

## Using the library

```python
from agentsg.cell import sorted_linear_key, sorted_linear_distance
from agentsg.cell import CellDatabase

# search key: sort(p_ij / sqrt(sum p)), in ångström
sorted_linear_key((78, 78, 37, 90, 90, 90))
d = sorted_linear_distance((78, 78, 37, 90, 90, 90), (79, 79, 38, 90, 90, 90))

# pdb_search distances are in this same metric
db = CellDatabase("data/pdb_cells.duckdb")
idx = db.build_index()
hits = idx.k_nearest((100, 100, 100, 90, 90, 90), k=20)
```

## What's here

- **Lattice search key** (`agentsg.cell.rootform`) — `sort(p_ij / sqrt(Σ p))`,
  one key per lattice, a rearrangement lower bound, Lipschitz, and linear in
  the cell edges. `pdb_search` reports distances in this metric. Kurlin's
  square-root products stay available with `stabilize="sqrt"`.
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
