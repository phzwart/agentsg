# Installation

Python 3.10 or newer, `pip`, and `git`. The installed package has no required
third-party libraries. DuckDB, NumPy, SciPy, Matplotlib, pytest, gemmi, and
spglib are optional extras.

## Clone and create a virtual environment

From the repository root (the directory that contains `README.md`):

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

## What to install

| Extra | Adds | Use it for |
|---|---|---|
| *(none)* | nothing | space groups, cells, Niggli, reindexing |
| `db` | duckdb, numpy, scipy | the PDB cell index |
| `plot` | matplotlib, numpy | ITA plate PNGs |
| `test` | pytest, gemmi, spglib, matplotlib, numpy, duckdb, scipy | the test suite |
| `mcp` | fastmcp ≥ 4 | the MCP server |
| `docs` | sphinx, myst-parser, furo, sphinx-copybutton | this documentation |
| `full` | `db`, `plot`, and `mcp` | PDB search, plates, and the MCP server |

Everything the running library needs beyond itself:

```bash
python -m pip install -e ".[full]"
```

A local checkout that can search PDB cells and draw plates:

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
without reinstalling. A released install from GitHub is:

```bash
python -m pip install "git+https://github.com/phzwart/agentsg.git"
```

Check that the package imports and that space group 96 resolves:

```bash
python -c "from agentsg import space_group; print(space_group(96))"
```

That prints the Hermann–Mauguin symbol `P 43 21 2`.

## PDB cell database

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

## Tests

From the repository root, after the `test` extra:

```bash
pytest -q
pytest -q -m "not slow"
```

The second command skips the long calibration and all-230 stress tests.
About 3600 tests are collected. gemmi and spglib are oracles in the tests.
They are not imported by the package.

## Build this site locally

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. The HTML tree is not committed.

## Publish on Read the Docs

The repository contains `.readthedocs.yaml`. Read the Docs uses it to install
the package with the `docs` extra and to build `docs/conf.py`.

1. Sign in at [readthedocs.org](https://readthedocs.org/) with the GitHub account that can see `phzwart/agentsg`.
2. Choose **Import a Project** and select that repository.
3. Leave the configuration file as `.readthedocs.yaml`. The first build publishes the site at `https://agentsg.readthedocs.io/` once the project slug is `agentsg`.

A push to the default branch rebuilds the site. Pull requests can be built
from the Read the Docs project settings.
