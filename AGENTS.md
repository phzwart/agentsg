# AGENTS.md

## What this is

Crystallographic space-group engine (exact rational arithmetic) and numeric
unit-cell math, plus a local HTTP API and an MCP server. Symmetry results are
exact. Cell edges, volumes, and lattice-search distances are floats.

Answer crystallography by importing the package or by calling the local API.
`GET /skill.md`, once a server is up, is the scientific playbook (LIMITATIONS,
t vs k, no invented Wyckoff letters). The same limits apply in process.

## Commands

From the repository root, with the virtualenv active. Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .                   # library: space groups, settings, cells
pip install -e ".[db,plot]"        # add the PDB index and ITA plate PNGs
pip install -e ".[db,plot,test]"   # add pytest, gemmi, and spglib
pip install -e ".[mcp,db]"         # add the MCP server
```

There is no `[dev]` extra. Tests live in `agentsg/tests`.

```bash
python -c "from agentsg import space_group; print(space_group(96))"
pytest -q
pytest -q -m "not slow"
pytest agentsg/tests/test_setting.py -q
```

`space_group(96)` prints `SpaceGroup(No. 96, 'P 43 21 2', Hall 'P 4nw 2abw')`.

PDB cells (206,214; cite [10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222)):

```bash
mkdir -p data
curl -L --fail -o data/pdb_cells.duckdb \
  "https://zenodo.org/records/22986222/files/pdb_cells.duckdb?download=1"
```

Local HTTP API. Leave it running. Skip this when `GET /health` already
returns `"status": "ok"`. Leave `AGENTSG_TOKEN` unset so localhost stays open.

```bash
export AGENTSG_DB="$PWD/data/pdb_cells.duckdb"
export AGENTSG_HOST=127.0.0.1
export AGENTSG_PORT=8765
export AGENTSG_PUBLIC_URL="http://127.0.0.1:8765"
export AGENTSG_API_NAME=agentsg

python -m agentsg.serve
```

Stdout says `agentsg serve on http://127.0.0.1:8765`. Health reports a `cells`
count near 206214. If the port is taken, change `AGENTSG_PORT` and
`AGENTSG_PUBLIC_URL` together.

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/skill.md
curl -sS "http://127.0.0.1:8765/v1/space-group?sg=96"
```

MCP, on `127.0.0.1:9877` with `/mcp`:

```bash
python -m agentsg.serve.mcp_app --db "$PWD/data/pdb_cells.duckdb"
```

`./serve-sg-mcp.sh`, `./serve-sg-muse.sh`, and `./serve-tunnel-front.sh` are
the public sg-mcp / sg-muse processes. They read `agentsg/serve.local.env`.

Concept graph. Commit the code first. `anchors.py` must be given `agentsg/src`
(its default source root is wrong). `build.py` exits when
`git status --porcelain` is non-empty. Commit `anchors.json`, then build, then
commit `agentsg/src/agentsg/kg/out`.

```bash
python agentsg/src/agentsg/kg/anchors.py agentsg/src
python agentsg/src/agentsg/kg/build.py
```

## Layout

Sources are `agentsg/src/agentsg/`. Tests are `agentsg/tests/` and mirror the
package.

- `space_groups.py`, `ita_settings.py`, `hall.py`, `setting.py` — the 230 groups, alternate settings, and change-of-basis notation
- `group.py`, `reflections.py`, `reflection_lattice.py`, `wyckoff.py`, `subgroups.py`, `semi_invariants.py`, `identify.py` — operators, absences, sites, t/k subgroups
- `cell/` — numeric cells, Selling keys, Niggli, reindexing, the PDB store (`celldb.py`), and ITA plates (`diagrams.py`)
- `serve/` — HTTP API (`app.py`, `handlers.py`), MCP wrappers (`mcp_app.py`), playbook (`skill.md`)
- `kg/` — concept text in `concepts.py`; generated `anchors.json` and `out/`

## Use it as a library

Quote the operators, conditions, orders, and edge types the functions return.
Wyckoff letters stay unassigned. `space_group("P21/n")` is the same group as
`space_group(14)` and `space_group("-P 2yn")`. On the HTTP API, an ambiguous
short name also sets `assumed` to `unique axis b`.

```python
from fractions import Fraction as Fr

from agentsg import space_group, SpaceGroupSetting, subgroup_graph
from agentsg.linalg import Vector3
from agentsg.reflections import reflection_conditions
from agentsg.wyckoff import multiplicity, site_symmetry_order

sg = space_group("P21/n")          # or 14, "P 1 21/n 1", "-P 2yn", "14:b2"
ops = list(sg.operations())
print(sg, sg.order(), reflection_conditions(ops))

x = Vector3((Fr(0), Fr(0), Fr(0)))
print(multiplicity(x, ops), site_symmetry_order(x, ops))

setting = SpaceGroupSetting.parse("C 2 2 21 (a/2+b/2,-a/2+b/2,c)")
print(setting.order(), list(setting.operations()))

graph = subgroup_graph(96)         # each edge has type "t" or "k"
```

PDB nearest neighbours need the `db` extra and `data/pdb_cells.duckdb`.
Distances are ångströms on `sort(p / sqrt(Σ p))`.

```python
from agentsg.cell import CellDatabase, sorted_linear_key

sorted_linear_key((78, 78, 37, 90, 90, 90))
db = CellDatabase("data/pdb_cells.duckdb")
hits = db.build_index().k_nearest((100, 100, 100, 90, 90, 90), k=20)
```

Plates need the `plot` extra. `ita_plate` returns a matplotlib figure. With
the server up, `GET /plates?sg=19` is the same drawing.

```python
from agentsg.cell.diagrams import ita_plate

fig = ita_plate(19, legend=True)
fig.savefig("p212121.png")
```

## How to call the local API

Base URL `http://127.0.0.1:8765`. No Bearer token unless the process was
started with `AGENTSG_TOKEN`.

1. `GET /health` — liveness and cell count.
2. `GET /skill.md` — standing playbook for every space-group, reflection, site, plate, cell, or PDB-lattice question.
3. `GET /api` — endpoint catalog. Use it; paths are not guessed.
4. Then a compute call, e.g. `GET /v1/space-group?sg=96`.

`POST` uses `Content-Type: application/json`. Discovery routes (`/health`,
`/skill.md`, `/openapi.json`, `/docs/muse.md`, `/v1/help`) work without auth
when no token is set.

Omit `AGENTSG_DB` and `/v1/pdb/search` returns 503; space-group and cell math
still work. Rebuild the database with
`python -m agentsg.cell.pdb_app build data/pdb_cells.duckdb` only when the
Zenodo download fails. A remote connector is only for a base URL and Bearer
token a human already gave you.

## Rules

- Operators and changes of basis stay exact rationals (`fractions.Fraction`). Cell edges, volumes, angles, and search distances are floats.
- Absences and semi-invariants are derived from the operators. New code and tests compare exact operator sets or rational matrices. Grid enumeration is not a proof.
- Plates and panel titles come from the operators and the setting table. Identity of a transformed group is the operator set of the matched Hall row.
- Wyckoff letters are unassigned. Systematic absences and International Tables text come from the call, not from memory.
- MCP wrappers in `serve/mcp_app.py` stay thin. A bad call raises `HttpError` so the tool result is `isError` and the text includes a corrected example.
- The local VM API listens on `127.0.0.1` with no token.

## Do not

- Hand-edit `agentsg/src/agentsg/kg/out/` or `anchors.json`.
- Rename an MCP tool or argument without leaving the old name as a deprecated alias for one release.
- Set `AGENTSG_TOKEN` for the local API.
- Commit `data/`, `agentsg/serve.local.env`, or LaTeX `*.aux` / `*.log` / `*.out`.

## Done means

1. `pytest -q` passes. A single file is `pytest agentsg/tests/<file>.py -q`.
2. Tool or HTTP behaviour changes get a `CHANGELOG.md` entry and a bump of `API_VERSION` in `agentsg/src/agentsg/serve/manifest.py`. The package version in `pyproject.toml` stays put unless the library release itself changes.
3. `serve/skill.md`, the MCP docstrings, and the `_mcp_playbook` replacements in `mcp_app.py` describe the same calls.
4. An anchored function that moved is refreshed with the concept-graph commands above, from a clean tree, so the payload's `dirty` flag is false.
