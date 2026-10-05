# agentsg

Python ≥ 3.10 space-group engine. Operators and changes of basis are exact
rationals (`fractions.Fraction`). Cell edges, volumes, and search distances
are floats. Sources: `agentsg/src/agentsg/`. Tests: `agentsg/tests/`.

Answer crystallography by importing the package or calling the local API.
Once a server is up, `GET /skill.md` is the scientific playbook. Wyckoff
letters stay unassigned.

## Commands

From the repository root, virtualenv active.

- Library: `pip install -e .`
- PDB index and plates: `pip install -e ".[db,plot]"`
- Tests: `pip install -e ".[db,plot,test]"`
- MCP: `pip install -e ".[mcp,db]"`
- Import check: `python -c "from agentsg import space_group; print(space_group(96))"`
- Full suite: `pytest -q`
- Skip the slow tests: `pytest -q -m "not slow"`
- One file: `pytest agentsg/tests/test_setting.py -q`
- One test: `pytest agentsg/tests/test_setting.py::test_name -q`
- MCP: `python -m agentsg.serve.mcp_app --db "$PWD/data/pdb_cells.duckdb"`
- Public processes: `./serve-sg-muse.sh`, `./serve-sg-mcp.sh`, `./serve-tunnel-front.sh` (they read `agentsg/serve.local.env`)

`space_group(96)` prints `SpaceGroup(No. 96, 'P 43 21 2', Hall 'P 4nw 2abw')`.

PDB snapshot, 206,214 cells, [10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222):

```bash
mkdir -p data
curl -L --fail -o data/pdb_cells.duckdb \
  "https://zenodo.org/records/22986222/files/pdb_cells.duckdb?download=1"
```

Local API. Leave `AGENTSG_TOKEN` unset. Skip this when
`curl -sS http://127.0.0.1:8765/health` already returns `"status": "ok"`.
Stdout says `agentsg serve on http://127.0.0.1:8765` and health reports a
`cells` count near 206214. If the port is taken, change `AGENTSG_PORT` and
`AGENTSG_PUBLIC_URL` together.

```bash
export AGENTSG_DB="$PWD/data/pdb_cells.duckdb"
export AGENTSG_HOST=127.0.0.1
export AGENTSG_PORT=8765
export AGENTSG_PUBLIC_URL="http://127.0.0.1:8765"
export AGENTSG_API_NAME=agentsg
python -m agentsg.serve
```

Concept graph. Commit the code first. Pass `agentsg/src` to `anchors.py`.
`build.py` exits while `git status --porcelain` is non-empty. Commit
`anchors.json`, then build, then commit `agentsg/src/agentsg/kg/out`.

```bash
python agentsg/src/agentsg/kg/anchors.py agentsg/src
python agentsg/src/agentsg/kg/build.py
```

## Architecture

- `space_groups.py`, `ita_settings.py`, `hall.py`, `setting.py` — the 230 groups, alternate settings, change of basis
- `group.py`, `reflections.py`, `wyckoff.py`, `subgroups.py`, `semi_invariants.py`, `identify.py` — derived from the operators
- `cell/` — numeric cells, Selling keys, the PDB store (`celldb.py`), ITA plates (`diagrams.py`)
- `serve/handlers.py` — HTTP behaviour. `serve/mcp_app.py` forwards arguments
- `serve/skill.md` — playbook served at `GET /skill.md`
- `kg/concepts.py` — concept text. `kg/anchors.json` and `kg/out/` are generated

```python
from agentsg import space_group, SpaceGroupSetting, subgroup_graph

sg = space_group("P21/n")       # same group as 14 and "-P 2yn"
list(sg.operations())
SpaceGroupSetting.parse("C 2 2 21 (a/2+b/2,-a/2+b/2,c)").operations()
subgroup_graph(96)              # each edge has type "t" or "k"
```

Further samples are in `agentsg/README.md`. With the server up, `GET /api`
is the endpoint catalog and `GET /plates?sg=19` draws the plate.

## Code style

- Symmetry code uses `fractions.Fraction`. Floats belong in `cell/`.
- Identify a space group by its closed operator set, exact rational 3×4 matrices, not by the Hermann–Mauguin string.
- Absences, site symmetry, and semi-invariants are derived from the operators.
- Orthorhombic panel titles are the ITA setting of that panel's change of basis.
- A bad HTTP or MCP call raises `HttpError` and names a corrected example, so the MCP result is `isError`.

## Testing

- `pytest -q` from the repository root is the check before the work is done.
- gemmi and spglib are oracles inside tests. The package does not import them.
- New symmetry tests compare operator sets. An hkl grid scan is not a proof.

## Security

- `agentsg/serve.local.env` holds the public-server token. Leave it unprinted and uncommitted.
- Leave `AGENTSG_TOKEN` unset on the local API. It binds `127.0.0.1`.
- Leave `data/*.duckdb` and LaTeX `*.aux`, `*.log`, `*.out` uncommitted.

## Git

- Commit messages are one imperative sentence, matching `git log`.
- A tool or HTTP behaviour change adds a `CHANGELOG.md` entry and bumps `API_VERSION` in `agentsg/src/agentsg/serve/manifest.py`. The package version in `pyproject.toml` changes for a library release.
- `serve/skill.md`, the MCP docstrings, and `_mcp_playbook` in `mcp_app.py` describe the same calls.
- Rebuild the concept graph with the commands above so the payload's `dirty` flag is false.
- Leave `kg/out/` and `anchors.json` unedited by hand.
- A renamed MCP tool or argument keeps the old name as a deprecated alias for one release.
