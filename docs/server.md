# HTTP API and MCP

The same library calls are available over HTTP. On localhost the server does
not require a token unless `AGENTSG_TOKEN` is set. Discovery routes
(`/health`, `/skill.md`, `/openapi.json`, `/docs/muse.md`, `/v1/help`) stay
open either way.

## Start the local server

```bash
export AGENTSG_DB="$PWD/data/pdb_cells.duckdb"
export AGENTSG_HOST=127.0.0.1
export AGENTSG_PORT=8765
export AGENTSG_PUBLIC_URL="http://127.0.0.1:8765"
export AGENTSG_API_NAME=agentsg

python -m agentsg.serve
```

Stdout should say `agentsg serve on http://127.0.0.1:8765`. Then:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/skill.md
curl -sS "http://127.0.0.1:8765/v1/space-group?sg=96"
```

`/health` returns `"status": "ok"` and a `cells` count near 206214 when the
database is in place. `GET /skill.md` is the scientific playbook for an agent
using the API. `GET /api` is the endpoint catalog. If port 8765 is taken,
change `AGENTSG_PORT` and `AGENTSG_PUBLIC_URL` together.

Plates need the `plot` extra. The database needs the `db` extra and the
Zenodo file. Without the file, space-group and cell calls still work and PDB
search returns HTTP 503.

When `AGENTSG_TOKEN` is set, compute routes expect
`Authorization: Bearer <token>`. Leave it unset for a local process.

Simple lookups accept GET query strings. A list body is POST JSON
(`Content-Type: application/json`).

| Route | Role |
|---|---|
| `GET /health` | liveness and cell count |
| `GET /skill.md` | scientific playbook |
| `GET /api`, `GET /v1/help`, `GET /openapi.json` | endpoint catalog |
| `GET /v1/space-group?sg=96` | one space group |
| `POST /v1/setting` | a symbol with a change of basis |
| `POST /v1/identify` | operations to an International Tables type |
| `GET /v1/site` | multiplicity and site symmetry of one point |
| `GET /v1/reflections` | reflection conditions, or one `hkl` |
| `GET /v1/harker` | Harker sections |
| `GET /v1/allowed-origins` | allowed origins and any floating origin |
| `GET /v1/subgroups` | translationengleiche and klassengleiche edges |
| `POST /v1/ita-plate` | plate PNG |
| `GET /v1/cell`, `POST /v1/lattice-symmetry` | cell math and metric symmetry |
| `POST /v1/compare`, `POST /v1/reindex` | cell relation and geometric ambiguity |
| `POST /v1/pdb/search`, `GET /v1/pdb/<pdbid>` | neighbour search and one deposited cell |
| `GET /v1/concept` | concept graph: search, neighbours, uses |

Do not guess further paths. Read `GET /api` on the running server.

## MCP

```bash
python -m pip install -e ".[mcp,db]"
python -m agentsg.serve.mcp_app --db "$PWD/data/pdb_cells.duckdb"
```

That listens on `127.0.0.1:9877`. The streamable HTTP endpoint is `/mcp`.
Override the bind address with `--host`, `--port`, or `AGENTSG_MCP_HOST` and
`AGENTSG_MCP_PORT`.
