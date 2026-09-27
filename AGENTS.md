# AGENTS.md

Standing instructions for an agent (including Facebook **Muse**) running in a
local VM. **Do not use a remote host.** Install this tree, start the HTTP API
on localhost, and answer crystallography only from that API.

After the server is up, `GET /skill.md` is the scientific playbook
(LIMITATIONS, t vs k, no invented Wyckoff letters). This file is only how to
**get the server running**.

## Already running?

```bash
curl -sS http://127.0.0.1:8765/health
```

If `"status": "ok"`, skip setup. Base URL is `http://127.0.0.1:8765`. No
Bearer token unless you started the process with `AGENTSG_TOKEN`.

## One-time setup

Needs Python ≥ 3.10, `pip`, and `curl` (or any HTTPS downloader).

```bash
# from a clone of this repository (or the tree Muse already has)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[db,plot]"        # DuckDB index + ITA plate PNGs

mkdir -p data
# 206,214 PDB cells + Kurlin roots (~41 MB). Cite 10.5281/zenodo.22986222
curl -L --fail -o data/pdb_cells.duckdb \
  "https://zenodo.org/records/22986222/files/pdb_cells.duckdb?download=1"
```

The DOI for that file is [10.5281/zenodo.22986222](https://doi.org/10.5281/zenodo.22986222).
Do not rebuild from RCSB unless the download fails.

## Start the server

Leave this process running (background job, or a second terminal). **Do not
set `AGENTSG_TOKEN`** in the VM — localhost stays open so you never need a
public URL or credentials.

```bash
export AGENTSG_DB="$PWD/data/pdb_cells.duckdb"
export AGENTSG_HOST=127.0.0.1
export AGENTSG_PORT=8765
export AGENTSG_PUBLIC_URL="http://127.0.0.1:8765"
export AGENTSG_API_NAME=agentsg
# unset AGENTSG_TOKEN

python -m agentsg.serve
```

Wait until stdout says `agentsg serve on http://127.0.0.1:8765` and
`GET /health` returns `"status": "ok"` and a `cells` count (about 206214).

If port 8765 is taken, pick another and set `AGENTSG_PORT` and
`AGENTSG_PUBLIC_URL` to match.

## How to call (local Muse)

Base URL: `http://127.0.0.1:8765`

1. `GET /health` — liveness and cell count.
2. `GET /skill.md` — standing playbook. Follow it for every space-group,
   reflection, site, plate, cell, or PDB-lattice question.
3. `GET /api` — endpoint catalog. Do not guess paths.
4. Then a compute call, e.g. `GET /v1/space-group?sg=96`.

`POST` uses `Content-Type: application/json`. Discovery routes
(`/health`, `/skill.md`, `/openapi.json`, `/docs/muse.md`, `/v1/help`) work
without auth when no token is set.

Never invent systematic absences or ITA Wyckoff letters. Never recite
International Tables from memory.

## Optional

- **No PDB search:** omit `AGENTSG_DB`. Space-group and cell math still work;
  `/v1/pdb/search` returns 503.
- **Rebuild instead of Zenodo:** `python -m agentsg.cell.pdb_app build data/pdb_cells.duckdb`
  (slow; hits RCSB).
- **Remote connector:** only if a human already gave you a base URL and a
  Bearer token. That is not this path.
