# agentsg — Muse Custom Connector

Base URL: `{{BASE_URL}}`

Auth: `Authorization: Bearer <token>` on every `/v1/*` call except `GET /v1/help`.
Store the token in Muse credentials. Do not put it in chat.

Content type: `application/json`

Read-only. No writes. Rate limit: 60 requests / minute / token (`429` + `Retry-After`).

This page is the contract. `GET /api` (Bearer) is the machine-readable catalog of every endpoint — call it instead of guessing. `GET /skill.md` is the standing playbook, including a **LIMITATIONS** preface and the **translationengleiche (t) vs klassengleiche (k)** distinction. `GET /v1/subgroups` derives finite-index t and k edges from operators (not the ITA A1 table). `GET /v1/help` is a short JSON index of the same catalog.

## Already connected — paste this into Muse

The connector can reach the API and still not know when to call it. Muse does not re-read tool docs on its own. Paste:

> You already have the agentsg connector. From now on start with GET /api (Bearer) and treat that catalog plus {{BASE_URL}}/skill.md as standing instructions, including the LIMITATIONS section. On every space-group, reflection, site, ITA plate, unit-cell, or PDB question, call the API before you answer. Never invent absences or Wyckoff letters. GET /skill.md and GET /api now, then look up space group 96 with GET /v1/space-group?sg=96 and tell me the Hermann–Mauguin symbol, crystal system, order, and the systematic absences — quoting the JSON, not your memory.

A working connector answers P 43 21 2, tetragonal, order 8, and absences from the response. A connector that describes 96 from memory has drifted — paste the block again.

## New connector — paste this into Muse

> Create a custom connector for agentsg (crystallographic space groups and unit cells). Read {{BASE_URL}}/docs/muse.md, {{BASE_URL}}/skill.md, {{BASE_URL}}/v1/help, and {{BASE_URL}}/openapi.json. Keep the skill as standing routing rules. Auth is Authorization: Bearer — I will give you the token in credentials, not in chat. Then GET /health and tell me how many PDB cells are indexed. After that, GET /v1/space-group?sg=96 and tell me the HM symbol, order, and reflection conditions from the JSON. Do not invent absences or Wyckoff letters.

## Connection test

```
curl {{BASE_URL}}/health
```

A live connector answers `"status": "ok"` and a `cells` count (about 206214). If you describe space groups from memory instead of calling the API, the connector failed.

## How to call

Simple lookups accept GET query strings. Anything with a list body is POST JSON.

Discovery (no token):

- `GET /health`
- `GET /openapi.json`
- `GET /docs/muse.md` (this file)
- `GET /skill.md` (when to call what; scientific caveats)
- `GET /v1/help` (JSON cookbook)

Compute (bearer required except `/v1/help`):

- Space group — `GET /v1/space-group?sg=96` or `POST /v1/space-group` `{"sg":96}` or `"P 43 21 2"`
- Setting — `POST /v1/setting` `{"setting":"P 21 21 2 (2a,b-a,c)"}`
- Identify — `POST /v1/identify` `{"ops":["x,y,z",...]}`. Returns the ITA **type**. `|det P|>1` means a conventional centred cell (primitive F222 → #22, not #16).
- Site — `GET /v1/site?sg=225&xyz=1/4,1/4,1/4`
- Reflections — `GET /v1/reflections?sg=96` or `...?hkl=1,0,0`
- Harker — `GET /v1/harker?sg=19`
- Subgroups — `GET /v1/subgroups?sg=96` (`kind=t|k|both`, `maximal=true`)
- ITA plate — `POST /v1/ita-plate` `{"sg":96,"legend":true}` then `GET` the returned `png_url`
- Cell — `GET /v1/cell?cell=79,79,38,90,90,90&sg=96`
- Lattice symmetry — `POST /v1/lattice-symmetry` `{"cell":[50,50,51,90,90,90]}`
- Compare — two cells; set `include_sublattices: true` for integer `M`
- Reindex — geometric ambiguity only
- PDB search — `POST /v1/pdb/search` `{"cell":[...],"sg":96,"cutoff":1.0}` or `"k":10`. The index is the sorted root key. `"plot":true` adds a mean-centred SVD scatter of the hit Kurlin roots (`plot_png_base64`, `xy`). `"return_cob":true` adds the change of basis from the query cell onto each deposited hit (`cob`, `cob_xyz`, `cob_residual`, and `cob_coset` when several proper operators match). Only determinant +1 settings are listed. `cob: null` means no proper operator matched the reduced cell within tolerance. See `GET /skill.md`.
- PDB lookup — `GET /v1/pdb/1ABC`
- Concept — `GET /v1/concept?q=Smith+normal+form` then `GET /v1/concept?id=smith_normal_form`. `GET /v1/concept/uses`, `/v1/concept/module`, `/v1/concept/receipt` for dependencies, a source file, and one ledger node. Sample questions and what to show the user are in `GET /skill.md` under **Concept questions**.
- Plate image — `GET /v1/ita-plate.png?sg=96&legend=true`

Errors are JSON `{ "error": "..." }` with 400 / 401 / 404 / 422 / 429 / 503.

## Worked curl

```
curl -s "{{BASE_URL}}/v1/space-group?sg=96" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

```
curl -s {{BASE_URL}}/v1/space-group \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sg":96}'
```
