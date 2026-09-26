---
name: agentsg
description: Call the agentsg REST API for every space-group, reflection, site, ITA plate, unit-cell, or PDB-lattice question. Never answer crystallography from memory.
---

# Standing rule

You have an HTTP connector named **agentsg**.

- Base URL: `https://sg-muse.mxagents.org`
- Auth: `Authorization: Bearer` (token already in credentials — never print it, never put it in chat)
- `Content-Type: application/json` on every POST
- Read-only. Rate limit 60/min (`429` + `Retry-After`)

**Call the API before you answer.** Start a session with `GET /api` (Bearer) and use that catalog — do not guess paths. Do not recite International Tables from memory. Do not invent systematic absences or Wyckoff letters. If you are unsure which compute endpoint, `GET /v1/space-group?sg=…`.

After any `4xx`/`5xx`, read the JSON `error` and retry with a corrected body. Do not invent a result.

# How to run a call

Prefer **GET with query params** for simple lookups. Use **POST JSON** when the body is a list or a setting string.

| Intent | Call this |
|---|---|
| Space-group info (ops, order, system, absences) | `GET /v1/space-group?sg=96` or `POST /v1/space-group` `{"sg":96}` |
| Is *hkl* allowed? phase, equivalents | `GET /v1/reflections?sg=96&hkl=1,0,0` |
| All reflection conditions | `GET /v1/reflections?sg=96` |
| Multiplicity / site symmetry at (x,y,z) | `GET /v1/site?sg=225&xyz=1/4,1/4,1/4` |
| Harker sections | `GET /v1/harker?sg=19` |
| Draw / show the ITA plate | `GET /plates?sg=19` (PNG) or `POST /v1/ita-plate` `{"sg":96,"legend":true}` then **GET the returned `png_url`** |
| Discover every endpoint | `GET /api` |
| Non-standard setting | `POST /v1/setting` `{"setting":"P 21 21 2 (2a,b-a,c)"}` |
| Identify operators | `POST /v1/identify` `{"ops":["x,y,z",...]}` |
| Cell volume, Niggli, root key | `GET /v1/cell?cell=79,79,38,90,90,90&sg=96` |
| What Bravais / holohedry is this noisy cell? | `POST /v1/lattice-symmetry` `{"cell":[50,50,51,90,90,90]}` |
| Are these two lattices the same? | `POST /v1/compare` with `cell_a` and `cell_b` |
| Native vs SeMet integer transform | `POST /v1/compare` with `include_sublattices: true` |
| Serial XFEL reindex / ambiguity | `POST /v1/reindex` |
| Find similar PDB cells | `POST /v1/pdb/search` with `sg` + `cutoff` or `k` |
| Look up 1ABC | `GET /v1/pdb/1ABC` |
| Remind yourself of this playbook | `GET /api` (full catalog) or `GET /skill.md` |

`sg` accepts an IT number (`96`), Hermann–Mauguin (`P 43 21 2` / `P43212`), or a Hall symbol.

# What to say from the JSON

**Space group** (`/v1/space-group`): quote `sg_number`, `sg_hm`, `crystal_system`, `centering`, `order`, `point_group_order`, `laue_class`. Summarize `reflection_conditions`. Do not dump every `ops` xyz unless asked; say how many there are (`order`) and name the generators in words (screws, glides, inversions).

**Reflections**: if `hkl` was sent, lead with `absent`, then `centric` / `phase`, then a few `equivalent_hkls`. If no `hkl`, summarize `conditions`.

**Site**: quote `multiplicity`, `site_symmetry_order`, and a few `orbit` points. `wyckoff_letter` is always `null` — say that letters are not assigned; you computed the orbit content.

**ITA plate**: name glyphs from `elements` (`type`, `symbol`, `axis`). Then GET `png_url` (same bearer) and display the PNG. Monoclinic defaults to `projection=b`.

**Cell**: quote `volume`, `niggli`, and `root_invariant`. If the cell is C/I/F/R, you **must** send `sg` so the server reduces to primitive first.

**Lattice symmetry**: quote `crystal_system` and `order` from the response. Do not call a noisy cell cubic unless the JSON does.

**Compare**: lead with `root_distance` in Å. G6 / similarity only if asked.

**Reindex**: list branches. `is_metric_symmetry: true` (residual 0) is true merohedry. You cannot pick the intensity branch. Say that you cannot decide P3₁ vs P3₂ from the cell alone.

**PDB search**: list `pdb_id`, `distance` (Å on the root invariant), `sg_hm`, `cell`.

# Worked calls

User: “What is space group 96?” / “Tell me about P4₃2₁2”

```
GET /v1/space-group?sg=96
```

or

```
POST /v1/space-group
{"sg": 96}
```

User: “What is systematically absent in P4₃2₁2?”

```
GET /v1/space-group?sg=96
GET /v1/reflections?sg=96
```

Summarize the condition strings. Do not invent 00l or h00 rules.

User: “Is (0,0,1) allowed in 96?”

```
GET /v1/reflections?sg=96&hkl=0,0,1
```

User: “Show me the ITA plate for 96.”

```
POST /v1/ita-plate
{"sg": 96, "legend": true}
```

Then GET `png_url` and display it.

User: “Site symmetry of (1/4,1/4,1/4) in Fm-3m”

```
GET /v1/site?sg=225&xyz=1/4,1/4,1/4
```

User: “Find PDB entries like HEWL tetragonal” (cell ≈ 79, 79, 38)

```
POST /v1/pdb/search
{"cell": [79, 79, 38, 90, 90, 90], "sg": 96, "k": 10}
```

# Hard rules

1. **Derive, do not tabulate.** Absences and site content come from operators. Never invent ITA Wyckoff letters (`a`, `b`, `c`, …).
2. **Centred cells need `sg`.** C/I/F/R conventional cells must be reduced to primitive before any root, PDB search, or lattice comparison. Always send `sg`.
3. **Kurlin over G6.** Distances are Å on the root invariant. G6 (Å²) is diagnostic only; do not lead with it.
4. **Geometry surfaces; intensities decide.** `/v1/reindex` lists branches. This API cannot pick the intensity branch.
5. **Root key is not a proof of identity** for every Voronoi type. Small distance means “same lattice for search,” not a theorem.
6. **Monoclinic ITA plates use `projection: "b"`.** The server already defaults monoclinic to `b`; other systems default to `c`.
7. **Explain, do not dump.** Translate JSON into ITA language. Quote numbers from the response.

# Units

- Cell edges Å, angles degrees.
- Root distance Å.
- Le Page δ degrees.
- Similarity distance dimensionless.
- G6 only if asked, and label Å² / diagnostic.
