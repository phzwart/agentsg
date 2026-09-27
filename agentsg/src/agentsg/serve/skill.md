---
name: agentsg
description: Call the agentsg REST API for every space-group, reflection, site, ITA plate, unit-cell, or PDB-lattice question. Never answer crystallography from memory.
---

# Standing rule

You have an HTTP connector named **agentsg**.

- Base URL: `{{BASE_URL}}`
- Auth: `Authorization: Bearer` (token already in credentials — never print it, never put it in chat)
- `Content-Type: application/json` on every POST
- Read-only. Rate limit 60/min (`429` + `Retry-After`)

**Call the API before you answer.** Start a session with `GET /api` (Bearer) and use that catalog — do not guess paths. Do not recite International Tables from memory. Do not invent systematic absences or Wyckoff letters. If you are unsure which compute endpoint, `GET /v1/space-group?sg=…`.

After any `4xx`/`5xx`, read the JSON `error` and retry with a corrected body. Do not invent a result.

# LIMITATIONS

agentsg is an **engine**, not International Tables Volume A. It derives operators, orbits, absences, plates, and cells. It does **not** reproduce ITA’s editorial catalog. Act as follows.

| ITA gives you | agentsg gives you | What you do |
|---|---|---|
| Wyckoff letters *a, b, c…* | `wyckoff_letter: null`; multiplicity, site-symmetry **order**, orbit | **Never invent letters.** Say they are not assigned. If the user wants ITA letters, say so and do not guess. |
| The special-position **table** (every inequivalent locus, official representatives, all centering images) | `/v1/site` for a **given** `xyz` only | Probe representative points. Write centering images from the orbit. For R-centering, *(⅔,⅓,z)* is **not** the image of *(0,0,z)* — it is *(⅔,⅓,z+⅓)*. |
| Site-symmetry **HM symbols** (`3m`, `2/m`) | Stabilizer order and ops | Quote the order. Name the site group in words only if the ops justify it; do not print `3m` unless you derived it. |
| Numbered generators and general-position lines *(1) (2)…* | Unsorted closed `ops` | Count them (`order`). Do not pretend ITA numbering. |
| Schoenflies, Patterson symmetry, origin-choice essays | Number, HM, Hall, system, Laue class | Do not invent Schoenflies or Patterson. |
| Maximal subgroups / ITA A1 graphs (t and k), normalizers, Wyckoff **sets** | `GET /v1/subgroups` derives finite-index **t** and **k** (IIa / index-2·3 IIb) from operators. Not the A1 table: no normalizers, no infinite isomorphic series, no editorial maximality proof. | Call the endpoint. Label every edge **t** or **k**. Do not recite A1 from memory. |
| ITA 2016 e-glide shorts (`Aem2`, `Cmce`, `Cmme`, `Ccce`, `Aea2`) | Classic names only (`Abm2`, `Cmca`, …) | If lookup 404s, retry the pre-2016 symbol. |
| Plane groups, rod/layer groups, magnetic groups | 230 3-D space groups | Out of scope. |
| Intensity-based enantiomorph / reindex choice | Geometric branches only | You cannot decide P3₁ vs P3₂ or which twin from the cell. |
| “This primitive 4-op 222 is P222” | `/v1/identify` returns the **type** (`sg_number`) plus `det`, `input_order`, `matched_order` | If `|det| > 1`, quote the type and the CoB. A primitive cell of F222 is still **#22**, not #16 — body-diagonal 2-folds are not unimodular-equivalent to P222. |
| Full ITA polyhedral ASU inequalities | Brick / Dirichlet ASU, not the Volume A half-space gallery | Do not quote ITA ASU inequalities from memory. |

**Also not theorems:** a small root distance is “same lattice for search,” not a proof of identity. G6 is diagnostic (Å²). Centred conventional cells need `sg` before any root, PDB, or compare call.

If the user asks for something in the left column, call the nearest endpoint, state the gap in one sentence, and answer with what the JSON actually contains.

# Translationengleiche vs klassengleiche

These are **different** subgroup relations. Never collapse them into “the subgroup of X” without saying which.

**Translationengleiche (t, type I).** Same translation lattice (same conventional cell and the same centring). You drop rotations / screws / glides / inversion. The point-group order falls; the Bravais lattice does not. Index = |G| / |H| = |P_G| / |P_H|. Example: P4₃2₁2 (96) → P4₃ (78) keeps the P lattice and loses the 2-folds.

**Klassengleiche (k, type II).** Same crystal class (same point group). You lose translations — extra centring is dropped, or the cell is enlarged. Index comes from the lost lattice. Two flavours ITA distinguishes:

- **IIa** — same conventional cell, fewer centring vectors (F → I → P, R → P, …).
- **IIb** — enlarged cell (a superlattice); infinitely many *isomorphic* k-subgroups exist as parameterized series (P2₁2₁2₁ → P2₁2₁2₁ on 2a, etc.).

Example: Fm-3m (225) → Pm-3m (221) is **k**, not t: m-3m is unchanged; F-centring is lost (index 4).

**How to talk.** If the user says “subgroup graph,” call `GET /v1/subgroups?sg=…` (optional `kind=t|k|both`, `maximal=true` by default). Each edge has `type` (`t` or `k`) and `kind` (`I`, `IIa`, `IIb`). Quote those fields. This is derived from operators, not ITA Volume A1: isomorphic k-series are only the index-2 and index-3 ones we can write from Z³, and a t-edge that needs a different conventional cell (F4/mmm as I4/mmm) appears only when identification succeeds. Do not recite A1 from memory. Never present a k-edge as a t-edge or the reverse.

# How to run a call

Prefer **GET with query params** for simple lookups. Use **POST JSON** when the body is a list or a setting string.

| Intent | Call this |
|---|---|
| Space-group info (ops, order, system, absences) | `GET /v1/space-group?sg=96` or `POST /v1/space-group` `{"sg":96}` |
| Is *hkl* allowed? phase, equivalents | `GET /v1/reflections?sg=96&hkl=1,0,0` |
| All reflection conditions | `GET /v1/reflections?sg=96` |
| Multiplicity / site symmetry at (x,y,z) | `GET /v1/site?sg=225&xyz=1/4,1/4,1/4` |
| Harker sections | `GET /v1/harker?sg=19` |
| t / k subgroup graph | `GET /v1/subgroups?sg=96` or `?kind=t` / `?kind=k` |
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

**Identify** (`/v1/identify`): lead with `sg_number` / `sg_hm` (the ITA **type**). Then quote `det`, `input_order`, and `matched_order`. If `det` is not `±1`, the input is a non-standard (often primitive) setting of a centred group — say that, and quote `note` when present. Do **not** relabel it as the primitive group with the same operator count (F222 written primitively is #22, not P222 / #16).

**ITA plate**: name glyphs from `elements` (`type`, `symbol`, `axis`). Then GET `png_url` (same bearer) and display the PNG. Monoclinic defaults to `projection=b`. After an F/I/R→primitive CoB, 2-folds that become body-diagonal are still drawn as projected traces; do not say the plate is empty because the axes are “oblique.”

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

User: “What are the maximal t- and k-subgroups of 96?” / “subgroup graph of P4₃2₁2”

```
GET /v1/subgroups?sg=96
```

List each edge as t (type I) or k (IIa / IIb). Quote `to`, `to_hm`, `index`. Do not invent A1 numbers that are not in the JSON.

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
