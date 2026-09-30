# agentsg concept knowledge graph (with receipts)

A knowledge graph of the crystallographic and mathematical concepts used in the
`agentsg` package, built pygrits-style: the domain content lives in a typed
payload document, and every claim in it has a receipt in a PROV-O / Web
Annotation ledger that points at verbatim evidence.

Built 2026-09-30 from `phzwart/agentsg` HEAD 38eea8c with the Smith-normal-form
patch applied to `semi_invariants.py` and `reflection_lattice.py` (the file
hashes in the evidence are of the patched tree).

## Contents

| file | what it is |
|---|---|
| `agentsg_concepts.json` | **payload**: 155 concepts, 324 typed relations, 481 code anchors, 233 external references |
| `concept-schema.json` | JSON Schema of the payload (its digest is the ledger's `schema_digest`) |
| `agentsg_kg.grits.jsonld` | **pygrits 0.6.2 ledger**, validated with `pygrits.validate()`: 1 plan, 1349 nodes |
| `neo4j/*.csv`, `neo4j/load.cypher`, `neo4j/queries.cypher` | Neo4j import package and competency queries |
| `builder/` | the catalogue (`concepts.py`), anchor extractor, fetch results and `build.py` — rerun to regenerate |

## Graph model

Concept nodes (`kind` ∈ crystallography 73, mathematics 65, algorithm 14,
data-structure 3) carry a builder-written one-sentence definition, aliases and
literature pointers named in the code. Concept–concept relationships are typed:
`USES`, `IS_A`, `PART_OF`, `SPECIALIZES`, `DUAL_OF`, `EQUIVALENT_TO`,
`RELATED_TO`, `DEFINED_BY`, `CONTRASTS_WITH`.

Software is evidence, not a node class of its own: each concept has
`HAS_CODE_EVIDENCE` edges to `CodeEvidence` nodes (module, symbol, line, the
verbatim first docstring line, GitHub URI, file SHA-256). Every public routine
that embodies a concept — SVD, Niggli/Selling reduction, HNF/SNF, rref,
cofactor inverses, KD-tree/NearTree, Dijkstra, Laplacian, Pearson, Reynolds
averaging, coset enumeration, … — is anchored this way; 44 of the 45 source
modules are covered (the server plumbing, tabulated data and PDB download
modules are deliberately not, except `serve/scatter.py` for the SVD).

External definitions are `Reference` nodes (`IUCr` or `Wikipedia` label) with
`DEFINED_IN` edges: URL, page title, one verbatim sentence with a
TextQuoteSelector (`quote`/`prefix`/`suffix`), a `status`, and a `receipt` id
into the ledger.

## How honest the receipts are

* Code anchors are `how: quote` with the file's SHA-256 and the exact
  substring; the builder verified every `exact` string is present in the file.
* Dictionary/Wikipedia sentences were transcribed from pages rendered to text
  by the fetch tool. The raw page bytes were not retained, so no content hash
  exists and these entities are `how: derived` with a rationale saying so
  (`status: quoted`). Where inline math or markup was visibly stripped they are
  `status: quoted-unverified-markup` (50 of 233). A random spot-check of the
  transcriptions against fresh fetches matched 3/3.
* 8 references were unreachable (the Wikipedia domain served cache-only and
  those pages were uncached: Hall notation, Pseudosymmetry, Laue group,
  Neumann's principle, Euclidean normalizer, Holohedry, Site symmetry, Wilson
  plot). They are recorded as `result: inconclusive` absence entities, and the
  affected concepts still carry their IUCr quote where one exists. Two concepts
  have no reachable external definition at all: `hall_symbol` and
  `extended_setting_notation`.
* Concept definitions are `how: derived` (builder paraphrase, rationale given);
  relations are `how: inferred`. Both point at the payload via
  `payload_ref` (SHA-256 of `agentsg_concepts.json`).

## Loading into Neo4j

Copy `neo4j/*.csv` into the database's `import/` directory and run
`load.cypher` (needs APOC for the last statement, or use the commented
per-type MERGE). `queries.cypher` holds seven competency queries for a chatbot
layer (where is X implemented; what does X rest on; definition with receipts;
gaps; full-text entry; dual pairs; everything a module touches).

## Regenerating

```
cd builder
python anchors.py /path/to/agentsg/agentsg/src      # verbatim docstring anchors
python build.py                                     # payload, ledger (validated), Neo4j CSVs
```
`concepts.py` is the hand-curated catalogue; add a concept there with its code
anchors and IUCr/Wikipedia titles, refetch (`fetched_*.json`), rebuild.
