"""Build the agentsg concept knowledge graph.

Outputs (in ./out, which the server loads):
  agentsg_concepts.json       payload: concepts, relations, code evidence, references
  concept-schema.json         JSON Schema of the payload
  agentsg_kg.grits.jsonld     pygrits ledger (receipts for every claim)
  neo4j/*.csv, load.cypher    Neo4j import package + competency queries
  README.md                   counts regenerated from the payload

Refuses to write when ``git status --porcelain`` is non-empty unless
``--allow-dirty`` is passed. The payload records the commit hash and dirty flag.
"""
from __future__ import annotations
import csv, hashlib, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True,
).strip())
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
from concepts import CONCEPTS  # noqa: E402
from literature import lookup as literature_lookup  # noqa: E402
from quotes import is_anaphoric, is_unreadable_markup, needles_for, page_matches, pick_sentence  # noqa: E402
import pygrits  # noqa: E402


def git_snapshot(allow_dirty: bool) -> dict:
    """Commit the graph was built from. Exit when the tree is dirty."""
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    )
    dirty = bool(porcelain.strip())
    if dirty and not allow_dirty:
        sys.exit(
            "refusing to build from a dirty tree "
            "(git status --porcelain is non-empty). Commit, or pass --allow-dirty."
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    return {
        "repository": "https://github.com/phzwart/agentsg",
        "commit": commit,
        "dirty": dirty,
    }

AGENT = "agentsg-kg-builder/0.1 (Claude Fable 5.1, Cowork session, 2026-09-30)"
PAYLOAD_SCHEMA_URL = "https://github.com/phzwart/agentsg/kg/concept-schema.json"
FETCH_DATE = "2026-09-30"
SNAPSHOT = git_snapshot("--allow-dirty" in sys.argv)
(OUT / "neo4j").mkdir(parents=True, exist_ok=True)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ inputs
anchors = json.load(open(HERE / "anchors.json"))
fetched = []
for p in sorted(HERE.glob("fetched_*.json")):
    fetched += json.load(open(p))
ref_by_key = {(r["source"], r["title"]): r for r in fetched}

# ------------------------------------------------------------------ payload
schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": PAYLOAD_SCHEMA_URL,
    "title": "agentsg concept graph payload",
    "type": "object",
    "required": ["version", "generated", "concepts"],
    "properties": {
        "version": {"type": "string"},
        "generated": {"type": "string"},
        "source_repository": {"type": "string"},
        "concepts": {"type": "array", "items": {"$ref": "#/$defs/concept"}},
    },
    "$defs": {
        "concept": {
            "type": "object",
            "required": ["id", "label", "kind", "definition", "relations", "code_evidence", "references"],
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                "label": {"type": "string"},
                "kind": {"enum": ["crystallography", "mathematics", "algorithm", "data-structure"]},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "definition": {"type": "string"},
                "literature": {"type": "array", "items": {"type": "string"}},
                "relations": {"type": "array", "items": {
                    "type": "object", "required": ["type", "target"],
                    "properties": {"type": {"enum": ["IS_A", "PART_OF", "USES", "DUAL_OF", "SPECIALIZES", "EQUIVALENT_TO", "RELATED_TO", "DEFINED_BY", "INSTANCE_OF", "CONTRASTS_WITH"]},
                                   "target": {"type": "string"}}}},
                "code_evidence": {"type": "array", "items": {
                    "type": "object", "required": ["module", "symbol", "line", "quote", "uri", "sha256"],
                    "properties": {"module": {"type": "string"}, "symbol": {"type": "string"},
                                   "kind": {"type": "string"}, "line": {"type": "integer"},
                                   "quote": {"type": "string"}, "uri": {"type": "string"},
                                   "sha256": {"type": "string"}}}},
                "references": {"type": "array", "items": {
                    "type": "object", "required": ["source", "title", "url", "status"],
                    "properties": {"source": {"enum": ["iucr", "wikipedia", "literature"]}, "title": {"type": "string"},
                                   "url": {"type": "string"}, "page_title": {"type": "string"},
                                   "status": {"enum": ["quoted", "quoted-unverified-markup", "unreadable-markup", "unreachable", "off-topic", "related", "citation"]},
                                   "quote": {"type": "string"}, "prefix": {"type": "string"},
                                   "suffix": {"type": "string"}, "truncated": {"type": "boolean"},
                                   "note": {"type": "string"}, "fetched": {"type": "string"},
                                   "doi": {"type": "string"}, "sha256": {"type": "string"}}}},
            },
        }
    },
}
schema_bytes = json.dumps(schema, indent=2).encode() + b"\n"
(OUT / "concept-schema.json").write_bytes(schema_bytes)

anch_by_concept: dict[str, list] = {}
for a in anchors["anchors"]:
    anch_by_concept.setdefault(a["concept"], []).append(a)

payload = {"version": "0.1.0", "generated": FETCH_DATE,
           "source_repository": SNAPSHOT["repository"],
           "snapshot": SNAPSHOT,
           "concepts": []}
ids = {c["id"] for c in CONCEPTS}
for c in CONCEPTS:
    refs = []
    # Wikipedia is the external definition when the page is about this concept.
    # IUCr is the fallback, and a mismatched dictionary page is marked related.
    for src, key in (("wikipedia", c["wiki"]), ("iucr", c["iucr"])):
        if not key:
            continue
        r = ref_by_key.get(("wiki" if src == "wikipedia" else "iucr", key))
        base = "https://en.wikipedia.org/wiki/" if src == "wikipedia" else "https://dictionary.iucr.org/"
        if r is None or not r["exists"]:
            refs.append({"source": src, "title": key, "url": base + key.replace(" ", "_"),
                         "status": "unreachable", "fetched": FETCH_DATE,
                         "note": (r or {}).get("note", "not fetched"), "quote": ""})
            continue
        note = r.get("note") or ""
        page_title = r.get("page_title") or key
        candidates = list(r.get("candidates") or [])
        if r.get("exact"):
            candidates.append(r["exact"])
        quote = pick_sentence(candidates, needles_for(c["label"], c["aliases"]))
        prefix = r.get("prefix") or ""
        suffix = r.get("suffix") or ""
        if quote != (r.get("exact") or ""):
            # prefix/suffix were taken around `exact`, not around another sentence
            prefix, suffix = "", ""
        if is_anaphoric(quote):
            quote, prefix, suffix = "", "", ""
            note = (note + " " if note else "") + "anaphoric sentence withheld; it does not define the concept on its own"
        matched = page_matches(page_title, c["label"], c["aliases"])
        if src == "wikipedia" and not matched:
            status, quote = "off-topic", ""
            note = (note + " " if note else "") + "page title does not match the concept label or aliases; lede not quoted"
        elif src == "iucr" and not matched:
            status = "related"
            note = (note + " " if note else "") + "dictionary page is adjacent to this concept, not its definition"
        elif not quote:
            status = "related" if src == "iucr" else "off-topic"
        elif is_unreadable_markup(quote, r.get("raw") or None):
            status = "unreadable-markup"
            quote, prefix, suffix = "", "", ""
            note = (note + " " if note else "") + "stripped markup left the sentence unreadable; quote withheld"
        else:
            status = "quoted-unverified-markup" if "unverified" in note else "quoted"
        item = {"source": src, "title": key, "url": r.get("final_url") or r["url"],
                "page_title": page_title, "status": status,
                "quote": quote, "prefix": prefix, "suffix": suffix,
                "truncated": bool(r.get("truncated")), "note": note.strip(), "fetched": FETCH_DATE}
        # A content hash upgrades a verified sentence to how=quote. Markup that
        # is still unverified stays derived: the hash would not be of the sentence.
        if status == "quoted" and r.get("sha256"):
            item["sha256"] = r["sha256"]
        refs.append(item)
    for citation in c["refs"]:
        hit = literature_lookup(citation)
        refs.append({
            "source": "literature",
            "title": citation,
            "url": hit.get("url") or "",
            "page_title": hit.get("title") or citation,
            "status": "citation",
            "quote": "",
            "prefix": "",
            "suffix": "",
            "truncated": False,
            "note": "bibliographic pointer from the concept's literature list; no abstract sentence was fetched",
            "fetched": FETCH_DATE,
            "doi": hit.get("doi") or "",
        })
    payload["concepts"].append({
        "id": c["id"], "label": c["label"], "kind": c["kind"], "aliases": c["aliases"],
        "definition": c["definition"], "literature": c["refs"],
        "relations": [{"type": t, "target": g} for t, g in c["rels"]],
        "code_evidence": [{"module": a["module"], "symbol": a["symbol"], "kind": a["kind"],
                           "line": a["line"], "quote": a["exact"], "uri": a["uri"], "sha256": a["sha256"]}
                          for a in anch_by_concept.get(c["id"], [])],
        "references": refs,
    })
payload_bytes = json.dumps(payload, indent=1, ensure_ascii=False).encode() + b"\n"
(OUT / "agentsg_concepts.json").write_bytes(payload_bytes)
payload_ref = {"uri": "https://github.com/phzwart/agentsg/kg/agentsg_concepts.json", "sha256": sha(payload_bytes)}

# ------------------------------------------------------------------ grits ledger
plan_id = "plan:agentsg-concept-kg-2026-09-30"
graph = [{
    "@id": plan_id, "@type": "prov:Plan",
    "name": "agentsg concept knowledge graph: extract crystallographic and mathematical concepts from the agentsg source, anchor each to verbatim docstrings, and attach IUCr Online Dictionary and Wikipedia definitions",
    "prompt_digest": sha((HERE / "prompt.txt").read_bytes()),
    "schema_digest": sha(schema_bytes),
    "agent": AGENT,
}]
n_quote = n_ref = n_absent = n_rel = 0
for pc in payload["concepts"]:
    cid = pc["id"]
    used = []
    for i, a in enumerate(pc["code_evidence"]):
        eid = f"evi:code:{cid}:{i}"
        graph.append({"@id": eid, "@type": "prov:Entity", "plan": plan_id, "how": "quote", "agent": AGENT,
                      "source": {"uri": a["uri"], "sha256": a["sha256"], "media_type": "text/x-python"},
                      "target": {"hasSource": a["uri"],
                                 "selector": {"@type": "oa:TextQuoteSelector", "exact": a["quote"]}},
                      "summary": f"{a['module']}:{a['symbol'] or '<module docstring>'} line {a['line']}"})
        used.append(eid); n_quote += 1
    lit_n = 0
    for r in pc["references"]:
        if r["source"] == "literature":
            lit_n += 1
            eid = f"evi:lit:{cid}:{lit_n}"
            where = r.get("url") or "no DOI or URL confirmed"
            graph.append({"@id": eid, "@type": "prov:Entity", "plan": plan_id, "how": "derived", "agent": AGENT,
                          "rationale": "Bibliographic pointer copied from the concept's literature list. No abstract sentence was fetched, so this is not a quote.",
                          "summary": f"{r['title']} {where}".strip()})
            n_ref += 1
            used.append(eid)
            continue
        eid = f"evi:ref:{cid}:{r['source']}"
        if r.get("sha256") and r.get("quote") and r["status"] == "quoted":
            sel = {"@type": "oa:TextQuoteSelector", "exact": r["quote"]}
            if r.get("prefix"): sel["prefix"] = r["prefix"]
            if r.get("suffix"): sel["suffix"] = r["suffix"]
            graph.append({"@id": eid, "@type": "prov:Entity", "plan": plan_id, "how": "quote", "agent": AGENT,
                          "source": {"uri": r["url"], "sha256": r["sha256"], "media_type": "application/json"},
                          "target": {"hasSource": r["url"], "selector": sel},
                          "summary": f"{r['source']} definition of '{r.get('page_title', r['title'])}'"})
            n_ref += 1
        elif r["status"] in ("unreachable", "off-topic", "unreadable-markup") or not r.get("quote"):
            if r["status"] == "off-topic":
                why = "page title does not match this concept, so the lede was not quoted"
            elif r["status"] == "unreadable-markup":
                why = "markup stripping left the sentence unreadable, so it was not quoted"
            elif r["status"] == "citation":
                why = "bibliographic pointer only"
            else:
                why = "the page could not be retrieved, so existence and wording are undetermined"
            graph.append({"@id": eid, "@type": "prov:Entity", "plan": plan_id, "agent": AGENT,
                          "result": "inconclusive",
                          "summary": f"Looked for a {r['source']} page titled '{r['title']}' at {r['url']} on {FETCH_DATE}; {why}. {r.get('note','')}".strip()})
            n_absent += 1
        else:
            sel = {"@type": "oa:TextQuoteSelector", "exact": r["quote"]}
            if r.get("prefix"): sel["prefix"] = r["prefix"]
            if r.get("suffix"): sel["suffix"] = r["suffix"]
            adjacent = " Adjacent dictionary page, not a definition of this concept." if r["status"] == "related" else ""
            rationale = (f"Sentence transcribed on {FETCH_DATE} from the page as rendered by the fetch tool (HTML converted to text); the raw page bytes were not retained, so no content hash is available and the entity is 'derived' rather than 'quote'. Verify against the live URL with the selector.{adjacent}"
                         + (" Inline markup (math, sub/superscripts, bold) may have been stripped in transcription: " + r["note"] if r["status"] == "quoted-unverified-markup" else "")
                         + (" The page's first sentence was truncated at a clause boundary." if r.get("truncated") else ""))
            graph.append({"@id": eid, "@type": "prov:Entity", "plan": plan_id, "how": "derived", "agent": AGENT,
                          "rationale": rationale,
                          "target": {"hasSource": r["url"], "selector": sel},
                          "summary": f"{r['source']} definition of '{r.get('page_title', r['title'])}'"})
            n_ref += 1
        used.append(eid)
    cent = f"ent:concept:{cid}"
    generated = [cent]
    graph.append({"@id": cent, "@type": "prov:Entity", "plan": plan_id, "how": "derived", "agent": AGENT,
                  "rationale": "Definition paraphrased by the builder from the anchored agentsg docstrings and the external dictionary sentences listed in the derivation's 'used'; the wording is the builder's, not a quotation.",
                  "summary": f"{pc['label']}: {pc['definition']}",
                  "plan_variable": f"pplan:vars/concept/{cid}",
                  "payload": PAYLOAD_SCHEMA_URL, "payload_ref": payload_ref})
    for rel in pc["relations"]:
        rid = f"ent:rel:{cid}:{rel['type']}:{rel['target']}"
        graph.append({"@id": rid, "@type": "prov:Entity", "plan": plan_id, "how": "inferred", "agent": AGENT,
                      "rationale": "Relation asserted by the builder from how the anchored code uses or defines the two concepts together; not stated verbatim in any single source.",
                      "summary": f"{cid} {rel['type']} {rel['target']}",
                      "payload": PAYLOAD_SCHEMA_URL, "payload_ref": payload_ref})
        generated.append(rid); n_rel += 1
    graph.append({"@id": f"act:derive:{cid}", "@type": "prov:Activity", "plan": plan_id, "kind": "derivation",
                  "performed_by": AGENT, "plan_step": "pplan:steps/derive_concept",
                  "used": used, "generated": generated,
                  "started_at": f"{FETCH_DATE}T00:00:00Z", "ended_at": f"{FETCH_DATE}T23:59:59Z"})

def _code_ids(concept_id: str) -> list[str]:
    for pc in payload["concepts"]:
        if pc["id"] == concept_id:
            return [f"evi:code:{concept_id}:{i}" for i in range(len(pc["code_evidence"]))]
    return []

for step, depends in (
    ("conventional", "conventional_cell"),
    ("primitive", "primitive_cell"),
    ("selling", "delaunay_selling_reduction"),
    ("root", "kurlin_root_form"),
    ("kdtree", "kd_tree"),
):
    used = _code_ids(depends)[:1]
    if not used or "ent:concept:pdb_lattice_search" not in {n["@id"] for n in graph}:
        continue
    graph.append({"@id": f"act:pdb:{step}", "@type": "prov:Activity", "plan": plan_id, "kind": "derivation",
                  "performed_by": AGENT, "plan_step": f"pplan:steps/pdb_search/{step}",
                  "used": used, "generated": ["ent:concept:pdb_lattice_search"],
                  "started_at": f"{FETCH_DATE}T00:00:00Z", "ended_at": f"{FETCH_DATE}T23:59:59Z"})

ledger = {"@context": "https://phzwart.github.io/pygrits/context.jsonld", "@graph": graph}
bundle = pygrits.load(ledger)
pygrits.validate(bundle)
pygrits.dump(bundle, OUT / "agentsg_kg.grits.jsonld")
print(f"ledger ok: {len(graph)} nodes ({n_quote} code quotes, {n_ref} reference quotes, {n_absent} unreachable refs, {n_rel} relations, {len(payload['concepts'])} concepts)")

# ------------------------------------------------------------------ Neo4j CSV
def w(name, rows, fields):
    with open(OUT / "neo4j" / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        wr.writeheader()
        for r in rows: wr.writerow({k: r.get(k, "") for k in fields})

w("concepts.csv", [{"id:ID(Concept)": c["id"], "label": c["label"], "kind": c["kind"],
                    "definition": c["definition"], "aliases": ";".join(c["aliases"]),
                    "literature": ";".join(c["literature"]), ":LABEL": "Concept"} for c in payload["concepts"]],
  ["id:ID(Concept)", "label", "kind", "definition", "aliases", "literature", ":LABEL"])
w("concept_relations.csv", [{":START_ID(Concept)": c["id"], ":END_ID(Concept)": r["target"], ":TYPE": r["type"],
                             "receipt": f"ent:rel:{c['id']}:{r['type']}:{r['target']}"}
                            for c in payload["concepts"] for r in c["relations"]],
  [":START_ID(Concept)", ":END_ID(Concept)", ":TYPE", "receipt"])
code_rows, code_edges = [], []
for c in payload["concepts"]:
    for i, a in enumerate(c["code_evidence"]):
        eid = f"evi:code:{c['id']}:{i}"
        code_rows.append({"id:ID(Evidence)": eid, "module": a["module"], "symbol": a["symbol"], "symbol_kind": a["kind"],
                          "line:int": a["line"], "quote": a["quote"], "uri": a["uri"], "sha256": a["sha256"], ":LABEL": "CodeEvidence"})
        code_edges.append({":START_ID(Concept)": c["id"], ":END_ID(Evidence)": eid, ":TYPE": "HAS_CODE_EVIDENCE"})
w("code_evidence.csv", code_rows, ["id:ID(Evidence)", "module", "symbol", "symbol_kind", "line:int", "quote", "uri", "sha256", ":LABEL"])
w("code_evidence_edges.csv", code_edges, [":START_ID(Concept)", ":END_ID(Evidence)", ":TYPE"])
ref_rows, ref_edges, seen = [], [], set()
_LABEL = {"iucr": "IUCr", "wikipedia": "Wikipedia", "literature": "Literature"}
lit_count: dict[str, int] = {}
for c in payload["concepts"]:
    for r in c["references"]:
        if r["source"] == "literature":
            lit_count[c["id"]] = lit_count.get(c["id"], 0) + 1
            receipt = f"evi:lit:{c['id']}:{lit_count[c['id']]}"
            slug = re.sub(r"[^a-z0-9]+", "", r["title"].lower())[:48]
            rid = f"ref:literature:{slug}"
        else:
            receipt = f"evi:ref:{c['id']}:{r['source']}"
            rid = f"ref:{r['source']}:{r['title']}"
        if rid not in seen:
            seen.add(rid)
            ref_rows.append({"id:ID(Reference)": rid, "source": r["source"], "title": r["title"],
                             "page_title": r.get("page_title", ""), "url": r["url"], "status": r["status"],
                             "quote": r.get("quote", ""), "prefix": r.get("prefix", ""), "suffix": r.get("suffix", ""),
                             "truncated:boolean": str(bool(r.get("truncated"))).lower(), "note": r.get("note", ""),
                             "fetched": r["fetched"], "doi": r.get("doi", ""),
                             ":LABEL": "Reference;" + _LABEL.get(r["source"], "Reference")})
        ref_edges.append({":START_ID(Concept)": c["id"], ":END_ID(Reference)": rid, ":TYPE": "DEFINED_IN",
                          "receipt": receipt})
w("references.csv", ref_rows, ["id:ID(Reference)", "source", "title", "page_title", "url", "status", "quote", "prefix", "suffix", "truncated:boolean", "note", "fetched", "doi", ":LABEL"])
w("reference_edges.csv", ref_edges, [":START_ID(Concept)", ":END_ID(Reference)", ":TYPE", "receipt"])

(OUT / "neo4j" / "load.cypher").write_text('''// agentsg concept knowledge graph -- Neo4j 5 loader (LOAD CSV; copy the CSVs into the import/ directory)
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:CodeEvidence) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT reference_id IF NOT EXISTS FOR (r:Reference) REQUIRE r.id IS UNIQUE;
CREATE FULLTEXT INDEX concept_text IF NOT EXISTS FOR (c:Concept) ON EACH [c.label, c.definition, c.aliases];
CREATE FULLTEXT INDEX reference_text IF NOT EXISTS FOR (r:Reference) ON EACH [r.title, r.quote];

LOAD CSV WITH HEADERS FROM 'file:///concepts.csv' AS row
MERGE (c:Concept {id: row.`id:ID(Concept)`})
SET c.label = row.label, c.kind = row.kind, c.definition = row.definition,
    c.aliases = [a IN split(row.aliases, ';') WHERE a <> ''],
    c.literature = [l IN split(row.literature, ';') WHERE l <> ''];

LOAD CSV WITH HEADERS FROM 'file:///code_evidence.csv' AS row
MERGE (e:CodeEvidence {id: row.`id:ID(Evidence)`})
SET e.module = row.module, e.symbol = row.symbol, e.symbol_kind = row.symbol_kind,
    e.line = toInteger(row.`line:int`), e.quote = row.quote, e.uri = row.uri, e.sha256 = row.sha256;

LOAD CSV WITH HEADERS FROM 'file:///code_evidence_edges.csv' AS row
MATCH (c:Concept {id: row.`:START_ID(Concept)`}), (e:CodeEvidence {id: row.`:END_ID(Evidence)`})
MERGE (c)-[:HAS_CODE_EVIDENCE]->(e);

LOAD CSV WITH HEADERS FROM 'file:///references.csv' AS row
MERGE (r:Reference {id: row.`id:ID(Reference)`})
SET r.source = row.source, r.title = row.title, r.page_title = row.page_title, r.url = row.url,
    r.status = row.status, r.quote = row.quote, r.prefix = row.prefix, r.suffix = row.suffix,
    r.truncated = toBoolean(row.`truncated:boolean`), r.note = row.note, r.fetched = row.fetched,
    r.doi = row.doi
WITH r, row CALL { WITH r, row WITH r, row WHERE row.source = 'iucr' SET r:IUCr }
WITH r, row CALL { WITH r, row WITH r, row WHERE row.source = 'wikipedia' SET r:Wikipedia }
WITH r, row CALL { WITH r, row WITH r, row WHERE row.source = 'literature' SET r:Literature }
RETURN count(r);

LOAD CSV WITH HEADERS FROM 'file:///reference_edges.csv' AS row
MATCH (c:Concept {id: row.`:START_ID(Concept)`}), (r:Reference {id: row.`:END_ID(Reference)`})
MERGE (c)-[d:DEFINED_IN]->(r) SET d.receipt = row.receipt;

// concept-to-concept relations: one relationship type per CSV :TYPE value
LOAD CSV WITH HEADERS FROM 'file:///concept_relations.csv' AS row
MATCH (a:Concept {id: row.`:START_ID(Concept)`}), (b:Concept {id: row.`:END_ID(Concept)`})
CALL apoc.merge.relationship(a, row.`:TYPE`, {}, {receipt: row.receipt}, b, {}) YIELD rel
RETURN count(rel);
// Without APOC, replace the last statement by one MERGE per type, e.g.
// MATCH ... WHERE row.`:TYPE` = 'USES' MERGE (a)-[:USES {receipt: row.receipt}]->(b);
''')
(OUT / "neo4j" / "queries.cypher").write_text('''// Competency queries for the chatbot layer
// 1. Where is a concept implemented?
MATCH (c:Concept {id: 'smith_normal_form'})-[:HAS_CODE_EVIDENCE]->(e) RETURN e.module, e.symbol, e.line, e.quote;
// 2. What does a routine rest on? (concepts reachable through USES from a starting concept)
MATCH p = (c:Concept {id: 'reflection_conditions'})-[:USES*1..3]->(d) RETURN DISTINCT d.id, length(p) AS depth ORDER BY depth;
// 3. Definition with receipts: paraphrase plus the IUCr / Wikipedia sentences
MATCH (c:Concept {id: 'allowed_origins'}) OPTIONAL MATCH (c)-[:DEFINED_IN]->(r) RETURN c.label, c.definition, collect({src: r.source, quote: r.quote, url: r.url, status: r.status});
// 4. Which concepts have no reachable external definition? (gaps to fill by hand)
MATCH (c:Concept) WHERE NOT (c)-[:DEFINED_IN]->(:Reference {status: 'quoted'}) RETURN c.id, c.label;
// 5. Free-text entry point for a chatbot
CALL db.index.fulltext.queryNodes('concept_text', 'origin shift semi-invariant') YIELD node, score RETURN node.id, node.label, score LIMIT 5;
// 6. Dual pairs and contrasts (good for explanation prompts)
MATCH (a)-[r:DUAL_OF|CONTRASTS_WITH]->(b) RETURN a.label, type(r), b.label;
// 7. Everything a module touches
MATCH (c:Concept)-[:HAS_CODE_EVIDENCE]->(e {module: 'agentsg/semi_invariants.py'}) RETURN DISTINCT c.id, c.label;
''')
n_concepts = len(payload["concepts"])
n_relations = sum(len(c["relations"]) for c in payload["concepts"])
n_anchors = len(code_rows)
n_references = len(ref_rows)
n_unverified = sum(1 for c in payload["concepts"] for r in c["references"] if r["status"] == "quoted-unverified-markup")
n_unreachable = sum(1 for c in payload["concepts"] for r in c["references"] if r["status"] == "unreachable")
(OUT / "README.md").write_text(
    "# agentsg concept knowledge graph\n\n"
    "Generated by `python build.py`. Do not edit files in this directory by hand.\n\n"
    f"- concepts: {n_concepts}\n"
    f"- relations: {n_relations}\n"
    f"- anchors: {n_anchors}\n"
    f"- references: {n_references}\n"
    f"- unverified: {n_unverified}\n"
    f"- unreachable: {n_unreachable}\n"
    f"- snapshot: {SNAPSHOT['commit']} dirty={str(SNAPSHOT['dirty']).lower()}\n"
)
print("neo4j:", n_concepts, "concepts,", n_anchors, "code evidence,", n_references, "references,", n_relations, "relations")
print("snapshot:", SNAPSHOT["commit"], "dirty=" + str(SNAPSHOT["dirty"]).lower())
lingering = [
    f"{c['id']} [{ref['source']}] {ref.get('quote','')}"
    for c in payload["concepts"] for ref in c["references"]
    if ref.get("quote") and is_anaphoric(ref["quote"])
]
print("anaphoric quotes remaining:", len(lingering))
for row in lingering:
    print("  -", row)
