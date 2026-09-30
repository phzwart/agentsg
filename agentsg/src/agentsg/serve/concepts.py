"""In-memory concept graph: search, cards, USES walks, module anchors, receipts.

The payload is ``concept_kg/agentsg_concepts.json``. Receipt ids point at nodes
in ``concept_kg/agentsg_kg.grits.jsonld``. Code hashes are that snapshot.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .http import HttpError

_TOKEN = re.compile(r"[a-z0-9]+")
_LOCK = threading.Lock()
_GRAPH: ConceptGraph | None = None


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _norm_id(text: str) -> str:
    return "_".join(_tokens(text))


def _soft_in(token: str, bag: set[str]) -> bool:
    if token in bag:
        return True
    if len(token) < 4:
        return False
    for word in bag:
        if len(word) < 4:
            continue
        if word.startswith(token) or token.startswith(word):
            return True
    return False


def _as_int(value: Any, default: int, name: str, lo: int, hi: int) -> int:
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise HttpError(400, f"{name} must be an integer") from exc
    if number < lo or number > hi:
        raise HttpError(400, f"{name} must be between {lo} and {hi}")
    return number


def kg_dir() -> Path:
    """Directory that holds the payload and the ledger."""
    env = os.environ.get("AGENTSG_KG", "").strip()
    if env:
        path = Path(env)
        if path.is_file():
            path = path.parent
        if (path / "agentsg_concepts.json").is_file():
            return path
        raise FileNotFoundError(f"AGENTSG_KG has no agentsg_concepts.json: {path}")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "concept_kg" / "agentsg_concepts.json"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError("concept_kg/agentsg_concepts.json not found")


class ConceptGraph:
    """Payload plus an id index of the receipt ledger."""

    def __init__(self, payload: dict[str, Any], ledger: dict[str, Any] | None):
        concepts = payload.get("concepts") or []
        self.payload = payload
        self.concepts: list[dict[str, Any]] = concepts
        self.by_id: dict[str, dict[str, Any]] = {c["id"]: c for c in concepts}
        raw_snapshot = payload.get("snapshot")
        if isinstance(raw_snapshot, dict):
            self.snapshot = raw_snapshot
        else:
            self.snapshot = {
                "repository": raw_snapshot or payload.get("source_repository") or "",
                "commit": None,
                "dirty": None,
            }
        self.generated = payload.get("generated") or ""
        self.nodes: dict[str, dict[str, Any]] = {}
        if ledger:
            for node in ledger.get("@graph") or []:
                ident = node.get("@id")
                if ident:
                    self.nodes[ident] = node

    @classmethod
    def load(cls, directory: Path | None = None) -> ConceptGraph:
        root = directory or kg_dir()
        payload_path = root / "agentsg_concepts.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        ledger_path = root / "agentsg_kg.grits.jsonld"
        ledger = None
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        return cls(payload, ledger)

    @property
    def n(self) -> int:
        return len(self.concepts)

    def _meta(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot,
            "generated": self.generated,
            "n_concepts": self.n,
        }

    def _bags(self, concept: dict[str, Any]) -> tuple[set[str], set[str]]:
        label_bag = set(_tokens(concept["id"].replace("_", " ")))
        label_bag |= set(_tokens(concept.get("label") or ""))
        for alias in concept.get("aliases") or []:
            label_bag |= set(_tokens(alias))
        definition = set(_tokens(concept.get("definition") or ""))
        return label_bag, definition

    def search(self, query: str, limit: int = 8) -> dict[str, Any]:
        qtokens = _tokens(query)
        if not qtokens:
            raise HttpError(400, "q is empty")
        qnorm = " ".join(qtokens)
        qid = "_".join(qtokens)
        hits: list[dict[str, Any]] = []
        for concept in self.concepts:
            cid = concept["id"]
            label_norm = " ".join(_tokens(concept.get("label") or ""))
            aliases = [" ".join(_tokens(a)) for a in concept.get("aliases") or []]
            if qid == cid or qnorm == cid.replace("_", " "):
                score = 100
            elif qnorm == label_norm:
                score = 90
            elif qnorm in aliases:
                score = 85
            else:
                label_bag, definition = self._bags(concept)
                label_hits = sum(1 for token in qtokens if _soft_in(token, label_bag))
                def_hits = sum(1 for token in qtokens if _soft_in(token, definition))
                if label_hits == 0 and def_hits == 0:
                    continue
                score = 10 * label_hits + 2 * def_hits
                if label_hits == len(qtokens):
                    score += 20
            hits.append({
                "id": cid,
                "label": concept.get("label") or cid,
                "kind": concept.get("kind"),
                "score": score,
                "definition": concept.get("definition") or "",
            })
        hits.sort(key=lambda row: (-row["score"], row["id"]))
        return {
            **self._meta(),
            "query": query,
            "hits": hits[:limit],
        }

    def _evidence(self, cid: str, concept: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for index, anchor in enumerate(concept.get("code_evidence") or []):
            rows.append({
                "module": anchor.get("module") or "",
                "symbol": anchor.get("symbol") or "",
                "kind": anchor.get("kind") or "",
                "line": anchor.get("line"),
                "quote": anchor.get("quote") or "",
                "uri": anchor.get("uri") or "",
                "sha256": anchor.get("sha256") or "",
                "receipt": f"evi:code:{cid}:{index}",
            })
        return rows

    def _references(self, cid: str, concept: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for ref in concept.get("references") or []:
            source = ref.get("source") or ""
            rows.append({
                "source": source,
                "title": ref.get("title") or "",
                "url": ref.get("url") or "",
                "status": ref.get("status") or "",
                "quote": ref.get("quote") or "",
                "receipt": f"evi:ref:{cid}:{source}",
            })
        return rows

    def _relations(self, cid: str, concept: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for rel in concept.get("relations") or []:
            target = rel.get("target") or ""
            kind = rel.get("type") or ""
            other = self.by_id.get(target)
            rows.append({
                "type": kind,
                "target": target,
                "target_label": (other or {}).get("label") or "",
                "receipt": f"ent:rel:{cid}:{kind}:{target}",
            })
        return rows

    def card(self, ident: str) -> dict[str, Any]:
        cid = ident.strip()
        concept = self.by_id.get(cid) or self.by_id.get(_norm_id(cid))
        if concept is None:
            raise HttpError(
                404,
                f"unknown concept id {ident!r}",
                {"hint": "GET /v1/concept?q=... to search labels and aliases"},
            )
        cid = concept["id"]
        return {
            **self._meta(),
            "id": cid,
            "label": concept.get("label") or cid,
            "kind": concept.get("kind"),
            "aliases": list(concept.get("aliases") or []),
            "definition": concept.get("definition") or "",
            "definition_receipt": f"ent:concept:{cid}",
            "literature": list(concept.get("literature") or []),
            "relations": self._relations(cid, concept),
            "code_evidence": self._evidence(cid, concept),
            "references": self._references(cid, concept),
            "note": (
                "definition is a builder paraphrase (definition_receipt). "
                "references[].status quoted is a transcribed sentence; "
                "quoted-unverified-markup may have lost inline math; "
                "unreachable means no sentence was retrieved. "
                "code_evidence sha256 values are the snapshot, not the live tree."
            ),
        }

    def uses(self, ident: str, depth: int) -> dict[str, Any]:
        concept = self.by_id.get(ident.strip()) or self.by_id.get(_norm_id(ident))
        if concept is None:
            raise HttpError(404, f"unknown concept id {ident!r}")
        cid = concept["id"]
        seen: dict[str, int] = {}
        frontier = [cid]
        for step in range(1, depth + 1):
            nxt: list[str] = []
            for node in frontier:
                for rel in self.by_id[node].get("relations") or []:
                    if rel.get("type") != "USES":
                        continue
                    target = rel.get("target") or ""
                    if target == cid or target in seen or target not in self.by_id:
                        continue
                    seen[target] = step
                    nxt.append(target)
            frontier = nxt
        rows = [
            {
                "id": target,
                "label": self.by_id[target].get("label") or target,
                "depth": seen[target],
                "definition": self.by_id[target].get("definition") or "",
            }
            for target in seen
        ]
        rows.sort(key=lambda row: (row["depth"], row["id"]))
        return {
            **self._meta(),
            "id": cid,
            "label": concept.get("label") or cid,
            "depth": depth,
            "uses": rows,
        }

    def module(self, name: str) -> dict[str, Any]:
        query = name.strip().replace("\\", "/").lstrip("./")
        if not query:
            raise HttpError(400, "module is required")
        found: list[dict[str, Any]] = []
        for concept in self.concepts:
            anchors = [
                anchor for anchor in self._evidence(concept["id"], concept)
                if _module_match(anchor["module"], query)
            ]
            if not anchors:
                continue
            found.append({
                "id": concept["id"],
                "label": concept.get("label") or concept["id"],
                "anchors": anchors,
            })
        if not found:
            raise HttpError(
                404,
                f"no concept is anchored in {query!r}",
                {"hint": "use a path like agentsg/semi_invariants.py or a bare filename"},
            )
        found.sort(key=lambda row: row["id"])
        return {**self._meta(), "module": query, "concepts": found}

    def receipt(self, ident: str) -> dict[str, Any]:
        key = ident.strip()
        if not key:
            raise HttpError(400, "id is required")
        node = self.nodes.get(key)
        if node is None:
            raise HttpError(
                404,
                f"unknown receipt {key!r}",
                {"hint": "pass definition_receipt, relations[].receipt, code_evidence[].receipt, or references[].receipt"},
            )
        return {
            **self._meta(),
            "receipt": key,
            "node": node,
            "note": (
                "Ledger node. how=quote is a verbatim substring of the file at "
                "source.sha256. how=derived is a builder paraphrase or a "
                "transcribed sentence without a content hash. how=inferred is "
                "a builder relation."
            ),
        }


def _module_match(stored: str, query: str) -> bool:
    path = stored.replace("\\", "/").lstrip("./")
    if path == query or path.endswith("/" + query):
        return True
    if "/" not in query and path.rsplit("/", 1)[-1] == query:
        return True
    return False


def get_graph() -> ConceptGraph:
    """Load the graph once. Missing files raise FileNotFoundError."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    with _LOCK:
        if _GRAPH is None:
            _GRAPH = ConceptGraph.load()
    return _GRAPH


def concept_count() -> int | None:
    try:
        return get_graph().n
    except FileNotFoundError:
        return None


def _graph_or_503() -> ConceptGraph:
    try:
        return get_graph()
    except FileNotFoundError as exc:
        raise HttpError(503, str(exc)) from exc


def concept_info(data: dict[str, Any]) -> dict[str, Any]:
    """Search when ``q`` is set; return the card when ``id`` is set."""
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    query = str(data.get("q") or "").strip()
    if ident:
        return graph.card(ident)
    if query:
        limit = _as_int(data.get("limit"), 8, "limit", 1, 20)
        return graph.search(query, limit)
    raise HttpError(400, "pass id (a concept id) or q (words to search)")


def concept_uses(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    depth = _as_int(data.get("depth"), 3, "depth", 1, 3)
    return graph.uses(ident, depth)


def concept_module(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    name = str(data.get("module") or "").strip()
    if not name:
        raise HttpError(400, "module is required")
    return graph.module(name)


def concept_receipt(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or data.get("receipt") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    if not graph.nodes:
        raise HttpError(503, "concept ledger agentsg_kg.grits.jsonld is not loaded")
    return graph.receipt(ident)
