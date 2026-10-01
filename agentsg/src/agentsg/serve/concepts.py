"""HTTP adapters over :class:`agentsg.kg.graph.ConceptGraph`."""
from __future__ import annotations

import threading
from typing import Any

from ..kg.graph import ConceptGraph, GraphError
from .http import HttpError

_LOCK = threading.Lock()
_GRAPH: ConceptGraph | None = None


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


def _call(fn, *args):
    try:
        return fn(*args)
    except GraphError as exc:
        raise HttpError(exc.status, str(exc), exc.extra or None) from exc


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


def concept_info(data: dict[str, Any]) -> dict[str, Any]:
    """Search when ``q`` is set; return the card when ``id`` is set."""
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    query = str(data.get("q") or "").strip()
    if ident:
        return _call(graph.card, ident)
    if query:
        limit = _as_int(data.get("limit"), 8, "limit", 1, 20)
        return _call(graph.search, query, limit)
    raise HttpError(400, "pass id (a concept id) or q (words to search)")


def concept_search(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    query = str(data.get("q") or "").strip()
    if not query:
        raise HttpError(400, "q is required")
    limit = _as_int(data.get("limit"), 8, "limit", 1, 20)
    return _call(graph.search, query, limit)


def concept_neighbors(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    depth = _as_int(data.get("depth"), 1, "depth", 1, 3)
    relation = str(data.get("relation") or "").strip() or None
    return _call(graph.neighbors, ident, relation, depth)


def concept_uses(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    depth = _as_int(data.get("depth"), 3, "depth", 1, 3)
    return _call(graph.uses, ident, depth)


def concept_used_by(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    depth = _as_int(data.get("depth"), 3, "depth", 1, 3)
    return _call(graph.used_by, ident, depth)


def concept_module(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    name = str(data.get("module") or "").strip()
    if not name:
        raise HttpError(400, "module is required")
    return _call(graph.module, name)


def concept_receipt(data: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_or_503()
    ident = str(data.get("id") or data.get("receipt") or "").strip()
    if not ident:
        raise HttpError(400, "id is required")
    if not graph.nodes:
        raise HttpError(503, "concept ledger agentsg_kg.grits.jsonld is not loaded")
    return _call(graph.receipt, ident)
