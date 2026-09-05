"""HTTP API for PDB lattice-similarity search.

Serves radius queries against a prebuilt DuckDB file (``pdb_cells.duckdb``)
using the Kurlin root invariant. Requires DuckDB (``pip install agentsg[db]``).

Computation pipeline (query and stored PDB rows use the same path)
-----------------------------------------------------------------
1. **Space group** — the Hermann-Mauguin symbol supplies the Bravais
   centring letter (P, A, B, C, I, F, R/H). This is required: without it a
   centred conventional cell would be Selling-reduced as if it were primitive,
   giving the wrong lattice and wrong root form.
2. **Primitive cell** — the conventional cell is reduced to the primitive
   lattice via the ITA Table 5.1.3.1 transforms (:mod:`agentsg.cell.primitive`).
3. **Selling / Delaunay reduction** — the primitive cell is reduced to an
   obtuse superbase (:func:`agentsg.cell.rootform.delaunay_superbase`).
4. **Root form** — conorms ``p_ij = -v_i·v_j`` → root products ``√p_ij`` →
   Kurlin root invariant (:func:`agentsg.cell.rootform.root_invariant`).
5. **Search** — Euclidean distance on the six root components (Å); a cKDTree
   index over precomputed database roots returns all PDB ids within ``cutoff``.

Command-line usage::

    python -m agentsg.cell.pdb_server --db data/pdb_cells.duckdb --port 8765

Endpoints
---------
GET /health
    Liveness check.

GET /search
    Query parameters: ``a,b,c,alpha,beta,gamma`` (required), ``sg`` (required:
    space-group number or Hermann-Mauguin symbol, for centring → primitive),
    ``cutoff`` (Å, required), and optional ``same_hm`` / ``same_sg``
    (``true``/``false``; when true, restrict hits to the same Hermann-Mauguin
    setting — not merely the same IT number).

POST /search
    JSON body with the same fields::

        {"cell": [79, 79, 38, 90, 90, 90], "sg": "P 21 21 21", "cutoff": 1.0}

Response (both methods)::

    {
      "cell": [79.0, 79.0, 38.0, 90.0, 90.0, 90.0],
      "sg_number": 96,
      "sg_hm": "P 43 21 2",
      "centering": "P",
      "primitive_cell": [79.0, 79.0, 38.0, 90.0, 90.0, 90.0],
      "cutoff": 1.0,
      "same_hm": true,
      "pipeline": {
        "centering_to_primitive": "ITA Table 5.1.3.1",
        "reduction": "selling_delaunay",
        "invariant": "kurlin_root"
      },
      "count": 3,
      "hits": [
        {"pdb_id": "1ABC", "distance": 0.12, "sg_number": 96, "sg_hm": "P 43 21 2"},
        ...
      ]
    }

Distances are root-product units (Å).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..space_groups import space_group
from .celldb import CellDatabase, RootIndex
from .primitive import lattice_letter, primitive_cell

_PIPELINE = {
    "centering_to_primitive": "ITA Table 5.1.3.1",
    "reduction": "selling_delaunay",
    "invariant": "kurlin_root",
}


def _parse_cell(raw) -> tuple[float, ...]:
    """Validate and convert raw cell tuple/list into 6 floats."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError("cell must be a list of six numbers (a,b,c,alpha,beta,gamma)")
    return tuple(float(x) for x in raw)


def _resolve_space_group(sg) -> tuple[int, str]:
    """Resolve space group number or Hermann-Mauguin string to (number, hm_string)."""
    if sg is None:
        raise ValueError("sg is required (space-group number or Hermann-Mauguin symbol)")
    key = int(sg) if isinstance(sg, str) and sg.strip().isdigit() else sg
    rec = space_group(key)
    return rec.number, rec.hermann_mauguin


def search_compatible(
    db: CellDatabase,
    index: RootIndex,
    *,
    cell: tuple[float, ...],
    cutoff: float,
    sg_number: int,
    sg_hm: str,
    same_hm: bool = False,
    same_sg: bool | None = None,
) -> dict[str, Any]:
    """Return PDB ids within ``cutoff`` Å (root distance) of ``cell``.

    The query cell is reduced to primitive using ``sg_hm`` (centering letter),
    then Selling/Delaunay-reduced to the Kurlin root invariant before distance
    is computed — matching how roots were stored at database build time.

    When ``same_hm`` is true (``same_sg`` is accepted as an alias), candidates
    are restricted to the same Hermann-Mauguin setting string, not the IT
    number — so ``C 1 2 1`` does not pull in ``I 1 2 1`` (both number 5).
    """
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if not sg_hm:
        raise ValueError("sg is required (space-group number or Hermann-Mauguin symbol)")
    if same_sg is not None:
        same_hm = bool(same_sg)
    prim = primitive_cell(cell, sg_hm)
    if same_hm:
        hits = db.within(cell, cutoff, match_sg_hm=sg_hm, sg_hm=sg_hm)
    else:
        hits = index.within(cell, cutoff, sg_hm=sg_hm)
    meta = db.lookup_cells([pid for pid, _ in hits])
    enriched = []
    for pid, dist in hits:
        rec = {"pdb_id": pid, "distance": dist}
        info = meta.get(pid)
        if info is not None:
            rec["sg_number"] = info["sg_number"]
            rec["sg_hm"] = info["sg_hm"]
            rec["cell"] = info["cell"]
        enriched.append(rec)
    return {
        "cell": list(cell),
        "sg_number": sg_number,
        "sg_hm": sg_hm,
        "centering": lattice_letter(sg_hm),
        "primitive_cell": list(prim),
        "cutoff": cutoff,
        "same_hm": same_hm,
        "pipeline": _PIPELINE,
        "count": len(enriched),
        "hits": enriched,
    }


def _parse_search_params(data: dict[str, Any]) -> dict[str, Any]:
    """Parse query-string or JSON search parameters into canonical keyword arguments."""
    if "cell" in data:
        cell = _parse_cell(data["cell"])
    else:
        try:
            cell = _parse_cell([
                data["a"], data["b"], data["c"],
                data["alpha"], data["beta"], data["gamma"],
            ])
        except KeyError as exc:
            raise ValueError(
                "provide cell=[a,b,c,alpha,beta,gamma] or a,b,c,alpha,beta,gamma"
            ) from exc
    cutoff = float(data["cutoff"])
    sg_number = None
    sg_hm = None
    if "sg" in data and data["sg"] is not None:
        sg_number, sg_hm = _resolve_space_group(data["sg"])
    elif "sg_number" in data and data["sg_number"] is not None:
        sg_number, sg_hm = _resolve_space_group(data["sg_number"])
    elif "sg_hm" in data and data["sg_hm"] is not None:
        sg_number, sg_hm = _resolve_space_group(data["sg_hm"])
    else:
        raise ValueError(
            "sg is required (space-group number or Hermann-Mauguin symbol); "
            "needed to resolve centring type before Selling reduction to root form"
        )
    # same_hm is the preferred name; same_sg kept as a backward-compatible alias
    same_raw = data.get("same_hm", data.get("same_sg", "false"))
    same_hm = str(same_raw).lower() in ("1", "true", "yes")
    return dict(
        cell=cell,
        cutoff=cutoff,
        sg_number=sg_number,
        sg_hm=sg_hm,
        same_hm=same_hm,
    )


class _PdbSearchServer:
    """Shared state for request handlers."""

    def __init__(self, db_path: str):
        self.db = CellDatabase(db_path)
        self.index = self.db.build_index()
        self.db_path = db_path
        self.n_cells = len(self.db)


def make_handler(state: _PdbSearchServer):
    """Construct an HTTP request handler closure bound to a search server state."""
    class Handler(BaseHTTPRequestHandler):
        """HTTP request handler for PDB cell search service."""
        server_version = "agentsg-pdb-server/0.1"

        def log_message(self, fmt, *args):
            """Log formatted message to standard error."""
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, status: int, payload: dict[str, Any]):
            """Serialize dictionary payload as JSON and write HTTP response."""
            body = json.dumps(payload, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            """Read and deserialize JSON body from incoming HTTP request."""
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("POST body required")
            return json.loads(self.rfile.read(length).decode())

        def do_GET(self):
            """Handle GET requests for /health and /search."""
            path = urlparse(self.path).path
            if path == "/health":
                self._json(200, {
                    "status": "ok",
                    "db": state.db_path,
                    "cells": state.n_cells,
                    "index_size": len(state.index),
                })
                return
            if path != "/search":
                self._json(404, {"error": "not found", "paths": ["/health", "/search"]})
                return
            qs = parse_qs(urlparse(self.path).query, keep_blank_values=False)
            flat = {k: v[0] for k, v in qs.items()}
            try:
                params = _parse_search_params(flat)
                result = search_compatible(state.db, state.index, **params)
                self._json(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

        def do_POST(self):
            """Handle POST requests for /search."""
            path = urlparse(self.path).path
            if path != "/search":
                self._json(404, {"error": "not found", "paths": ["/health", "/search"]})
                return
            try:
                data = self._read_json()
                params = _parse_search_params(data)
                result = search_compatible(state.db, state.index, **params)
                self._json(200, result)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                self._json(500, {"error": str(exc)})

    return Handler


def run_server(db_path: str, host: str = "127.0.0.1", port: int = 8765):
    """Load the database, build the root index, and serve HTTP requests."""
    state = _PdbSearchServer(db_path)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"agentsg PDB search server on http://{host}:{port}")
    print(f"  database: {db_path} ({state.n_cells} cells)")
    print("  GET  /health")
    print("  GET  /search?a=79&b=79&c=38&alpha=90&beta=90&gamma=90&sg=P212121&cutoff=1.0&same_hm=true")
    print('  POST /search  {"cell":[...],"sg":"P 21 21 21","cutoff":1.0,"same_hm":true}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        state.db.close()
        httpd.server_close()


def _cli(argv=None):
    """Command-line entry point for running the HTTP PDB search server."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="pdb_cells.duckdb",
                   help="path to pdb_cells.duckdb (default: pdb_cells.duckdb)")
    p.add_argument("--host", default="127.0.0.1", help="bind address")
    p.add_argument("--port", type=int, default=8765, help="listen port")
    args = p.parse_args(argv)
    run_server(args.db, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
