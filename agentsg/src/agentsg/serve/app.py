"""Threading HTTP server for the Muse symmetry API."""
from __future__ import annotations

import argparse
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import handlers
from .accesslog import LOGGED_HEADER, log_exchange, log_path
from .concepts import (
    concept_count,
    concept_info,
    concept_module,
    concept_neighbors,
    concept_receipt,
    concept_search,
    concept_used_by,
    concept_uses,
)
from .http import (
    HttpError,
    RateLimiter,
    check_bearer,
    parse_tokens,
    parse_json_body,
    query_params,
    query_to_body,
    send_bytes,
    send_json,
    send_text,
)
from .manifest import (
    API_VERSION,
    apply_site,
    build_api_manifest,
    open_paths,
    routed_paths,
)
from .openapi import build_openapi
from .plates import plates_available, render_ita_png

_HERE = Path(__file__).resolve().parent
_OPEN_PATHS = open_paths() | {"/favicon.ico"}


class ServerState:
    """Shared process state (optional PDB index)."""

    def __init__(self, db_path: str | None, token: str | None):
        self.token = parse_tokens(token)
        self.db_path = db_path
        self.db = None
        self.index = None
        self.n_cells = 0
        self.limiter = RateLimiter()
        if db_path and Path(db_path).exists():
            from ..cell.celldb import CellDatabase
            self.db = CellDatabase(db_path, read_only=True)
            self.index = self.db.build_index()
            self.n_cells = len(self.db)


def _pkg_text(name: str) -> str:
    return apply_site((_HERE / name).read_text(encoding="utf-8"))


def make_handler(state: ServerState):
    """Build a request handler bound to ``state``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"agentsg-serve/{API_VERSION}"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _norm(self, path: str) -> str:
            if path != "/" and path.endswith("/"):
                return path.rstrip("/")
            return path

        def _auth(self, path: str) -> str:
            path = self._norm(path)
            key = check_bearer(self, state.token if path not in _OPEN_PATHS else None)
            if path not in _OPEN_PATHS:
                try:
                    state.limiter.check(key)
                except HttpError as exc:
                    if exc.status == 429:
                        raise
            return key

        def _dispatch_get(self, path: str, qs: dict):
            path = self._norm(path)
            if path == "/":
                return {
                    "service": "agentsg",
                    "docs": "/docs/muse.md",
                    "skill": "/skill.md",
                    "help": "/v1/help",
                    "api": "/api",
                    "openapi": "/openapi.json",
                    "health": "/health",
                    "read_only": True,
                    "note": (
                        "Discover endpoints with GET /api (Bearer). "
                        "Then call a compute path before answering. "
                        "Space-group lookup: GET /v1/space-group?sg=96. "
                        "ITA plate PNG: GET /plates?sg=19."
                    ),
                }
            if path == "/favicon.ico":
                raise HttpError(404, "not found")
            if path == "/health":
                return {
                    "status": "ok",
                    "api_version": API_VERSION,
                    "read_only": True,
                    "db": state.db_path,
                    "cells": state.n_cells,
                    "index_size": 0 if state.index is None else len(state.index),
                    "plates": plates_available(),
                    "concepts": concept_count(),
                    "auth_required": bool(state.token),
                }
            if path == "/openapi.json":
                return build_openapi()
            if path == "/docs/muse.md":
                return ("text", _pkg_text("muse.md"))
            if path == "/skill.md":
                return ("text", _pkg_text("skill.md"))
            if path == "/v1/help":
                return handlers.help_catalog()
            if path == "/api":
                return build_api_manifest()
            if path == "/plates":
                return ("png", handlers.plates_png(qs))
            if path == "/v1/space-group":
                return handlers.space_group_info(query_to_body(qs))
            if path == "/v1/reflections":
                return handlers.reflections_info(query_to_body(qs))
            if path == "/v1/site":
                return handlers.site_info(query_to_body(qs))
            if path == "/v1/harker":
                return handlers.harker_info(query_to_body(qs))
            if path == "/v1/allowed-origins":
                return handlers.allowed_origins_info(query_to_body(qs))
            if path == "/v1/subgroups":
                return handlers.subgroups_info(query_to_body(qs))
            if path == "/v1/ita-plate":
                body = query_to_body(qs)
                q = handlers.plate_query_string(body)
                return handlers.ita_plate_json(body, png_query=q)
            if path == "/v1/cell":
                return handlers.cell_info(query_to_body(qs))
            if path == "/v1/lattice-symmetry":
                return handlers.lattice_symmetry_info(query_to_body(qs))
            if path == "/v1/concept":
                return concept_info(query_to_body(qs))
            if path == "/v1/concept/search":
                return concept_search(query_to_body(qs))
            if path == "/v1/concept/neighbors":
                return concept_neighbors(query_to_body(qs))
            if path == "/v1/concept/uses":
                return concept_uses(query_to_body(qs))
            if path == "/v1/concept/used-by":
                return concept_used_by(query_to_body(qs))
            if path == "/v1/concept/module":
                return concept_module(query_to_body(qs))
            if path == "/v1/concept/receipt":
                return concept_receipt(query_to_body(qs))
            if path == "/v1/ita-plate.png":
                sg, proj, legend, cent = handlers.plate_png_args(qs)
                png = render_ita_png(sg, projection=proj, legend=legend, show_centring=cent)
                return ("png", png)
            if path.startswith("/v1/pdb/") and path != "/v1/pdb/search":
                pid = path[len("/v1/pdb/"):].strip("/")
                return handlers.pdb_lookup(state, pid)
            if path in ("/search", "/v1/pdb/search"):
                return handlers.pdb_search(state, query_to_body(qs) if qs else qs)
            raise HttpError(404, "not found", {"paths": sorted(routed_paths())})

        def _dispatch_post(self, path: str, data: dict):
            path = self._norm(path)
            if path == "/v1/space-group":
                return handlers.space_group_info(data)
            if path == "/v1/setting":
                return handlers.setting_info(data)
            if path == "/v1/identify":
                return handlers.identify_ops(data)
            if path == "/v1/site":
                return handlers.site_info(data)
            if path == "/v1/reflections":
                return handlers.reflections_info(data)
            if path == "/v1/harker":
                return handlers.harker_info(data)
            if path == "/v1/allowed-origins":
                return handlers.allowed_origins_info(data)
            if path == "/v1/subgroups":
                return handlers.subgroups_info(data)
            if path == "/v1/ita-plate":
                q = handlers.plate_query_string(data)
                return handlers.ita_plate_json(data, png_query=q)
            if path == "/v1/cell":
                return handlers.cell_info(data)
            if path == "/v1/lattice-symmetry":
                return handlers.lattice_symmetry_info(data)
            if path == "/v1/compare":
                return handlers.compare_info(data)
            if path == "/v1/reindex":
                return handlers.reindex_info(data)
            if path == "/v1/concept":
                return concept_info(data)
            if path == "/v1/concept/search":
                return concept_search(data)
            if path == "/v1/concept/neighbors":
                return concept_neighbors(data)
            if path == "/v1/concept/uses":
                return concept_uses(data)
            if path == "/v1/concept/used-by":
                return concept_used_by(data)
            if path == "/v1/concept/module":
                return concept_module(data)
            if path == "/v1/concept/receipt":
                return concept_receipt(data)
            if path in ("/search", "/v1/pdb/search"):
                return handlers.pdb_search(state, data)
            raise HttpError(404, "not found", {"paths": sorted(routed_paths())})

        def _record(self, status: int, body: bytes | None = None) -> None:
            if self.headers.get(LOGGED_HEADER) == "1":
                return
            parsed = urlparse(self.path)
            log_exchange(
                service="http",
                method=self.command,
                path=self._norm(parsed.path),
                status=status,
                headers=self.headers,
                peer=self.client_address[0],
                host=self.headers.get("Host", ""),
                body=body,
                query=query_params(self.path),
            )

        def do_GET(self):
            parsed = urlparse(self.path)
            path = self._norm(parsed.path)
            status = 500
            try:
                self._auth(path)
                qs = query_params(self.path)
                result = self._dispatch_get(path, qs)
                self._write_result(result)
                status = 200
            except HttpError as exc:
                status = exc.status
                send_json(self, exc.status, exc.payload)
            except (ValueError, KeyError, TypeError) as exc:
                status = 400
                send_json(self, 400, {"error": str(exc)})
            except ImportError as exc:
                status = 503
                send_json(self, 503, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                send_json(self, 500, {"error": str(exc)})
            finally:
                self._record(status)

        def do_POST(self):
            path = self._norm(urlparse(self.path).path)
            status = 500
            raw = b""
            try:
                self._auth(path)
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b""
                data = parse_json_body(raw)
                result = self._dispatch_post(path, data)
                self._write_result(result)
                status = 200
            except HttpError as exc:
                status = exc.status
                send_json(self, exc.status, exc.payload)
            except (ValueError, KeyError, TypeError) as exc:
                status = 400
                send_json(self, 400, {"error": str(exc)})
            except ImportError as exc:
                status = 503
                send_json(self, 503, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                send_json(self, 500, {"error": str(exc)})
            finally:
                self._record(status, raw)

        def _write_result(self, result):
            if isinstance(result, tuple) and result[0] == "text":
                send_text(self, 200, result[1])
            elif isinstance(result, tuple) and result[0] == "png":
                send_bytes(self, 200, result[1], "image/png")
            else:
                send_json(self, 200, result)

    return Handler


def run_server(db_path: str | None = None, host: str = "127.0.0.1",
               port: int = 8765, token: str | None = None,
               public_url: str | None = None, api_name: str | None = None):
    """Load optional PDB index and serve forever."""
    if public_url:
        os.environ["AGENTSG_PUBLIC_URL"] = public_url.rstrip("/")
    if api_name:
        os.environ["AGENTSG_API_NAME"] = api_name
    state = ServerState(db_path, token)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"agentsg serve on http://{host}:{port}")
    if state.db_path:
        print(f"  database: {state.db_path} ({state.n_cells} cells)")
    print(f"  plates: {plates_available()}")
    print(f"  concepts: {concept_count()}")
    print(f"  auth: {'bearer required' if state.token else 'open (no AGENTSG_TOKEN)'}")
    if log_path():
        print(f"  access log: {log_path()}")
    print("  GET /health  /api  /plates  /openapi.json  /skill.md")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if state.db is not None:
            state.db.close()
        httpd.server_close()


def _cli(argv=None):
    p = argparse.ArgumentParser(description="agentsg HTTP API for Muse")
    p.add_argument("--db", default=os.environ.get("AGENTSG_DB", ""),
                   help="path to pdb_cells.duckdb (optional)")
    p.add_argument("--host", default=os.environ.get("AGENTSG_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("AGENTSG_PORT", "8765")))
    p.add_argument("--token", default=os.environ.get("AGENTSG_TOKEN", ""),
                   help="Comma-separated bearer tokens (or AGENTSG_TOKEN)")
    p.add_argument("--public-url", default=os.environ.get("AGENTSG_PUBLIC_URL", ""),
                   help="Public base URL for docs, OpenAPI, and curl examples")
    p.add_argument("--name", default=os.environ.get("AGENTSG_API_NAME", "agentsg"),
                   help="Connector name in GET /api")
    args = p.parse_args(argv)
    db = args.db or None
    token = args.token or None
    run_server(
        db, host=args.host, port=args.port, token=token,
        public_url=args.public_url or None, api_name=args.name or None,
    )
    return 0
