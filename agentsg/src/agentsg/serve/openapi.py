"""OpenAPI 3.1 document generated from ``manifest.ENDPOINTS``."""
from __future__ import annotations

from typing import Any

from .manifest import (
    API_VERSION,
    DEFAULT_SERVER,
    ENDPOINTS,
    build_api_manifest,
    routed_paths,
)

__all__ = ["API_VERSION", "DEFAULT_SERVER", "build_openapi", "routed_paths",
           "build_api_manifest"]

_TYPE = {
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "string": "string",
    "array": "array",
}


def _schema(param: dict[str, Any]) -> dict[str, Any]:
    typ = _TYPE.get(param["type"], "string")
    schema: dict[str, Any] = {"type": typ, "description": param["description"]}
    if typ == "array":
        schema["items"] = {"type": "number"}
    return schema


def _parameter(param: dict[str, Any], loc: str) -> dict[str, Any]:
    return {
        "name": param["name"],
        "in": loc,
        "required": bool(param["required"]),
        "schema": _schema(param),
        "description": param["description"],
    }


def _operation(entry: dict[str, Any], method: str) -> dict[str, Any]:
    op_ids = entry.get("_op_ids") or {}
    op_id = op_ids.get(method) or f"{method.lower()}{entry['path']}"
    op: dict[str, Any] = {
        "operationId": op_id,
        "summary": entry["description"],
        "x-read-only": True,
        "responses": {
            "200": {"description": "image/png" if entry.get("_png") else "ok"},
            "400": {"description": "bad request"},
            "401": {"description": "unauthorized"},
        },
    }
    if not entry.get("_auth", True):
        op["security"] = []
    params = entry["params"]
    if method == "GET":
        loc = "path" if "{" in entry["path"] else "query"
        op["parameters"] = [_parameter(p, "path" if p["name"] in entry["path"] else loc)
                            for p in params]
    else:
        props = {p["name"]: _schema(p) for p in params}
        required = [p["name"] for p in params if p["required"]]
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        op["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": schema}},
        }
    return op


def build_openapi() -> dict[str, Any]:
    """Return the OpenAPI 3.1 document as a dict."""
    paths: dict[str, Any] = {}
    for entry in ENDPOINTS:
        item: dict[str, Any] = {}
        for method in entry["methods"]:
            item[method.lower()] = _operation(entry, method)
        paths[entry["path"]] = item
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "agentsg symmetry API",
            "version": API_VERSION,
            "description": (
                "Read-only crystallographic space-group and lattice tools. "
                "Call GET /api (Bearer) for the machine-readable catalog "
                "before guessing endpoints. Same catalog generates this spec."
            ),
        },
        "servers": [{"url": DEFAULT_SERVER}],
        "security": [{"bearerAuth": []}],
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "AGENTSG_TOKEN. Discovery GET routes listed with security: [] are open.",
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                    "required": ["error"],
                },
            },
        },
        "paths": paths,
    }
