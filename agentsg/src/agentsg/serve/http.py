"""Minimal JSON HTTP helpers (stdlib)."""
from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qs, urlparse


class HttpError(Exception):
    """HTTP error with status code and JSON body."""

    def __init__(self, status: int, error: str, extra: dict[str, Any] | None = None):
        super().__init__(error)
        self.status = status
        self.payload = {"error": error}
        if extra:
            self.payload.update(extra)


class RateLimiter:
    """Per-token sliding window (default 60 requests / 60 s)."""

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.max_per_minute:
            raise HttpError(429, "rate limit exceeded", {"retry_after": 60})
        q.append(now)


def parse_json_body(raw: bytes) -> dict[str, Any]:
    """Parse a JSON object body."""
    if not raw:
        raise HttpError(400, "POST body required")
    try:
        data = json.loads(raw.decode())
    except json.JSONDecodeError as exc:
        raise HttpError(400, f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HttpError(400, "JSON body must be an object")
    return data


def read_json_body(handler) -> dict[str, Any]:
    """Read and parse a JSON POST body."""
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise HttpError(400, "POST body required")
    return parse_json_body(handler.rfile.read(length))


def query_params(path: str) -> dict[str, str]:
    """First-value query string map."""
    qs = parse_qs(urlparse(path).query, keep_blank_values=False)
    return {k: v[0] for k, v in qs.items()}


_SG_KEYS = {"sg", "sg_a", "sg_b"}
_LIST_INT = {"hkl"}
_LIST_FLOAT = {"cell", "cell_a", "cell_b"}
_BOOL_KEYS = {
    "legend", "show_centring", "include_sublattices", "include_g6",
    "same_hm", "same_sg", "maximal", "plot", "return_cob",
}
_INT_KEYS = {"k"}
_FLOAT_KEYS = {"cutoff", "max_delta", "length_tol_pct", "angle_tol_deg"}


def query_to_body(qs: dict[str, str]) -> dict[str, Any]:
    """Coerce GET query strings into the same shapes POST JSON uses."""
    data: dict[str, Any] = {}
    for key, raw in qs.items():
        val = raw.strip()
        if key in _SG_KEYS:
            data[key] = int(val) if val.lstrip("-").isdigit() else val
        elif key in _LIST_INT:
            data[key] = [int(x) for x in val.replace(" ", "").split(",") if x]
        elif key in _LIST_FLOAT:
            data[key] = [float(x) for x in val.replace(" ", "").split(",") if x]
        elif key in _BOOL_KEYS:
            data[key] = val.lower() in ("1", "true", "yes")
        elif key in _INT_KEYS:
            data[key] = int(val)
        elif key in _FLOAT_KEYS:
            data[key] = float(val)
        else:
            data[key] = val
    return data


def send_json(handler, status: int, payload: dict[str, Any] | list) -> None:
    """Write a JSON response with CORS headers."""
    body = json.dumps(payload, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    if status == 429:
        handler.send_header("Retry-After", "60")
    handler.end_headers()
    handler.wfile.write(body)


def send_bytes(handler, status: int, body: bytes, content_type: str) -> None:
    """Write a raw (e.g. PNG or markdown) response."""
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def send_text(handler, status: int, text: str, content_type: str = "text/markdown; charset=utf-8") -> None:
    """Write a UTF-8 text response."""
    send_bytes(handler, status, text.encode(), content_type)


def parse_tokens(raw: str | Iterable[str] | None) -> frozenset[str]:
    """Split a comma-separated token string into bearer tokens.

    Whitespace around each entry is ignored. Empty entries are dropped.
    """
    if raw is None:
        return frozenset()
    parts = raw.split(",") if isinstance(raw, str) else raw
    return frozenset(part.strip() for part in parts if str(part).strip())


def check_bearer(handler, token: str | Iterable[str] | None) -> str:
    """Require ``Authorization: Bearer`` when server tokens are configured.

    ``token`` may be one string or several comma-separated values. Any match
    is accepted. Returns the presented token (or ``anonymous`` if auth is off).
    """
    header = handler.headers.get("Authorization", "")
    presented = ""
    if header.lower().startswith("bearer "):
        presented = header[7:].strip()
    elif header.lower().startswith("token "):
        presented = header[6:].strip()
    allowed = parse_tokens(token)
    if allowed:
        if not presented or not any(
            secrets.compare_digest(presented, item) for item in allowed
        ):
            raise HttpError(401, "invalid or missing bearer token")
        return presented
    return presented or "anonymous"
