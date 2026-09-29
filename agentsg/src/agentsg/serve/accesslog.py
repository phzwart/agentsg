"""Append-only request log.

Enabled when ``AGENTSG_ACCESS_LOG`` is a file path. Each line is one JSON
object: when, which host, the client address, the path, the status, and a
short summary of the question. The bearer token is never written; a caller
with a token is stored as the first 12 hex digits of its SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

_LOCK = threading.Lock()
LOGGED_HEADER = "X-Agentsg-Access-Logged"
_DROP = {
    "authorization", "token", "access_token", "api_key", "password", "secret",
}


def log_path() -> str:
    """Destination file, or empty when logging is off."""
    return os.environ.get("AGENTSG_ACCESS_LOG", "").strip()


def client_ip(headers, peer: str) -> str:
    """Visitor address from Cloudflare or a proxy, otherwise the TCP peer."""
    for name in ("CF-Connecting-IP", "X-Forwarded-For"):
        raw = headers.get(name) if headers is not None else None
        if raw:
            return str(raw).split(",")[0].strip()
    return peer


def caller_id(authorization: str | None) -> str:
    """Stable label for a bearer token, or ``anonymous`` when there is none."""
    header = authorization or ""
    presented = ""
    lowered = header.lower()
    if lowered.startswith("bearer "):
        presented = header[7:].strip()
    elif lowered.startswith("token "):
        presented = header[6:].strip()
    if not presented:
        return "anonymous"
    return hashlib.sha256(presented.encode()).hexdigest()[:12]


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).lower() in _DROP:
                continue
            out[str(key)] = _scrub(item, depth + 1)
        return out
    if isinstance(value, list):
        items = [_scrub(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"…{len(value) - 12} more")
        return items
    if isinstance(value, str) and len(value) > 180:
        return value[:180] + "…"
    return value


def question(body: bytes | None, query: dict | None = None) -> dict:
    """Short description of what was asked. Secrets are dropped."""
    ask: dict[str, Any] = {}
    if query:
        ask.update(_scrub(query))
    if not body:
        return ask
    try:
        data = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ask
    if not isinstance(data, dict):
        return ask
    params = data.get("params")
    if isinstance(data.get("method"), str) and isinstance(params, dict):
        ask["mcp_method"] = data["method"]
        if "name" in params:
            ask["tool"] = params.get("name")
        arguments = params.get("arguments")
        if isinstance(arguments, dict):
            ask.update(_scrub(arguments))
        return ask
    ask.update(_scrub(data))
    return ask


def write_access(record: dict) -> None:
    """Append one JSON line. No-op when ``AGENTSG_ACCESS_LOG`` is unset."""
    path = log_path()
    if not path:
        return
    record = dict(record)
    record.setdefault(
        "ts", datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
    directory = os.path.dirname(path)
    with _LOCK:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def log_exchange(
    *,
    service: str,
    method: str,
    path: str,
    status: int,
    headers,
    peer: str,
    host: str = "",
    body: bytes | None = None,
    query: dict | None = None,
) -> None:
    """Record one finished request."""
    if log_path() == "":
        return
    try:
        write_access({
            "service": service,
            "host": host,
            "client": client_ip(headers, peer),
            "peer": peer,
            "caller": caller_id(headers.get("Authorization") if headers else None),
            "method": method,
            "path": path,
            "status": status,
            "ask": question(body, query),
        })
    except OSError as exc:
        sys.stderr.write(f"access log failed: {exc}\n")
