"""Host-based front door for the one Cloudflare tunnel.

``cloudflared tunnel run --url`` sends every published hostname to a single
origin. This process splits that origin:

- ``sg-mcp.mxagents.org`` → the FastMCP server (no API key)
- every other host → the bearer-token HTTP API

The MCP side is rate-limited per ``CF-Connecting-IP`` (default 60/minute).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .accesslog import LOGGED_HEADER, log_exchange, log_path
from .http import RateLimiter

_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


def upstream_for_host(host: str, *, mcp_port: int, muse_port: int) -> tuple[str, int, bool]:
    """Return ``(host, port, is_mcp)`` for a request Host header."""
    name = host.split(":")[0].strip().lower()
    if name == "sg-mcp.mxagents.org" or name.startswith("sg-mcp."):
        return "127.0.0.1", mcp_port, True
    return "127.0.0.1", muse_port, False


def make_front_handler(
    *,
    mcp_port: int,
    muse_port: int,
    rate_per_minute: int = 60,
):
    """Build a reverse-proxy handler. ``rate_per_minute`` 0 disables the MCP limit."""
    limiter = RateLimiter(rate_per_minute) if rate_per_minute > 0 else None
    opener = build_opener(ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "agentsg-front/0.1"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self):
            self._forward()

        def do_POST(self):
            self._forward()

        def do_PUT(self):
            self._forward()

        def do_DELETE(self):
            self._forward()

        def do_OPTIONS(self):
            self._forward()

        def do_HEAD(self):
            self._forward()

        def _client_key(self) -> str:
            for name in ("CF-Connecting-IP", "X-Forwarded-For"):
                raw = self.headers.get(name)
                if raw:
                    return raw.split(",")[0].strip()
            return self.client_address[0]

        def _send_json(self, status: int, payload: dict, extra: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _record(self, status: int, body: bytes, host_header: str) -> None:
            parsed = urlparse(self.path)
            log_exchange(
                service="front",
                method=self.command,
                path=parsed.path or "/",
                status=status,
                headers=self.headers,
                peer=self.client_address[0],
                host=host_header,
                body=body,
                query={k: v[0] for k, v in parse_qs(parsed.query).items()},
            )

        def _forward(self) -> None:
            host_header = self.headers.get("Host", "")
            dest_host, dest_port, is_mcp = upstream_for_host(
                host_header, mcp_port=mcp_port, muse_port=muse_port,
            )
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            if is_mcp and limiter is not None:
                try:
                    limiter.check(self._client_key())
                except Exception as exc:
                    status = getattr(exc, "status", 429)
                    payload = getattr(exc, "payload", {"error": str(exc)})
                    self._send_json(status, payload, {"Retry-After": "60"})
                    self._record(status, body, host_header)
                    return
            url = f"http://{dest_host}:{dest_port}{self.path}"
            data = body if body else None
            req = Request(url, data=data, method=self.command)
            for key, value in self.headers.items():
                if key.lower() in _HOP or key.lower() in ("host", LOGGED_HEADER.lower()):
                    continue
                req.add_header(key, value)
            req.add_header("Host", host_header or f"{dest_host}:{dest_port}")
            req.add_header(LOGGED_HEADER, "1")
            status = 502
            try:
                resp = opener.open(req, timeout=180)
            except HTTPError as exc:
                resp = exc
            except URLError as exc:
                self._send_json(502, {"error": f"upstream unavailable: {exc.reason}"})
                self._record(502, body, host_header)
                return
            try:
                payload = resp.read()
                status = getattr(resp, "status", None) or resp.getcode()
                self.send_response(status)
                for key, value in resp.headers.items():
                    if key.lower() in _HOP:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
            finally:
                resp.close()
                self._record(status, body, host_header)

    return Handler


def run_front(
    host: str = "127.0.0.1",
    port: int = 9880,
    mcp_port: int = 9877,
    muse_port: int = 9876,
    rate_per_minute: int = 60,
):
    """Listen until interrupted."""
    httpd = ThreadingHTTPServer(
        (host, port),
        make_front_handler(
            mcp_port=mcp_port,
            muse_port=muse_port,
            rate_per_minute=rate_per_minute,
        ),
    )
    print(f"agentsg front on http://{host}:{port}")
    print(f"  sg-mcp.* → 127.0.0.1:{mcp_port} (no API key)")
    print(f"  other hosts → 127.0.0.1:{muse_port}")
    if log_path():
        print(f"  access log: {log_path()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Host router in front of the Cloudflare tunnel")
    parser.add_argument("--host", default=os.environ.get("AGENTSG_FRONT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("AGENTSG_FRONT_PORT", "9880")))
    parser.add_argument("--mcp-port", type=int,
                        default=int(os.environ.get("AGENTSG_MCP_PORT", "9877")))
    parser.add_argument("--muse-port", type=int,
                        default=int(os.environ.get("AGENTSG_PORT", "9876")))
    parser.add_argument("--rate", type=int,
                        default=int(os.environ.get("AGENTSG_MCP_RATE", "60")),
                        help="MCP requests per minute per client IP (0 disables)")
    args = parser.parse_args(argv)
    run_front(
        host=args.host,
        port=args.port,
        mcp_port=args.mcp_port,
        muse_port=args.muse_port,
        rate_per_minute=args.rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
