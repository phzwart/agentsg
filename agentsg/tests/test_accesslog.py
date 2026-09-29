"""Request log: question, client address, no bearer token."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from agentsg.serve.accesslog import caller_id, client_ip, question, write_access
from agentsg.serve.app import ServerState, make_handler


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _fetch(url, data=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = json.dumps(data).encode()
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode())


def _server(token=None):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ServerState(None, token)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_client_ip_prefers_cloudflare():
    headers = _Headers({"X-Forwarded-For": "198.51.100.2, 10.0.0.1",
                        "CF-Connecting-IP": "203.0.113.9"})
    assert client_ip(headers, "127.0.0.1") == "203.0.113.9"


def test_caller_id_hides_the_token():
    label = caller_id("Bearer secret-token")
    assert label != "secret-token"
    assert "secret" not in label
    assert label == caller_id("Bearer secret-token")
    assert caller_id("") == "anonymous"


def test_question_drops_secrets_and_names_mcp_tool():
    body = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "pdb_search",
            "arguments": {"sg": "P 1 21 1", "token": "nope", "k": 5},
        },
    }).encode()
    ask = question(body, {"token": "also-nope", "sg": "96"})
    assert ask["tool"] == "pdb_search"
    assert ask["sg"] == "P 1 21 1"
    assert ask["k"] == 5
    assert "token" not in ask
    assert "nope" not in json.dumps(ask)


def test_http_access_log_records_the_question(tmp_path, monkeypatch):
    path = tmp_path / "access.jsonl"
    monkeypatch.setenv("AGENTSG_ACCESS_LOG", str(path))
    httpd, base = _server(token="sekret")
    try:
        status, _body = _fetch(
            f"{base}/v1/space-group",
            {"sg": 96, "token": "do-not-store"},
            token="sekret",
        )
        assert status == 200
    finally:
        httpd.shutdown()
    text = path.read_text()
    assert "sekret" not in text
    assert "do-not-store" not in text
    line = json.loads(text.splitlines()[-1])
    assert line["service"] == "http"
    assert line["path"] == "/v1/space-group"
    assert line["status"] == 200
    assert line["ask"]["sg"] == 96
    assert line["caller"] == caller_id("Bearer sekret")
    assert line["client"] == "127.0.0.1"


def test_marked_request_is_not_logged_twice(tmp_path, monkeypatch):
    path = tmp_path / "access.jsonl"
    monkeypatch.setenv("AGENTSG_ACCESS_LOG", str(path))
    write_access({"service": "front", "path": "/health", "status": 200})
    state = ServerState(None, None)
    handler = make_handler(state).__new__(make_handler(state))
    handler.path = "/health"
    handler.command = "GET"
    handler.headers = {"X-Agentsg-Access-Logged": "1", "Host": "sg-muse"}
    handler.client_address = ("127.0.0.1", 9)
    handler._record(200)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["service"] == "front"
