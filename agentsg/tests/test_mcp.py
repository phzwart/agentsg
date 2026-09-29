"""FastMCP tools and the tunnel front door."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentsg.serve.app import ServerState
from agentsg.serve.front import make_front_handler, upstream_for_host
from agentsg.serve.mcp_app import build_mcp


def test_upstream_for_host():
    assert upstream_for_host("sg-mcp.mxagents.org", mcp_port=9877, muse_port=9876) == (
        "127.0.0.1", 9877, True,
    )
    assert upstream_for_host("SG-MCP.MXAGENTS.ORG:443", mcp_port=1, muse_port=2)[2] is True
    host, port, is_mcp = upstream_for_host("sg-muse.mxagents.org", mcp_port=1, muse_port=9876)
    assert (host, port, is_mcp) == ("127.0.0.1", 9876, False)


def test_mcp_tools_have_no_auth_and_call_handlers():
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    mcp = build_mcp(ServerState(None, None), public_url="https://sg-mcp.mxagents.org")
    assert mcp.auth is None

    async def _run():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            space = await client.call_tool("space_group", {"sg": "96"})
            book = await client.call_tool("playbook", {})
            with pytest.raises(ToolError):
                await client.call_tool("pdb_lookup", {"pdb_id": "1LYZ"})
            return names, tools, space.data, book.data

    names, tools, space, book = asyncio.run(_run())
    assert "space_group" in names
    assert "pdb_search" in names
    pdb = next(tool for tool in tools if tool.name == "pdb_search")
    schema = getattr(pdb, "input_schema", None) or pdb.inputSchema
    assert "plot" in schema["properties"]
    assert "return_cob" in schema["properties"]
    assert "ita_plate" in names
    assert "ita_plate_image" not in names
    assert space["sg_number"] == 96
    assert "reflection_conditions" in space
    assert "There is no API key" in book
    assert "bearer" not in book.lower()
    assert "authorization" not in book.lower()
    assert "LIMITATIONS" in book
    assert "scatter PNG" in book
    assert "plate PNG" in book
    assert "ita_plate_image" not in book


def test_mcp_returns_ita_png_and_cell_plot(tmp_path):
    pytest.importorskip("fastmcp")
    pytest.importorskip("matplotlib")
    pytest.importorskip("duckdb")
    from fastmcp import Client

    from agentsg.cell.celldb import CellDatabase

    path = tmp_path / "cells.duckdb"
    db = CellDatabase(str(path))
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("LYZ2", (80.0, 79.0, 38.2, 90, 90, 90), 96, "P 43 21 2")
    db.close()
    mcp = build_mcp(ServerState(str(path), None))

    async def _run():
        async with Client(mcp) as client:
            plate = await client.call_tool_mcp("ita_plate", {"sg": "96"})
            plotted = await client.call_tool_mcp("pdb_search", {
                "sg": "96",
                "cell": [79, 79, 38, 90, 90, 90],
                "k": 2,
                "plot": True,
            })
            return plate, plotted

    plate, plotted = asyncio.run(_run())

    def png_blocks(result):
        return [block for block in result.content if getattr(block, "type", None) == "image"]

    assert png_blocks(plate)
    assert png_blocks(plate)[0].mime_type == "image/png"
    assert png_blocks(plate)[0].data
    elements = plate.structured_content["elements"]
    screws = [
        el["location"][0]
        for el in elements
        if el["symbol"] == "2_1" and abs(el["axis"][1] - 1.0) < 1e-6
    ]
    assert any(abs(x - 0.25) < 1e-6 for x in screws)
    assert any(abs(x - 0.75) < 1e-6 for x in screws)
    assert any(
        el["symbol"] == "2"
        and abs(el["axis"][0]) == 1
        and abs(el["axis"][1]) == 1
        and abs(el["location"][0]) < 1e-6
        and abs(el["location"][1]) < 1e-6
        for el in elements
    )
    assert png_blocks(plotted)
    assert png_blocks(plotted)[0].mime_type == "image/png"
    assert plotted.structured_content["png"] == "included"
    assert "plot_png_base64" not in plotted.structured_content
    assert "svd" in plotted.structured_content


def test_front_routes_by_host():
    def echo(label: str):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = label.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                return

        return Handler

    muse = ThreadingHTTPServer(("127.0.0.1", 0), echo("muse"))
    mcp = ThreadingHTTPServer(("127.0.0.1", 0), echo("mcp"))
    front = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_front_handler(
            mcp_port=mcp.server_address[1],
            muse_port=muse.server_address[1],
            rate_per_minute=0,
        ),
    )
    threads = []
    for httpd in (muse, mcp, front):
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        import urllib.request
        front_port = front.server_address[1]

        def fetch(host: str) -> bytes:
            req = urllib.request.Request(f"http://127.0.0.1:{front_port}/health")
            req.add_header("Host", host)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.read()

        assert fetch("sg-mcp.mxagents.org") == b"mcp"
        assert fetch("sg-muse.mxagents.org") == b"muse"
    finally:
        for httpd in (front, mcp, muse):
            httpd.shutdown()
