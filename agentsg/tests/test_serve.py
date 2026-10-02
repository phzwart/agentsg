"""HTTP serve API (Muse connector) tests."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from agentsg.serve.app import ServerState, make_handler
from agentsg.serve.manifest import API_VERSION, build_api_manifest
from agentsg.serve.openapi import build_openapi, routed_paths
from agentsg.serve.serialize import parse_cell, parse_frac, parse_xyz_point, vec_to_json
from http.server import ThreadingHTTPServer


@pytest.fixture
def sample_db(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from agentsg.cell.celldb import CellDatabase
    path = tmp_path / "test.duckdb"
    db = CellDatabase(str(path))
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("LYZ2", (79.0, 79.0, 38.0, 90, 90, 90), 96, "P 43 21 2")
    db.close()
    return str(path)


def _fetch(url, data=None, token=None, method=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is None:
        req = urllib.request.Request(url, headers=headers, method=method or "GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                return resp.status, json.loads(body.decode())
            return resp.status, body
    body = json.dumps(data).encode()
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode())


def _server(db_path=None, token=None):
    state = ServerState(db_path, token)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return state, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_serialize_frac_and_cell():
    assert str(parse_frac("1/4")) == "1/4"
    assert parse_cell([79, 79, 38, 90, 90, 90])[0] == 79.0
    v = parse_xyz_point(["1/4", "1/4", "1/4"])
    assert vec_to_json(v) == ["1/4", "1/4", "1/4"]


def test_openapi_covers_routes():
    spec = build_openapi()
    paths = routed_paths()
    assert "/health" in paths
    assert "/api" in paths
    assert "/plates" in paths
    assert "/v1/space-group" in paths
    assert "/v1/help" in paths
    assert "/v1/ita-plate.png" in paths
    manifest = build_api_manifest()
    assert manifest["api_version"] == API_VERSION == "0.3.0"
    assert "/v1/subgroups" in paths
    assert {e["path"] for e in manifest["endpoints"]} == paths
    assert spec["openapi"].startswith("3.")
    assert "bearerAuth" in spec["components"]["securitySchemes"]
    assert "get" in spec["paths"]["/v1/space-group"]
    assert spec["paths"]["/v1/space-group"]["post"]["operationId"] == "spaceGroup"


def test_site_from_env(monkeypatch):
    monkeypatch.setenv("AGENTSG_PUBLIC_URL", "https://example.test")
    monkeypatch.setenv("AGENTSG_API_NAME", "demo")
    from agentsg.serve.manifest import apply_site, build_api_manifest
    from agentsg.serve.openapi import build_openapi
    man = build_api_manifest()
    assert man["name"] == "demo"
    assert any("https://example.test/v1/space-group" in e["example"]
               for e in man["endpoints"])
    assert build_openapi()["servers"] == [{"url": "https://example.test"}]
    assert apply_site("{{BASE_URL}}/skill.md") == "https://example.test/skill.md"


def test_http_discovery_and_space_group():
    state, httpd, base = _server()
    try:
        status, health = _fetch(f"{base}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["read_only"] is True
        assert health["api_version"] == "0.3.0"
        status, root = _fetch(f"{base}/")
        assert status == 200
        assert root["docs"] == "/docs/muse.md"
        status, health_slash = _fetch(f"{base}/health/")
        assert status == 200
        assert health_slash["status"] == "ok"

        status, spec = _fetch(f"{base}/openapi.json")
        assert status == 200
        assert "/v1/space-group" in spec["paths"]

        status, md = _fetch(f"{base}/skill.md")
        assert status == 200
        assert b"Wyckoff" in md
        assert b"LIMITATIONS" in md
        assert b"Translationengleiche" in md
        assert b"Klassengleiche" in md
        assert b"GET /v1/space-group" in md
        assert b"/v1/subgroups" in md

        status, sg = _fetch(f"{base}/v1/space-group", {"sg": 96})
        assert status == 200
        assert sg["sg_number"] == 96
        assert sg["order"] == 8
        assert "hkl" in sg["reflection_conditions"] or sg["reflection_conditions"]

        status, sg_get = _fetch(f"{base}/v1/space-group?sg=96")
        assert status == 200
        assert sg_get["sg_number"] == 96
        assert sg_get["sg_hm"] == sg["sg_hm"]

        status, help_ = _fetch(f"{base}/v1/help")
        assert status == 200
        assert help_["calls"]
        assert any("space-group" in c["path"] for c in help_["calls"])
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_bearer_required():
    state, httpd, base = _server(token="secret")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/v1/space-group", {"sg": 19})
        assert exc.value.code == 401
        status, sg = _fetch(f"{base}/v1/space-group", {"sg": 19}, token="secret")
        assert status == 200
        assert sg["sg_number"] == 19
        status, health = _fetch(f"{base}/health")
        assert status == 200
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_multiple_bearer_tokens():
    state, httpd, base = _server(token="secret, testtoken73")
    try:
        assert state.token == frozenset({"secret", "testtoken73"})
        for presented in ("secret", "testtoken73"):
            status, sg = _fetch(
                f"{base}/v1/space-group", {"sg": 19}, token=presented,
            )
            assert status == 200
            assert sg["sg_number"] == 19
        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/v1/space-group", {"sg": 19}, token="other")
        assert exc.value.code == 401
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_site_reflections_harker_cell():
    state, httpd, base = _server()
    try:
        status, site = _fetch(f"{base}/v1/site", {
            "sg": 225, "xyz": ["1/4", "1/4", "1/4"],
        })
        assert status == 200
        assert site["multiplicity"] == 8
        assert site["site_symmetry_order"] == 24
        assert site["wyckoff_letter"] is None

        status, ref = _fetch(f"{base}/v1/reflections", {"sg": 96, "hkl": [0, 0, 1]})
        assert status == 200
        assert "absent" in ref

        status, hk = _fetch(f"{base}/v1/harker", {"sg": 19})
        assert status == 200
        assert hk["loci"]
        status, origins = _fetch(f"{base}/v1/allowed-origins?sg=225")
        assert status == 200
        assert origins["n_origins"] == 2
        assert ["0", "0", "0"] in origins["origins"]
        assert ["0", "0", "1/2"] in origins["origins"]
        assert origins["floating_origin"] == []
        assert "allowed_origins" in origins["concepts"]
        status, graph = _fetch(f"{base}/v1/subgroups?sg=96")
        assert status == 200
        assert graph["sg_number"] == 96
        types = {e["type"] for e in graph["edges"]}
        kids = {e["to"] for e in graph["edges"]}
        assert "t" in types
        assert 78 in kids or 18 in kids
        status, graph_post = _fetch(f"{base}/v1/subgroups", {"sg": 96, "kind": "t"})
        assert status == 200
        assert all(e["type"] == "t" for e in graph_post["edges"])

        status, cell = _fetch(f"{base}/v1/cell", {
            "cell": [79, 79, 38, 90, 90, 90], "sg": 96,
        })
        assert status == 200
        assert cell["volume"] > 0
        assert len(cell["root_invariant"]) == 6
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_lattice_compare_reindex():
    state, httpd, base = _server()
    try:
        status, ls = _fetch(f"{base}/v1/lattice-symmetry", {
            "cell": [50, 50, 50, 90, 90, 90],
        })
        assert status == 200
        assert ls["order"] >= 2

        status, cmp_ = _fetch(f"{base}/v1/compare", {
            "cell_a": [50, 50, 50, 90, 90, 90],
            "cell_b": [51, 51, 51, 90, 90, 90],
        })
        assert status == 200
        assert cmp_["root_distance"] > 0

        status, rx = _fetch(f"{base}/v1/reindex", {
            "sg": 75, "cell": [50, 50, 80, 90, 90, 90],
        })
        assert status == 200
        assert rx["operators"]
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_identify_and_setting():
    state, httpd, base = _server()
    try:
        status, sg = _fetch(f"{base}/v1/space-group", {"sg": 19})
        status, ident = _fetch(f"{base}/v1/identify", {"ops": sg["ops"]})
        assert ident["sg_number"] == 19
        assert ident["input_order"] == 4
        assert ident["matched_order"] == 4
        assert ident["det"] in (1, -1, "1", "-1")
        assert "note" not in ident

        status, prim = _fetch(f"{base}/v1/setting", {
            "setting": "F 2 2 2 ((y+z)/2,(x+z)/2,(x+y)/2)",
        })
        assert status == 200
        assert prim["order"] == 4
        status, ident22 = _fetch(f"{base}/v1/identify", {"ops": prim["ops"]})
        assert ident22["sg_number"] == 22
        assert ident22["input_order"] == 4
        assert ident22["matched_order"] == 16
        assert abs(int(ident22["det"])) == 4
        assert "note" in ident22

        status, st = _fetch(f"{base}/v1/setting", {
            "setting": "P 21 21 21 (b,c,a)",
        })
        assert status == 200
        assert "P" in st["P"][0] or st["cob"]

        status, f432 = _fetch(f"{base}/v1/setting", {
            "setting": "F 4 2 3 ((y+z)/2,(x+z)/2,(x+y)/2)",
        })
        assert status == 200
        assert f432["base_sg_number"] == 209
        assert f432["det"] == "1/4"
        assert f432["order"] == 24
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_pdb_search_and_lookup(sample_db):
    state, httpd, base = _server(sample_db)
    try:
        status, result = _fetch(f"{base}/v1/pdb/search", {
            "cell": [79, 79, 38, 90, 90, 90], "sg": 96, "cutoff": 0.5,
        })
        assert status == 200
        assert result["count"] == 2
        status, row = _fetch(f"{base}/v1/pdb/LYZ2")
        assert row["pdb_id"] == "LYZ2"
        status, knn = _fetch(f"{base}/v1/pdb/search", {
            "cell": [79, 79, 38, 90, 90, 90], "sg": 96, "k": 1,
        })
        assert knn["count"] == 1
        assert "plot_png_base64" not in knn
        status, plotted = _fetch(f"{base}/v1/pdb/search", {
            "cell": [79, 79, 38, 90, 90, 90], "sg": 96, "cutoff": 0.5,
            "plot": True,
        })
        assert status == 200
        assert plotted["count"] == 2
        assert plotted["svd"]["n"] == 2
        assert plotted["svd"]["feature"] == "root_invariant r0..r5"
        assert len(plotted["svd"]["variance_frac"]) >= 1
        for hit in plotted["hits"]:
            assert len(hit["xy"]) == 2
        assert len(plotted["query_xy"]) == 2
        if plotted.get("plot_png_base64"):
            import base64
            raw = base64.b64decode(plotted["plot_png_base64"])
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        httpd.shutdown()
        state.db.close()


def test_pdb_search_returns_cob_for_same_lattice(tmp_path):
    """A true reindexing gets a COB; a nearby impostor gets cob null."""
    from agentsg.cell.celldb import CellDatabase
    from agentsg.cell.metric import UnitCell
    from agentsg.cell.selling_cob import _metric_of, parse_cob_columns
    from agentsg.cell.sublattice import apply_to_cell

    base = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    swapped = apply_to_cell(base, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    near = (40.4, 50.3, 60.2, 90.0, 90.0, 90.0)
    path = tmp_path / "cob.duckdb"
    db = CellDatabase(str(path))
    db.add_cell("SWAP", swapped, 19, "P 21 21 21")
    db.add_cell("NEAR", near, 19, "P 21 21 21")
    db.close()

    state, httpd, base_url = _server(str(path))
    try:
        status, plain = _fetch(f"{base_url}/v1/pdb/search", {
            "cell": list(base), "sg": "P 21 21 21", "k": 2,
        })
        assert status == 200
        assert all("cob" not in hit for hit in plain["hits"])
        status, result = _fetch(f"{base_url}/v1/pdb/search", {
            "cell": list(base), "sg": "P 21 21 21", "k": 2, "return_cob": True,
        })
        assert status == 200
        assert result["gates"]["length_tol_pct"] == 0.75
        assert result["gates"]["angle_tol_deg"] == 0.5
        assert result["gates"]["angle_sigma_deg"] == 0.05
        assert result["gates"]["boundary_rel"] == 1e-3
        by_id = {hit["pdb_id"]: hit for hit in result["hits"]}
        assert by_id["NEAR"]["cob"] is None
        swap = by_id["SWAP"]
        assert swap["cob"] is not None
        assert swap["cob_xyz"]
        P = parse_cob_columns(
            f"{n}/{d}" for row in swap["cob"] for n, d in row
        )
        Gp = _metric_of(UnitCell(*base).metric_tensor(), P)
        G_swap = UnitCell(*swapped).metric_tensor()
        for a in range(3):
            for b in range(3):
                assert Gp[a][b] == pytest.approx(G_swap[a][b], rel=1e-8, abs=1e-6)
        if "cob_coset" in swap:
            assert len(swap["cob_coset"]) > 1
            assert swap["cob_coset"][0]["cob"] == swap["cob"]
        status, radius = _fetch(f"{base_url}/v1/pdb/search", {
            "cell": list(base), "sg": "P 21 21 21", "cutoff": 1.0,
            "return_cob": True,
        })
        assert status == 200
        by_id = {hit["pdb_id"]: hit for hit in radius["hits"]}
        assert by_id["SWAP"]["cob"] == swap["cob"]
        assert by_id["NEAR"]["cob"] is None
    finally:
        httpd.shutdown()
        state.db.close()


def test_ita_plate_png():
    pytest.importorskip("matplotlib")
    state, httpd, base = _server()
    try:
        status, meta = _fetch(f"{base}/v1/ita-plate", {"sg": 19, "legend": True})
        assert status == 200
        assert meta["elements"]
        assert meta["png_url"].startswith("/v1/ita-plate.png")
        status, p43212 = _fetch(f"{base}/v1/ita-plate", {"sg": 96})
        assert status == 200
        screws = [
            el["location"][0]
            for el in p43212["elements"]
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
            for el in p43212["elements"]
        )
        status, png = _fetch(f"{base}/v1/ita-plate.png?sg=19&legend=true")
        assert status == 200
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        status, plate = _fetch(f"{base}/plates?sg=19")
        assert status == 200
        assert plate[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_api_manifest_auth_and_plates_errors():
    pytest.importorskip("matplotlib")
    state, httpd, base = _server(token="secret")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/api")
        assert exc.value.code == 401

        status, manifest = _fetch(f"{base}/api", token="secret")
        assert status == 200
        assert manifest["name"] == "agentsg"
        assert manifest["api_version"] == "0.3.0"
        assert manifest["auth"] == {"scheme": "bearer", "header": "Authorization"}
        by_path = {e["path"]: e for e in manifest["endpoints"]}
        assert "/search" in by_path
        assert "/plates" in by_path
        assert "/v1/space-group" in by_path
        assert "/v1/reflections" in by_path
        assert "/v1/site" in by_path
        assert "/v1/identify" in by_path
        assert "/v1/setting" in by_path
        assert "/v1/harker" in by_path
        assert "/v1/subgroups" in by_path
        assert "/v1/compare" in by_path
        assert "/v1/reindex" in by_path
        search = by_path["/search"]
        assert "79" in search["example"] and "P212121" in search["example"]
        assert search["side_effects"] == "reads database"
        assert by_path["/plates"]["example"]
        assert by_path["/v1/space-group"]["side_effects"] == "none"

        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/plates?sg=999", token="secret")
        assert exc.value.code == 404
        err = json.loads(exc.value.read().decode())
        assert "error" in err

        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/plates", token="secret")
        assert exc.value.code == 400
        err = json.loads(exc.value.read().decode())
        assert "sg" in err["error"]

        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/plates?sg=19&projection=z", token="secret")
        assert exc.value.code == 400
        err = json.loads(exc.value.read().decode())
        assert "projection" in err["error"]
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()
