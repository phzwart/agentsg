"""HTTP PDB search server tests."""
import json
import threading
import urllib.error
import urllib.request

import pytest

duckdb = pytest.importorskip("duckdb")

from agentsg.cell.celldb import CellDatabase
from agentsg.cell.pdb_server import (
    _parse_search_params,
    make_handler,
    search_compatible,
    _PdbSearchServer,
)
from http.server import ThreadingHTTPServer


@pytest.fixture
def sample_db(tmp_path):
    path = tmp_path / "test.duckdb"
    db = CellDatabase(str(path))
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("LYZ2", (79.0, 79.0, 38.0, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("ORTH", (59.1, 68.5, 30.5, 90, 90, 90), 19, "P 21 21 21")
    db.add_cell("FAR", (100.0, 100.0, 100.0, 90, 90, 90), 96, "P 43 21 2")
    db.close()
    return str(path)


def test_within_radius(sample_db):
    db = CellDatabase(sample_db)
    hits = db.within((79.0, 79.0, 38.0, 90, 90, 90), 0.5,
                     sg_number=96, sg_hm="P 43 21 2")
    ids = {pid for pid, _ in hits}
    assert ids == {"LYZ1", "LYZ2"}
    db.close()


def test_search_compatible(sample_db):
    state = _PdbSearchServer(sample_db)
    result = search_compatible(
        state.db, state.index,
        cell=(79.0, 79.0, 38.0, 90, 90, 90),
        cutoff=0.5,
        sg_number=96,
        sg_hm="P 43 21 2",
    )
    ids = {h["pdb_id"] for h in result["hits"]}
    assert ids == {"LYZ1", "LYZ2"}
    state.db.close()


def test_search_compatible_same_hm(sample_db):
    state = _PdbSearchServer(sample_db)
    result = search_compatible(
        state.db, state.index,
        cell=(79.0, 79.0, 38.0, 90, 90, 90),
        cutoff=5.0,
        sg_number=96,
        sg_hm="P 43 21 2",
        same_hm=True,
    )
    ids = {h["pdb_id"] for h in result["hits"]}
    assert "LYZ1" in ids and "LYZ2" in ids
    assert "ORTH" not in ids
    assert all(h["sg_hm"] == "P 43 21 2" for h in result["hits"])
    assert result["same_hm"] is True
    state.db.close()


def test_same_hm_excludes_other_settings_of_same_number(tmp_path):
    """Same IT number can host several HM settings; same_hm keeps only one."""
    path = tmp_path / "sg18.duckdb"
    db = CellDatabase(str(path))
    # SG 18 settings are all P-centred, so similar conventional cells stay close
    # in root space — unlike C vs I of number 5, which differ after primitive reduction.
    db.add_cell("ASET", (70.0, 90.0, 100.0, 90, 90, 90), 18, "P 21 21 2")
    db.add_cell("BSET", (70.1, 90.1, 100.1, 90, 90, 90), 18, "P 2 21 21")
    db.close()
    state = _PdbSearchServer(str(path))
    result = search_compatible(
        state.db, state.index,
        cell=(70.0, 90.0, 100.0, 90, 90, 90),
        cutoff=5.0,
        sg_number=18,
        sg_hm="P 21 21 2",
        same_hm=True,
    )
    ids = {h["pdb_id"] for h in result["hits"]}
    assert "ASET" in ids
    assert "BSET" not in ids
    any_hm = search_compatible(
        state.db, state.index,
        cell=(70.0, 90.0, 100.0, 90, 90, 90),
        cutoff=5.0,
        sg_number=18,
        sg_hm="P 21 21 2",
        same_hm=False,
    )
    any_ids = {h["pdb_id"] for h in any_hm["hits"]}
    assert "ASET" in any_ids and "BSET" in any_ids
    state.db.close()


def test_search_reports_centering_and_primitive(sample_db):
    state = _PdbSearchServer(sample_db)
    result = search_compatible(
        state.db, state.index,
        cell=(80.0, 90.0, 100.0, 90.0, 90.0, 90.0),
        cutoff=1.0,
        sg_number=5,
        sg_hm="C 1 2 1",
    )
    assert result["centering"] == "C"
    assert result["pipeline"]["reduction"] == "selling_delaunay"
    assert len(result["primitive_cell"]) == 6
    assert result["primitive_cell"] != list(result["cell"])
    state.db.close()


def test_parse_search_params_requires_sg():
    with pytest.raises(ValueError, match="sg is required"):
        _parse_search_params({
            "a": "79", "b": "79", "c": "38",
            "alpha": "90", "beta": "90", "gamma": "90",
            "cutoff": "1.0",
        })


def test_parse_search_params():
    params = _parse_search_params({
        "a": "79", "b": "79", "c": "38",
        "alpha": "90", "beta": "90", "gamma": "90",
        "sg": "P212121",
        "cutoff": "1.0",
        "same_hm": "true",
    })
    assert params["cell"] == (79.0, 79.0, 38.0, 90.0, 90.0, 90.0)
    assert params["sg_number"] == 19
    assert params["sg_hm"] == "P 21 21 21"
    assert params["same_hm"] is True
    # same_sg alias still accepted
    alias = _parse_search_params({
        "cell": [79, 79, 38, 90, 90, 90],
        "sg": 19,
        "cutoff": 1.0,
        "same_sg": "true",
    })
    assert alias["same_hm"] is True
    assert alias["sg_hm"] == "P 21 21 21"


def _fetch(url, data=None):
    if data is None:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_http_endpoints(sample_db):
    state = _PdbSearchServer(sample_db)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        status, health = _fetch(f"{base}/health")
        assert status == 200
        assert health["cells"] == 4

        url = (f"{base}/search?a=79&b=79&c=38&alpha=90&beta=90&gamma=90"
               f"&sg=96&cutoff=0.5")
        status, result = _fetch(url)
        assert status == 200
        assert result["count"] == 2

        status, result = _fetch(
            f"{base}/search",
            {"cell": [79, 79, 38, 90, 90, 90], "sg": 96, "cutoff": 0.5},
        )
        assert status == 200
        assert result["count"] == 2

        with pytest.raises(urllib.error.HTTPError) as exc:
            _fetch(f"{base}/search?a=1&b=2&c=3&alpha=90&beta=90&gamma=90&cutoff=1")
        assert exc.value.code == 400
        body = json.loads(exc.value.read().decode())
        assert "sg is required" in body["error"]
    finally:
        httpd.shutdown()
        state.db.close()
