"""Static geo 0.1 site: plane neighbours and a tiny archive."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agentsg.explorer.build_static import LIMIT, _write, build, plane_neighbors


def test_plane_neighbors_drop_self_and_rank_by_distance():
    xy = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=np.float32)
    idx, dist = plane_neighbors(xy, 2)
    assert set(idx[0].tolist()) == {1, 2}
    assert set(idx[3].tolist()) == {1, 2}
    assert dist[0, 0] <= dist[0, 1]
    assert idx[0, 0] != 0


def test_write_rejects_a_file_at_the_pages_limit(tmp_path):
    with pytest.raises(SystemExit):
        _write(tmp_path / "big.bin", b"x" * LIMIT)


def test_build_writes_a_hashed_site(tmp_path):
    plane = tmp_path / "cells.npz"
    cell = (50.0, 40.0, 30.0, 90.0, 90.0, 90.0)
    np.savez_compressed(
        plane,
        xy=np.array([[0, 0], [0.1, 0], [1, 1]], dtype=np.float32),
        pdb_id=np.array(["AAAA", "BBBB", "CCCC"], dtype="U4"),
        a=np.array([cell[0]] * 3),
        b=np.array([cell[1]] * 3),
        c=np.array([cell[2]] * 3),
        alpha=np.array([cell[3]] * 3),
        beta=np.array([cell[4]] * 3),
        gamma=np.array([cell[5]] * 3),
        sg_number=np.array([1, 19, 96], dtype=np.int32),
        sg_hm=np.array(["P 1", "P 21 21 21", "P 43 21 2"], dtype="U32"),
        deposit_group=np.array(["", "G1", ""], dtype=object),
        uniprots=np.array([
            [("P12345", "Lysozyme", 129)],
            [],
            [("P12345", "Lysozyme", 129), ("Q99999", "Other", 40)],
        ], dtype=object),
        snapshot=np.array("test-snapshot"),
    )
    out = tmp_path / "site"
    manifest = build(plane, out, k=2)
    assert manifest["n"] == 3
    assert manifest["k"] == 2
    assert manifest["plane"] == "geo 0.1"
    data = out / "data"
    for role, records in manifest["files"].items():
        for record in records:
            blob = (data / record["name"]).read_bytes()
            assert len(blob) < LIMIT
            assert len(blob) == record["bytes"]
    meta = json.loads((data / "meta.json").read_text())
    assert meta["pdb_id"] == ["AAAA", "BBBB", "CCCC"]
    assert meta["deposit"][1] == "DEPOSITION_GROUP_G1"
    assert meta["uniAcc"][0] == "P12345"
    assert meta["uniAcc"][1] == ""
    page = Path(__file__).resolve().parents[2] / "site" / "index.html"
    text = page.read_text()
    assert "lasso" in text
    assert "cellfig" in text
