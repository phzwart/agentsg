"""celldb tests. DB tests need duckdb; PDB-fetch tests need network (skipped if
either is unavailable)."""
import pytest

duckdb = pytest.importorskip("duckdb")

from agentsg.cell.celldb import CellDatabase


def test_add_and_query_in_memory():
    db = CellDatabase(":memory:")
    # a small synthetic set incl. the lysozyme tetragonal family
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("LYZ2", (79.0, 79.0, 38.0, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("ORTH", (59.1, 68.5, 30.5, 90, 90, 90), 19, "P 21 21 21")
    assert len(db) == 3
    res = db.nearest((79.0, 79.0, 38.0, 90, 90, 90), k=3)
    ids = [pid for pid, _ in res]
    # the two lysozyme cells rank first, orthorhombic last
    assert ids[0] in ("LYZ1", "LYZ2") and ids[1] in ("LYZ1", "LYZ2")
    assert ids[-1] == "ORTH"
    db.close()


def test_sg_prefilter():
    db = CellDatabase(":memory:")
    db.add_cell("A", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("B", (59.1, 68.5, 30.5, 90, 90, 90), 19, "P 21 21 21")
    res = db.nearest((79.0, 79.0, 38.0, 90, 90, 90), k=5, sg_number=96)
    assert [pid for pid, _ in res] == ["A"]
    db.close()


def test_volume_band_prefilter():
    db = CellDatabase(":memory:")
    db.add_cell("small", (30, 30, 30, 90, 90, 90), 195, "P 2 3")
    db.add_cell("big", (300, 300, 300, 90, 90, 90), 195, "P 2 3")
    res = db.nearest((31, 31, 31, 90, 90, 90), k=5, volume_band=0.3)
    ids = [pid for pid, _ in res]
    assert "small" in ids and "big" not in ids
    db.close()


def test_nearest_with_supercells_identifies_relations():
    """Volume-spanning search finds isometric, supercell (x2, x3) relations and
    labels each with the correct index and sublattice matrix."""
    from agentsg.cell.sublattice import apply_to_cell
    base = (40.0, 50.0, 60.0, 90, 90, 90)
    sup2 = apply_to_cell(base, [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    sup3 = apply_to_cell(base, [[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    iso = (50.0, 40.0, 60.0, 90, 90, 90)                # axis swap -> isometric
    db = CellDatabase(":memory:")
    db.add_cell("ISO", iso, 16, "ortho")
    db.add_cell("SUP2", sup2, 16, "x2")
    db.add_cell("SUP3", sup3, 16, "x3")
    db.add_cell("UNREL", (37, 44, 71, 88, 97, 101), 2, "unrel")
    res = {r["pdb_id"]: r for r in db.nearest_with_supercells(base, k=8, max_index=3)}
    assert res["ISO"]["index"] == 1 and res["ISO"]["relation"] == "isometric"
    assert res["ISO"]["distance"] < 1e-6
    assert res["SUP2"]["index"] == 2 and res["SUP2"]["relation"] == "db_is_supercell"
    assert res["SUP2"]["distance"] < 1e-3
    assert res["SUP3"]["index"] == 3 and res["SUP3"]["relation"] == "db_is_supercell"
    assert "UNREL" not in res                           # unrelated cell excluded
    db.close()


def test_nearest_with_supercells_finds_sublattice_direction():
    """If the DB holds a SMALLER cell that the query is a supercell of, the query
    search reports it as db_is_sublattice."""
    from agentsg.cell.sublattice import apply_to_cell
    small = (40.0, 50.0, 60.0, 90, 90, 90)
    query = apply_to_cell(small, [[1, 0, 0], [0, 1, 0], [0, 0, 2]])   # query = 2x small
    db = CellDatabase(":memory:")
    db.add_cell("SMALL", small, 16, "ortho")
    res = {r["pdb_id"]: r for r in db.nearest_with_supercells(query, k=5, max_index=3)}
    assert "SMALL" in res
    assert res["SMALL"]["index"] == 2
    assert res["SMALL"]["relation"] == "db_is_sublattice"
    db.close()
