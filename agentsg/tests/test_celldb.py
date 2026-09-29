"""celldb tests. DB tests need duckdb; PDB-fetch tests need network (skipped if
either is unavailable)."""
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("scipy")

from agentsg.cell.celldb import CellDatabase


def test_add_and_query_in_memory():
    db = CellDatabase(":memory:")
    # a small synthetic set incl. the lysozyme tetragonal family
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("LYZ2", (79.0, 79.0, 38.0, 90, 90, 90), 96, "P 43 21 2")
    db.add_cell("ORTH", (59.1, 68.5, 30.5, 90, 90, 90), 19, "P 21 21 21")
    assert len(db) == 3
    row = db.sql("SELECT r0,s0 FROM cells WHERE pdb_id='LYZ1'")[0]
    assert row[1] is not None
    assert row[0] != row[1] or row[0] == 0
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


def test_add_cell_stores_one_selling_reduced_cell():
    """red_* is the obtuse basis of the primitive lattice, not the deposited cell."""
    from fractions import Fraction

    from agentsg.cell.metric import UnitCell
    from agentsg.cell.selling_cob import COB_COLUMNS, _metric_of, parse_cob_columns

    conv = (79.723, 90.46, 69.404, 90.0, 98.19, 90.0)
    hm = "C 1 2 1"
    db = CellDatabase(":memory:")
    assert db.add_cell("10GS", conv, 5, hm)
    row = db.sql(
        "SELECT red_a, red_b, red_c, red_alpha, red_beta, red_gamma "
        "FROM cells WHERE pdb_id='10GS'"
    )[0]
    cob_row = db.sql(
        "SELECT " + ", ".join(COB_COLUMNS) + " FROM cells WHERE pdb_id='10GS'"
    )[0]
    P = parse_cob_columns(cob_row)
    assert P.det() in (Fraction(1, 2), Fraction(-1, 2))
    G_red = _metric_of(UnitCell(*conv).metric_tensor(), P)
    G_stored = UnitCell(*row).metric_tensor()
    for a in range(3):
        for b in range(3):
            assert G_red[a][b] == pytest.approx(G_stored[a][b], rel=1e-8, abs=1e-6)
    assert abs(row[0] - conv[0]) > 1.0
    db.close()


def test_selling_reduced_cell_is_obtuse():
    from agentsg.cell.rootform import selling_reduced_cell

    red = selling_reduced_cell((10.0, 12.0, 14.0, 70.0, 80.0, 60.0))
    assert min(red[3], red[4], red[5]) >= 90.0 - 1e-6


def test_reference_orbit_cob_maps_settings():
    """Composed COB carries the reference metric onto another setting of it."""
    from agentsg.cell.metric import UnitCell
    from agentsg.cell.selling_cob import _metric_of, cob_xyz, match_operators, reference_orbit
    from agentsg.cell.sublattice import apply_to_cell

    base = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    swapped = apply_to_cell(base, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    hm = "P 21 21 21"
    db = CellDatabase(":memory:")
    assert db.add_cell("BASE", base, 19, hm)
    assert db.add_cell("SWAP", swapped, 19, hm)
    orbit = reference_orbit(base, hm)
    rec = db.lookup_reductions(["SWAP"])["SWAP"]
    ops = match_operators(orbit, rec["red"], rec["cob"])
    assert ops
    matrices = [P for P, _res in ops]
    assert all(P.det() > 0 for P in matrices)
    assert cob_xyz(matrices[0]).count("-") == min(cob_xyz(P).count("-") for P in matrices)
    base_rec = db.lookup_reductions(["BASE"])["BASE"]
    self_ops = match_operators(orbit, base_rec["red"], base_rec["cob"])
    assert cob_xyz(self_ops[0][0]) == "(a,b,c)"
    assert self_ops[0][1] == pytest.approx(0.0, abs=1e-6)
    G_swap = UnitCell(*swapped).metric_tensor()
    for P, _res in ops:
        Gp = _metric_of(UnitCell(*base).metric_tensor(), P)
        for a in range(3):
            for b in range(3):
                assert Gp[a][b] == pytest.approx(G_swap[a][b], rel=1e-8, abs=1e-6)
    db.close()


def test_cob_coset_keeps_only_proper_rotations():
    """A chiral reindexing must not lead with a determinant −1 setting."""
    from agentsg.cell.selling_cob import cob_xyz, match_operators, reference_orbit
    from agentsg.cell.sublattice import apply_to_cell

    base = (40.96, 18.65, 22.52, 90.0, 90.77, 90.0)
    swapped = apply_to_cell(base, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    hm = "P 1 21 1"
    db = CellDatabase(":memory:")
    assert db.add_cell("SELF", base, 4, hm)
    assert db.add_cell("SWAP", swapped, 4, hm)
    orbit = reference_orbit(base, hm)
    for pid in ("SELF", "SWAP"):
        rec = db.lookup_reductions([pid])[pid]
        ops = match_operators(orbit, rec["red"], rec["cob"])
        assert ops
        assert all(P.det() > 0 for P, _res in ops)
        spellings = [cob_xyz(P) for P, _res in ops]
        assert "(c,b,a)" not in spellings
        assert "(-c,b,-a)" not in spellings
    swap = match_operators(
        orbit,
        db.lookup_reductions(["SWAP"])["SWAP"]["red"],
        db.lookup_reductions(["SWAP"])["SWAP"]["cob"],
    )
    assert cob_xyz(swap[0][0]) in ("(c,-b,a)", "(-c,-b,-a)")
    db.close()


def test_cob_matches_a_close_reduced_cell_and_reports_residual():
    """1JXU-sized noise still gets an operator; a 1% cell does not."""
    from agentsg.cell.selling_cob import annotate_search_hits, reference_orbit

    base = (40.96, 18.65, 22.52, 90.0, 90.77, 90.0)
    near = (40.90, 18.59, 22.40, 90.0, 90.80, 90.0)
    far = (42.0, 19.5, 24.0, 90.0, 92.0, 90.0)
    hm = "P 1 21 1"
    db = CellDatabase(":memory:")
    assert db.add_cell("NEAR", near, 4, hm)
    assert db.add_cell("FAR", far, 4, hm)
    hits = [
        {"pdb_id": "NEAR", "distance": 0.2},
        {"pdb_id": "FAR", "distance": 2.0},
    ]
    annotate_search_hits(db, base, hm, hits)
    by_id = {hit["pdb_id"]: hit for hit in hits}
    assert by_id["NEAR"]["cob"] is not None
    assert by_id["NEAR"]["cob_residual"] < 0.75
    assert by_id["NEAR"]["cob_xyz"]
    assert by_id["FAR"]["cob"] is None
    assert by_id["FAR"]["cob_residual"] is None
    assert reference_orbit(base, hm).labeled
    db.close()


def test_backfill_selling_cells():
    db = CellDatabase(":memory:")
    db.add_cell("LYZ1", (79.1, 79.1, 37.9, 90, 90, 90), 96, "P 43 21 2")
    db.sql("UPDATE cells SET red_a=NULL WHERE pdb_id='LYZ1'")
    assert db.backfill_selling_cells() == 1
    row = db.sql("SELECT red_a, red_b, red_c FROM cells WHERE pdb_id='LYZ1'")[0]
    assert row[0] == pytest.approx(79.1)
    assert row[1] == pytest.approx(79.1)
    assert row[2] == pytest.approx(37.9)
    assert db.backfill_selling_cells() == 0
    db.close()
