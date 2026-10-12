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


def _random_sl3(rng, steps=4):
    """A det +1 integer matrix with small entries.

    Each step adds one column into another. Entries stay small so a 1e-6
    parameter perturbation is not amplified into a large metric error.
    """
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for _ in range(steps):
        i, j = rng.sample((0, 1, 2), 2)
        k = rng.choice((-1, 1))
        for row in M:
            row[i] += k * row[j]
    return M


def _perturb_rel(cell, rng, eps=1e-6):
    return tuple(x * (1.0 + eps * rng.uniform(-1.0, 1.0)) for x in cell)


def test_reindexed_v4_v5_query_keeps_cob():
    """A noisy unimodular reindexing of a V4 or V5 cell still gets an operator.

    The closure is computed on the query. Without a widened zero tolerance the
    reindexed cell looks like V1 and cob is null.
    """
    from fractions import Fraction

    from agentsg.cell.metric import UnitCell
    from agentsg.cell.selling_cob import _metric_of, annotate_search_hits
    from agentsg.cell.sublattice import apply_to_cell
    from agentsg.lattice_symmetry import lattice_symmetry
    from agentsg.linalg import Matrix3

    rng = __import__("random").Random(422)
    # (deposited cell, stored HM, IT number). P centring, so the primitive
    # cell is the cell itself.
    stored = (
        ((79.1, 79.1, 37.9, 90.0, 90.0, 90.0), "P 43 21 2", 96),
        ((40.0, 50.0, 60.0, 90.0, 90.0, 90.0), "P 21 21 21", 19),
        ((80.0, 80.0, 40.0, 90.0, 90.0, 120.0), "P 6 2 2", 177),
    )
    # The reported lysozyme reindexing, plus random det +1 matrices.
    seeds = [
        [[1, 1, 0], [0, 1, 1], [1, 2, 2]],
        _random_sl3(rng),
        _random_sl3(rng),
    ]
    db = CellDatabase(":memory:")
    for n, (cell, hm, number) in enumerate(stored):
        assert db.add_cell(f"C{n}", cell, number, hm)
        M = seeds[n]
        query = _perturb_rel(apply_to_cell(cell, M), rng)
        hits = [{"pdb_id": f"C{n}", "distance": 0.0}]
        annotate_search_hits(db, query, "P 1", hits)
        hit = hits[0]
        assert hit["cob"] is not None, hm
        proper = sum(
            1 for op in lattice_symmetry(cell).operations if op.W.det() > 0
        )
        assert len(hit["cob_coset"]) == proper
        if cell[0] == cell[1] and cell[5] == 90.0:
            assert proper == 8
            assert hit["cob_xyz"] == "(2*a-2*b+c,b-c,a-b+c)"
        P = Matrix3([[Fraction(num, den) for num, den in row] for row in hit["cob"]])
        Gp = _metric_of(UnitCell(*query).metric_tensor(), P)
        Gdep = UnitCell(*cell).metric_tensor()
        # Compare against the largest metric entry. A near-zero component of an
        # orthogonal cell has no relative scale of its own, and the query is
        # already perturbed at 1e-6.
        scale = max(abs(Gdep[i][j]) for i in range(3) for j in range(3))
        for a in range(3):
            for b in range(3):
                assert abs(Gp[a][b] - Gdep[a][b]) <= 1e-5 * scale
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


def _cell():
    return (79.1, 79.1, 37.9, 90, 90, 90)


def _uniprot_rows(db, pdb_id):
    rows = db.sql("SELECT uniprots FROM cells WHERE pdb_id = ?", [pdb_id])
    value = rows[0][0]
    if value is None:
        return None
    return [(item["acc"], item["name"], item["n_res"]) for item in value]


def test_migrate_adds_uniprots_column(tmp_path):
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE cells (pdb_id VARCHAR, r0 DOUBLE, l0 DOUBLE)")
    con.execute("INSERT INTO cells VALUES ('A', 1, 1)")
    con.close()
    db = CellDatabase(str(path))
    cols = {row[0] for row in db.sql("DESCRIBE cells")}
    assert "uniprots" in cols
    assert "deposit_group" in cols
    assert _uniprot_rows(db, "A") is None
    db.close()


def test_backfill_uniprot_lists():
    chain = """# test release
PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END
one\tA\tP00698\t1\t40\t1\t40\t1\t40
two\tA\tP00720\t1\t10\t1\t10\t1\t10
two\tB\tP00720\t1\t30\t1\t\t1\t30
two\tA\tP02185\t1\t25\t1\t25\t1\t25
ord\tA\tP00020\t1\t10\t1\t10\t1\t10
ord\tA\tP00010\t1\t10\t1\t10\t1\t10
noname\tA\tP99999\t1\t5\t1\t5\t1\t5
"""
    names = """Entry\tEntry Name\tProtein names
P00698\tLYSC_CHICK\tLysozyme C (EC 3.2.1.17) (Allergen Gal d 4)
P00720\tENLYS_BPT4\tEndolysin (EC 3.2.1.17)
P02185\tMYG_PHYCA\tMyoglobin
P00010\tEARLY_TEST\tEarlier protein
P00020\tLATE_TEST\tLater protein
"""
    db = CellDatabase(":memory:")
    for pdb_id in ("ONE", "TWO", "ORD", "NONAME", "NONE"):
        db.add_cell(pdb_id, _cell(), 96, "P 43 21 2")
    n = db.backfill_uniprot(chain_uniprot=chain, names=names, source="r1")
    assert n == 5
    assert _uniprot_rows(db, "ONE") == [("P00698", "Lysozyme C", 40)]
    assert _uniprot_rows(db, "TWO") == [
        ("P00720", "Endolysin", 40),
        ("P02185", "Myoglobin", 25),
    ]
    assert _uniprot_rows(db, "ORD") == [
        ("P00010", "Earlier protein", 10),
        ("P00020", "Later protein", 10),
    ]
    assert _uniprot_rows(db, "NONAME") == [("P99999", None, 5)]
    assert _uniprot_rows(db, "NONE") == []
    assert db.sql("SELECT value FROM schema_meta WHERE key = 'uniprot_source'") == [("r1",)]

    db.add_cell("NEW", _cell(), 96, "P 43 21 2")
    assert _uniprot_rows(db, "NEW") is None
    changed = """# later
PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END
one\tA\tP00998\t1\t5
new\tA\tP00007\t1\t12
"""
    changed_names = """Entry\tEntry Name\tProtein names
P00007\tNEW_TEST\tNew protein
P00998\tOTHER_TEST\tOther protein
"""
    n = db.backfill_uniprot(chain_uniprot=changed, names=changed_names, source="r2")
    assert n == 1
    assert _uniprot_rows(db, "ONE") == [("P00698", "Lysozyme C", 40)]
    assert _uniprot_rows(db, "NEW") == [("P00007", "New protein", 12)]
    assert _uniprot_rows(db, "NONE") == []

    n = db.backfill_uniprot(
        chain_uniprot=changed, names=changed_names, source="r3", refresh=True,
    )
    assert n == 6
    assert _uniprot_rows(db, "ONE") == [("P00998", "Other protein", 5)]
    assert db.sql("SELECT value FROM schema_meta WHERE key = 'uniprot_source'") == [("r3",)]
    db.close()


def test_pick_deposit_group_ignores_other_memberships():
    from agentsg.cell.deposit_group import (
        deposit_groups_from_entries,
        deposition_group_label,
        pick_deposit_group,
    )

    memberships = [
        {"aggregation_method": "matching_uniprot_accession", "group_id": "P00698"},
        {"aggregation_method": "matching_deposit_group_id", "group_id": "G_1002344"},
        {"aggregation_method": "matching_deposit_group_id", "group_id": "G_1000001"},
    ]
    assert pick_deposit_group(memberships) == "G_1000001"
    assert pick_deposit_group(None) == ""
    assert deposition_group_label("G_1002344") == "DEPOSITION_GROUP_G_1002344"
    found = deposit_groups_from_entries(
        [
            {"rcsb_id": "7i66", "rcsb_entry_group_membership": memberships[:2]},
            {"rcsb_id": "1LYZ", "rcsb_entry_group_membership": None},
        ],
        ["7I66", "1LYZ", "GONE"],
    )
    assert found == {"7I66": "G_1002344", "1LYZ": "", "GONE": ""}


def test_backfill_deposit_group_is_independent_of_uniprot():
    chain = """PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END
one\tA\tP00698\t1\t40
"""
    names = """Entry\tEntry Name\tProtein names
P00698\tLYSC_CHICK\tLysozyme C
"""
    db = CellDatabase(":memory:")
    for pdb_id in ("ONE", "NONE", "BARE"):
        db.add_cell(pdb_id, _cell(), 96, "P 43 21 2")
    db.backfill_uniprot(chain_uniprot=chain, names=names, source="r")
    db.add_cell("NEW", _cell(), 96, "P 43 21 2")
    assert _uniprot_rows(db, "NEW") is None
    groups = {"NONE": "G_1002344", "BARE": "", "ONE": "G_9", "NEW": "G_7"}
    n = db.backfill_deposit_group(groups=groups, source="test")
    assert n == 4
    assert _uniprot_rows(db, "ONE") == [("P00698", "Lysozyme C", 40)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='ONE'") == [("G_9",)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='NONE'") == [("G_1002344",)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='BARE'") == [("",)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='NEW'") == [("G_7",)]
    assert db.backfill_deposit_group(groups={"NONE": "G_1"}, source="again") == 0
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='ONE'") == [("G_9",)]
    n = db.backfill_deposit_group(
        groups={"NONE": "G_2", "BARE": "G_3", "ONE": "G_8", "NEW": "G_7"},
        source="refresh", refresh=True,
    )
    assert n == 4
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='NONE'") == [("G_2",)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='BARE'") == [("G_3",)]
    assert db.sql("SELECT deposit_group FROM cells WHERE pdb_id='ONE'") == [("G_8",)]
    assert _uniprot_rows(db, "ONE") == [("P00698", "Lysozyme C", 40)]
    assert db.sql("SELECT value FROM schema_meta WHERE key='deposit_group_source'") == [("refresh",)]
    db.close()


def test_backfill_uniprot_sums_segments():
    chain = """PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END
ab\tA\tP00698\t1\t100\t1\t100\t1\t100
ab\tA\tP00698\t101\t110\t101\t110\t101\t110
ab\tB\tP00698\t1\t40\t1\t40\t1\t40
ab\tA\tP00720\t\t\t\t\t1\t50
"""
    names = """P00698\tLysozyme C
P00720\tEndolysin
"""
    db = CellDatabase(":memory:")
    db.add_cell("AB", _cell(), 96, "P 43 21 2")
    db.backfill_uniprot(chain_uniprot=chain, names=names, source="chains")
    # Chain A is two segments, 100 + 10, and chain B adds 40 of the same accession.
    assert _uniprot_rows(db, "AB") == [
        ("P00698", "Lysozyme C", 150),
        ("P00720", "Endolysin", 50),
    ]
    db.close()
