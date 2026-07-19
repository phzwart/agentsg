"""Cell comparison tests, reproducing sec 3.2 of Zwart/GK/Adams 2006.

Native P2_1 2_1 2_1  (61.8 97.7 148.9 90 90 90) vs
SeMet1 P2_1          (115.5 149.0 115.6 90 115 90)
must give the index-2 sublattice solution M = [[2,1,0],[0,1,0],[0,0,1]]
resulting in a cell ~115.6 115.6 148.9 90 90 115.4 (the paper's SeMet Niggli cell).
"""
import pytest
from agentsg.cell.compare import compare_cells, CellMatch
from agentsg.cell.metric import UnitCell


NATIVE = (61.8, 97.7, 148.9, 90, 90, 90)
SEMET1 = (115.5, 149.0, 115.6, 90, 115, 90)


def test_paper_native_vs_semet1():
    res = compare_cells(NATIVE, SEMET1, length_tol_pct=3.0, angle_tol_deg=6.0)
    assert res["index"] == 2
    assert abs(res["volume_ratio"] - 2.0) < 0.02
    assert len(res["solutions"]) >= 1
    best = res["solutions"][0]
    assert best.M == ((2, 1, 0), (0, 1, 0), (0, 0, 1))
    # resulting cell matches the paper's SeMet Niggli cell
    a, b, c, al, be, ga = best.resulting_cell
    assert abs(a - 115.6) < 1.0 and abs(b - 115.6) < 1.0 and abs(c - 148.9) < 1.0
    assert abs(al - 90) < 1.0 and abs(be - 90) < 1.0 and abs(ga - 115.4) < 1.0
    # tiny deviations
    assert best.max_length_dev < 1.0
    assert best.max_angle_dev < 1.0


def test_reduced_cells_match_paper():
    res = compare_cells(NATIVE, SEMET1)
    lego = res["lego_cell"]
    target = res["target_cell"]
    # lego is the native orthorhombic reduced cell
    assert abs(lego[0] - 61.8) < 0.5 and abs(lego[2] - 148.9) < 0.5
    # target is the SeMet monoclinic Niggli cell ~115.5 115.6 149 90 90 115
    assert abs(target[5] - 115.0) < 1.0


def test_identical_cells_index_one():
    cell = (50, 60, 70, 90, 90, 90)
    res = compare_cells(cell, cell)
    assert res["index"] == 1
    assert any(s.max_length_dev < 1e-6 and s.max_angle_dev < 1e-6 for s in res["solutions"])


def test_no_relation_when_volumes_incommensurate():
    # volume ratio far from integer -> no sublattice search
    res = compare_cells((50, 50, 50, 90, 90, 90), (57.3, 61.1, 66.9, 90, 90, 90))
    assert res["solutions"] == []


def test_supercell_is_recovered():
    # build an exact index-3 supercell of a triclinic lego and recover it
    base = (10, 11, 12, 88, 92, 103)
    from agentsg.cell.sublattice import generate_sublattices, apply_to_cell
    M = generate_sublattices(3)[4]
    big = apply_to_cell(base, M)
    res = compare_cells(base, big, length_tol_pct=1.0, angle_tol_deg=1.0)
    assert res["index"] == 3
    assert len(res["solutions"]) >= 1
    assert res["solutions"][0].max_length_dev < 1.0


def test_volume_ratio_orientation_independent():
    # comparison is symmetric: A vs B and B vs A give the same lego/target
    r1 = compare_cells(NATIVE, SEMET1)
    r2 = compare_cells(SEMET1, NATIVE)
    assert r1["lego_cell"] == r2["lego_cell"]
    assert r1["target_cell"] == r2["target_cell"]
