"""Reindexing coset + twin-law tests.

A reindexing operator between two settings of the same lattice is never unique:
the valid operators form the coset P.H where H is the lattice symmetry group.
Twin laws are the cosets of the crystal Laue group in the lattice holohedry.
"""
import math
import pytest
from agentsg.cell.reindex import (
    reindexing_operators, reindexing_operator, twin_laws, _transform_metric,
    _matmul, _int_inv,
)
from agentsg.cell.metric import UnitCell
from agentsg.cell.rootform import root_distance


def _cell_of(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)),
            ang(G[0][1] / (a * b)))


def test_int_inverse():
    M = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    Minv = _int_inv(M)
    prod = _matmul(M, Minv)
    assert prod == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_reindexing_coset_size_equals_holohedry():
    """The reindexing coset has |H| operators (H = lattice symmetry group)."""
    base = (50.0, 50.0, 70.0, 90, 90, 90)              # tetragonal, |H|=16
    G = UnitCell(*base).metric_tensor()
    B = _cell_of(_transform_metric(G, ((0, 1, 0), (1, 0, 0), (0, 0, 1))))
    ops = reindexing_operators(base, B)
    assert len(ops) == 16
    # every operator maps A onto B
    for P in ops:
        assert root_distance(_cell_of(_transform_metric(G, P)), B) < 1e-6


def test_reindexing_contains_true_operator():
    base = (40.0, 50.0, 60.0, 85, 95, 100)             # triclinic, |H|=2
    G = UnitCell(*base).metric_tensor()
    P_true = ((1, 1, 0), (0, 1, 0), (0, 0, 1))
    B = _cell_of(_transform_metric(G, P_true))
    ops = set(reindexing_operators(base, B))
    assert P_true in ops
    assert len(ops) == 2                               # triclinic: {P, -P}


def test_reindexing_none_for_different_lattices():
    a = (40.0, 50.0, 60.0, 90, 90, 90)
    b = (41.0, 55.0, 67.0, 88, 93, 97)                 # unrelated
    assert reindexing_operator(a, b) is None


def test_reindexing_single_representative():
    base = (50.0, 50.0, 70.0, 90, 90, 90)
    G = UnitCell(*base).metric_tensor()
    B = _cell_of(_transform_metric(G, ((0, 1, 0), (1, 0, 0), (0, 0, 1))))
    P = reindexing_operator(base, B)
    assert P is not None
    assert root_distance(_cell_of(_transform_metric(G, P)), B) < 1e-6


@pytest.mark.parametrize("sgkey,cell,n_domains", [
    ("P4", (50, 50, 70, 90, 90, 90), 2),        # tetragonal merohedral twin
    ("P212121", (40, 50, 60, 90, 90, 90), 1),   # orthorhombic: no merohedral twin
    ("P1", (40, 50, 60, 85, 95, 100), 1),       # triclinic: none
    ("P3", (50, 50, 70, 90, 90, 120), 4),       # trigonal on hexagonal: index 4
    ("P6", (50, 50, 70, 90, 90, 120), 2),       # hexagonal Laue 6/m in 6/mmm
])
def test_twin_law_counts(sgkey, cell, n_domains):
    tl = twin_laws(sgkey, cell)
    assert len(tl) == n_domains
    # first representative is always the identity (untwinned domain)
    assert tl[0] == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_twin_law_operators_are_lattice_symmetries():
    """Each twin law must be a metric symmetry of the lattice (M^T G M ~ G)."""
    cell = (50, 50, 70, 90, 90, 90)
    G = UnitCell(*cell).metric_tensor()
    for M in twin_laws("P4", cell):
        assert root_distance(_cell_of(_transform_metric(G, M)), cell) < 1e-6
