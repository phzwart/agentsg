"""strain_closure: widened Selling closure in the strain norm, tiers, reindexing, sort spectrum."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from fractions import Fraction

from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.strain_closure import (
    LEPAGE_STRAIN, _data, adjacency_radius, reduce_cell, reindex, sort_spectrum, widened_closure,
)

WORKED = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)
NEEDLE = (5.583, 5.21, 131.965, 90.0, 90.0, 91.118)
PLATE = (100.0, 100.1, 5.0, 90.03, 89.98, 90.05)


def test_shipped_operator_data():
    U, L1, L2, K1, K2 = _data()
    assert len(U) == 6096 and len(K1) == 127 and len(K2) == 2755
    assert np.abs(U).max() == 3 and K1 <= K2


def test_lepage_band():
    assert abs(LEPAGE_STRAIN - np.sqrt(2) * np.sin(np.radians(3))) < 1e-15


def test_tiers_and_sizes():
    W = widened_closure(WORKED)
    assert W.tier == 1 and len(W) == 16
    assert widened_closure((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2").tier == 1
    Wn = widened_closure(NEEDLE)
    assert Wn.tier == 2 and len(Wn) == 56
    Wp = widened_closure(PLATE)
    assert Wp.tier == 2 and len(Wp) == 52
    # the uncertified L1 filter is the tier-1 answer where tier 1 holds, and misses members where it does not
    assert {tuple(map(tuple, m.superbase)) for m in widened_closure(WORKED, certify=False).members} == \
           {tuple(map(tuple, m.superbase)) for m in W.members}
    assert len(widened_closure(NEEDLE, certify=False)) < len(Wn)


def test_adjacency_radius():
    assert abs(adjacency_radius(reduce_cell(NEEDLE).G) - 0.0557) < 2e-4
    assert adjacency_radius(reduce_cell(WORKED).G) == 0.25


def test_reindex_recovers_the_true_operator():
    rng = np.random.default_rng(3)
    for cell in (WORKED, NEEDLE, PLATE, (40.0, 40.3, 80.0, 90, 90, 90)):
        W = widened_closure(cell)
        GR = np.array(UnitCell(*cell).metric_tensor(), float)
        for _ in range(4):
            while True:
                P = rng.integers(-2, 3, (3, 3))
                if round(abs(np.linalg.det(P))) == 1:
                    break
            G = P.T @ GR @ P
            w, V = np.linalg.eigh(G)
            Gh = V @ np.diag(w ** .5) @ V.T
            X = rng.normal(size=(3, 3))
            X = (X + X.T) / 2
            X *= rng.uniform(0.005, 0.06) / np.linalg.norm(X)
            cands, _ = reindex(W, params_from_metric((Gh @ (np.eye(3) + X) @ Gh).tolist()))
            truth = [[Fraction(int(x)) for x in row] for row in P]
            assert any([list(r) for r in c.operator.rows] == truth for c in cands)


def test_sort_spectrum():
    spec = sort_spectrum((40.0, 40.3, 80.0, 90, 90, 90))
    assert spec["O5"] < 1e-9 and abs(spec["T5"] - 0.01057) < 2e-5
    spec = sort_spectrum((100.0, 100.2, 99.9, 90, 90, 90), "F 2 2 2")
    assert abs(spec["C3"] - 0.00432) < 2e-5


def test_indexing_classes_reproduce_the_merohedral_ambiguities():
    from agentsg.cell.strain_closure import indexing_classes

    def relabel(cell):
        G = np.array(UnitCell(*cell).metric_tensor(), float)
        P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        return params_from_metric((P.T @ G @ P * 1.002).tolist())

    expected = {   # (|H_eps|, |Laue|, number of distinct indexings)
        ((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2"): (16, 16, 1),
        ((40.0, 40.3, 80.0, 90, 90, 90), "P 21 21 21"): (16, 8, 2),     # pseudo-tetragonal
        ((50.0, 50.0, 80.0, 90, 90, 120), "P 3"): (24, 6, 4),
        ((50.0, 50.0, 80.0, 90, 90, 120), "P 31 2 1"): (24, 12, 2),
        ((60.0, 60.0, 60.0, 90, 90, 90), "P 21 3"): (48, 24, 2),
        (WORKED, "P 1"): (4, 2, 2),
    }
    for (cell, sg), (h, l, n) in expected.items():
        r = indexing_classes(widened_closure(cell, sg), relabel(cell), sg)
        assert (r.symmetry_order, r.laue_order, len(r.classes)) == (h, l, n), (sg, r)
        assert r.closed and r.laue_in_symmetry
        assert all(c.size == l for c in r.classes)
        assert r.classes[0].representative.price < 1e-6
