"""strain_closure: widened Selling closure in the strain norm, layers, reindexing, sort spectrum."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from fractions import Fraction

from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.strain_closure import (
    LEPAGE_PRICE, LEPAGE_STRAIN, _data, adjacency_radius, band_for, price_spectrum, reduce_cell, reindex,
    sort_spectrum, widened_closure,
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
    assert LEPAGE_PRICE == 2 * LEPAGE_STRAIN
    assert abs(band_for(0.074) - 0.074 / 0.926) < 1e-12


def test_layers_and_sizes():
    W = widened_closure(WORKED)
    assert W.layer == 1 and W.tier == 1 and len(W) == 16
    assert widened_closure((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2").layer == 1
    Wn = widened_closure(NEEDLE)
    assert Wn.layer == 2 and len(Wn) == 56
    Wp = widened_closure(PLATE)
    assert Wp.layer == 2 and len(Wp) == 52
    # the uncertified L1 filter is the layer-1 answer where layer 1 holds, and misses members where it does not
    assert {tuple(map(tuple, m.superbase)) for m in widened_closure(WORKED, certify=False).members} == \
           {tuple(map(tuple, m.superbase)) for m in W.members}
    assert len(widened_closure(NEEDLE, certify=False)) < len(Wn)


def test_adjacency_radius():
    assert abs(adjacency_radius(reduce_cell(NEEDLE).G) - 0.0557) < 2e-4
    assert adjacency_radius(reduce_cell(WORKED).G) == 0.25


def test_reindex_recovers_the_true_operator():
    rng = np.random.default_rng(3)
    for cell in (WORKED, NEEDLE, PLATE, (40.0, 40.3, 80.0, 90, 90, 90)):
        W = widened_closure(cell, eps=band_for(LEPAGE_STRAIN))
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
            new = params_from_metric((Gh @ (np.eye(3) + X) @ Gh).tolist())
            cands, _ = reindex(W, new, accept=LEPAGE_STRAIN)
            truth = [[Fraction(int(x)) for x in row] for row in P]
            assert any([list(r) for r in c.operator.rows] == truth for c in cands)
            det = int(round(np.linalg.det(P)))
            proper, _ = reindex(W, new, accept=LEPAGE_STRAIN, proper=True)
            assert all(c.determinant == 1 for c in proper)
            assert (any([list(r) for r in c.operator.rows] == truth for c in proper)) == (det == 1)
    with pytest.raises(ValueError):
        reindex(widened_closure(WORKED), WORKED, accept=0.09)            # needs a wider reference


def test_scale_fit_and_scale_invariance():
    W = widened_closure(WORKED)
    G = np.array(UnitCell(*WORKED).metric_tensor(), float)
    cands, _ = reindex(W, params_from_metric((1.02 * G).tolist()))      # a 1 % error in the length scale
    assert cands[0].price < 1e-9 and abs(cands[0].scale - 1.02) < 1e-9  # removed by the scale fit
    assert len(widened_closure(tuple(1.3 * x for x in WORKED[:3]) + WORKED[3:])) == len(W)   # S_eps(sG) = S_eps(G)


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
        W = widened_closure(cell, sg, eps=band_for(LEPAGE_STRAIN))      # widened at 0.080, accepted at 0.074
        r = indexing_classes(W, relabel(cell), sg, accept=LEPAGE_STRAIN)
        assert (r.symmetry_order, r.group_order, len(r.classes)) == (h, l, n), (sg, r)
        assert r.closed and r.group_in_symmetry
        assert all(c.size == l for c in r.classes)
        assert r.classes[0].representative.price < 1e-6
    # anomalous data: the point group replaces the Laue group. For a chiral group the improper
    # candidates are dropped (proper=True) and the count does not change; for an acentric
    # non-chiral group the inversion separates indexings and the count doubles.
    cases = {((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2"): (True, 1, 1),
             ((50.0, 50.0, 80.0, 90, 90, 120), "P 3"): (True, 4, 4),
             ((40.0, 40.0, 70.0, 90, 90, 90), "P 4 m m"): (None, 1, 2),
             ((50.0, 50.0, 80.0, 90, 90, 120), "P 3 1 m"): (None, 2, 4)}
    for (cell, sg), (proper, n_merge, n_anom) in cases.items():
        W = widened_closure(cell, sg, eps=band_for(LEPAGE_STRAIN))
        r0 = indexing_classes(W, relabel(cell), sg, accept=LEPAGE_STRAIN, proper=proper)
        r1 = indexing_classes(W, relabel(cell), sg, accept=LEPAGE_STRAIN, proper=proper, anomalous=True)
        assert (len(r0.classes), len(r1.classes)) == (n_merge, n_anom) and r1.anomalous, sg
    # chiral: improper candidates are dropped, which halves H and leaves the class count
    W = widened_closure((50.0, 50.0, 80.0, 90, 90, 120), "P 3", eps=band_for(LEPAGE_STRAIN))
    r = indexing_classes(W, relabel((50.0, 50.0, 80.0, 90, 90, 120)), "P 3", accept=LEPAGE_STRAIN, proper=True)
    assert r.symmetry_order == 12 and len(r.classes) == 4


def test_price_spectrum_and_the_twofold_rule():
    """a pseudo-twofold's price is twice its sort distance (Proposition A9); the price
    spectrum of the worked cell has its gap between the pseudo-mP and pseudo-oP twofolds."""
    spec = price_spectrum(widened_closure(WORKED, eps=0.2))
    assert max(spec[:2]) < 1e-12                                           # +-I
    assert abs(spec[2] - 2 * sort_spectrum(WORKED)["M4"]) < 1e-4 and abs(spec[2] - spec[3]) < 1e-12
    assert abs(spec[4] - 2 * sort_spectrum(WORKED)["O5"]) < 2e-3            # the pseudo-oP twofolds
