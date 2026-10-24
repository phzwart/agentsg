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
    assert abs(LEPAGE_PRICE - 2 * np.sqrt(2) * np.arctanh(np.sin(np.radians(3)))) < 1e-15
    assert abs(band_for(0.074) - np.expm1(0.074)) < 1e-15


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
        assert all(c.determinant == 1 for c in reindex(W, cell, accept=LEPAGE_STRAIN)[0])   # proper by default
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
            cands, _ = reindex(W, new, accept=LEPAGE_STRAIN, proper=None)
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

    expected = {   # (|H_eps| proper, |Laue|, number of distinct indexings)
        ((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2"): (8, 16, 1),
        ((40.0, 40.3, 80.0, 90, 90, 90), "P 21 21 21"): (8, 8, 2),     # pseudo-tetragonal
        ((50.0, 50.0, 80.0, 90, 90, 120), "P 3"): (12, 6, 4),
        ((50.0, 50.0, 80.0, 90, 90, 120), "P 31 2 1"): (12, 12, 2),
        ((60.0, 60.0, 60.0, 90, 90, 90), "P 21 3"): (24, 24, 2),
        (WORKED, "P 1"): (2, 2, 2),
    }
    for (cell, sg), (h, l, n) in expected.items():
        W = widened_closure(cell, sg, eps=band_for(LEPAGE_STRAIN))      # widened at 0.080, accepted at 0.074
        r = indexing_classes(W, relabel(cell), sg, accept=LEPAGE_STRAIN)
        assert (r.symmetry_order, r.group_order, len(r.classes)) == (h, l, n), (sg, r)
        assert r.closed and r.group_in_symmetry
        assert all(c.size == l // 2 for c in r.classes)                 # the proper half of each class
        assert all(c.representative.determinant == 1 for c in r.classes)
        assert r.classes[0].representative.price < 1e-6
    # anomalous data: the point group replaces the Laue group. Only proper operators are
    # returned (the default); for a chiral group the count does not change, for an acentric
    # non-chiral group the inversion separates indexings and the count doubles.
    cases = {((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2"): (True, 1, 1),
             ((50.0, 50.0, 80.0, 90, 90, 120), "P 3"): (True, 4, 4),
             ((40.0, 40.0, 70.0, 90, 90, 90), "P 4 m m"): (True, 1, 2),
             ((50.0, 50.0, 80.0, 90, 90, 120), "P 3 1 m"): (True, 2, 4)}
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
    spec = price_spectrum(widened_closure(WORKED, eps=0.2), proper=None)
    assert max(spec[:2]) < 1e-12                                           # +-I
    assert abs(spec[2] / (2 * sort_spectrum(WORKED)["M4"]) - 1) < 0.01 and abs(spec[2] - spec[3]) < 1e-12
    assert abs(spec[4] / (2 * sort_spectrum(WORKED)["O5"]) - 1) < 0.03      # the pseudo-oP twofolds (geodesic vs linearised)
    proper = price_spectrum(widened_closure(WORKED, eps=0.2))
    assert len(proper) == len(spec) // 2 and max(proper[:1]) < 1e-12


def test_strain_from_esds_and_certified_spectrum():
    from agentsg.cell.strain_closure import certified_sort_spectrum, strain_from_esds
    # 0.1 % on each edge and 0.05 degrees on each angle of a 30 x 40 x 50 A cell
    r = strain_from_esds(WORKED, (0.03, 0.04, 0.05, 0.05, 0.05, 0.05))
    assert abs(r - 0.00407) < 1e-4                  # sqrt(3 x 0.002^2 + 3 x (sqrt2 x 0.05 deg)^2)
    spec, d1 = certified_sort_spectrum(NEEDLE)
    assert abs(d1 - 0.0557) < 2e-4


def test_scale_bound_and_volume_check():
    """a doubled cell is not a reindexing: the fitted scale is bounded, and primitive cells of
    different volume are refused with an explanation (a wrong centring symbol)."""
    W = widened_closure((60.0, 60.0, 60.0, 90, 90, 90), "P 21 3")
    with pytest.raises(ValueError, match="volume"):
        reindex(W, (120.0, 120.0, 120.0, 90, 90, 90))                    # a superlattice
    G = np.array(UnitCell(60.0, 60.0, 60.0, 90, 90, 90).metric_tensor(), float)
    big = params_from_metric((1.15 ** 2 * G).tolist())                     # 15 % longer edges, same shape
    assert reindex(W, big, max_scale_change=None)[0]
    with pytest.raises(ValueError, match="volume"):
        reindex(W, big)                                                    # beyond the 10 % scale limit
    Wc = widened_closure((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1")
    with pytest.raises(ValueError, match="centring"):
        reindex(Wc, (50.0, 30.0, 40.0, 90, 105, 90), None)                # the C cell read as primitive
    assert reindex(Wc, (50.0, 30.0, 40.0, 90, 105, 90))[0]               # default: the reference's centring


def test_proper_group_order():
    from agentsg.cell.strain_closure import indexing_classes
    W = widened_closure((79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2")
    r = indexing_classes(W, (79.1, 79.1, 37.9, 90, 90, 90), "P 43 21 2")
    assert (r.group_order, r.proper_group_order, r.symmetry_order) == (16, 8, 8)


def test_centring_and_input_checks():
    """R 3 :R is primitive; R 3 :R and R 3 :H settings of one lattice reindex; unreadable
    symbols and malformed cells raise; the new cell inherits the reference's centring."""
    from agentsg.cell.strain_closure import indexing_classes, reduce_cell
    from agentsg.cell.selling_cob import _PRIM_EXACT
    hexa = (80.0, 80.0, 120.0, 90, 90, 120)
    T = np.array([[float(x) for x in r] for r in _PRIM_EXACT["R"]])
    Gh = np.array(UnitCell(*hexa).metric_tensor(), float)
    rcell = params_from_metric((T.T @ Gh @ T).tolist())                    # the rhombohedral axes
    v_h = np.sqrt(np.linalg.det(reduce_cell(hexa, "R 3 :H").G))
    v_r = np.sqrt(np.linalg.det(reduce_cell(rcell, "R 3 :R").G))
    assert abs(v_h / v_r - 1) < 1e-6
    W = widened_closure(hexa, "R 3 :H")
    assert reindex(W, rcell, "R 3 :R")[0]
    for bad in [(10, 20, 30, 90, 90), (-10, 20, 30, 90, 90, 90),
                (10, 20, 30, 90, 90, 190), (10, 20, 30, 10, 10, 170)]:
        with pytest.raises(ValueError):
            reduce_cell(bad)
    with pytest.warns(UserWarning, match="radians"):
        reduce_cell((10, 20, 30, 1.57, 1.57, 1.57))
    with pytest.raises(ValueError, match="centring"):
        reduce_cell((10, 20, 30, 90, 90, 90), "Q 2")
    Wc = widened_closure((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1")
    r = indexing_classes(Wc, (50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1")
    assert len(r.classes) == 1


def test_ball_guard_and_universal_operators():
    """a thin needle at a large band is refused quickly; the 6096 column operators of U,
    used as B_R W B_N^-1, contain the true operator between two reduced cells."""
    from agentsg.cell.strain_closure import universal_operators, reduce_cell
    with pytest.raises(ValueError, match="anisotropic"):
        widened_closure((3.0, 3.0, 800.0, 90, 90, 90), eps=0.2)
    U = universal_operators()
    assert U.shape == (6096, 3, 3) and np.abs(U).max() == 3
    rng = np.random.default_rng(3)
    cell = (30.0, 40.0, 50.0, 90, 100, 90)
    G = np.array(UnitCell(*cell).metric_tensor(), float)
    for _ in range(20):
        while True:
            M = rng.integers(-2, 3, (3, 3))
            if round(np.linalg.det(M)) == 1:
                break
        new = params_from_metric((M.T @ G @ M).tolist())
        BR = np.array([[float(x) for x in r] for r in reduce_cell(cell).basis.rows])
        BN = np.array([[float(x) for x in r] for r in reduce_cell(new).basis.rows])
        Ps = np.round(np.einsum("ij,njk,kl->nil", BR, U, np.linalg.inv(BN))).astype(int)
        Gn = M.T @ G @ M
        ok = [P for P in Ps if np.allclose(P.T @ G @ P, Gn, rtol=1e-6, atol=1e-6)]
        assert ok                                    # some lattice symmetry composed with M


def test_rhombohedral_axes_ties_and_scale_bounds():
    """'R 3' on a rhombohedral-axes cell is read on those axes (with a warning); equal-price
    candidates come nearest the identity first; the volume check matches the edge-scale limit
    in both directions."""
    from agentsg.cell.strain_closure import reduce_cell
    from agentsg.cell.selling_cob import _PRIM_EXACT
    hexa = (80.0, 80.0, 120.0, 90, 90, 120)
    T = np.array([[float(x) for x in r] for r in _PRIM_EXACT["R"]])
    rcell = params_from_metric((T.T @ np.array(UnitCell(*hexa).metric_tensor(), float) @ T).tolist())
    with pytest.warns(UserWarning, match="rhombohedral axes"):
        v1 = np.linalg.det(reduce_cell(rcell, "R 3").G)
    assert abs(v1 / np.linalg.det(reduce_cell(hexa, "R 3 :H").G) - 1) < 1e-6
    W = widened_closure((50.0, 50.0, 80.0, 90, 90, 120), "P 3")
    c, _ = reindex(W, (50.0, 50.0, 80.0, 90, 90, 120))
    assert [[int(x) for x in r] for r in c[0].operator.rows] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    G = np.array(UnitCell(60.0, 60.0, 60.0, 90, 90, 90).metric_tensor(), float)
    Wc = widened_closure((60.0, 60.0, 60.0, 90, 90, 90))
    small = params_from_metric((0.905 ** 2 * G).tolist())
    assert reindex(Wc, small)[0]                                         # edges 0.905: within 10 %


def test_round6_api():
    """None means primitive in both reindex and indexing_classes; an R symbol on a cell of neither
    shape raises; an empty result warns."""
    from agentsg.cell.strain_closure import indexing_classes, reduce_cell
    Wc = widened_closure((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1")
    with pytest.raises(ValueError, match="volume"):
        indexing_classes(Wc, (50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1", None)
    assert indexing_classes(Wc, (50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1").classes
    assert np.linalg.det(reduce_cell((80.0, 80.0, 120.0, 90, 90, 119.4), "R 3").G) > 0   # near-hexagonal: hexagonal axes
    with pytest.raises(ValueError, match="neither hexagonal"):
        reduce_cell((50.0, 60.0, 70.0, 80, 85, 100), "R 3")
    Wi = widened_closure((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1")
    with pytest.warns(UserWarning, match="no candidate"):
        reindex(Wi, (50.0, 30.0, 40.0, 90, 105, 90), "I 1 2 1")
