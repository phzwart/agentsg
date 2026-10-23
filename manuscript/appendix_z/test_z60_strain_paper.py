"""z:strain-paper — the numbers of the restructured paper that are stated in the
strain norm ||G^-1/2 dG G^-1/2||_F (agentsg.cell.strain_closure).

(i) The worked filter: the sixteen members of the idealised closure of the
measured worked cell, their depths (strain distance to obtuse) under the measured
metric, and the size of the widened set as the band grows.
(ii) Sort spectra: the worked cell, and tetragonal lysozyme indexed in P1; the
Bravais type is the last sort before the widest gap.
(iii) The seven BLDConograph samples (z51; Table S5): strain distance to the program's
Bravais type, the next type, and where the widest gap lies.
(iv) Propositions A9 and A10 and Lemma A8: prices of symmetries of order 2, 3, 4 and 6,
the twofold rule in the geodesic distance, the exact product and coset bounds, the
overlap bound; Le Page's obliquity against prices and sort distances; null
distributions of prices under noise; centred references.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.zcheck]

from agentsg.cell.strain_closure import (  # noqa: E402
    BRAVAIS, LEPAGE_STRAIN, distance_to_obtuse, sort_spectrum, widened_closure,
)
from test_z44_widened_complete import S0, _cn, _generate  # noqa: E402

WORKED = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)
LYSO_P1 = (79.06, 79.13, 37.93, 90.03, 89.96, 90.05)


def _type_ladder(cell):
    by = {}
    for s, d in sort_spectrum(cell).items():
        by[BRAVAIS[s]] = min(by.get(BRAVAIS[s], 9.0), d)
    return sorted(by.items(), key=lambda kv: kv[1])


def _gap_after(ladder):
    """the type after which the widest log-gap in distance opens (aP excluded)."""
    d = [max(v, 1e-6) for _, v in ladder]
    g = [np.log10(d[i + 1] / d[i]) for i in range(1, len(d) - 1)]
    return ladder[int(np.argmax(g)) + 1][0]


def test_worked_filter_in_strain():
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    G = np.array(UnitCell(*WORKED).metric_tensor())
    F0 = np.array(canonical_superbase(WORKED)[0], float)
    Gf = F0[1:4] @ G @ F0[1:4].T
    p0 = _cn(np.array(S0, float), Gf)
    _, frames = _generate(p0, 4 * 43.2, 43.2)
    depth = sorted(100 * distance_to_obtuse(Gf, S) for S in frames.values())
    assert np.allclose(depth, [0, 0.28, 0.49, 0.62, 0.78, 0.78, 3.76, 4.56,
                               5.06, 5.08, 5.08, 5.09, 5.09, 5.11, 5.11, 5.11], atol=0.006)
    assert all(d < 100 * LEPAGE_STRAIN for d in depth)            # all sixteen at Le Page's 3 degrees
    sizes = [len(widened_closure(WORKED, eps=e)) for e in (0.001, 0.003, 0.006, 0.01, 0.03, LEPAGE_STRAIN)]
    assert sizes == [1, 2, 3, 6, 6, 16]


def test_sort_spectra_of_the_two_figure_cells():
    sp = sort_spectrum(WORKED)
    want = {"A2": 0.28, "M4": 0.777, "A3": 3.757, "O5": 5.111}
    for s, v in want.items():
        assert abs(100 * sp[s] - v) < 0.01, s
    assert min(v for s, v in sp.items() if BRAVAIS[s] not in ("aP", "mP", "oP")) > 0.30
    assert _gap_after(_type_ladder(WORKED)) == "mP"
    sl = sort_spectrum(LYSO_P1)
    assert abs(100 * sl["T5"] - 0.215) < 0.005
    near = [s for s, v in sl.items() if v < 0.003]
    assert len(near) == 12 and "T5" in near                       # aP x3, mP, mC x5, oS, oP, tP
    assert min(v for v in sl.values() if v > 0.003) > 0.43
    assert _gap_after(_type_ladder(LYSO_P1)) == "tP"


CONOGRAPH = [   # name, cell, type, strain % to the type, next type, its strain %, type before the widest gap
    ("HERMES_Sr327_250K", (3.8417, 3.8524, 10.389, 100.41, 100.53, 90.046), "tI", 0.84, "mP", 25.6, "tI"),
    ("1339prbasrcuo_bs1", (3.8792, 3.8830, 11.654, 90.018, 90.039, 90.057), "tP", 0.22, "oF", 31.5, "tP"),
    ("sample3", (16.332, 16.348, 16.357, 109.41, 109.52, 109.34), "cI", 0.58, "cF", 57.6, "cI"),
    ("CeO2_matsushita", (3.7250, 3.7269, 3.7286, 90.100, 119.96, 119.91), "cF", 0.30, "cI", 47.0, "cF"),
    ("M402-pf-119673", (10.654, 22.895, 23.003, 60.600, 89.377, 89.119), "hP", 2.94, "oI", 33.8, "hP"),
    ("Acrinol-B-D8CuKa1", (9.1568, 9.1712, 20.946, 81.634, 81.751, 80.086), "mC", 0.31, "oF", 20.6, "mC"),
    ("Alumina676", (4.7428, 4.7802, 5.1093, 116.89, 90.170, 119.69), "hR", 3.58, "oI", 14.9, "mC"),
]


def test_conograph_samples_in_strain():
    for name, cell, bt, d, nxt, dn, gap in CONOGRAPH:
        lad = _type_ladder(cell)
        names = [k for k, _ in lad]
        i = names.index(bt)
        assert abs(100 * lad[i][1] - d) < 0.006, name
        assert lad[i + 1][0] == nxt and abs(100 * lad[i + 1][1] - dn) < 0.06, name
        assert _gap_after(lad) == gap, name
        assert lad[i][1] < LEPAGE_STRAIN < lad[i + 1][1], name        # Le Page's band selects the program's type


def test_twofold_price_is_twice_the_sort_distance():
    """Table 8 and Fig. 3 (c): the second indexing class costs the pseudo-symmetry, about
    twice the distance to the sort that the twofold fixes."""
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.strain_closure import indexing_classes
    for cell, sg, sort, d, price in (((30.02, 39.97, 50.03, 90.25, 92.05, 89.80), "P 1", "M4", 0.777, 1.55),
                                     ((40.0, 40.3, 80.0, 90, 90, 90), "P 21 21 21", "T5", 1.06, 2.11)):
        ref = widened_closure(cell, sg)
        G = np.array(UnitCell(*cell).metric_tensor(), float)
        r = indexing_classes(ref, params_from_metric((G * 1.002).tolist()), sg)
        prices = sorted(100 * c.representative.price for c in r.classes)
        assert len(prices) == 2 and prices[0] < 1e-6 and abs(prices[1] - price) < 0.01, (sg, prices)
        assert abs(100 * sort_spectrum(cell)[sort] - d) < 0.01
        assert abs(prices[1] / d - 2) < 0.02


def test_single_projection_is_not_enough_on_the_needle():
    """Section 4.3 / 5.3: on the z57 needle at small bands the single projection onto
    the most special face within 4 eps misses members that the union over all
    (frame, face) pairs, and the exact ball enumeration (Lemma A6), contain."""
    import test_z44_widened_complete as t
    from test_z58_projection_closure import NEEDLE, construct, construct_union
    from test_z59_adjacency import superbases_within
    G = t._gram(NEEDLE)
    for eps in (0.0005, 0.002, 0.005, 0.01):
        truth = set(superbases_within(G, eps))
        uni, _ = construct_union(G, eps)
        one, _ = construct(G, eps)
        assert uni == truth, eps
        if eps in (0.002, 0.005):
            assert len(one) == 6 and len(truth) == 8


def test_alumina_gap_below_hR():
    lad = _type_ladder((4.7428, 4.7802, 5.1093, 116.89, 90.170, 119.69))
    d = dict(lad)
    assert abs(100 * d["mC"] - 0.43) < 0.006 and abs(100 * d["hR"] - 3.58) < 0.006


def test_centred_references_in_conventional_settings():
    """Reindexing with centred references given in their conventional settings: the
    reference is widened as (cell, space group), the new cell is a random primitive
    basis of the same lattice with a random metric error, and the true (rational)
    operator from the conventional basis to the new basis is among the candidates."""
    from fractions import Fraction
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.strain_closure import band_for, primitive_matrix, reduce_cell, reindex
    rng = np.random.default_rng(7)
    refs = [((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1"), ((30.0, 40.0, 50.0, 90, 90, 90), "I 2 2 2"),
            ((60.0, 60.0, 90.0, 90, 90, 120), "R 3 :H"), ((80.0, 80.0, 80.0, 90, 90, 90), "F 2 3"),
            ((40.0, 40.0, 70.0, 90, 90, 90), "I 41"), ((30.0, 50.0, 80.0, 90, 90, 90), "F d d d")]
    for cell, sg in refs:
        ref = widened_closure(cell, sg, eps=band_for(LEPAGE_STRAIN))
        Gc = np.array(UnitCell(*cell).metric_tensor(), float)
        C = np.array([[float(x) for x in row] for row in primitive_matrix(sg).rows])   # primitive basis in conventional
        for _ in range(4):
            while True:
                U = rng.integers(-2, 3, (3, 3))
                if round(abs(np.linalg.det(U))) == 1:
                    break
            P = C @ U                                                  # conventional -> new primitive basis
            G = P.T @ Gc @ P
            w, V = np.linalg.eigh(G)
            Gh = V @ np.diag(w ** .5) @ V.T
            X = rng.normal(size=(3, 3)); X = (X + X.T) / 2
            X *= rng.uniform(0.002, 0.03) / np.linalg.norm(X)
            new = params_from_metric((Gh @ (np.eye(3) + X) @ Gh).tolist())
            cands, _ = reindex(ref, new, accept=LEPAGE_STRAIN)
            truth = [[Fraction(x).limit_denominator(12) for x in row] for row in P]
            assert any([list(r) for r in c.operator.rows] == truth for c in cands), (sg, len(cands))


def test_twofold_price_is_twice_the_distance_to_its_fixed_space():
    """Proposition A9 on a beta scan: for a P 1 2 1 cell 30 x 40 x 50 A the pseudo-oP
    twofold's price is 2 sqrt2 sin(beta - 90) to first order, twice the distance to the
    mP->oP wall, so a price band of 0.074 offers it up to beta = 91.5 deg and the band
    LEPAGE_PRICE = 0.148 up to 93 deg."""
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.strain_closure import LEPAGE_PRICE, band_for, indexing_classes
    for beta in (91.0, 91.4, 91.6, 92.5, 93.5):
        cell = (30.0, 40.0, 50.0, 90, beta, 90)
        d_oP = sort_spectrum(cell)["O5"]
        expect = np.sqrt(2) * np.sin(np.radians(beta - 90))
        assert abs(d_oP - expect) < 0.002 * expect + 1e-5
        for accept in (LEPAGE_STRAIN, LEPAGE_PRICE):
            ref = widened_closure(cell, "P 1 2 1", eps=band_for(accept))
            G = np.array(UnitCell(*cell).metric_tensor(), float)
            r = indexing_classes(ref, params_from_metric((G * 1.001).tolist()), "P 1 2 1", accept=accept)
            offered = len(r.classes) == 2
            assert offered == (2 * d_oP <= accept * (1 + 0.02)), (beta, accept, len(r.classes))
            if offered:
                assert abs(r.classes[1].representative.price - 2 * d_oP) < 0.03 * 2 * d_oP


def test_overlap_band_eps_over_one_minus_eps():
    """Lemma A8 as implemented: a price of eps measured at the new cell's metric is at
    most eps/(1-eps) at the predicted metric, so the reference widened at band_for(eps)
    holds the new cell's reduced superbase; the bound is attained when the error has a single
    non-zero eigenvalue, not for a pure scale."""
    from agentsg.cell.strain_closure import _strain
    for eps in (0.05, 0.1, 0.2):
        G = np.eye(3)
        Gp = G * (1 - eps / np.sqrt(3))                     # price eps at G
        p_at_G, _ = _strain(G, Gp, fit_scale=False)
        p_at_Gp, _ = _strain(Gp, G, fit_scale=False)
        assert abs(p_at_G - eps) < 1e-12 and p_at_Gp <= eps / (1 - eps) + 1e-12
        assert p_at_Gp > eps                                 # the two distances differ at second order
        G1 = np.diag([1 - eps, 1.0, 1.0])                    # one eigenvalue: the bound is attained
        assert abs(_strain(G, G1, fit_scale=False)[0] - eps) < 1e-12
        assert abs(_strain(G1, G, fit_scale=False)[0] - eps / (1 - eps)) < 1e-12


CONOGRAPH_AND_LYSO = [c for _, c, *_ in CONOGRAPH] + [LYSO_P1, WORKED]


def test_adjacency_radius_reaches_the_cap_for_the_spectrum_cells():
    """Section 5.2: for the seven BLDConograph cells, the lysozyme cell and the worked cell
    delta_1 reaches the quarter cap, so a sort distance reported as '>25 %' is a lower
    bound (no superbase outside the first layer is within 0.25) as well as an upper bound
    over the frames of U."""
    from agentsg.cell.strain_closure import adjacency_radius, reduce_cell
    for cell in CONOGRAPH_AND_LYSO:
        assert adjacency_radius(reduce_cell(cell).G) == 0.25, cell


def test_conograph_ladders():
    """Table S5 (supporting information): the types up to the program's, with their strain distances (%)."""
    want = {"HERMES_Sr327_250K": [("mC", 0.31), ("oI", 0.74), ("oF", 0.83), ("tI", 0.84)],
            "1339prbasrcuo_bs1": [("mP", 0.11), ("mC", 0.14), ("oS", 0.17), ("oP", 0.18), ("tP", 0.22)],
            "sample3": [("mC", 0.23), ("hR", 0.39), ("oF", 0.39), ("oI", 0.49), ("tI", 0.49), ("cI", 0.58)],
            "CeO2_matsushita": [("mC", 0.09), ("oI", 0.17), ("oF", 0.17), ("tI", 0.18), ("hR", 0.26), ("cF", 0.30)],
            "M402-pf-119673": [("mC", 0.99), ("mP", 2.24), ("oS", 2.37), ("hP", 2.94)],
            "Acrinol-B-D8CuKa1": [("mC", 0.31)],
            "Alumina676": [("mC", 0.43), ("hR", 3.58)]}
    for name, cell, *_ in CONOGRAPH:
        lad = [(k, round(100 * v, 2)) for k, v in _type_ladder(cell) if k != "aP"]
        got = lad[:len(want[name])]
        assert {k for k, _ in got} == {k for k, _ in want[name]}, name
        d = dict(got)
        assert all(abs(d[k] - v) < 0.006 for k, v in want[name]), (name, got)


def test_price_of_an_order_n_symmetry():
    """Proposition A9 for rotations of order n: the price of g is |lambda - 1| times the
    distance to V_g on each eigenspace of X -> g^T X g, so the ratio is 2 for a twofold,
    sqrt3 for a threefold, between sqrt2 and 2 for a fourfold, and between 1 and sqrt3 for
    a sixfold (norm at a metric G* that g preserves)."""
    from agentsg.cell.metric import UnitCell
    rng = np.random.default_rng(1)
    hexG = np.array(UnitCell(50, 50, 80, 90, 90, 120).metric_tensor())
    cubG = np.array(UnitCell(40, 40, 40, 90, 90, 90).metric_tensor())
    cases = {"2": (hexG, np.diag([-1, -1, 1]), 2.0, 2.0), "3": (hexG, np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]), 3 ** .5, 3 ** .5),
             "4": (cubG, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), 2 ** .5, 2.0),
             "6": (hexG, np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]]), 1.0, 3 ** .5)}
    for name, (Gs, g, lo, hi) in cases.items():
        assert np.allclose(g.T @ Gs @ g, Gs)
        Gi = np.linalg.inv(Gs)
        ip = lambda X, Y: np.trace(Gi @ X @ Gi @ Y)
        n = next(k for k in range(1, 7) if np.allclose(np.linalg.matrix_power(g, k), np.eye(3)))
        rs = []
        for _ in range(500):
            A = rng.normal(size=(3, 3)); X = (A + A.T) / 2
            Xf = sum(np.linalg.matrix_power(g, k).T @ X @ np.linalg.matrix_power(g, k) for k in range(n)) / n
            D = g.T @ X @ g - X
            rs.append(np.sqrt(ip(D, D)) / np.sqrt(ip(X - Xf, X - Xf)))
        assert lo - 1e-9 <= min(rs) and max(rs) <= hi + 1e-9 and max(rs) - min(rs) <= hi - lo + 1e-9, name
        assert min(rs) < lo + 0.05 and max(rs) > hi - 0.05


def test_lepage_obliquity_against_prices_and_sort_distance():
    """30 x 40 x 50 A, alpha = beta = gamma = 92 deg: each axis has Le Page obliquity 2.88 deg,
    each twofold a price 0.1426 (2 sqrt2 sin 2.88 deg = 0.1421), within 0.148; the oP sort
    distance, which combines the three, is 0.0885, above 0.074; the mP distance 0.0711."""
    import math
    from agentsg.cell.metric import UnitCell
    from agentsg.cell.strain_closure import _strain
    cell = (30, 40, 50, 92, 92, 92)
    G = np.array(UnitCell(*cell).metric_tensor()); Gi = np.linalg.inv(G)
    for i in range(3):
        e = np.eye(3)[i]; v = Gi @ e
        delta = math.degrees(math.acos((e @ G @ v) / math.sqrt(e @ G @ e) / math.sqrt(v @ G @ v)))
        g = -np.eye(3); g[i, i] = 1
        price, _ = _strain(G, g.T @ G @ g, fit_scale=False)
        assert abs(delta - 2.88) < 0.005 and abs(price - 0.1426) < 5e-4 and price < 2 * LEPAGE_STRAIN
    sp = sort_spectrum(cell)
    assert abs(sp["O5"] - 0.0885) < 5e-4 and sp["O5"] > LEPAGE_STRAIN and abs(sp["M4"] - 0.0711) < 5e-4


def test_null_distributions_of_prices():
    """Two independent measurements of an exact monoclinic lattice, each with isotropic strain
    noise of rms sigma: the true operator's price (after the scale fit) has median 1.2 sigma
    and 99th percentile 2.2 sigma; the price of the exact twofold in one measurement has
    median 1.0 sigma and 99th percentile 2.5 sigma."""
    from agentsg.cell.metric import UnitCell
    from agentsg.cell.strain_closure import _strain
    rng = np.random.default_rng(0)
    G = np.array(UnitCell(30, 40, 50, 90, 92, 90).metric_tensor()); w, V = np.linalg.eigh(G)
    Gh = V @ np.diag(w ** .5) @ V.T
    g = np.diag([-1, 1, -1]); s = 0.005

    def sym():
        A = rng.normal(size=(3, 3)); return (A + A.T) / 2 * s / np.sqrt(6)
    tp, sp = [], []
    for _ in range(8000):
        G1 = Gh @ (np.eye(3) + sym()) @ Gh; G2 = Gh @ (np.eye(3) + sym()) @ Gh
        tp.append(_strain(G2, G1)[0] / s); sp.append(_strain(G1, g.T @ G1 @ g)[0] / s)
    assert abs(np.median(tp) - 1.2) < 0.05 and abs(np.percentile(tp, 99) - 2.2) < 0.1
    assert abs(np.median(sp) - 1.0) < 0.05 and abs(np.percentile(sp, 99) - 2.5) < 0.12


def test_exact_product_and_coset_bounds():
    """Proposition A10, exact (prices without the scale fit): price(h1 h2) <= p1 + p2 + p1 p2,
    and the price of h P0 at G_N lies in [(1 - p0) price(h) - p0, (1 + p0) price(h) + p0]."""
    from agentsg.cell.strain_closure import _strain
    rng = np.random.default_rng(3)

    def price(G, h):
        return _strain(G, h.T @ G @ h, fit_scale=False)[0]
    for _ in range(400):
        A = rng.normal(size=(3, 3)); G = A @ A.T + 0.5 * np.eye(3)
        h1 = np.eye(3) + 0.15 * rng.normal(size=(3, 3)); h2 = np.eye(3) + 0.15 * rng.normal(size=(3, 3))
        p1, p2 = price(G, h1), price(G, h2)
        assert price(G, h1 @ h2) <= p1 + p2 + p1 * p2 + 1e-12
        P0 = np.eye(3) + 0.1 * rng.normal(size=(3, 3))
        X = rng.normal(size=(3, 3)); X = (X + X.T) / 2
        B = P0.T @ G @ P0
        w, V = np.linalg.eigh(B); Bh = V @ np.diag(w ** .5) @ V.T
        GN = Bh @ (np.eye(3) + rng.uniform(0.001, 0.2) * X / np.linalg.norm(X)) @ Bh
        p0 = _strain(GN, B, fit_scale=False)[0]
        if p0 >= 1:
            continue
        ph = price(G, h1)
        q = _strain(GN, (h1 @ P0).T @ G @ (h1 @ P0), fit_scale=False)[0]
        assert (1 - p0) * ph - p0 - 1e-12 <= q <= (1 + p0) * ph + p0 + 1e-12


def test_twofold_price_is_twice_the_geodesic_distance():
    """Proposition A9 (ii): in the affine-invariant geodesic distance the geodesic midpoint of
    G and g^T G g is fixed by the twofold g, and no metric fixed by g is closer to G."""
    from scipy.linalg import expm, logm, sqrtm
    from scipy.optimize import minimize
    rng = np.random.default_rng(4)
    g = np.diag([-1.0, 1.0, -1.0])                                 # twofold along b

    def dist(A, B):
        Ai = np.linalg.inv(np.real(sqrtm(A)))
        return float(np.linalg.norm(np.real(logm(Ai @ B @ Ai))))
    for _ in range(6):
        A = rng.normal(size=(3, 3)); G = A @ A.T + np.eye(3)
        S = g.T @ G @ g
        Gh = np.real(sqrtm(G)); Ghi = np.linalg.inv(Gh)
        m = Gh @ np.real(sqrtm(Ghi @ S @ Ghi)) @ Gh                 # geodesic midpoint
        assert np.allclose(g.T @ m @ g, m, atol=1e-9)
        assert abs(dist(G, m) - dist(G, S) / 2) < 1e-9

        def to_fixed(x):                                            # metrics with G12 = G23 = 0
            return expm(np.array([[x[0], 0, x[1]], [0, x[2], 0], [x[1], 0, x[3]]]))
        best = min(minimize(lambda x: dist(G, to_fixed(x)), rng.normal(size=4) * 0.1,
                            method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-11, maxiter=4000)).fun
                   for _ in range(2))
        assert best >= dist(G, m) - 1e-6
