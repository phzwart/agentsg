"""z:strain-paper — the numbers of the restructured paper that are stated in the
strain norm ||G^-1/2 dG G^-1/2||_F (agentsg.cell.strain_closure).

(i) The worked filter: the sixteen members of the idealised closure of the
measured worked cell, their depths (strain distance to obtuse) under the measured
metric, and the size of the widened set as the band grows.
(ii) Sort spectra: the worked cell, and tetragonal lysozyme indexed in P1; the
Bravais type is the last sort before the widest gap.
(iii) The seven BLDConograph samples (z51): strain distance to the program's
Bravais type, the next type, and where the widest gap lies.
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
    """Figure 4 (e), (f): the second indexing class costs the pseudo-symmetry, about
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
