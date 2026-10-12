"""z:closure-metric — the closure minimum is a congruence statistic, not a metric."""
from __future__ import annotations

import math
import random

import pytest

pytestmark = [pytest.mark.zcheck]


def _gram_from_p(p):
    """Gram of v1,v2,v3 from conorms ordered (p01,p02,p03,p12,p13,p23)."""
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _cell_from_p(p):
    from agentsg.cell.metric import params_from_metric
    return params_from_metric(_gram_from_p(p))


def _rand_cell(rng):
    return (rng.uniform(20, 60), rng.uniform(20, 60), rng.uniform(20, 60),
            rng.uniform(75, 105), rng.uniform(75, 105), rng.uniform(75, 105))


def test_sandwich_on_random_pairs():
    from agentsg.cell.rootform import sorted_linear_distance
    from agentsg.cell.canonical import closure_distance

    rng = random.Random(0)
    eq = 0
    for _ in range(120):
        A, B = _rand_cell(rng), _rand_cell(rng)
        s = sorted_linear_distance(A, B)
        _, D, _ = closure_distance(A, B)
        assert D >= s - 1e-9
        if abs(D - s) <= 1e-6 * max(1.0, s):
            eq += 1
    assert eq <= 10  # equality is the exception, not the rule


def test_near_face_discontinuity():
    from agentsg.cell.rootform import sorted_linear_distance
    from agentsg.cell.canonical import closure_distance

    for delta, ratio_floor in ((0.5, 2.5), (0.1, 10.0), (0.02, 50.0)):
        A = _cell_from_p((4.0, 7.0, 9.0, 5.0, 6.0, delta))
        B = _cell_from_p((4.0, 7.0, 9.0, 5.0, 6.0, -delta))
        s = sorted_linear_distance(A, B)
        _, D0, _ = closure_distance(A, B)
        _, Dw, _ = closure_distance(A, B, boundary_rel=0.2)
        assert s < 0.2  # the lattices are genuinely close
        assert D0 > 0.4  # the obtuse-restricted minimum is stuck
        assert Dw < 0.25  # widening recovers the closeness
        assert D0 / Dw > ratio_floor


def test_hfq_widening_sequence():
    from agentsg.cell.rootform import sorted_linear_distance
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.selling_cob import _primitive_or_self

    A = _primitive_or_self((107.100, 62.300, 57.100, 90.0, 95.70, 90.0),
                           "C 1 2 1")
    B = _primitive_or_self((61.200, 61.200, 53.100, 82.60, 87.30, 60.00),
                           "P 1")
    s = sorted_linear_distance(A, B)
    _, D0, _ = closure_distance(A, B)
    assert abs(s - 5.434) < 0.1 and abs(D0 - s) < 0.02
    vals = [closure_distance(A, B, boundary_rel=r)[1]
            for r in (0.02, 0.06, 0.12, 0.3)]
    assert vals[0] >= vals[1] >= vals[2] >= vals[3]
    assert vals[1] < 4.6 and vals[2] < 2.9  # the key does not bound these


def test_interior_certificate_contrapositive():
    """Widening improves only where a lattice margin lies below the minimum."""
    from agentsg.cell.rootform import sorted_linear_key
    from agentsg.cell.canonical import closure_distance

    rng = random.Random(1)
    improved = 0
    for _ in range(40):
        p1 = [rng.uniform(3, 9) for _ in range(6)]
        p2 = [x + rng.uniform(-0.3, 0.3) for x in p1]
        A, B = _cell_from_p(p1), _cell_from_p(p2)
        _, D0, _ = closure_distance(A, B)
        _, Dw, _ = closure_distance(A, B, boundary_rel=0.15)
        if Dw < D0 - 1e-9:
            improved += 1
            margin = min(
                min(x for x in sorted_linear_key(c) if x > 1e-9)
                for c in (A, B))
            assert margin <= D0 + 1e-6, (margin, D0)
    assert improved <= 10
