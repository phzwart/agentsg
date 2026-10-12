"""z:deformation — the certified, symmetrised deformation distance (the ideal)."""
from __future__ import annotations

import math
import random

import pytest

pytestmark = [pytest.mark.zcheck]


def _gram_from_p(p):
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _cell_from_p(p):
    from agentsg.cell.metric import params_from_metric
    return params_from_metric(_gram_from_p(p))


def test_near_face_ideal_is_exact_and_continuous():
    from agentsg.cell.deformation import deformation_distance

    for delta in (0.1, 0.02):
        A = _cell_from_p((4.0, 7.0, 9.0, 5.0, 6.0, delta))
        B = _cell_from_p((4.0, 7.0, 9.0, 5.0, 6.0, -delta))
        r = deformation_distance(A, B)
        assert r["certified"]
        # one conorm changes by 2*delta: Frobenius on G is exactly 4*delta
        assert abs(r["d"] - 4.0 * delta) < 0.01 * 4.0 * delta
        assert abs(r["d_AB"] - r["d_BA"]) < 1e-6
        assert abs(r["Delta_cell"][3] + 8.54 * delta) < 0.03  # alpha, degrees
        assert r["d"] == max(r["d_AB"], r["d_BA"])


def test_hfq_certified_with_entries_at_most_one():
    from agentsg.cell.deformation import deformation_distance
    from agentsg.cell.selling_cob import _primitive_or_self

    A = _primitive_or_self((107.100, 62.300, 57.100, 90.0, 95.70, 90.0),
                           "C 1 2 1")
    B = _primitive_or_self((61.200, 61.200, 53.100, 82.60, 87.30, 60.00),
                           "P 1")
    r = deformation_distance(A, B)
    assert r["certified"] and r["radius_used"] == 1
    assert r["radius_bound"] < 2.0  # the entries-<=1 ball provably suffices
    assert 400.0 < r["d"] < 600.0
    # the raw minimum is direction dependent; the report symmetrises it
    assert abs(r["d_AB"] - r["d_BA"]) > 0.01 * r["d"]
    assert r["d"] == max(r["d_AB"], r["d_BA"])


def test_ladder_closure_value_upper_bounds_the_ideal():
    import numpy as np
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.deformation import deformation_distance
    from agentsg.cell.metric import UnitCell

    rng = random.Random(3)
    for _ in range(15):
        p1 = [rng.uniform(3, 9) for _ in range(6)]
        p2 = [x + rng.uniform(-0.5, 0.5) for x in p1]
        A, B = _cell_from_p(p1), _cell_from_p(p2)
        GA = np.array(UnitCell(*A).metric_tensor())
        GB = np.array(UnitCell(*B).metric_tensor())
        P, _, _ = closure_distance(A, B)  # jointly obtuse rung
        M = np.array(P, dtype=float)
        rung = float(np.linalg.norm(GA - M.T @ GB @ M))
        ideal = deformation_distance(A, B)["d_AB"]
        assert ideal <= rung + 1e-9


def test_ball_counts_and_radius_regimes():
    import math
    from agentsg.cell.deformation import _metric, _radius, _unimodular_ball

    assert len(_unimodular_ball(1)) == 6960      # of 3^9 = 19683 candidates
    assert len(_unimodular_ball(2)) == 135408
    H = _metric((41.8, 41.8, 233.0, 90.0, 90.0, 120.0))
    assert 7.0 < _radius(H, H, 0.0) < 9.0        # entry ball infeasible here
