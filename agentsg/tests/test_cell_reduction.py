"""Niggli reduction tests: lattice invariance, change-of-basis correctness,
idempotence, and validation against gemmi."""
import math
import random
import pytest
from agentsg.cell.reduction import niggli_reduce
from agentsg.cell.metric import UnitCell


def _metric(p):
    return UnitCell(*p).metric_tensor()


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _T(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def _det(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def test_change_of_basis_reproduces_reduced_metric():
    for p in [(6, 7, 8, 85, 95, 100), (5, 5, 5, 60, 60, 60), (10, 10, 10, 89, 89, 89)]:
        red, M = niggli_reduce(*p)
        G = _metric(p)
        Gred_via_M = _matmul(_matmul(_T(M), G), M)
        Gred_direct = _metric(red)
        for i in range(3):
            for j in range(3):
                assert math.isclose(Gred_via_M[i][j], Gred_direct[i][j], abs_tol=1e-6)
        assert abs(round(_det(M))) == 1     # unimodular


def test_idempotent():
    random.seed(11)
    for _ in range(50):
        p = (random.uniform(4, 12), random.uniform(4, 12), random.uniform(4, 12),
             random.uniform(70, 110), random.uniform(70, 110), random.uniform(70, 110))
        try:
            r1, _ = niggli_reduce(*p)
        except (ValueError, RuntimeError):
            continue
        r2, _ = niggli_reduce(*r1)
        for x, y in zip(r1, r2):
            assert math.isclose(x, y, abs_tol=1e-6)


gemmi = pytest.importorskip("gemmi")


def _gemmi_niggli(p):
    gv = gemmi.GruberVector(gemmi.UnitCell(*p), None)
    gv.niggli_reduce()
    return gv.cell_parameters()


def test_matches_gemmi_on_random_cells():
    random.seed(23)
    n = 0
    while n < 150:
        p = (random.uniform(4, 12), random.uniform(4, 12), random.uniform(4, 12),
             random.uniform(70, 110), random.uniform(70, 110), random.uniform(70, 110))
        if gemmi.UnitCell(*p).volume < 1:
            continue
        n += 1
        red, _ = niggli_reduce(*p)
        gred = _gemmi_niggli(p)
        for x, y in zip(red, gred):
            assert math.isclose(x, y, abs_tol=1e-4), (p, red, gred)


def test_reduced_cell_is_niggli_shaped():
    # reduced cell: a <= b <= c (Niggli main condition on lengths)
    red, _ = niggli_reduce(9, 5, 7, 80, 100, 95)
    a, b, c = red[:3]
    assert a <= b + 1e-6 and b <= c + 1e-6
