"""Tests for the Grosse-Kunstleve/Sauter/Adams Niggli reduction (niggli_gk).

The distinguishing guarantee of niggli_gk over niggli_reduce is that the tracked
change of basis M satisfies M^T G M == G_red *exactly*, for all inputs including
heavily transformed cells. These tests assert that invariant directly, cross-
check the reduced parameters against spglib, and confirm det(M) is unimodular by
exact integer arithmetic (float determinants are unreliable on large-integer M).
"""
import math
import random

import numpy as np
import pytest

from agentsg.cell.reduction import niggli_gk, niggli_reduce
from agentsg.cell.metric import UnitCell, params_from_metric

try:
    import spglib
    HAVE_SPGLIB = True
except Exception:  # pragma: no cover
    HAVE_SPGLIB = False


# ---- helpers ---------------------------------------------------------------
def _G(cell):
    return np.array(UnitCell(*cell).metric_tensor())


def _cart(cell):
    return np.array(UnitCell(*cell).orthogonalization_matrix()).T  # rows a,b,c


def _idet(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _cob_invariant_error(cell):
    red, M = niggli_gk(cell)
    Mn = np.array(M, dtype=float)
    return np.abs(Mn.T @ _G(cell) @ Mn - _G(red)).max(), M, red


def _scramble(cell, rng, max_entry=2, max_elong=6.0):
    """Apply a random small-integer unimodular COB, rejecting illegible cells."""
    G = _G(cell)
    while True:
        M = np.array([[rng.randint(-max_entry, max_entry) for _ in range(3)]
                      for _ in range(3)])
        d = round(np.linalg.det(M))
        if d not in (1, -1):
            continue
        Gp = M.T @ G @ M
        try:
            p = params_from_metric(Gp.tolist())
        except Exception:
            continue
        if max(p[:3]) / min(p[:3]) > max_elong:
            continue
        if any(a < 15 or a > 165 for a in p[3:]):
            continue
        return params_from_metric(Gp.tolist())


def _spglib_niggli(cell):
    red = spglib.niggli_reduce(_cart(cell))
    return params_from_metric((red @ red.T).tolist())


# ---- fixed reference cells -------------------------------------------------
FIXED = [
    (10.0, 6.0, 8.0, 90.0, 80.0, 90.0),      # simple monoclinic
    (8.0, 6.0, 11.0, 90.0, 90.3, 90.0),      # P21-like, beta ~ 90
    (5.0, 5.0, 5.0, 90.0, 90.0, 90.0),       # cubic
    (7.0, 9.0, 11.0, 85.0, 95.0, 100.0),     # triclinic
    (12.0, 12.0, 4.0, 90.0, 90.0, 120.0),    # hexagonal-ish
    (9.63, 14.46, 5.59, 116.92, 78.71, 85.4),   # previously non-converging
    (12.86, 12.89, 9.67, 77.15, 63.24, 83.0),   # previously non-converging
]


@pytest.mark.parametrize("cell", FIXED)
def test_cob_invariant_fixed(cell):
    err, M, red = _cob_invariant_error(cell)
    scale = np.abs(_G(red)).max()
    assert err <= 1e-6 * scale, (cell, err, M)


@pytest.mark.parametrize("cell", FIXED)
def test_det_unimodular_fixed(cell):
    _, M = niggli_gk(cell)
    assert _idet(M) in (1, -1), (cell, M, _idet(M))


@pytest.mark.parametrize("cell", FIXED)
def test_reduction_is_stationary(cell):
    red, _ = niggli_gk(cell)
    red2, M2 = niggli_gk(red)
    for x, y in zip(red, red2):
        assert math.isclose(x, y, abs_tol=1e-6)
    # re-reducing an already-reduced cell is the identity CoB
    assert M2 == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_cob_invariant_scrambled_fuzz():
    """The core guarantee: M^T G M == G_red on heavily transformed cells."""
    rng = random.Random(12345)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    worst = 0.0
    for _ in range(400):
        base = tuple(p + rng.uniform(-0.4, 0.4) for p in parent[:3]) + parent[3:]
        cell = _scramble(base, rng)
        err, M, red = _cob_invariant_error(cell)
        worst = max(worst, err / max(1.0, np.abs(_G(red)).max()))
        assert _idet(M) in (1, -1)
    assert worst < 1e-6, worst


def test_det_exact_integer_not_float():
    """Regression: float determinants misreport det on large-integer M; the
    exact integer determinant must be unimodular."""
    rng = random.Random(7)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    seen_large = False
    for _ in range(200):
        cell = _scramble(parent, rng)
        _, M = niggli_gk(cell)
        if max(abs(x) for row in M for x in row) >= 3:
            seen_large = True
        assert _idet(M) in (1, -1)
    assert seen_large  # ensure the test actually exercised large-entry M


@pytest.mark.skipif(not HAVE_SPGLIB, reason="spglib not available")
@pytest.mark.parametrize("cell", FIXED)
def test_params_match_spglib_fixed(cell):
    red, _ = niggli_gk(cell)
    sp = _spglib_niggli(cell)
    for x, y in zip(red, sp):
        assert math.isclose(x, y, abs_tol=1e-3 * max(red[:3]))


@pytest.mark.skipif(not HAVE_SPGLIB, reason="spglib not available")
def test_params_match_spglib_fuzz():
    rng = random.Random(99)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    for _ in range(200):
        cell = _scramble(parent, rng)
        red, _ = niggli_gk(cell)
        sp = _spglib_niggli(cell)
        worst = max(abs(red[i] - sp[i]) for i in range(6))
        assert worst < 1e-3 * max(red[:3]), (cell, red, sp)


def test_agrees_with_niggli_reduce_fully():
    """niggli_reduce is a wrapper around niggli_gk: params and CoB match."""
    rng = random.Random(2024)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    for _ in range(200):
        cell = _scramble(parent, rng)
        red_gk, M_gk = niggli_gk(cell)
        red_ref, M_ref = niggli_reduce(*cell)
        assert M_gk == M_ref
        for x, y in zip(red_gk, red_ref):
            assert math.isclose(x, y, abs_tol=1e-12)
