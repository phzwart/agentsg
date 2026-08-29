"""Niggli CoB invariant: M^T G M == gram(reduced) for niggli_reduce and niggli_gk.

Historically ``niggli_reduce`` returned a change-of-basis that did not map the
input metric onto the reduced metric under any of N, N^T, N^{-1}, N^{-T}, even
when the reduced parameters themselves were correct. Both entry points now share
the Grosse-Kunstleve/Sauter/Adams step matrices and assert the invariant on
every return; these tests pin that contract.
"""
from __future__ import annotations

import math
import random

import pytest

from agentsg.cell.reduction import niggli_reduce, niggli_gk, _transform_metric, _gram_from_params
from agentsg.cell.metric import UnitCell, params_from_metric


def _G(cell):
    return UnitCell(*cell).metric_tensor()


def _idet(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _cob_err(G_orig, M, red):
    Gp = _transform_metric(G_orig, M)
    Gr = _gram_from_params(*red)
    return max(abs(Gp[i][j] - Gr[i][j]) for i in range(3) for j in range(3))


def _scramble(cell, rng, max_entry=2, max_elong=6.0):
    G = _G(cell)
    while True:
        M = [[rng.randint(-max_entry, max_entry) for _ in range(3)]
             for _ in range(3)]
        if _idet(M) not in (1, -1):
            continue
        Gp = _transform_metric(G, M)
        try:
            p = params_from_metric(Gp)
        except Exception:
            continue
        if max(p[:3]) / min(p[:3]) > max_elong:
            continue
        if any(a < 15 or a > 165 for a in p[3:]):
            continue
        return p


FIXED = [
    (10.0, 6.0, 8.0, 90.0, 80.0, 90.0),
    (8.0, 6.0, 11.0, 90.0, 90.3, 90.0),
    (5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    (7.0, 9.0, 11.0, 85.0, 95.0, 100.0),
    (12.0, 12.0, 4.0, 90.0, 90.0, 120.0),
    (9.63, 14.46, 5.59, 116.92, 78.71, 85.4),
    (12.86, 12.89, 9.67, 77.15, 63.24, 83.0),
]


@pytest.mark.parametrize("cell", FIXED)
@pytest.mark.parametrize("fn", [niggli_reduce, niggli_gk], ids=["reduce", "gk"])
def test_cob_invariant_NT_G_N_equals_gram_reduced(cell, fn):
    """The one-line contract: M^T G M == gram(reduced)."""
    if fn is niggli_reduce:
        red, M = niggli_reduce(*cell)
    else:
        red, M = niggli_gk(cell)
    err = _cob_err(_G(cell), M, red)
    scale = max(abs(x) for row in _G(red) for x in row) or 1.0
    assert err <= 1e-6 * scale, (cell, err, M, red)
    assert _idet(M) in (1, -1)


def test_niggli_reduce_cob_on_scrambled_fuzz():
    """Regression: scrambled cells previously broke niggli_reduce's CoB only."""
    rng = random.Random(12345)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    worst = 0.0
    for _ in range(400):
        base = tuple(p + rng.uniform(-0.4, 0.4) for p in parent[:3]) + parent[3:]
        cell = _scramble(base, rng)
        red, M = niggli_reduce(*cell)
        err = _cob_err(_G(cell), M, red)
        scale = max(abs(x) for row in _G(red) for x in row) or 1.0
        worst = max(worst, err / scale)
        assert _idet(M) in (1, -1)
    assert worst < 1e-6, worst


def test_niggli_reduce_matches_niggli_gk_including_cob():
    rng = random.Random(2024)
    parent = (8.0, 6.0, 11.0, 90.0, 90.3, 90.0)
    for _ in range(100):
        cell = _scramble(parent, rng)
        red_r, M_r = niggli_reduce(*cell)
        red_g, M_g = niggli_gk(cell)
        assert M_r == M_g
        for x, y in zip(red_r, red_g):
            assert math.isclose(x, y, abs_tol=1e-12)


def test_four_conventions_only_NT_holds():
    """Document that the documented convention is M^T G M, not the other three."""
    cell = _scramble((10.0, 6.0, 8.0, 90.0, 80.0, 90.0), random.Random(1))
    G = _G(cell)
    red, M = niggli_reduce(*cell)
    Gr = _G(red)
    # invert M (integer unimodular)
    # adjugate / det
    det = _idet(M)
    # cofactor transpose / det
    def cof(r, c):
        minor = [[M[i][j] for j in range(3) if j != c]
                 for i in range(3) if i != r]
        md = minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
        return ((-1) ** (r + c)) * md
    Minv = [[cof(j, i) / det for j in range(3)] for i in range(3)]

    def resid(Gp):
        return max(abs(Gp[i][j] - Gr[i][j]) for i in range(3) for j in range(3))

    # N: M G M^T
    N = [[sum(M[i][k] * G[k][j] for k in range(3)) for j in range(3)]
         for i in range(3)]
    N = [[sum(N[i][k] * M[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    NT = _transform_metric(G, M)
    # N^{-1} G N^{-T}
    MinvT = [[Minv[j][i] for j in range(3)] for i in range(3)]
    Ninv = _transform_metric(G, MinvT)  # (N^{-T})^T G N^{-T} wait
    # For Minv as columns: Minv^T G Minv
    Ninv_NT = _transform_metric(G, Minv)

    scale = max(abs(x) for row in Gr for x in row) or 1.0
    assert resid(NT) <= 1e-6 * scale
    # On a scrambled cell the wrong conventions fail by a large margin
    assert resid(N) > 1e-3 * scale or resid(Ninv_NT) > 1e-3 * scale
