"""Dirichlet / Voronoi ASU and sphericity optimiser."""
from __future__ import annotations
import random

import pytest

from agentsg import space_group
from agentsg.asu import build_dirichlet_asu, optimize_asu
from agentsg.cell.metric import UnitCell
from agentsg.linalg import Vector3

_CELL = UnitCell(10, 10, 10, 90, 90, 90)


@pytest.mark.parametrize("n,tol", [
    (1, 0.02),
    (2, 0.03),
    (4, 0.03),
    (19, 0.03),
    (225, 0.003),
])
def test_volume_fraction_approx_one_over_order(n, tol):
    ops = list(space_group(n).operations())
    asu = build_dirichlet_asu(ops, _CELL, space="direct")
    nmc = 15000 if n < 200 else 50000
    vf = asu.volume_fraction(n=nmc, seed=1)
    assert abs(vf - 1.0 / len(ops)) < tol


@pytest.mark.parametrize("n", (1, 2, 4, 19, 225))
def test_exactly_one_orbit_representative(n):
    ops = list(space_group(n).operations())
    asu = build_dirichlet_asu(ops, _CELL, space="direct")
    rng = random.Random(0)
    for _ in range(80 if n < 200 else 30):
        x = Vector3((rng.random(), rng.random(), rng.random()))
        orbit = []
        seen = set()
        for op in ops:
            y = ((op.W @ x) + op.w).mod1()
            if y.v not in seen:
                seen.add(y.v)
                orbit.append(y)
        count = sum(1 for y in orbit if asu.contains(y))
        assert count == 1


def test_optimizer_at_least_matches_identity_gauge():
    """On a skinny cell, the search includes the identity origin gauge."""
    skinny = UnitCell(5.0, 5.0, 40.0, 90.0, 90.0, 90.0)
    ops = list(space_group(4).operations())  # P21
    n_sample = 800
    base = build_dirichlet_asu(ops, skinny).sphericity(n=n_sample, seed=0)
    opt = optimize_asu(ops, skinny, space="direct", score="sphericity", n_sample=n_sample)
    assert opt.score >= base - 1e-12
    assert 0.0 <= opt.score <= 1.0
    assert "sphericity" in opt.metrics


def test_reciprocal_dirichlet_smoke():
    ops = list(space_group(19).operations())
    asu = build_dirichlet_asu(ops, _CELL, space="reciprocal")
    vf = asu.volume_fraction(n=8000, seed=2)
    # Laue group of P212121 is mmm, order 8
    assert abs(vf - 1.0 / 8) < 0.04


def test_near_degenerate_inertia_eigenvalues():
    """Near-equal SPD eigenvalues must not collapse to Cardano triple-bogus roots."""
    from agentsg.asu import _eigh3_sorted

    # Diagonal, spread ~1e-8: old path hit p≈0 and returned three identical
    # wrong values (~1.0000076) instead of the mean (~1).
    A = [[1.0, 0.0, 0.0], [0.0, 1.0 + 1e-8, 0.0], [0.0, 0.0, 1.0 + 2e-8]]
    ev = _eigh3_sorted(A)
    mean = (3.0 + 3e-8) / 3.0
    assert all(abs(x - mean) < 1e-9 for x in ev)
    assert abs(ev[2] - ev[0]) < 1e-7

    # Noisy near-isotropic SPD: every eigenvalue stays near 1 (sphericity ~1).
    rng = random.Random(0)
    for _ in range(200):
        eps = 1e-10
        M = [
            [1.0 + rng.uniform(-eps, eps), rng.uniform(-eps, eps), rng.uniform(-eps, eps)],
            [0.0, 1.0 + rng.uniform(-eps, eps), rng.uniform(-eps, eps)],
            [0.0, 0.0, 1.0 + rng.uniform(-eps, eps)],
        ]
        M[1][0] = M[0][1]
        M[2][0] = M[0][2]
        M[2][1] = M[1][2]
        ev = _eigh3_sorted(M)
        assert all(abs(x - 1.0) < 1e-6 for x in ev)
        if ev[2] > 0:
            assert ev[0] / ev[2] > 0.999
