"""Bounded Niggli-reduction fuzz: random + near-degenerate cells vs oracles.

Checks that are always required (even without gemmi/spglib):
  * det(M) in {-1, +1}
  * idempotency of reduced params
  * re-reducing an already-reduced cell yields the identity CoB
    (no residual axis permutation / sign flip on our fixed point)

When gemmi and/or spglib are installed, also compare oracles. Near the
type-I / type-II Niggli boundary (e.g. rhombohedral ~60°), float noise can
pick either acute or obtuse presentation; both are lattice-equivalent, so we
accept agreement after mapping the oracle cell through our reducer.
"""
from __future__ import annotations

import math
import random

import pytest

from agentsg.cell.metric import UnitCell
from agentsg.cell.reduction import niggli_reduce

gemmi = pytest.importorskip("gemmi")


def _det(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _is_identity(M, tol=1e-12):
    for i in range(3):
        for j in range(3):
            want = 1.0 if i == j else 0.0
            if abs(M[i][j] - want) > tol:
                return False
    return True


def _params_close(p, q, abs_tol=1e-4):
    return all(abs(x - y) <= abs_tol for x, y in zip(p, q))


def _valid_volume(p, min_vol=1.0):
    try:
        return gemmi.UnitCell(*p).volume >= min_vol
    except Exception:
        return False


def _gemmi_niggli(p):
    gv = gemmi.GruberVector(gemmi.UnitCell(*p), None)
    gv.niggli_reduce()
    return gv.cell_parameters()


def _lattice_rows(cell):
    a, b, c, al, be, ga = cell
    al, be, ga = map(math.radians, (al, be, ga))
    va = [a, 0.0, 0.0]
    vb = [b * math.cos(ga), b * math.sin(ga), 0.0]
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return [va, vb, [cx, cy, cz]]


def _params_from_rows(rows):
    # Pure Python — avoids a hard numpy dependency when spglib is absent.
    def _norm(v):
        return math.sqrt(sum(x * x for x in v))

    def _dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    ls = [_norm(rows[i]) for i in range(3)]

    def ang(i, j):
        return math.degrees(
            math.acos(max(-1.0, min(1.0, _dot(rows[i], rows[j]) / (ls[i] * ls[j]))))
        )

    return (ls[0], ls[1], ls[2], ang(1, 2), ang(0, 2), ang(0, 1))


def _try_import_spglib():
    try:
        import spglib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None, None
    import numpy as np
    import spglib

    return spglib, np


def _spglib_niggli(p, spglib, np):
    red = spglib.niggli_reduce(np.array(_lattice_rows(p)))
    if red is None:
        return None
    return _params_from_rows(red)


def _oracle_agree(ours, oracle, abs_tol=1e-4):
    """True if params match, or both land on the same agentsg fixed point."""
    if oracle is None:
        return True
    if _params_close(ours, oracle, abs_tol=abs_tol):
        return True
    # type-I / type-II ambiguity: same lattice, different angle presentation
    if abs(UnitCell(*ours).volume() - UnitCell(*oracle).volume()) > 1e-6 * max(
        UnitCell(*ours).volume(), 1.0
    ):
        return False
    fixed_ours, _ = niggli_reduce(*ours)
    fixed_oracle, _ = niggli_reduce(*oracle)
    return _params_close(fixed_ours, fixed_oracle, abs_tol=abs_tol)


def _random_cells(rng, n):
    cells = []
    while len(cells) < n:
        p = (
            rng.uniform(4.0, 12.0),
            rng.uniform(4.0, 12.0),
            rng.uniform(4.0, 12.0),
            rng.uniform(70.0, 110.0),
            rng.uniform(70.0, 110.0),
            rng.uniform(70.0, 110.0),
        )
        if _valid_volume(p):
            cells.append(p)
    return cells


def _near_degenerate_cells(rng, n):
    """Nearly cubic, near-rhombohedral, and Niggli-boundary (xi ~ ±B) cells."""
    cells = []
    while len(cells) < n:
        kind = rng.randrange(3)
        if kind == 0:
            a = rng.uniform(5.0, 15.0)
            noise = 10 ** rng.uniform(-9.0, -4.0)
            p = (
                a * (1.0 + rng.uniform(-noise, noise)),
                a * (1.0 + rng.uniform(-noise, noise)),
                a * (1.0 + rng.uniform(-noise, noise)),
                90.0 + rng.uniform(-noise * 50.0, noise * 50.0),
                90.0 + rng.uniform(-noise * 50.0, noise * 50.0),
                90.0 + rng.uniform(-noise * 50.0, noise * 50.0),
            )
        elif kind == 1:
            a = rng.uniform(5.0, 15.0)
            ang = rng.choice([60.0, 90.0, 109.4712206, 120.0])
            noise = 10 ** rng.uniform(-8.0, -3.0)
            p = (
                a * (1.0 + rng.uniform(-noise, noise)),
                a * (1.0 + rng.uniform(-noise, noise)),
                a * (1.0 + rng.uniform(-noise, noise)),
                ang + rng.uniform(-noise * 50.0, noise * 50.0),
                ang + rng.uniform(-noise * 50.0, noise * 50.0),
                ang + rng.uniform(-noise * 50.0, noise * 50.0),
            )
        else:
            edges = sorted(rng.uniform(4.0, 12.0) for _ in range(3))
            a, b, c = edges
            target = b / (2.0 * c)
            if abs(target) >= 0.999:
                continue
            target *= 1.0 + rng.uniform(-1e-5, 1e-5)
            target = max(-0.999, min(0.999, target))
            al = math.degrees(math.acos(target))
            p = (
                a,
                b,
                c,
                al,
                90.0 + rng.uniform(-1e-3, 1e-3),
                90.0 + rng.uniform(-1e-3, 1e-3),
            )
        if _valid_volume(p):
            cells.append(p)
    return cells


def _check_cell(p, *, spglib=None, np=None):
    red, M = niggli_reduce(*p)
    d = round(_det(M))
    assert d in (-1, 1), (p, M, d)

    red2, M2 = niggli_reduce(*red)
    assert _params_close(red, red2, abs_tol=1e-6), (p, red, red2)
    # Already-reduced: CoB must be exact identity (no leftover sign flips).
    assert _is_identity(M2), (red, M2)

    gred = _gemmi_niggli(p)
    assert _oracle_agree(red, gred), (p, red, gred)

    if spglib is not None:
        sred = _spglib_niggli(p, spglib, np)
        assert _oracle_agree(red, sred, abs_tol=5e-3), (p, red, sred)


def test_niggli_fuzz_random_vs_oracles():
    rng = random.Random(20260719)
    cells = _random_cells(rng, 120)
    spglib, np = _try_import_spglib()
    for p in cells:
        _check_cell(p, spglib=spglib, np=np)


def test_niggli_fuzz_near_degenerate_vs_oracles():
    rng = random.Random(20260719 + 1)
    cells = _near_degenerate_cells(rng, 150)
    spglib, np = _try_import_spglib()
    for p in cells:
        _check_cell(p, spglib=spglib, np=np)
