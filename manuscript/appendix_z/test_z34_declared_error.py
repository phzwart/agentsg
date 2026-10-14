"""z:declared-error — relating reduced cells under a declared error radius D.

Error model: G' = G_Niggli + D*E with E random symmetric, ||E||_F = 1 (the
ideal's own Frobenius norm, so D, the sqrt(3) D widening and R(D) are
commensurate). On the Scotty bases: the Niggli basis jumps chambers at
errors far below the noise floor; the radius-one ball and the widened
closure both recover the relation, but the ball's certificate R(D) < 2 is
an anisotropy budget D < 4*lmin - lmax that is negative for 7 of the 17
Scotty cells, while the widening certificate is anisotropy-independent.
"""
from __future__ import annotations

import csv
import itertools
import math
import os
import random

import pytest

pytestmark = [pytest.mark.zcheck]

_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "agentsg",
                    "tests", "data", "scotty_cases.csv")

_NEGATIVE = {"2d4k", "1vdp", "5wey", "4nl6", "1jr7", "2r6s", "8cdf"}


def _np():
    import numpy
    return numpy


def _metric(cell):
    from agentsg.cell.metric import UnitCell
    return _np().array(UnitCell(*cell).metric_tensor(), dtype=float)


def _bases():
    from agentsg.cell.selling_cob import _primitive_or_self
    out = {}
    with open(os.path.normpath(_CSV)) as fh:
        for r in csv.DictReader(fh):
            cell = (float(r["a"]), float(r["b"]), float(r["c"]),
                    float(r["alpha"]), float(r["beta"]), float(r["gamma"]))
            out[r["pdb"]] = _primitive_or_self(cell, r["sg_hm"])
    return out


def _dmax(G):
    lam = _np().linalg.eigvalsh(G)
    return 4.0 * lam[0] - lam[-1], math.sqrt(lam[-1] / lam[0])


def test_anisotropy_budget_of_the_scotty_bases():
    """D_max = 4*lmin - lmax of the Niggli Gram: negative for 7 of 17."""
    from agentsg.cell.reduction import niggli_reduce

    neg = set()
    vals = {}
    for pdb, prim in _bases().items():
        red, _ = niggli_reduce(*prim)
        d, r0 = _dmax(_metric(red))
        vals[pdb] = d
        if d < 0:
            neg.add(pdb)
    assert neg == _NEGATIVE, neg
    assert 1600 < vals["2yht"] < 1900          # the certified workhorse
    assert 800 < vals["4amt"] < 1000
    assert vals["2d4k"] < -900                 # lysozyme: deeply uncertified


def _rand_sym_unit(rng):
    np = _np()
    A = np.array([[rng.gauss(0, 1) for _ in range(3)] for _ in range(3)])
    E = 0.5 * (A + A.T)
    return E / np.linalg.norm(E)


def _sperms():
    np = _np()
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                S[p, i] = s
            out.append(S)
    return out


def _sweep(pdb, D, n, seed):
    """Return per-direction records for one Scotty base at one error radius."""
    np = _np()
    from agentsg.cell.canonical import canonical_superbase, closure_distance
    from agentsg.cell.deformation import _ball_min, _radius, _unimodular_ball
    from agentsg.cell.metric import params_from_metric
    from agentsg.cell.reduction import niggli_reduce

    prim = _bases()[pdb]
    red0, _ = niggli_reduce(*prim)
    GN = _metric(red0)
    C0, _ = canonical_superbase(red0)
    maxvon = max(abs(np.array(v) @ GN @ np.array(v)) for v in C0)
    ball1 = _unimodular_ball(1)
    sperms = _sperms()
    rng = random.Random(seed)
    recs = []
    while len(recs) < n:
        E = _rand_sym_unit(rng)
        Gp = GN + D * E
        if np.linalg.eigvalsh(Gp)[0] <= 0:
            continue
        cellp = tuple(params_from_metric(Gp.tolist()))
        try:
            redp, Np = niggli_reduce(*cellp)
        except RuntimeError:
            continue
        GP = _metric(redp)
        v1, _ = _ball_min(GN, GP, ball1)
        jumped = not any(np.allclose(np.array(Np, float), S) for S in sperms)
        br = math.sqrt(3.0) * D / maxvon
        _, _, resid = closure_distance(red0, redp, boundary_rel=br)
        recs.append({
            "ball_ok": v1 <= 1.05 * D + 1e-6,
            "certified": _radius(GN, GP, v1) < 2.0,
            "jumped": jumped,
            "closure_ok": resid is not None and resid <= 1.05 * D + 1e-6,
        })
    return recs


def test_anisotropic_column_search_on_the_needle():
    """How far to search, per column: each column of M is a B-lattice vector
    whose squared length matches the corresponding diagonal of G_A within D
    (|E_jj| <= ||E||_F = D). On the P6_3 needle against itself at D = 50 A^2
    the three shells hold 6, 6 and 2 vectors (72 column triples) while the
    isotropic entry box at floor(R) = 7 holds 15^9 matrices; the next
    candidate enters the long shell only at D ~ 125 A^2, the in-plane stack
    of length 232.7 A that shadows the 233 A axis."""
    np = _np()
    H = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
    G = _metric(H)
    lam = np.linalg.eigvalsh(G)
    D = 50.0
    R = math.sqrt((lam[-1] + D) / lam[0])
    assert 7.0 < R < 8.0
    box = (2 * math.floor(R) + 1) ** 9
    assert box == 15 ** 9

    # enumerate all lattice vectors with coordinates up to the per-column bound
    nmax = int(math.floor(math.sqrt((G[2, 2] + D) / lam[0])))
    rng_n = range(-nmax, nmax + 1)
    vecs = np.array([(i, j, k) for i in rng_n for j in rng_n for k in rng_n
                     if (i, j, k) != (0, 0, 0)], dtype=float)
    lens = np.einsum("ni,ij,nj->n", vecs, G, vecs)
    counts = []
    for jj in range(3):
        target = G[jj, jj]
        counts.append(int(np.sum(np.abs(lens - target) <= D)))
    assert counts == [6, 6, 2]
    assert counts[0] * counts[1] * counts[2] == 72

    # the near-miss that anisotropy hides: in-plane stacks at norm-form 31
    gap = G[2, 2] - 31.0 * G[0, 0]
    assert 120.0 < gap < 130.0          # enters the c-shell only at D ~ 125
    n_wide = int(np.sum(np.abs(lens - G[2, 2]) <= 130.0))
    assert n_wide > 2                   # and then it genuinely enters


def test_certified_base_2yht():
    """2yht (D_max ~ 1739): at D = 20 A^2 (0.3% of ||G||) the Niggli basis
    jumps in most directions; ball-1 recovers and is certified; the widened
    closure at sqrt(3) D recovers with its own certificate."""
    recs = _sweep("2yht", D=20.0, n=8, seed=13)
    assert all(r["ball_ok"] for r in recs)
    assert all(r["certified"] for r in recs)
    assert sum(r["jumped"] for r in recs) >= 3
    assert sum(r["closure_ok"] for r in recs) >= 6


def test_uncertified_base_2d4k_lysozyme():
    """2d4k (D_max < 0): ball-1 still finds the right value but can never
    prove it -- R(D) >= 2 at every error level; the widened closure
    recovers with the anisotropy-independent certificate."""
    recs = _sweep("2d4k", D=20.0, n=5, seed=13)
    assert all(r["ball_ok"] for r in recs)
    assert not any(r["certified"] for r in recs)
    assert sum(r["closure_ok"] for r in recs) >= 4
