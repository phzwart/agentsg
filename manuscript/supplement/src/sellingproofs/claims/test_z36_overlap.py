"""z36 — from a metric error to a band on the conorms (supporting information, Section S4).

A metric error dG shifts the conorm of a member with coordinate vectors u_i, u_j by exactly
-u_i^T dG u_j, at most c_ij xi with c_ij = ||sym(u_i u_j^T)||_F; the largest constant on the
zero conorms of each zero pattern, and the frame-free bound of Proposition S4.7.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools
import math
import random

import pytest

from ._helpers import TYPE_CONORMS, cell_from_conorms  # noqa: E402


def _np():
    import numpy
    return numpy


def _gram_from_p(p):
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _conorms(member, G):
    np = _np()
    W = [np.array(v, float) for v in member]
    return [-(W[i] @ G @ W[j]) for i in range(4) for j in range(i + 1, 4)]


def _canonset(members):
    out = set()
    for m in members:
        best = None
        for perm in itertools.permutations(range(4)):
            for s in (1, -1):
                t = tuple(tuple(int(s * x) for x in m[p]) for p in perm)
                if best is None or t < best:
                    best = t
        out.add(best)
    return out


@claim("z36", "SI §S4, From a metric error to a band", kind="exhaustive")
def test_conorm_shift_identity_and_coverage_constant():
    r"""A metric error dG shifts the conorm p_ij by exactly -u_i^T dG u_j, and the coverage
    bound tau_cov read off the closure holds for every error of norm xi.  The largest constant
    c_ij on a zero conorm is sqrt2 over the one-zero patterns, sqrt5 over the two-zero patterns,
    and 2 sqrt3 in six of the sixteen three-zero patterns (the paths with d in the interior),
    where an error of norm xi pushes a conorm below -3 xi, so tau = 3 xi is not a worst-case
    guarantee."""
    np = _np()
    from agentsg.cell.canonical import _closure_for_match
    from agentsg.cell.metric import UnitCell
    SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def connected(es):
        adj = {v: set() for v in range(4)}
        for a, b in es:
            adj[a].add(b)
            adj[b].add(a)
        seen, st = {0}, [0]
        while st:
            x = st.pop()
            for y in adj[x] - seen:
                seen.add(y)
                st.append(y)
        return len(seen) == 4

    def cn(M, G):
        return np.array([-M[i] @ G @ M[j] for i, j in SL])

    rnd = np.random.default_rng(36)
    wall_max, coord_max, path_cuboid = 0.0, 0, []
    by_zeros = {1: [], 2: [], 3: []}
    for r in range(4):
        for Z in itertools.combinations(range(6), r):
            if not connected([SL[k] for k in range(6) if k not in Z]):
                continue
            p = rnd.uniform(900, 2500, 6)
            p[list(Z)] = 0
            cell = tuple(cell_from_conorms(list(p)))
            G = np.array(UnitCell(*cell).metric_tensor())
            Ms = [np.array(M, float) for M in _closure_for_match(cell, boundary_rel=0.0)]
            coord_max = max(coord_max, int(max(np.abs(M).max() for M in Ms)))
            kap, cmax = 0.0, {}
            for M in Ms:
                pm = cn(M, G)
                for n, (i, j) in enumerate(SL):
                    u, v = M[i], M[j]
                    c = ((u @ u * (v @ v) + (u @ v) ** 2) / 2) ** 0.5
                    if abs(pm[n]) < 1e-9 * np.abs(G).max():
                        kap = max(kap, c)
                    E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2
                    assert np.isclose(cn(M, G + E)[n] - pm[n], -u @ E @ v)
            wall_max = max(wall_max, kap)
            if r:
                by_zeros[r].append(kap)
            zero_edges = [SL[k] for k in Z]
            deg = [sum(v in e for e in zero_edges) for v in range(4)]
            is_path = (r == 3 and connected(zero_edges) and max(deg) == 2
                       and sorted(deg) == [1, 1, 2, 2] and deg[0] == 2)  # v0 interior
            if is_path:
                path_cuboid.append(kap)
            else:
                assert kap <= 5 ** 0.5 + 1e-9, Z
            # coverage certificate vs an adversarial and random perturbations
            eta = 10.0
            tcov = max(max(((M[i] @ M[i] * (M[j] @ M[j]) + (M[i] @ M[j]) ** 2) / 2) ** 0.5
                           * eta - cn(M, G)[n] for n, (i, j) in enumerate(SL)) for M in Ms)
            for _ in range(20):
                E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2; E *= eta / np.linalg.norm(E)
                assert min(cn(M, G + E).min() for M in Ms) >= -tcov - 1e-9
            if is_path:
                M, n = max(((M, n) for M in Ms for n, (i, j) in enumerate(SL)
                            if abs(cn(M, G)[n]) < 1e-6),
                           key=lambda t: np.linalg.norm(np.outer(t[0][SL[t[1]][0]], t[0][SL[t[1]][1]])
                                                        + np.outer(t[0][SL[t[1]][1]], t[0][SL[t[1]][0]])))
                u, v = M[SL[n][0]], M[SL[n][1]]
                S = (np.outer(u, v) + np.outer(v, u)) / 2
                worst = min(cn(Mm, G + eta * S / np.linalg.norm(S)).min() for Mm in Ms)
                assert worst < -3 * eta and worst >= -tcov - 1e-9
    assert coord_max == 3                     # member vectors are not in {-1,0,1}
    assert abs(wall_max - 12 ** 0.5) < 1e-9
    assert path_cuboid and all(abs(k - 12 ** 0.5) < 1e-9 for k in path_cuboid)
    assert abs(max(by_zeros[1]) - 2 ** 0.5) < 1e-9
    assert abs(max(by_zeros[2]) - 5 ** 0.5) < 1e-9
    assert len(by_zeros[3]) == 16 and sum(abs(k - 12 ** 0.5) < 1e-9 for k in by_zeros[3]) == 6
    assert len(path_cuboid) == 6


@claim("z36", "SI §S4, Proposition S4.7", kind="sampled")
def test_frame_free_bound_sqrt2_eta_star():
    r"""Proposition S4.7: measured in a member's own basis, where the error has Frobenius norm
    eta*, no conorm of the member shifts by more than sqrt2 eta*."""
    np = _np()
    from agentsg.cell.canonical import _closure_for_match
    from agentsg.cell.metric import UnitCell
    S0 = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    cs = [((S0[i] @ S0[i] * (S0[j] @ S0[j]) + (S0[i] @ S0[j]) ** 2) / 2) ** 0.5
          for i in range(4) for j in range(i + 1, 4)]
    assert abs(max(cs) - 2 ** 0.5) < 1e-12
    rnd = np.random.default_rng(37)
    for Z in [(), (0,), (0, 1), (0, 5), (0, 1, 4), (3, 4, 5), (0, 3, 5)]:
        p = rnd.uniform(900, 2500, 6)
        p[list(Z)] = 0
        cell = tuple(cell_from_conorms(list(p)))
        G = np.array(UnitCell(*cell).metric_tensor())
        Ms = [np.array(M, float) for M in _closure_for_match(cell, boundary_rel=0.0)]
        for _ in range(100):
            E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2
            eta_star = max(np.linalg.norm(M[[k for k in range(4) if k != o]] @ E
                                          @ M[[k for k in range(4) if k != o]].T)
                           for M in Ms for o in range(4))
            shift = max(np.abs(np.array(_conorms(M, G + E)) - np.array(_conorms(M, G))).max()
                        for M in Ms)
            assert shift <= 2 ** 0.5 * eta_star + 1e-9
