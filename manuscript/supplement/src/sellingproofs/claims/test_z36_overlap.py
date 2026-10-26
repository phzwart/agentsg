"""z:overlap — the widened closure is a union of on-cone Kurlin guarantees.

Two directions. Overlap: for lattices within eps (Frobenius on G, shared
frame), each one's EXACT closure lies inside the other's widened closure at
tau = 3 eps, same integer coordinates -- because closure-member coordinates
have entries <= 1, so each conorm moves by at most 3 eps. Clipping: every
widened member is an EXACT obtuse superbase (hence a Kurlin-certified
closure member) of an explicitly constructed lattice within 2*sqrt(6)*tau.
So no guarantee is ever used off the cone: the band is bookkeeping for a
family of exact closures, one per admissible perturbation.
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


@claim("z36", "SI §S4", kind="sampled")
def test_parent_closure_inside_widened_daughter():
    r"""On all five Voronoi types, the closure of a parent lattice lies inside the widened set of a measured daughter, with penetration at most 3 eta."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase, superbase_variants
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.selling_closure import selling_superbase_closure

    rng = random.Random(3)
    for vt in (1, 2, 3, 4, 5):
        for _ in range(2):
            cell = cell_from_conorms(
                [x * (25 + 10 * rng.random()) for x in TYPE_CONORMS[vt]()])
            G = np.array(UnitCell(*cell).metric_tensor())
            A = np.array([[rng.gauss(0, 1) for _ in range(3)]
                          for _ in range(3)])
            E = 0.5 * (A + A.T)
            E /= np.linalg.norm(E)
            eps = 0.01 * np.linalg.norm(G)
            Gp = G + eps * E
            if np.linalg.eigvalsh(Gp)[0] <= 0:
                continue
            cellp = tuple(params_from_metric(Gp.tolist()))
            parent = selling_superbase_closure(cell)
            pen = max(max(0.0, -min(_conorms(m, Gp))) for m in parent)
            assert pen <= 3.0 * eps + 1e-9, (vt, pen / eps)
            C0, _ = canonical_superbase(cellp)
            scale = max(abs(np.array(v) @ Gp @ np.array(v)) for v in C0)
            wid = superbase_variants(cellp, boundary_rel=3.0 * eps / scale,
                                     max_variants=512)
            assert _canonset(parent) <= _canonset(wid), vt


@claim("z36", "SI §S4", kind="sampled")
def test_clipping_yields_exact_member_of_nearby_lattice():
    r"""Clipping the negative conorms of a widened member gives an exact obtuse superbase of a lattice within 2 sqrt6 tau of the original."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase, superbase_variants
    from agentsg.cell.metric import UnitCell

    delta = 0.8  # two small conorms put the lattice near walls
    cell = cell_from_conorms((delta, 50.0, 75.0, 100.0, 125.0, 2.0 * delta))
    G = np.array(UnitCell(*cell).metric_tensor())
    C0, _ = canonical_superbase(cell)
    scale = max(abs(np.array(v) @ G @ np.array(v)) for v in C0)
    tau = 3.0 * (3.0 * delta)  # a band wide enough to cross both walls
    wid = superbase_variants(cell, boundary_rel=tau / scale, max_variants=512)
    n_clipped = 0
    for m in wid:
        p = _conorms(m, G)
        if min(p) >= -1e-9:
            continue  # already exactly obtuse
        assert min(p) >= -tau * 2.0 - 1e-6  # within the (2x) band
        # rebuild the member Gram in its own frame and clip there
        order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pm = dict(zip(order, p))
        q = [max(0.0, pm[k]) for k in order]
        Hc = np.array(_gram_from_p(q))
        Hm = np.array(_gram_from_p([pm[k] for k in order]))
        n_clipped += 1
        # exact obtuseness of the standard superbase under the clipped Gram
        std = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        assert min(_conorms(std, Hc)) >= -1e-9
        assert np.linalg.eigvalsh(Hc)[0] > 0
        assert np.linalg.norm(Hc - Hm) <= 2.0 * math.sqrt(6.0) * (2.0 * tau)
    assert n_clipped >= 1  # the band genuinely contained non-obtuse frames


@claim("z36", "SI §S4, From a metric error to a band", kind="exhaustive")
def test_conorm_shift_identity_and_coverage_constant():
    r"""A metric error dG shifts the conorm p_ij by exactly -u_i^T dG u_j; the worst-case constants are sqrt2 (one zero), sqrt5 (two zeros), and 2 sqrt3 in six of the sixteen cuboid frames."""
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


@claim("z36", "SI §S4, From a metric error to a band", kind="sampled")
def test_frame_free_bound_sqrt2_eta_star():
    r"""Measured in each member's own frame, no conorm of any member shifts by more than sqrt2 eta*."""
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
