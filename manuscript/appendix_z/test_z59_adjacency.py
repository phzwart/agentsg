"""z:adjacency — completeness of project--enumerate--filter in the strain norm,
proved up to the adjacency radius of the measured cell.

Lemma A12 (ball). If F is within eps < 1 of obtuse for G, every vector of F in
class c has |u|^2_G <= (1+eps)/(1-eps) mu_c(G), mu_c the squared length of F0's
partial sum in that class. (Prop. A11 with u = v gives (1-eps)|u|^2_G <=
|u|^2_G' <= (1+eps)|u|^2_G for the metric G' within eps that makes F obtuse;
F's vectors are shortest in their classes for G', and F0's are for G.)
So S_eps(G) is a finite set, enumerable over the lattice vectors in that ball.

Theorem A12 (adjacency). Call F adjacent to F0 when some positive definite
metric makes both obtuse; then F = W F0 with W in the universal set U (Appendix
D), since both are members of that lattice's closure. Let delta(G) be the
smallest strain distance from G to a superbase not adjacent to F0. If
eps < min(delta(G), 1/4): every F in S_eps(G) is adjacent (its nearest obtuse
metric is within eps < delta), so S_eps(G) = {W F0 : W in U, within eps}; and
no projection within 4 eps < 1 is singular (a singular metric differs from G
by strain >= 1 along its null vector), so project--enumerate--filter returns
S_eps(G): F's own frame and zero pattern are among the (frame, face) pairs
projected onto, at distance <= eps, and F is the base superbase of that
projection's closure. delta(G) is computed exactly from the ball of Lemma A12
at eps = 1/4.

Checked here: the ball bound on every member found; delta for needles, plates
and isotropic cells (delta ~ sqrt2 x shortest/longest edge); for eps below
delta the filtered U.F0 equals the ball-enumerated S_eps and equals the
procedure's output; above delta the procedure can still be exact (the plates
at 3 degrees) but U.F0 is not.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytestmark = [pytest.mark.zcheck]

import test_z44_widened_complete as t  # noqa: E402
from test_z58_projection_closure import (  # noqa: E402
    NEEDLE, W_ALL, _reduced, construct_union, member_distance,
)

SL = t.SL
S0 = np.array(t.S0, float)
UF0 = {t._key(np.concatenate([-(w.sum(0, keepdims=True)), w]).astype(float)) for w in W_ALL}


def class_minima(G):
    """squared length of the standard superbase's partial sum in each of the seven
    parity classes: {parity tuple of the F0 coordinates: mu_c}."""
    out = {}
    for s in itertools.product((0, 1), repeat=4):
        if 0 < sum(s) < 4 and s[0] == 0:
            v = np.array(s) @ S0
            out[tuple(int(x) % 2 for x in v)] = float(v @ G @ v)
    return out


def ball_vectors(G, eps):
    """lattice vectors u with |u|^2 <= (1+eps)/(1-eps) mu_c(G) in their own class c
    (Lemma A12), as integer rows in F0 coordinates."""
    mu = class_minima(G)
    f = (1 + eps) / (1 - eps)
    R2 = f * max(mu.values())
    Gi = np.linalg.inv(G)
    bound = np.floor(np.sqrt(R2 * np.diag(Gi))).astype(int)
    rng = [np.arange(-b, b + 1) for b in bound]
    V = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T.astype(float)
    V = V[np.any(V.astype(int) % 2 != 0, 1)]            # the doubled class holds no superbase vector
    n2 = np.einsum("ia,ab,ib->i", V, G, V)
    lim = np.array([f * mu[tuple(int(x) % 2 for x in v)] for v in V])
    return V[n2 <= lim * (1 + 1e-12)]


def _near_obtuse_superbases(G, eps, V):
    """(key, superbase, distance) for every superbase with vectors in V whose conorms
    pass the Prop. A11 prefilter and whose strain distance to obtuse is <= eps."""
    n = np.sqrt(np.einsum("ia,ab,ib->i", V, G, V))
    ok = -(V @ G @ V.T) >= -eps * np.outer(n, n) - 1e-9 * abs(G).max()
    idx = {tuple(v.astype(int)): i for i, v in enumerate(V)}
    seen = set()
    for a in range(len(V)):
        bs = np.where(ok[a])[0]
        bs = bs[bs > a]
        for b in bs:
            cs = bs[(bs > b) & ok[b, bs]]
            if not len(cs):
                continue
            D = -(V[a] + V[b] + V[cs])
            js = np.array([idx.get(tuple(d.astype(int)), -1) for d in D])
            good = js >= 0
            good[good] &= ok[js[good], a] & ok[js[good], b] & ok[js[good], cs[good]]
            for c in cs[good]:
                Bm = np.array([V[a], V[b], V[c]])
                if abs(round(np.linalg.det(Bm))) != 1:
                    continue
                S = np.vstack([-(V[a] + V[b] + V[c]), Bm])
                k = t._key(S)
                if k in seen:
                    continue
                seen.add(k)
                d = member_distance(G, S)
                if d <= eps * (1 + 1e-9):
                    yield k, S, d


def superbases_within(G, eps):
    """{key: superbase}: the exact S_eps(G), by Lemma A12."""
    return {k: S for k, S, _ in _near_obtuse_superbases(G, eps, ball_vectors(G, eps))}


def adjacency_radius(G, cap=0.25, layer=None):
    """delta_k(G): smallest strain distance to a superbase outside the given layer
    (default U.F0), searched exactly in the ball of Lemma A12 at eps = cap;
    (delta, superbase), delta = cap when none is closer."""
    layer = UF0 if layer is None else layer
    best = (cap, None)
    for k, S, d in _near_obtuse_superbases(G, cap, ball_vectors(G, cap)):
        if k not in layer and d < best[0]:
            best = (d, S.astype(int).tolist())
    return best


def layers(kmax):
    """L_1 = U.F0, L_k = U.L_(k-1): sets of superbase keys in F0 coordinates."""
    Ws = [w.astype(float) for w in W_ALL]
    frontier = {t._key(S): S for S in (np.concatenate([-(w.sum(0, keepdims=True)), w]) for w in Ws)}
    seen = dict(frontier)
    L = [set(seen)]
    for _ in range(kmax - 1):
        new = {}
        for S in frontier.values():
            B = S[1:]
            for w in Ws:
                S2 = np.concatenate([-((w @ B).sum(0, keepdims=True)), w @ B])
                k = t._key(S2)
                if k not in seen:
                    seen[k] = S2
                    new[k] = S2
        frontier = new
        L.append(set(seen))
    return L


def filtered_U(G, eps):
    out = set()
    for w in W_ALL:
        S = np.concatenate([-(w.sum(0, keepdims=True)), w]).astype(float)
        if member_distance(G, S) <= eps * (1 + 1e-9):
            out.add(t._key(S))
    return out


CELLS = {
    "needle 5.6x5.2x132": (lambda: t._gram(NEEDLE), 0.0557),
    "plate 100x100.1x5": (lambda: _reduced((100.0, 100.1, 5.0, 90.03, 89.98, 90.05)), 0.0501),
    "plate 150x120x4.8": (lambda: _reduced((150.0, 120.0, 4.8, 90.0, 90.4, 90.0)), 0.0430),
    "92deg cell 30x40x50": (lambda: _reduced((30.02, 39.97, 50.03, 90.25, 92.05, 89.80)), 0.25),
    "needle 10x10x100": (lambda: _reduced((10.0, 10.0, 100.0, 90, 90, 90)), 0.1393),
    "needle 10x10x30": (lambda: _reduced((10.0, 10.0, 30.0, 90, 90, 90)), 0.25),
}


def test_ball_lemma_and_adjacency_radius():
    for name, (make, delta_expected) in CELLS.items():
        G = make()
        delta, _ = adjacency_radius(G)
        assert abs(delta - delta_expected) < 2e-4, (name, delta)
        mu = class_minima(G)
        for eps in (0.01, 0.05):
            mem = superbases_within(G, eps)
            bound = (1 + eps) / (1 - eps)
            for S in mem.values():
                for u in S:
                    assert float(u @ G @ u) <= bound * mu[tuple(int(x) % 2 for x in u)] * (1 + 1e-9), name


def test_adjacency_theorem_below_delta_and_its_failure_above():
    for name, (make, _) in CELLS.items():
        G = make()
        delta, _ = adjacency_radius(G)
        for eps in (0.6 * delta, 0.9 * delta):
            truth = set(superbases_within(G, eps))
            assert filtered_U(G, eps) == truth, (name, eps)
            got, _ = construct_union(G, eps)
            assert got == truth, (name, eps)
    # above delta the filtered U.F0 misses members, the projection does not (the plates at 3 deg)
    for name in ("plate 100x100.1x5", "plate 150x120x4.8"):
        G = CELLS[name][0]()
        eps = np.sin(np.radians(3.0))
        truth = set(superbases_within(G, eps))
        assert not filtered_U(G, eps) >= truth
        got, _ = construct_union(G, eps)
        assert got == truth
