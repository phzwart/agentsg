"""z:appendix-statements — the appendix statements that had no direct check of their own.

(i) Terms: a superbase is obtuse exactly when each of its seven partial sums is a shortest
vector of its parity class; at a zero conorm a.b = 0 the class of a+b has the two shortest
pairs +-(a+b), +-(a-b), and the flip at ab gives an obtuse superbase whose partial sum in
that class is b-a.
(ii) Lemma A3: on each of Delone's 24 sorts the obtuse superbases are exactly those reached
from the reduced one by flips at zero conorms, and every symmetry maps the reduced superbase
to one of them.
(iii) Proposition A4: |u^T dG v| <= eta |u| |v| when ||G^-1/2 dG G^-1/2||_F = eta.
(iv) Proposition A10 (iii)-(v), by brute-force enumeration of the self-maps and candidates
(no use of U or of the closure): a gap makes H_eps a group and the candidates one coset, and
the sandwich between a reference and a nearby symmetric lattice.
(v) Proposition A11 (i): the bound on the conorms of d for a type-II Niggli cell.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from agentsg.cell import strain_closure as sc

S0 = np.array(sc._S0, float)                       # rows d, a, b, c in the basis (a, b, c)
SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSETS = [s for s in itertools.product((0, 1), repeat=3) if any(s)]
SORTS = list(sc._SORT_PATTERNS.items())


def _rand_conorms(pattern, rng):
    p = np.zeros(6)
    for cl in sc._classes(pattern):
        p[cl] = rng.uniform(5, 100)
    return p


def _rand_G(rng, anis=3.0):
    p = rng.uniform(1, 100, 6) * np.exp(rng.uniform(0, np.log(anis), 6))
    return sc._gram(p)


def _vectors(G, R2):
    """all non-zero integer x with x^T G x <= R2."""
    r = int(np.ceil(np.sqrt(R2 / np.linalg.eigvalsh(G)[0]))) + 1
    X = np.array([x for x in itertools.product(range(-r, r + 1), repeat=3) if any(x)], float)
    q = np.einsum("ni,ij,nj->n", X, G, X)
    return X[q <= R2 * (1 + 1e-12)]


def _class_minima(G):
    """{parity pattern: squared length of the shortest vector of the class}."""
    big = max(np.diag(G)) * 9
    X = _vectors(G, big)
    q = np.einsum("ni,ij,nj->n", X, G, X)
    out = {}
    for x, v in zip(X.astype(int), q):
        k = tuple(x % 2)
        if any(k):
            out[k] = min(out.get(k, np.inf), v)
    return out


def _partials(B):
    """the seven partial sums of the superbase with basis columns B (in the basis (a, b, c))."""
    return [B @ np.array(s, float) for s in SUBSETS]


def _superbase_conorms(B, G):
    V = np.vstack([-(B.sum(1)), B.T])
    return np.array([-V[i] @ G @ V[j] for i, j in SL])


# (i) Terms ---------------------------------------------------------------------------

@claim("z61", "SI §S12, Terms", kind="sampled")
def test_obtuse_iff_partial_sums_are_shortest():
    r"""A superbase is obtuse exactly when each of its seven partial sums is a shortest vector of its parity class."""
    rng = np.random.default_rng(61)
    U = sc.universal_operators().astype(float)
    k0 = sc._key(S0)
    swaps = np.array([W for W in U if sc._key(np.vstack([-(W.sum(1)), W.T])) == k0])
    assert len(swaps) == 48
    n_obtuse = n_not = 0
    for _ in range(40):
        G = _rand_G(rng)
        mins = _class_minima(G)
        for W in np.concatenate([swaps, U[rng.choice(len(U), 60, replace=False)]]):
            obtuse = _superbase_conorms(W, G).min() >= -1e-9 * np.abs(G).max()
            shortest = all(abs(w @ G @ w - mins[tuple(w.astype(int) % 2)]) <= 1e-9 * np.abs(G).max()
                           for w in _partials(W))
            assert obtuse == shortest
            n_obtuse += obtuse; n_not += not obtuse
    assert n_obtuse >= 40 * 48 and n_not >= 1000


@claim("z61", "SI §S12, Terms", kind="sampled")
def test_zero_conorm_gives_two_shortest_pairs_and_the_flip():
    r"""At a zero conorm a.b = 0 the class of a+b has the shortest pairs +-(a+b) and +-(a-b), and the flip at ab gives an obtuse superbase whose partial sum there is b-a."""
    rng = np.random.default_rng(62)
    for _ in range(200):
        p = rng.uniform(5, 100, 6); p[3] = 0.0              # p_ab = 0
        G = sc._gram(p)
        mins = _class_minima(G)
        a, b = np.eye(3)[0], np.eye(3)[1]
        assert np.isclose((a + b) @ G @ (a + b), mins[(1, 1, 0)])
        assert np.isclose((a - b) @ G @ (a - b), mins[(1, 1, 0)])
        # flip at ab: a -> -a, b -> b, c -> c + a (and d -> d + a)
        B = np.column_stack([-a, b, np.eye(3)[2] + a])
        assert _superbase_conorms(B, G).min() >= -1e-9 * p.max()
        sums = [tuple(w.astype(int)) for w in _partials(B)]
        assert tuple((b - a).astype(int)) in sums or tuple((a - b).astype(int)) in sums


# (ii) Lemma A3 -----------------------------------------------------------------------

def _flip(V, i, j):
    """Selling's flip of the superbase rows V at the pair (i, j)."""
    W = V.copy()
    W[i] = -V[i]
    for k in range(4):
        if k not in (i, j):
            W[k] = V[k] + V[i]
    return W


@claim("z61", "main Lemma A3", kind="exhaustive")
def test_generation_by_flips_on_every_sort():
    r"""Lemma A3: on each of the 24 sorts the obtuse superbases are exactly those reached by flips at zero conorms, and every symmetry maps the reduced superbase to one of them."""
    rng = np.random.default_rng(63)
    U = sc.universal_operators().astype(float)
    for name, pattern in SORTS:
        G = sc._gram(_rand_conorms(pattern, rng))
        tol = 1e-9 * np.abs(G).max()
        seen = {sc._key(S0)}
        todo = [S0]
        while todo:
            V = todo.pop()
            for i, j in SL:
                for (s, t) in ((i, j), (j, i)):
                    if abs(V[s] @ G @ V[t]) <= tol:
                        F = _flip(V, s, t)
                        k = sc._key(F)
                        if k not in seen:
                            seen.add(k); todo.append(F)
        closure = set()
        for W in U:
            V = np.vstack([-(W.sum(1)), W.T])
            if _superbase_conorms(W, G).min() >= -tol:
                closure.add(sc._key(V))
        assert seen == closure, name
        for W in U:
            if np.abs(W.T @ G @ W - G).max() <= tol:
                assert sc._key(np.vstack([-(W.sum(1)), W.T])) in seen, name


# (iii) Proposition A4 -----------------------------------------------------------------

@claim("z61", "main Proposition A4", kind="sampled")
def test_strain_norm_bounds_every_dot_product():
    r"""Proposition A4: |u^T dG v| <= eta |u| |v| when the strain norm of dG is eta."""
    rng = np.random.default_rng(64)
    for _ in range(2000):
        G = _rand_G(rng, 10.0)
        w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
        X = rng.normal(size=(3, 3)); X = X + X.T
        eta = rng.uniform(0.001, 0.5); X *= eta / np.linalg.norm(X)
        dG = Gh @ X @ Gh
        u, v = rng.integers(-4, 5, 3).astype(float), rng.integers(-4, 5, 3).astype(float)
        assert abs(u @ dG @ v) <= eta * np.sqrt(u @ G @ u) * np.sqrt(v @ G @ v) * (1 + 1e-12) + 1e-12


# (iv) Proposition A10 (iii)-(v) ---------------------------------------------------------

def _unit(G):
    return G / np.cbrt(np.linalg.det(G))


def _maps(GR, GN, eps, proper=True):
    """all unimodular P (proper, or both signs) with geodesic price(GN, P^T GR P) <= eps,
    GR and GN of unit determinant: each column has |P e_i|^2_R <= e^eps |e_i|^2_N."""
    cols = [_vectors(GR, np.exp(eps) * GN[i, i]) for i in range(3)]
    out = []
    for x in cols[0]:
        for y in cols[1]:
            xy = np.cross(x, y)
            d = cols[2] @ xy
            for z in cols[2][np.abs(np.abs(d) - 1) < 1e-9]:
                P = np.column_stack([x, y, z])
                if proper and np.linalg.det(P) < 0:
                    continue
                if sc._price(GN, P.T @ GR @ P)[0] <= eps:
                    out.append(tuple(P.astype(int).ravel()))
    return set(out)


def _strained(G, nu, rng):
    """a metric at geodesic distance nu from G, of the same determinant."""
    w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
    X = rng.normal(size=(3, 3)); X = X + X.T; X -= np.trace(X) / 3 * np.eye(3)
    X *= nu / np.linalg.norm(X)
    l, Q = np.linalg.eigh(X)
    return Gh @ (Q @ np.diag(np.exp(l)) @ Q.T) @ Gh


def _mat(t):
    return np.array(t, float).reshape(3, 3)


@claim("z61", "main Proposition A10(v)", kind="sampled")
def test_sandwich_between_reference_and_symmetric_lattice():
    r"""Proposition A10(v), by brute-force enumeration: Aut(G0) lies in H_2nu(G_R) and H_eps(G_R) in H_(eps+2nu)(G0)."""
    rng = np.random.default_rng(65)
    for name, pattern in SORTS:
        G0 = _unit(sc._gram(_rand_conorms(pattern, rng)))
        nu = rng.uniform(0.002, 0.03)
        GR = _strained(G0, nu, rng)
        aut = {k for k in _maps(G0, G0, 1e-9, proper=False)}
        HR = _maps(GR, GR, 2 * nu + 1e-9, proper=False)
        assert aut <= HR, name                                           # Aut(G0) in H_2nu(G_R)
        eps = 2 * nu + rng.uniform(0, 0.05)
        HR_eps = _maps(GR, GR, eps, proper=False)
        H0 = _maps(G0, G0, eps + 2 * nu + 1e-9, proper=False)
        assert HR_eps <= H0, name                                         # H_eps(G_R) in H_eps+2nu(G0)
        if H0 == aut:
            assert HR_eps == aut, name


@claim("z61", "main Proposition A10(iii)-(iv)", kind="sampled")
def test_gap_gives_a_group_and_one_coset():
    r"""Proposition A10(iii)-(iv), by brute-force enumeration: with a price gap H_eps is a group and the candidates are one coset H_eps P0."""
    rng = np.random.default_rng(66)
    n_group = n_coset = 0
    for name, pattern in SORTS:
        for _ in range(3):
            GR = _unit(_strained(sc._gram(_rand_conorms(pattern, rng)), rng.uniform(0.005, 0.05), rng))
            top = 0.4
            H = {k: sc._price(GR, _mat(k).T @ GR @ _mat(k))[0] for k in _maps(GR, GR, top)}
            prices = sorted(H.values())
            for eps in np.linspace(0.01, top / 2, 20):
                if not any(eps < q <= 2 * eps for q in prices):           # (iii)
                    He = [k for k, q in H.items() if q <= eps]
                    for a in He:
                        for b in He:
                            assert tuple((_mat(a) @ _mat(b)).astype(int).ravel()) in He
                    n_group += 1
            P0 = np.eye(3)[rng.permutation(3)]
            P0[:, 0] += rng.integers(-1, 2) * P0[:, 1]
            if np.linalg.det(P0) < 0:
                P0[:, 2] *= -1
            p0 = rng.uniform(0.002, 0.02)
            GN = _unit(_strained(P0.T @ GR @ P0, p0, rng))
            p0 = sc._price(GN, P0.T @ GR @ P0)[0]
            for eps in np.linspace(0.03, top / 2 - p0, 10):
                if not any(eps - p0 < q <= eps + p0 for q in prices):     # (iv)
                    C = _maps(GR, GN, eps)
                    coset = {tuple((_mat(k) @ P0).astype(int).ravel()) for k, q in H.items() if q <= eps}
                    assert C == coset, (name, eps)
                    n_coset += 1
    assert n_group > 100 and n_coset > 50


# (v) Proposition A11 (i) -----------------------------------------------------------------

@claim("z61", "main Proposition A11(i)", kind="sampled")
def test_type_ii_niggli_bound_on_d():
    r"""Proposition A11(i): for a type-II Niggli cell the conorm of d with x is at least x.x/2 minus the smaller product of x, and conorms on d exceed small conorms off d."""
    rng = np.random.default_rng(67)
    n = 0
    while n < 3000:
        aa, bb, cc = np.sort(rng.uniform(100, 3000, 3))
        ab = -rng.uniform(0, aa / 2); ac = -rng.uniform(0, aa / 2); bc = -rng.uniform(0, bb / 2)
        G = np.array([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]])
        if np.linalg.eigvalsh(G)[0] <= 0:
            continue
        n += 1
        B = np.eye(3)
        p = _superbase_conorms(B, G)
        assert p.min() >= -1e-9                                                   # obtuse
        prods = {(0, 1): abs(ab), (0, 2): abs(ac), (1, 2): abs(bc)}
        for x in range(3):
            others = [prods[tuple(sorted((x, y)))] for y in range(3) if y != x]
            assert p[x] >= G[x, x] / 2 - min(others) - 1e-9
        tau = sorted(prods.values())[1] * 1.0001
        if tau < aa / 4:
            off = [q for q in p[3:] if q <= tau]
            assert all(pd > q for pd in p[:3] for q in off)


@claim("z61", "main Appendix A, the enumeration", "SI §S12", "main §3.3", kind="exhaustive")
def test_first_layer_counted_by_hand():
    """The 127 superbases of the first layer, pattern by pattern: the reduced cell, one new
    superbase for each of the 6 single zeros, 3 for each of the 12 pairs sharing a vector and
    each of the 4 triangles, 6 for each of the 12 paths, none for the 3 disjoint pairs;
    1 + 6 + 36 + 12 + 72 = 127."""
    import collections
    U = sc.universal_operators()
    faces = [Z for r in range(4) for Z in itertools.combinations(range(6), r) if _connected(Z)]
    assert len(faces) == 38
    rng = np.random.default_rng(611)
    members = {}
    for Z in faces:
        p = rng.uniform(1, 2, 6); p[list(Z)] = 0.0
        G = sc._gram(p)
        keys = set()
        for W in U:
            B = W.T.astype(float); V = np.vstack([-B.sum(0), B])
            if min(-V[i] @ G @ V[j] for i, j in SL) > -1e-9:
                keys.add(sc._key(V))
        members[Z] = keys
    new = collections.defaultdict(list)
    union = set()
    for Z in faces:
        sub = set().union(*[members[Y] for Y in faces if set(Y) < set(Z)]) if Z else set()
        new[_kind(Z)].append(len(members[Z] - sub))
        union |= members[Z]
    assert {k: (len(v), sorted(set(v))) for k, v in new.items()} == {
        "none": (1, [1]), "one": (6, [1]), "sharing": (12, [3]), "disjoint": (3, [0]),
        "triangle": (4, [3]), "path": (12, [6])}
    assert len(union) == 127 == 1 + 6 + 36 + 12 + 72


def _connected(Z):
    par = list(range(4))

    def f(x):
        while par[x] != x:
            x = par[x]
        return x
    for k in range(6):
        if k not in Z:
            i, j = SL[k]
            par[f(i)] = f(j)
    return len({f(i) for i in range(4)}) == 1


def _kind(Z):
    import collections
    E = [SL[k] for k in Z]
    if len(E) == 0:
        return "none"
    if len(E) == 1:
        return "one"
    deg = collections.Counter(x for e in E for x in e)
    if len(E) == 2:
        return "disjoint" if len(deg) == 4 else "sharing"
    return "triangle" if len(deg) == 3 else "path"
