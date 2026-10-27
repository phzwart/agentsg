"""z68 — the Niggli window (Proposition A12) and the way from a reduced cell to the Niggli
cell (Corollary A13).

The finite computation is in sellingproofs.niggli_window: for each of the 38 faces, the
vectors shortest in their parity classes (exact, by Lemma A1), the ordered bases among them
that are Minkowski reduced somewhere on the face (a linear program; a basis is dropped only
with an exact Farkas certificate), and the superbases among them that are obtuse somewhere on
the face (likewise).  The exhaustive claims check every pair; the sampled claims run the
library's own Selling and Niggli reductions on seeded cells.
"""
from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from sellingproofs import niggli_window as nw
from sellingproofs.registry import claim

from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.reduction import niggli_gk

S0 = np.array(sc._S0, float)


@cache
def _summary():
    return nw.summary()


def _floats(M):
    return np.array([[float(x) for x in r] for r in (M.rows if hasattr(M, "rows") else M)])


def _unimodular(rng, k=2):
    while True:
        U = rng.integers(-k, k + 1, (3, 3))
        if round(abs(np.linalg.det(U))) == 1:
            return U


def _strained(G, eta, rng):
    w, V = np.linalg.eigh(G)
    Gh = V @ np.diag(w ** .5) @ V.T
    X = rng.normal(size=(3, 3)); X = X + X.T; X *= eta / np.linalg.norm(X)
    return Gh @ (np.eye(3) + X) @ Gh


@claim("z68", "main Proposition A12", kind="exhaustive")
def test_class_shortest_vectors_depend_on_the_face_only():
    """The vectors shortest in their parity classes, ties included, depend on the zero
    pattern only: 14 for the truncated octahedron, 16 for the hexarhombic dodecahedron, 18
    for the rhombic dodecahedron, 20 for the hexagonal prism and 26 for the cuboid, and on
    seeded metrics of every face they are exactly the class minima found by enumeration."""
    sizes = {nw.voronoi_type(Z): len(nw.class_shortest(Z)) for Z in nw.FACES}
    assert sizes == {"truncated octahedron": 14, "hexarhombic dodecahedron": 16,
                     "rhombic dodecahedron": 18, "hexagonal prism": 20, "cuboid": 26}
    rng = np.random.default_rng(680)
    X = np.array([x for x in itertools.product(range(-6, 7), repeat=3) if any(v % 2 for v in x)])
    for Z in nw.FACES:
        V = {tuple(v) for v in nw.class_shortest(Z)}
        for _ in range(3):
            p = rng.uniform(1, 10, 6) * np.exp(rng.uniform(0, np.log(4), 6))
            p[list(Z)] = 0
            G = sc._gram(p)
            q = np.einsum("ni,ij,nj->n", X, G, X)
            found = set()
            for cls in itertools.product((0, 1), repeat=3):
                if any(cls):
                    sel = np.all(X % 2 == cls, axis=1)
                    found |= {tuple(x) for x in X[sel][q[sel] <= q[sel].min() * (1 + 1e-9)]}
            assert found == V, Z


@claim("z68", "main Proposition A12", "SI §S11", kind="exhaustive", needs=("slow",))
def test_niggli_window_on_exact_lattices():
    """Proposition A12(i): on every face, any two ordered bases of class-shortest vectors that
    are Minkowski reduced somewhere on the face differ by a matrix with entries in
    {-1, 0, 1}: 174 912 ordered bases, 8 256 Minkowski reduced somewhere, 3 849 216 pairs.
    Every Niggli cell is one of these bases, so two Niggli cells of metrics with the same zero
    pattern, in the coordinates of a common obtuse superbase, differ by such a matrix."""
    S = _summary()
    assert sum(s["triples"] for s in S) == 174912
    assert sum(s["admissible"] for s in S) == 8256
    assert sum(s["window_pairs"] for s in S) == 3849216
    assert max(s["window_max"] for s in S) == 1


@claim("z68", "main Proposition A12", "SI §S11", kind="certified", needs=("slow",))
def test_dropped_bases_have_farkas_certificates():
    """Every ordered basis of class-shortest vectors left out of Proposition A12 and
    Corollary A13, 166 656 of them, is left out with an exact rational Farkas certificate that
    its Minkowski conditions fail everywhere on its face; none is left out without one.  The
    obtuse superbases, found exactly from their partial sums, are also exactly those that a
    linear program finds obtuse somewhere on the face."""
    S = _summary()
    assert sum(s["certified"] for s in S) == 166656
    assert sum(s["uncertified"] for s in S) == 0
    assert all(s["superbases_lp_agree"] for s in S)


@claim("z68", "main Corollary A13", "main §5.4", "SI §S11", kind="exhaustive", needs=("slow",))
def test_from_a_reduced_cell_to_the_niggli_cell():
    """Corollary A13: from every obtuse superbase of every metric, in every one of its 24
    ordered bases, the Niggli cell is reached by a matrix with entries in {-1, 0, 1}.  The
    superbases obtuse somewhere on a face number 1, 2, 6 (two zeros sharing a vector), 3 (two
    disjoint zeros) and 16 up to sign, as in Lemma A2, and the pairs checked number
    6 856 704.  The matrices that occur are exactly the 1 344 ordered bases drawn from the 14
    vectors +-(u1, u2, u3, u1+u2, u1+u3, u2+u3, u1+u2+u3), the signed partial sums of the
    superbase, so the Niggli vectors are three of the partial sums of every obtuse superbase;
    672 have determinant +1, and T and -T give the same cell."""
    S = _summary()
    counts = {}
    for s in S:
        counts.setdefault(s["type"], set()).add(s["superbases"] // 2)
    assert counts == {"truncated octahedron": {1}, "hexarhombic dodecahedron": {2},
                      "hexagonal prism": {6}, "rhombic dodecahedron": {3}, "cuboid": {16}}
    assert sum(s["landing_pairs"] for s in S) == 6856704
    assert max(s["landing_max"] for s in S) == 1
    found = set().union(*(s["landing_matrices"] for s in S))
    sums = [np.array(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]]
    vecs = sums + [-v for v in sums]
    closed = {tuple(np.array([vecs[i], vecs[j], vecs[k]]).T.ravel())
              for i, j, k in itertools.permutations(range(14), 3)
              if abs(round(np.linalg.det(np.array([vecs[i], vecs[j], vecs[k]])))) == 1}
    assert found == closed and len(closed) == 1344
    assert sum(round(np.linalg.det(np.array(t).reshape(3, 3))) == 1 for t in closed) == 672


@claim("z68", "main Corollary A13", kind="sampled")
def test_reductions_of_the_library_agree():
    """The library's Selling reduction and its Niggli reduction (Grosse-Kunstleve et al.,
    2004), run on seeded cells on and slightly off every face, in random bases, give a Niggli
    cell that satisfies the ten Minkowski conditions and lies within {-1, 0, 1} of each of the
    24 ordered bases of the reduced superbase."""
    rng = np.random.default_rng(681)
    worst, n = 0, 0
    for Z in nw.FACES:
        for rep in range(10):
            p = rng.uniform(1, 10, 6) * np.exp(rng.uniform(0, np.log(3), 6))
            p[list(Z)] = 0
            if rep % 2:
                p = p + rng.uniform(0, 1e-3, 6) * p.max() * np.isin(range(6), Z)
            U = _unimodular(rng)
            cell = tuple(params_from_metric((U.T @ sc._gram(p) @ U).tolist()))
            M = _floats(niggli_gk(cell)[1])
            Bs = _floats(sc.reduce_cell(cell).basis)
            T0 = np.linalg.solve(Bs, M)
            # the Niggli cell satisfies the ten Minkowski conditions (relative tolerance 1e-9)
            Gr = Bs.T @ np.array(UnitCell(*cell).metric_tensor(), float) @ Bs
            A = nw.minkowski_rows(np.rint(T0).astype(int))
            pr = sc.conorms(S0, Gr)
            assert np.all(A @ pr >= -1e-9 * np.abs(pr).sum()), (Z, rep)
            for i, j, k in itertools.permutations(range(4), 3):
                worst = max(worst, np.abs(np.rint(np.linalg.solve(S0[[i, j, k]].T, T0))).max())
                n += 1
    assert n == 38 * 10 * 24 and worst <= 1


RHO = {"mP": ((30, 40, 50, 90, 100, 90), 10.4), "oC": ((25, 50, 25, 90, 106.3, 90), 9.4),
       "hP": ((40, 40, 60, 90, 90, 120), 23.5), "oP": ((30, 40, 50, 90, 90, 90), 30.5),
       "tP": ((40, 40, 60, 90, 90, 90), 38.1), "cP": ((40, 40, 40, 90, 90, 90), 50.0),
       "aP": ((30, 40, 50, 85, 95, 100), 1.9)}


@cache
def _rho(name):
    Z, gap, adm = nw.rho(RHO[name][0])
    return min(gap, adm)


@claim("z68", "main Proposition A12", "main §5.4", "SI Table S21", kind="sampled", needs=("slow",))
def test_window_radius_examples():
    """Proposition A12(ii), the radius rho below which two measurements of one lattice have
    Niggli cells within {-1, 0, 1} of each other (computed numerically): 10.4 % for the
    monoclinic 30 x 40 x 50 A, 100 deg cell, 9.4 % for a C-centred orthorhombic lattice, 23.5 % hP,
    30.5 % oP, 38.1 % tP, 50 % cP and 1.9 % for a generic triclinic cell."""
    for name, (_, r) in RHO.items():
        assert abs(100 * _rho(name) - r) < 0.05, (name, 100 * _rho(name))


@claim("z68", "main Proposition A12", kind="sampled", needs=("slow",))
def test_window_on_measured_cells_within_the_radius():
    """Two measurements of each example lattice, each within 0.9 rho of it and the second in a
    random basis, have Niggli cells related by a matrix with entries in {-1, 0, 1}."""
    rng = np.random.default_rng(682)
    for name, (cell, _) in RHO.items():
        G = np.array(UnitCell(*cell).metric_tensor(), float)
        Bs = _floats(sc.reduce_cell(cell).basis)
        G = Bs.T @ G @ Bs                                      # primitive (the oC cell is given primitive)
        r = _rho(name)
        for _ in range(40):
            U = _unimodular(rng)
            c1 = tuple(params_from_metric(_strained(G, 0.9 * r * rng.uniform(), rng).tolist()))
            c2 = tuple(params_from_metric((U.T @ _strained(G, 0.9 * r * rng.uniform(), rng) @ U).tolist()))
            M1, M2 = _floats(niggli_gk(c1)[1]), _floats(niggli_gk(c2)[1])
            T = np.linalg.solve(M1, U @ M2)             # Niggli cell of c1 to that of c2
            assert np.abs(np.rint(T)).max() <= 1, (name, T)


@claim("z68", "main Corollary A13", "SI §S11", kind="sampled")
def test_niggli_from_selling_on_many_cells():
    """The library's Niggli cell read off the reduced superbase agrees with its Niggli
    reduction on 15 200 seeded cells (400 per zero pattern: on the face, slightly to either
    side, and with tied conorms, in random bases), of both Niggli types, with a right-handed
    basis; 516 of the 672 right-handed matrices occur."""
    import warnings
    rng = np.random.default_rng(5)
    seen, n, types = set(), 0, set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for Z in nw.FACES:
            for rep in range(400):
                p = rng.uniform(1, 10, 6) * np.exp(rng.uniform(0, np.log(8), 6))
                p[list(Z)] = 0
                if rep % 4 == 1:
                    p = p + rng.uniform(0, 1e-3, 6) * p.max() * np.isin(range(6), Z)
                if rep % 4 == 2:
                    p = p + rng.uniform(-1e-3, 0, 6) * p.max() * np.isin(range(6), Z)
                if rep % 4 == 3:
                    q = np.ones(6)
                    q[rng.choice(6, 2, replace=False)] = rng.integers(1, 4, 2)
                    p = q * rng.uniform(1, 3)
                U = _unimodular(rng)
                cell = tuple(params_from_metric((U.T @ sc._gram(p) @ U).tolist()))
                a = sc.niggli_from_selling(cell)
                b, _ = niggli_gk(cell)
                assert max(abs(x - y) / max(1, abs(y)) for x, y in zip(a.cell, b)) <= 1e-6, (Z, rep)
                assert a.basis.det() == 1
                seen.add(tuple(tuple(int(x) for x in r) for r in a.operator.rows))
                types.add(a.niggli_type)
                n += 1
    assert n == 15200 and types == {"I", "II"}
    assert len(seen) == 516
