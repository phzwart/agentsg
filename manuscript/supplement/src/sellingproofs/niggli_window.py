"""The finite computation behind Proposition A12 (the Niggli window) and Corollary A13
(from a reduced cell to the Niggli cell).

Coordinates.  Fix a face Z of the cone of conorms: a set of slots whose conorms vanish,
with the remaining slots connected (the 38 zero patterns).  A metric G* on Z has an obtuse
superbase F0 = (v0, v1, v2, v3) with conorms p, p_Z = 0 and p_i > 0 off Z.  Lattice vectors
are written in the basis (v1, v2, v3) of F0, and G* = sum_k p_k E_k with E_k the Gram
matrix of slot k (agentsg's ``_gram``).  Every squared length and every scalar product is
then a linear form in p.

V(Z).  By Lemma A1 a vector with superbase coefficients s + 2k (s in {0,1}^4, s0 = k0 = 0)
is shortest in its parity class for G* exactly when dk (dk + ds) = 0 on every slot ij with
p_ij > 0.  This depends on Z only, and it bounds |k_i| <= 3, so the vectors shortest in
their classes are listed exactly in the window [-7, 7]^3.

Admissible triples.  An ordered triple B = (a, b, c) of V(Z) that is a basis is Minkowski
reduced at p when |a| <= |b| <= |c|, |b| <= |b +- a| and |c| <= |c + x a + y b|
(x, y in {-1, 0, 1}, not both zero).  These ten conditions are linear in p; the triple is
*admissible on Z* when they hold at some p in the relative interior of Z (p_Z = 0,
p_i >= 1 off Z, the conditions being homogeneous).  Every Niggli cell is Minkowski reduced
(Gruber, 1973), and its vectors are shortest in their classes, so every Niggli cell of every
metric on Z is an admissible triple of Z.

Exactness.  Admissibility is decided by a linear program.  A triple the program accepts is
kept, which can only enlarge the set that is checked.  A triple it rejects is dropped only
with an exact Farkas certificate: rational y >= 0 with y^T A <= 0 on the free conorms and
y^T A != 0, so that y^T (A p) < 0 for every p in the relative interior, and the conditions
A p >= 0 cannot all hold.  The conclusions are therefore exact.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from functools import lru_cache

import numpy as np

SLOTS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _gram(k: int) -> np.ndarray:
    from agentsg.cell import strain_closure as sc
    return np.asarray(sc._gram(np.eye(6)[k]), float)


E = None


def _E():
    global E
    if E is None:
        E = [np.rint(_gram(k)).astype(int) for k in range(6)]
    return E


def connected(Z) -> bool:
    par = list(range(4))

    def f(x):
        while par[x] != x:
            x = par[x]
        return x
    for k in range(6):
        if k not in Z:
            i, j = SLOTS[k]
            par[f(i)] = f(j)
    return len({f(i) for i in range(4)}) == 1


FACES = [Z for r in range(4) for Z in itertools.combinations(range(6), r) if connected(Z)]


def voronoi_type(Z) -> str:
    if len(Z) == 0:
        return "truncated octahedron"
    if len(Z) == 1:
        return "hexarhombic dodecahedron"
    if len(Z) == 2:
        a, b = (set(SLOTS[k]) for k in Z)
        return "rhombic dodecahedron" if not a & b else "hexagonal prism"
    return "cuboid"


@lru_cache(maxsize=None)
def class_shortest(Z) -> np.ndarray:
    """V(Z): the vectors (in the basis v1, v2, v3 of F0) shortest in their nonzero parity
    classes for every metric on face Z, ties included."""
    out = []
    for x in itertools.product(range(-7, 8), repeat=3):
        if not any(v % 2 for v in x):
            continue
        c = np.array((0,) + x)
        s = c % 2
        k = (c - s) // 2
        if all(k_dk(k, s, i, j) == 0 for m, (i, j) in enumerate(SLOTS) if m not in Z):
            out.append(x)
    return np.array(out, dtype=int)


def k_dk(k, s, i, j):
    dk = k[i] - k[j]
    return dk * (dk + s[i] - s[j])


def norm_row(u) -> np.ndarray:
    E_ = _E()
    return np.array([u @ E_[k] @ u for k in range(6)], dtype=int)


def dot_row(u, v) -> np.ndarray:
    E_ = _E()
    return np.array([u @ E_[k] @ v for k in range(6)], dtype=int)


def minkowski_rows(B) -> np.ndarray:
    """The ten Minkowski conditions on the columns (a, b, c) of B, as rows A with A p >= 0."""
    a, b, c = B.T
    rows = [norm_row(b) - norm_row(a), norm_row(c) - norm_row(b)]
    rows += [norm_row(b + x * a) - norm_row(b) for x in (-1, 1)]
    rows += [norm_row(c + x * a + y * b) - norm_row(c)
             for x, y in itertools.product((-1, 0, 1), repeat=2) if x or y]
    return np.array(rows, dtype=int)


def feasible(A, Z) -> bool:
    from scipy.optimize import linprog
    r = linprog(np.zeros(6), A_ub=-A, b_ub=np.zeros(len(A)),
                bounds=[(0, 0) if k in Z else (1, None) for k in range(6)], method="highs")
    return r.status == 0


def farkas_certificate(A, Z):
    """Rational y >= 0 with y^T A <= 0 on the free conorms and y^T A != 0, verified exactly;
    None if no such y was found."""
    from scipy.optimize import linprog
    free = [k for k in range(6) if k not in Z]
    Af = A[:, free]
    m, n = Af.shape
    r = linprog(Af.sum(1), A_ub=np.vstack([Af.T, np.ones((1, m))]), b_ub=np.r_[np.zeros(n), 1],
                bounds=[(0, None)] * m, method="highs")
    if r.status != 0:
        return None
    for den in (10 ** 3, 10 ** 6, 10 ** 9):
        y = [Fraction(v).limit_denominator(den) if v > 1e-12 else Fraction(0) for v in r.x]
        s = [sum(y[i] * int(Af[i, j]) for i in range(m)) for j in range(n)]
        if all(v <= 0 for v in s) and any(v < 0 for v in s):
            return y
    return None


def ordered_bases(V):
    for t in itertools.combinations(range(len(V)), 3):
        for perm in itertools.permutations(t):
            B = V[list(perm)].T
            if abs(round(np.linalg.det(B))) == 1:
                yield B


@lru_cache(maxsize=None)
def admissible(Z):
    """(triples kept, number of ordered bases of V(Z), number dropped with a verified
    certificate, number rejected without one)."""
    kept, total, certified, uncertified = [], 0, 0, 0
    for B in ordered_bases(class_shortest(Z)):
        total += 1
        A = minkowski_rows(B)
        if feasible(A, Z):
            kept.append(B)
        elif farkas_certificate(A, Z) is not None:
            certified += 1
        else:
            uncertified += 1
            kept.append(B)                      # never drop a triple without a certificate
    return tuple(kept), total, certified, uncertified


def obtuse_rows(F) -> np.ndarray:
    """The six conditions -v_i.v_j >= 0 of an obtuse superbase, as rows linear in p."""
    return np.array([-dot_row(F[i], F[j]) for i, j in itertools.combinations(range(4), 2)],
                    dtype=int)


@lru_cache(maxsize=None)
def superbases(Z):
    """(obtuse, candidates, LP agrees): the superbases (v0, v1, v2, v3), v0+v1+v2+v3 = 0,
    with all four vectors in V(Z), and among them the obtuse ones, those whose seven partial
    sums are all in V(Z) (exact: a superbase is obtuse exactly when its partial sums are
    shortest in their classes, step (3) of the enumeration of Appendix A).  As a cross-check,
    whether the obtuse ones are exactly those for which the six conditions v_i.v_j <= 0 hold
    somewhere on Z by linear programming.  Each unordered superbase appears four times, once
    for each choice of v0."""
    V = class_shortest(Z)
    Vs = {tuple(v) for v in V}
    obtuse, total, agree = [], 0, True
    for t in itertools.combinations(range(len(V)), 3):
        v1, v2, v3 = V[list(t)]
        v0 = -(v1 + v2 + v3)
        if tuple(v0) not in Vs or abs(round(np.linalg.det(np.array([v1, v2, v3])))) != 1:
            continue
        F = (v0, v1, v2, v3)
        total += 1
        sums = [v1 + v2, v1 + v3, v2 + v3]
        ok = all(tuple(s) in Vs or tuple(-s) in Vs for s in sums)
        if ok:
            obtuse.append(F)
        agree &= ok == feasible(obtuse_rows(F), Z)
    return tuple(obtuse), total, agree


@lru_cache(maxsize=None)
def candidate_superbases(Z):
    """Every superbase with all four vectors in V(Z), obtuse on Z or not (each unordered
    superbase four times, once for each choice of v0)."""
    V = class_shortest(Z)
    Vs = {tuple(v) for v in V}
    out = []
    for t in itertools.combinations(range(len(V)), 3):
        v1, v2, v3 = V[list(t)]
        v0 = -(v1 + v2 + v3)
        if tuple(v0) in Vs and abs(round(np.linalg.det(np.array([v1, v2, v3])))) == 1:
            out.append((v0, v1, v2, v3))
    return tuple(out)


def landing_max_any_superbase(Z):
    """The largest entry of the matrix from an ordered basis of any candidate superbase of Z,
    obtuse or not, to a kept triple."""
    kept = np.array(admissible(Z)[0], float)
    big = 0
    for F in candidate_superbases(Z):
        for i, j, k in itertools.permutations(range(4), 3):
            Bi = np.linalg.inv(np.array([F[i], F[j], F[k]]).T)
            big = max(big, int(np.abs(np.rint(np.einsum("ij,njk->nik", Bi, kept))).max()))
    return big


def window_entries(Z):
    """The largest entry of B^-1 B' over all pairs of kept triples of Z, and the pair count."""
    kept = admissible(Z)[0]
    inv = [np.linalg.inv(B) for B in kept]
    big = 0
    for Bi in inv:
        for B2 in kept:
            big = max(big, int(np.abs(np.rint(Bi @ B2)).max()))
    return big, len(kept) ** 2


def landing_entries(Z):
    """The largest entry of the matrix from an ordered basis of a superbase of Z to a kept
    triple, over all, the number of pairs, and the distinct matrices from the basis
    (v1, v2, v3) of F0 itself."""
    kept = admissible(Z)[0]
    big, pairs = 0, 0
    for F in superbases(Z)[0]:
        for i, j, k in itertools.permutations(range(4), 3):
            Bi = np.linalg.inv(np.array([F[i], F[j], F[k]]).T)
            for B in kept:
                pairs += 1
                big = max(big, int(np.abs(np.rint(Bi @ B)).max()))
    from_F0 = {tuple(map(tuple, B)) for B in kept}
    return big, pairs, from_F0


def rho(cell):
    """rho(G*) = min(eta_gap, eta_adm) for the lattice of ``cell`` (computed numerically).
    eta_gap = (g - 1)/(g + 1), g the smallest ratio over the classes of the squared length of
    the shortest vector outside V(Z) to that of the class minimum; eta_adm the strain distance
    from G* to the nearest metric at which a triple of V(Z) not admissible on Z is Minkowski
    reduced."""
    from scipy.optimize import minimize
    from agentsg.cell import strain_closure as sc
    red = sc.reduce_cell(cell)
    G = np.asarray(red.G, float)
    S0 = np.array(sc._S0, float)
    p0 = np.asarray(sc.conorms(S0, G), float)
    Z = tuple(k for k in range(6) if p0[k] < 1e-9 * p0.max())
    V = class_shortest(Z)
    Vs = {tuple(v) for v in V}
    X = np.array([x for x in itertools.product(range(-6, 7), repeat=3) if any(v % 2 for v in x)])
    q = np.einsum("ni,ij,nj->n", X, G, X)
    g = np.inf
    for cls in itertools.product((0, 1), repeat=3):
        if not any(cls):
            continue
        sel = np.all(X % 2 == cls, axis=1)
        inV = np.array([tuple(x) in Vs for x in X[sel]])
        g = min(g, q[sel][~inV].min() / q[sel][inV].min())
    eta_gap = (g - 1) / (g + 1)
    M = sc._metric_M(G[None])[0]
    kept = {tuple(map(tuple, B)) for B in admissible(Z)[0]}
    best = np.inf
    for B in ordered_bases(V):
        if tuple(map(tuple, B)) in kept:
            continue
        A = minkowski_rows(B).astype(float)
        cons = {"type": "ineq", "fun": lambda p, A=A: A @ p, "jac": lambda p, A=A: A}
        r = minimize(lambda p: (p - p0) @ M @ (p - p0), p0 + 1e-3 * p0.max(),
                     jac=lambda p: 2 * M @ (p - p0), constraints=[cons], method="SLSQP",
                     options={"maxiter": 200, "ftol": 1e-14})
        if r.success:
            best = min(best, float(np.sqrt(max(r.fun, 0))))
    return Z, float(eta_gap), best


def face_summary(Z) -> dict:
    """Every number of Proposition A12(i) and Corollary A13 for one face."""
    kept, total, cert, uncert = admissible(Z)
    sk, stot, sagree = superbases(Z)
    wbig, wpairs = window_entries(Z)
    lbig, lpairs, _ = landing_entries(Z)
    mats = set()
    for F in sk:
        for i, j, k in itertools.permutations(range(4), 3):
            Bi = np.linalg.inv(np.array([F[i], F[j], F[k]]).T)
            for B in kept:
                mats.add(tuple(np.rint(Bi @ B).astype(int).ravel()))
    return dict(face=Z, type=voronoi_type(Z), V=len(class_shortest(Z)), triples=total,
                admissible=len(kept), certified=cert, uncertified=uncert, window_pairs=wpairs,
                window_max=wbig, superbase_candidates=stot // 4, superbases=len(sk) // 4,
                superbases_lp_agree=sagree,
                landing_pairs=lpairs, landing_max=lbig, landing_matrices=mats,
                landing_max_any=landing_max_any_superbase(Z))


def summary(processes: int | None = None) -> list[dict]:
    """face_summary for all 38 faces, in parallel."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(processes) as ex:
        return list(ex.map(face_summary, FACES))
