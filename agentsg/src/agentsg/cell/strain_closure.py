"""Widened Selling closure in the strain norm, with tiered completeness guarantees.

The strain norm of a metric perturbation dG at G is

    ||dG||_G = || G^(-1/2) dG G^(-1/2) ||_F ,

the same in every basis. A superbase F is *within eps of obtuse* for G when some
metric G' with ||G' - G||_G <= eps makes F obtuse (all six conorms >= 0); the
distance is a non-negative least-squares problem in the six conorms of F. The
*widened closure* S_eps(G) is the set of such superbases (manuscript, Section 3.2).

Tiers (manuscript, Section 3.3 and Theorem A12):

* tier 1 -- filter the first adjacency layer L1 = U.F0 (127 superbases up to
  relabelling). Proved complete when no superbase outside L1 is within eps, which
  :func:`widened_closure` certifies by an exact search of the finite ball of
  Lemma A12. For cells of ordinary shape this holds up to eps = 0.25.
* tier 2 -- filter the second layer L2 (2755 superbases), certified the same way;
  if that certificate also fails, enumerate the ball of Lemma A12 exactly.

The default band is Le Page-style: one cell angle changed by 3 degrees, a strain of
sqrt(2) sin 3 deg = 0.074 (:data:`LEPAGE_STRAIN`).

Reindexing (:func:`reindex`) pairs the new cell's single reduced superbase with every
member of the reference's widened closure and every relabelling; by the overlap
corollary the true operator is among the candidates whenever the two metrics differ
by at most the band. Candidates are ranked by their strain price.

Needs numpy and scipy (``pip install agentsg[db]``).
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

from ..linalg import Matrix3
from .metric import UnitCell
from .selling_cob import _primitive_or_self, primitive_matrix, selling_matrix

LEPAGE_STRAIN = math.sqrt(2.0) * math.sin(math.radians(3.0))
"""Strain of one cell angle changed by 3 degrees (Le Page-style tier-1 band)."""

_SL = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))      # slots da, db, dc, ab, ac, bc
_S0 = ((-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def _np():
    try:
        import numpy as np
    except ImportError as exc:                             # pragma: no cover
        raise ImportError("strain_closure needs numpy and scipy: pip install agentsg[db]") from exc
    return np


@lru_cache(maxsize=1)
def _data():
    np = _np()
    path = os.path.join(os.path.dirname(__file__), "data", "selling_operators.json")
    with open(path) as fh:
        raw = json.load(fh)
    U = np.array(raw["U"], dtype=float)
    L1 = np.array(raw["L1"], dtype=float)
    L2 = np.array(raw["L2"], dtype=float)
    return U, L1, L2, {_key(S) for S in L1}, {_key(S) for S in L2}


def _key(S):
    a = tuple(sorted(tuple(int(round(x)) for x in v) for v in S))
    b = tuple(sorted(tuple(-int(round(x)) for x in v) for v in S))
    return min(a, b)


def _gram(p):
    np = _np()
    return np.array([[p[0] + p[3] + p[4], -p[3], -p[4]],
                     [-p[3], p[1] + p[3] + p[5], -p[5]],
                     [-p[4], -p[5], p[2] + p[4] + p[5]]], float)


def _E():
    np = _np()
    return np.array([_gram(np.eye(6)[k]) for k in range(6)])


def conorms(S, G):
    """six conorms -u_i.u_j of the superbase S (4 rows, reduced-cell coordinates) under G."""
    np = _np()
    S = np.asarray(S, float)
    P = -S @ G @ S.T
    return np.array([P[i, j] for i, j in _SL])


def _metric_M(Gf):
    """(n,6,6): the strain norm as a quadratic form on conorm perturbations, in frames with Grams Gf."""
    np = _np()
    Gi = np.linalg.inv(Gf)
    A = np.einsum("nab,kbc->nkac", Gi, _E())
    return np.einsum("nkab,nlba->nkl", A, A)


def distance_to_obtuse(G, S):
    """strain distance from G to the nearest metric for which the superbase S is obtuse."""
    np = _np()
    from scipy.optimize import nnls
    q = conorms(S, G)
    if q.min() >= 0:
        return 0.0
    S = np.asarray(S, float)
    M = _metric_M((S[1:] @ G @ S[1:].T)[None])[0]
    R = np.linalg.cholesky(M).T
    x, _ = nnls(R, R @ q)
    r = x - q
    return float(np.sqrt(max(r @ M @ r, 0.0)))


# ---- reduction ----------------------------------------------------------------
@dataclass
class ReducedCell:
    """A cell, its primitive Selling-reduced basis and metric.

    ``basis`` (exact, 3x3) has the reduced basis vectors as columns in the input
    (conventional) basis, so that ``G_red = basis^T G_input basis``."""
    cell: tuple
    sg_hm: str | None
    basis: Matrix3
    G: object                                           # numpy (3,3)


def reduce_cell(cell, sg_hm=None) -> ReducedCell:
    np = _np()
    cell = tuple(float(x) for x in cell)
    prim = _primitive_or_self(cell, sg_hm)
    B = primitive_matrix(sg_hm) @ selling_matrix(prim)
    Bf = np.array([[float(x) for x in row] for row in B.rows])
    Gin = np.array(UnitCell(*cell).metric_tensor(), float)
    G = Bf.T @ Gin @ Bf
    p = conorms(np.array(_S0, float), G)
    G = _gram(np.maximum(p, 0.0))                       # clean round-off on exact walls
    return ReducedCell(cell, sg_hm, B, G)


# ---- Lemma A12: the finite ball and the certificate ----------------------------
def _class_minima(G):
    np = _np()
    S0 = np.array(_S0, float)
    out = {}
    for s in itertools.product((0, 1), repeat=4):
        if 0 < sum(s) < 4 and s[0] == 0:
            v = np.array(s) @ S0
            out[tuple(int(x) % 2 for x in v)] = float(v @ G @ v)
    return out


def _ball_vectors(G, eps):
    np = _np()
    mu = _class_minima(G)
    f = (1 + eps) / (1 - eps)
    R2 = f * max(mu.values())
    bound = np.floor(np.sqrt(R2 * np.diag(np.linalg.inv(G)))).astype(int)
    V = np.array(np.meshgrid(*[np.arange(-b, b + 1) for b in bound], indexing="ij")).reshape(3, -1).T
    V = V[np.any(V % 2 != 0, 1)].astype(float)
    n2 = np.einsum("ia,ab,ib->i", V, G, V)
    lim = np.array([f * mu[tuple(int(x) % 2 for x in v)] for v in V])
    return V[n2 <= lim * (1 + 1e-12)]


def _ball_superbases(G, eps):
    """every superbase within eps of obtuse (exact, Lemma A12): yields (key, S, distance)."""
    np = _np()
    V = _ball_vectors(G, eps)
    n = np.sqrt(np.einsum("ia,ab,ib->i", V, G, V))
    ok = -(V @ G @ V.T) >= -eps * np.outer(n, n) - 1e-9 * abs(G).max()   # Prop. A11 prefilter
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
                k = _key(S)
                if k in seen:
                    continue
                seen.add(k)
                d = distance_to_obtuse(G, S)
                if d <= eps * (1 + 1e-9):
                    yield k, S, d


def adjacency_radius(G, layer=1, cap=0.25):
    """delta_layer(G): strain distance to the nearest superbase outside the layer (exact
    search in the ball of Lemma A12 at ``cap``); returns ``cap`` when none is closer."""
    _, _, _, K1, K2 = _data()
    keys = K1 if layer == 1 else K2
    best = cap
    for k, _S, d in _ball_superbases(G, cap):
        if k not in keys and d < best:
            best = d
    return best


# ---- the widened closure --------------------------------------------------------
@dataclass
class Member:
    superbase: object        # numpy (4,3): integer rows in reduced-cell coordinates
    depth: float             # strain distance to obtuse


@dataclass
class WidenedClosure:
    reduced: ReducedCell
    eps: float
    tier: int                # 1 or 2: smallest layer holding the whole set; 3: neither; 0: uncertified L1 filter
    members: list = field(default_factory=list)

    def __len__(self):
        return len(self.members)


def widened_closure(cell, sg_hm=None, eps=LEPAGE_STRAIN, certify=True) -> WidenedClosure:
    """S_eps of the cell, and the tier that answers it.

    eps must be below 1/4. With ``certify`` (default) the ball of Lemma A12 is searched,
    which is exact; ``tier`` then records the smallest adjacency layer that contains the
    whole set (1: the 127 superbases of L1, 2: the 2755 of L2, 3: neither), i.e. how far
    a layer filter alone would have been complete. Without ``certify`` only L1 is
    filtered (fast, 127 candidates) and ``tier`` is 0: complete only if eps is below the
    cell's adjacency radius, which is not checked."""
    if not 0 <= eps < 0.25:
        raise ValueError("eps must be in [0, 0.25)")
    red = cell if isinstance(cell, ReducedCell) else reduce_cell(cell, sg_hm)
    G = red.G
    _, L1, L2, K1, K2 = _data()
    if not certify:
        members = [Member(S, d) for S in L1 if (d := distance_to_obtuse(G, S)) <= eps * (1 + 1e-9)]
        members.sort(key=lambda m: m.depth)
        return WidenedClosure(red, eps, 0, members)
    outside1 = outside2 = False
    exact = {}
    for k, S, d in _ball_superbases(G, eps):
        exact[k] = (S, d)
        outside1 |= k not in K1
        outside2 |= k not in K2
    tier = 1 if not outside1 else (2 if not outside2 else 3)
    if tier < 3:
        layer = L1 if tier == 1 else L2
        members = []
        for S in layer:
            d = distance_to_obtuse(G, S)
            if d <= eps * (1 + 1e-9):
                members.append(Member(S, d))
        if {_key(m.superbase) for m in members} != set(exact):   # consistency: the certificate and the filter agree
            raise RuntimeError("layer filter and ball enumeration disagree")
    else:
        members = [Member(S, d) for S, d in exact.values()]
    members.sort(key=lambda m: m.depth)
    return WidenedClosure(red, eps, tier, members)


# ---- reindexing ------------------------------------------------------------------
@dataclass
class Candidate:
    operator: Matrix3        # P with (a_N b_N c_N) = (a_R b_R c_R) P, exact
    price: float             # strain || G_N^-1/2 (s P^T G_R P - G_N) G_N^-1/2 ||_F after the best scale s
    scale: float
    depth: float             # the member's depth in the reference's widened closure
    determinant: int


def _relabellings(S):
    """the 48 bases (three of the four vectors, in order, with a common sign) of a superbase."""
    np = _np()
    for d in range(4):
        rest = [i for i in range(4) if i != d]
        for perm in itertools.permutations(rest):
            B = S[list(perm)]
            yield B
            yield -B


def _strain(G_new, G_pred):
    np = _np()
    w, V = np.linalg.eigh(G_new)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    X = Gm @ (G_pred - G_new) @ Gm
    return float(np.linalg.norm(X))


def reindex(reference, new_cell, new_sg_hm=None, eps=LEPAGE_STRAIN, accept=None, fit_scale=True):
    """Candidate operators from the reference to a new cell, ranked by strain price.

    ``reference`` is a :class:`WidenedClosure` (from :func:`widened_closure`). The new
    cell is reduced once. Returns (candidates, tier): every candidate whose price is
    at most ``accept`` (default ``eps``), each P mapping the reference's input basis to
    the new cell's input basis, ``h_N = P^T h_R``."""
    np = _np()
    accept = eps if accept is None else accept
    ref = reference
    new = reduce_cell(new_cell, new_sg_hm)
    G_R_in = np.array(UnitCell(*ref.reduced.cell).metric_tensor(), float)
    G_N_in = np.array(UnitCell(*new.cell).metric_tensor(), float)
    BN_inv = new.basis.inverse()
    out, seen = [], set()
    for m in ref.members:
        for Brows in _relabellings(np.asarray(m.superbase)):
            Mt = Matrix3([[Fraction(int(round(Brows[j][i]))) for j in range(3)] for i in range(3)])
            if Mt.det() not in (1, -1):
                continue
            P = ref.reduced.basis @ Mt @ BN_inv
            Pf = np.array([[float(x) for x in row] for row in P.rows])
            G_pred = Pf.T @ G_R_in @ Pf
            s = 1.0
            if fit_scale:
                s = float(np.sum(G_pred * G_N_in) / np.sum(G_pred * G_pred))
            price = _strain(G_N_in, s * G_pred)
            if price > accept:
                continue
            k = tuple(P.rows)
            if k in seen:
                continue
            seen.add(k)
            out.append(Candidate(P, price, s, m.depth, int(P.det())))
    out.sort(key=lambda c: c.price)
    return out, ref.tier


# ---- distances to the 24 Delone sorts --------------------------------------------
_SORT_PATTERNS = {
    "C1": "rrrrrr", "C3": "rr0rr0", "C5": "000rrr", "H4": "00rrrs", "T1": "rrsrrs", "T2": "rr0rrs",
    "T5": "000rrs", "R1": "rrrsss", "R3": "rr0sr0", "O1A": "rrsrrt", "O1B": "rstrst", "O2": "rs0srt",
    "O3": "rs0rs0", "O4": "00rsst", "O5": "000rst", "M1A": "rrsttu", "M1B": "rstrsu", "M2A": "rs0stu",
    "M2B": "rs0rst", "M3": "rs0ts0", "M4": "00rstu", "A1": "rstuvw", "A2": "rs0tuv", "A3": "rs0tu0",
}
BRAVAIS = {"C1": "cI", "C3": "cF", "C5": "cP", "H4": "hP", "T1": "tI", "T2": "tI", "T5": "tP", "R1": "hR",
           "R3": "hR", "O1A": "oF", "O1B": "oI", "O2": "oI", "O3": "oI", "O4": "oS", "O5": "oP",
           "M1A": "mC", "M1B": "mC", "M2A": "mC", "M2B": "mC", "M3": "mC", "M4": "mP",
           "A1": "aP", "A2": "aP", "A3": "aP"}
_S6_TO_SLOT = (5, 4, 3, 0, 1, 2)                      # SELLA order bc,ac,ab,ad,bd,cd -> slots


def _classes(pattern):
    cls = {}
    for n, ch in enumerate(pattern):
        if ch != "0":
            cls.setdefault(ch, []).append(_S6_TO_SLOT[n])
    return list(cls.values())


@lru_cache(maxsize=1)
def _L_all():
    np = _np()
    U = _data()[0]
    Ufull = np.concatenate([-(U.sum(1))[:, None, :], U], 1)
    L = np.zeros((len(U), 6, 6))
    for m in range(6):
        E = _gram(np.eye(6)[m])
        P = -np.einsum("nia,ab,njb->nij", Ufull, E, Ufull)
        L[:, :, m] = np.stack([P[:, i, j] for i, j in _SL], 1)
    return np.round(L)


def sort_spectrum(cell, sg_hm=None):
    """{sort: strain distance} from the cell to each of Delone's 24 sorts, minimised over the
    frames of the universal operator set (manuscript, 'Distance to each sort')."""
    np = _np()
    red = cell if isinstance(cell, ReducedCell) else reduce_cell(cell, sg_hm)
    Q = _L_all() @ conorms(np.array(_S0, float), red.G)
    M = _metric_M(np.stack([_gram(q) for q in Q]))
    out = {}
    for name, pat in _SORT_PATTERNS.items():
        classes = _classes(pat)
        A = np.zeros((6, len(classes)))
        for c, cl in enumerate(classes):
            A[cl, c] = 1
        AMA = np.einsum("ka,nkl,lb->nab", A, M, A)
        AMq = np.einsum("ka,nkl,nl->na", A, M, Q)
        y = np.linalg.solve(AMA, AMq[..., None])[..., 0]
        X = y @ A.T
        for k in np.where((y < -1e-12).any(1))[0]:
            active = list(range(len(classes)))
            while True:
                Aa = A[:, active]
                yy = np.linalg.solve(Aa.T @ M[k] @ Aa, Aa.T @ M[k] @ Q[k]) if active else np.zeros(0)
                if (yy >= -1e-12).all():
                    break
                active.pop(int(np.argmin(yy)))
            X[k] = Aa @ yy if active else 0.0
        R = X - Q
        out[name] = float(np.sqrt(np.maximum(np.einsum("nk,nkl,nl->n", R, M, R), 0)).min())
    return out


__all__ = [
    "LEPAGE_STRAIN", "BRAVAIS", "ReducedCell", "reduce_cell", "conorms", "distance_to_obtuse",
    "adjacency_radius", "Member", "WidenedClosure", "widened_closure", "Candidate", "reindex",
    "sort_spectrum",
]
