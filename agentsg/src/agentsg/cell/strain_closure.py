"""Widened Selling closure in the strain norm, enumerated exactly.

The strain norm of a metric perturbation dG at G is

    ||dG||_G = || G^(-1/2) dG G^(-1/2) ||_F ,

the same in every basis (the linearised affine-invariant metric). A superbase F is
*within eps of obtuse* for G when some metric G' with ||G' - G||_G <= eps makes F
obtuse (all six conorms >= 0); the distance is a non-negative least-squares problem
in the six conorms of F. The *widened closure* S_eps(G) is the set of such
superbases (manuscript, Section 4.2).

:func:`widened_closure` enumerates S_eps exactly from the finite ball of Lemma A6
(every vector of a member has squared length at most (1+eps)/(1-eps) times the
shortest vector of its parity class). It also reports the *layer* of the result
(manuscript, Theorem A7): 1 when every member lies in L1 = U.F0, the 127 superbases
adjacent to the reduced cell; 2 when every member lies in L2 = U.L1 (2755); 3
otherwise. The layer describes how far from the reduced cell the set reaches; it is
not needed for completeness, which the ball enumeration provides for every eps < 1/4.

Bands. :data:`LEPAGE_STRAIN` = sqrt(2) sin 3 deg = 0.074 is the strain of one cell
angle changed by 3 degrees, the distance at which a pseudo-twofold with Le Page's
obliquity of 3 degrees sits from the sort it would fix. A twofold's *price* (its
strain as a self-map of the cell) is twice that distance (manuscript, Proposition
A9), so a reindexing band of 0.074 on prices admits pseudo-twofolds of obliquity up
to 1.5 degrees; :data:`LEPAGE_PRICE` = 0.148 admits those up to 3 degrees.

Reindexing (:func:`reindex`) pairs the new cell's single reduced superbase with every
member of the reference's widened closure and every relabelling, and prices each
candidate at the new cell's metric after the best common scale. If the true operator
has price at most ``accept`` there, it is among the candidates provided the reference
was widened at ``accept / (1 - accept)`` (manuscript, Corollary A8); :func:`band_for`
gives that number. Candidates with determinant -1 are dropped for a chiral crystal
(``proper=True``), and :func:`indexing_classes` groups the candidates by the
reference's Laue group (merging without anomalous signal) or point group (anomalous
data).

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
"""Strain of one cell angle changed by 3 degrees: the sort distance of a pseudo-twofold
with Le Page's obliquity of 3 degrees."""

LEPAGE_PRICE = 2.0 * LEPAGE_STRAIN
"""Price of a twofold of obliquity 3 degrees (twice its sort distance, Proposition A9)."""


def band_for(accept):
    """The band at which to widen a reference so that every operator of price at most
    ``accept``, measured at the new cell's metric, is among the candidates
    (Corollary A8): accept / (1 - accept)."""
    if not 0 <= accept < 1:
        raise ValueError("accept must be in [0, 1)")
    return accept / (1.0 - accept)

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


# ---- Lemma A6: the finite ball and the certificate ----------------------------
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
    """every superbase within eps of obtuse (exact, Lemma A6): yields (key, S, distance)."""
    np = _np()
    V = _ball_vectors(G, eps)
    n = np.sqrt(np.einsum("ia,ab,ib->i", V, G, V))
    ok = -(V @ G @ V.T) >= -eps * np.outer(n, n) - 1e-9 * abs(G).max()   # Prop. A4 prefilter
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
    search in the ball of Lemma A6 at ``cap``); returns ``cap`` when none is closer."""
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
    layer: int               # 1 or 2: smallest adjacency layer holding the whole set; 3: neither; 0: uncertified L1 filter
    members: list = field(default_factory=list)

    def __len__(self):
        return len(self.members)

    @property
    def tier(self):          # deprecated name, one release
        return self.layer


def widened_closure(cell, sg_hm=None, eps=LEPAGE_STRAIN, certify=True) -> WidenedClosure:
    """S_eps of the cell, exactly, and the adjacency layer that holds it.

    eps must be below 1/4. With ``certify`` (default) the ball of Lemma A6 is
    enumerated, which is exact for every such eps; ``layer`` then records the smallest
    adjacency layer that contains the whole set (1: the 127 superbases of L1, 2: the 2755
    of L2, 3: neither). Without ``certify`` only L1 is filtered (fast, 127 depths) and
    ``layer`` is 0: complete only if eps is below the cell's adjacency radius delta_1,
    which is not checked (Theorem A7)."""
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
        lay = L1 if tier == 1 else L2
        members = []
        for S in lay:
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
    scale: float             # s minimising the price (fitted in the strain norm)
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


def _strain(G_new, G_pred, fit_scale=True):
    """price of G_pred against G_new, || G_new^-1/2 (s G_pred - G_new) G_new^-1/2 ||_F,
    with s the scale minimising it (in the same norm) when ``fit_scale``; returns
    (price, s)."""
    np = _np()
    w, V = np.linalg.eigh(G_new)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    Y = Gm @ G_pred @ Gm
    s = float(np.trace(Y) / np.sum(Y * Y)) if fit_scale else 1.0
    return float(np.linalg.norm(s * Y - np.eye(3))), s


def reindex(reference, new_cell, new_sg_hm=None, accept=None, proper=None, fit_scale=True, eps=None):
    """Candidate operators from the reference to a new cell, ranked by strain price.

    ``reference`` is a :class:`WidenedClosure` (from :func:`widened_closure`). The new
    cell is reduced once. Returns (candidates, layer): every candidate whose price,
    measured at the new cell's metric after the best common scale, is at most
    ``accept``; each P maps the reference's input basis to the new cell's input basis,
    ``h_N = P^T h_R``. The default ``accept`` is the largest band the reference
    certifies, ``ref.eps / (1 + ref.eps)`` (Corollary A8; see :func:`band_for`).
    ``proper=True`` keeps only candidates of determinant +1 (a chiral crystal);
    ``None`` keeps both signs. ``eps`` is a deprecated alias of ``accept``."""
    np = _np()
    ref = reference
    if accept is None:
        accept = eps if eps is not None else ref.eps / (1.0 + ref.eps)
    if accept > ref.eps / (1.0 + ref.eps) * (1 + 1e-9):
        raise ValueError(f"accept={accept:.4f} needs the reference widened at band_for(accept)="
                         f"{band_for(accept):.4f}; it was widened at {ref.eps:.4f}")
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
            det = int(P.det())
            if proper is True and det != 1:
                continue
            if proper is False and det != -1:
                continue
            Pf = np.array([[float(x) for x in row] for row in P.rows])
            G_pred = Pf.T @ G_R_in @ Pf
            price, s = _strain(G_N_in, G_pred, fit_scale)
            if price > accept:
                continue
            k = tuple(P.rows)
            if k in seen:
                continue
            seen.add(k)
            out.append(Candidate(P, price, s, m.depth, det))
    out.sort(key=lambda c: c.price)
    return out, ref.layer


def price_spectrum(reference, proper=None):
    """Prices of every self-map of the reference within its band (the operators H_eps
    that carry the reference onto itself), sorted; the gaps in this list are where a
    band gives a group (Proposition A10)."""
    ref = reference
    selfc, _ = reindex(ref, ref.reduced.cell, ref.reduced.sg_hm, proper=proper)
    return sorted(c.price for c in selfc)


# ---- indexing classes: the coset structure ---------------------------------------
@dataclass
class IndexingClass:
    representative: Candidate    # the cheapest operator of the class
    operators: list              # every candidate operator in the class (exact)
    size: int


@dataclass
class IndexingResult:
    classes: list                # one IndexingClass per class of the symmetry group, cheapest first
    layer: int
    symmetry_order: int          # |H_eps|: operators mapping the reference onto itself within the band
    group_order: int             # |L|: order of the reference's Laue group (or point group, anomalous=True)
    closed: bool                 # H_eps is closed under composition (a group) at this band
    group_in_symmetry: bool      # every operator of L maps the reference onto itself within the band
    anomalous: bool = False

    @property
    def tier(self):              # deprecated names, one release
        return self.layer

    @property
    def laue_order(self):
        return self.group_order

    @property
    def laue_in_symmetry(self):
        return self.group_in_symmetry


def _symmetry_matrices(space_group_key, anomalous=False):
    """the point group of the space group (anomalous=True) or its Laue group (default), as
    exact 3x3 matrices in the input setting."""
    from ..group import point_group
    from ..space_groups import space_group
    rot = []
    for W in point_group(space_group(space_group_key).operations()):
        R = Matrix3([[Fraction(x) for x in row] for row in W.rows])
        rot.append(R)
        if not anomalous:
            rot.append(Matrix3([[-x for x in row] for row in R.rows]))
    out, seen = [], set()
    for R in rot:
        if R.rows not in seen:
            seen.add(R.rows)
            out.append(R)
    return out


def _laue_matrices(space_group_key):
    return _symmetry_matrices(space_group_key, anomalous=False)


def indexing_classes(reference, new_cell, space_group_key, new_sg_hm=None, accept=None,
                     proper=None, anomalous=False, eps=None):
    """Candidate operators from the reference to a new cell, grouped into the classes that
    are physically distinct for a crystal of the reference's space group.

    Within a gap of the reference's price spectrum the candidates are one coset
    H_eps.P0 of the band's (pseudo-)symmetry group H_eps, the operators that carry the
    reference onto itself within the band (Proposition A10; ``closed`` reports whether
    H_eps is closed under composition at this band). Two candidates that differ on the
    left by an element of the reference's Laue group L, P2 = l.P1, index the crystal the
    same way, so the distinct answers are the classes of L in the candidate set,
    [H_eps : L] of them when the coset statement holds; choosing between them is a
    question for the intensities (indexing ambiguity, twinning). With ``anomalous=True``
    the point group replaces the Laue group, since the inversion then changes the
    indexing. ``space_group_key`` is the reference's space group in its input setting.
    ``proper=True`` drops improper candidates (a chiral crystal)."""
    ref = reference
    if accept is None:
        accept = eps if eps is not None else ref.eps / (1.0 + ref.eps)
    cands, tier = reindex(ref, new_cell, new_sg_hm, accept=accept, proper=proper)
    selfc, _ = reindex(ref, ref.reduced.cell, ref.reduced.sg_hm, accept=accept, proper=proper)
    H = {c.operator.rows for c in selfc}
    closed = all((a.operator @ b.operator).rows in H for a in selfc for b in selfc)
    L = _symmetry_matrices(space_group_key, anomalous)
    laue_in = all(l.rows in H for l in L)
    by_op = {c.operator.rows: c for c in cands}
    classes, covered = [], set()
    for c in cands:                                   # cheapest first
        if c.operator.rows in covered:
            continue
        orbit = []
        for l in L:
            q = (l @ c.operator).rows
            if q in by_op and q not in covered:
                covered.add(q)
                orbit.append(by_op[q])
        orbit.sort(key=lambda x: x.price)
        classes.append(IndexingClass(orbit[0], [x.operator for x in orbit], len(orbit)))
    return IndexingResult(classes, tier, len(H), len(L), closed, laue_in, anomalous)


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
    "IndexingClass", "IndexingResult", "indexing_classes", "sort_spectrum",
]
