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
(manuscript, Remark A8): 1 when every member lies in L1 = U.F0, the 127 superbases
adjacent to the reduced cell; 2 when every member lies in L2 = U.L1 (2755); 3
otherwise. The layer describes how far from the reduced cell the set reaches; it is
not needed for completeness, which the ball enumeration provides for every eps < 1/4.

Prices. An operator P from a reference to a new cell is priced by the affine-invariant
(geodesic) distance between the new metric and the predicted one, after the best common
scale: with l the logarithms of the eigenvalues of G_N^-1/2 P^T G_R P G_N^-1/2, the price
is || l - mean(l) ||. This is a true distance on cell shapes, the same in every basis
and symmetric in the two cells, so prices of products and cosets obey the triangle
inequality exactly (manuscript, Proposition A10). To first order it equals the strain
norm above.

Bands. :data:`LEPAGE_STRAIN` = sqrt(2) sin 3 deg = 0.074 is the strain distance at which
a pseudo-twofold with Le Page's obliquity of 3 degrees sits from the metrics it would fix.
A twofold's price is exactly twice its geodesic distance to those metrics (Proposition
A9): :data:`LEPAGE_PRICE` = 2 sqrt(2) artanh(sin 3 deg) = 0.148 is the price of a twofold
of obliquity 3 degrees. For reindexing data sets of one crystal form the band is set by
non-isomorphism (in the PDB the strain between cells of one crystal form after the scale
fit has a 90th percentile near 0.04 and a 95th near 0.07), not by the cell esds.

Reindexing (:func:`reindex`) pairs the new cell's single reduced superbase with every
member of the reference's widened closure and every relabelling, and prices each
candidate. If the true operator has price at most ``accept``, it is among the candidates
provided the reference was widened at ``exp(accept) - 1`` (manuscript, Lemma A7);
:func:`band_for` gives that number. Only proper operators are returned by default, and
a fitted scale that changes the edge lengths by more than ``max_scale_change`` is
rejected (a doubled cell is not a reindexing). :func:`indexing_classes` groups the
candidates by the reference's Laue group (merging without anomalous signal) or point
group (anomalous data).

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
from .primitive import lattice_letter
from .selling_cob import _PRIM_EXACT, selling_matrix

LEPAGE_STRAIN = math.sqrt(2.0) * math.sin(math.radians(3.0))
"""Strain of one cell angle changed by 3 degrees: the sort distance of a pseudo-twofold
with Le Page's obliquity of 3 degrees."""

LEPAGE_PRICE = 2.0 * math.sqrt(2.0) * math.atanh(math.sin(math.radians(3.0)))
"""Price of a twofold of obliquity 3 degrees: twice its geodesic distance to the metrics
it fixes (Proposition A9), 2 sqrt(2) artanh(sin 3 deg) = 0.148."""


DEFAULT_ACCEPT = 0.074
"""Default price band of :func:`reindex` and :func:`indexing_classes` (7.4 %), the band of the
paper's tables; the PDB non-isomorphism quantiles (90th 4.2 %, 95th 6.7 %) bound it from below."""

MAX_BALL = 5000
"""Largest number of lattice vectors in the ball of Lemma A6 that the exact enumeration
accepts; above it a ValueError is raised instead of running for minutes (thin needles at
large bands). Pass ``max_ball=None`` to :func:`widened_closure` to lift it."""


def band_for(accept):
    """The band at which to widen a reference so that every operator of price at most
    ``accept`` is among the candidates (Lemma A7): exp(accept) - 1."""
    if not 0 <= accept < math.log(1.25):
        raise ValueError("accept must be in [0, log(1.25)); the widened set is computed below a strain of 0.25")
    return math.expm1(accept)


def accept_for(eps):
    """The largest price band that a reference widened at ``eps`` certifies: log(1 + eps)."""
    return math.log1p(eps)

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


def universal_operators():
    """The 6096 matrices of U as integer column operators W (3x3), for a lookup between two
    Selling-reduced cells: with B_R, B_N the reduced bases (columns, in each cell's input
    basis), the candidates are P = B_R W B_N^-1 (Remark A8). The stored data hold each
    element as superbase rows; W is their transpose."""
    np = _np()
    return np.round(_data()[0]).astype(int).transpose(0, 2, 1)


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


def _check_cell(cell):
    """Six finite cell parameters, edges > 0, angles in degrees, positive-definite metric."""
    try:
        cell = tuple(float(x) for x in cell)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a cell is six numbers a, b, c, alpha, beta, gamma; got {cell!r}") from exc
    if len(cell) != 6 or not all(math.isfinite(x) for x in cell):
        raise ValueError(f"a cell is six finite numbers a, b, c, alpha, beta, gamma; got {cell!r}")
    if min(cell[:3]) <= 0:
        raise ValueError(f"cell edges must be positive; got {cell[:3]}")
    if not all(0 < x < 180 for x in cell[3:]):
        raise ValueError(f"cell angles must lie strictly between 0 and 180 degrees; got {cell[3:]}")
    if max(cell[3:]) < 2 * math.pi:                      # legal (a very oblique basis) but suspicious
        import warnings
        warnings.warn(f"cell angles {cell[3:]} are all below 2 pi degrees: if they are radians, give them in degrees",
                      stacklevel=3)
    ca, cb, cg = (math.cos(math.radians(x)) for x in cell[3:])
    if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg <= 0:
        raise ValueError(f"the angles {cell[3:]} do not form a cell (the metric is not positive definite)")
    return cell


def _centring(sg_hm, cell=None):
    """Lattice letter of a space-group symbol: P for none, P for a rhombohedral setting
    (':R'), raising on a symbol whose lattice letter is not recognised. An R symbol without a
    setting suffix on a cell that is not hexagonal-shaped (alpha = beta = 90, gamma = 120
    degrees) is read on rhombohedral axes, with a warning."""
    if not sg_hm:
        return "P"
    s = str(sg_hm).strip()
    if s.replace(" ", "").upper().endswith(":R"):
        return "P"                                      # rhombohedral axes: already primitive
    try:
        letter = lattice_letter(s)
    except ValueError as exc:
        raise ValueError(f"cannot read a lattice centring from the space-group symbol {sg_hm!r}") from exc
    if letter == "R" and cell is not None and ":" not in s:
        a, b, c, al, be, ga = cell
        hexa = abs(al - 90) < 5 and abs(be - 90) < 5 and abs(ga - 120) < 5 and abs(a / b - 1) < 0.05
        rhombo = (max(a, b, c) / min(a, b, c) < 1.05 and max(al, be, ga) - min(al, be, ga) < 2
                  and abs(al - 90) > 1)
        if rhombo and not hexa:
            import warnings
            warnings.warn(f"{sg_hm!r} on a rhombohedral-shaped cell: read on rhombohedral axes "
                          "(give 'R ... :R' or ':H' to be explicit)", stacklevel=3)
            return "P"
        if not hexa:
            raise ValueError(f"{sg_hm!r} on a cell {cell} that is neither hexagonal- nor rhombohedral-shaped: "
                             "give the setting explicitly ('R ... :H' or 'R ... :R')")
    return letter


def reduce_cell(cell, sg_hm=None) -> ReducedCell:
    """The cell's primitive Selling-reduced basis and metric. ``sg_hm`` gives the centring
    (P, A, B, C, I, F, R; 'R ... :R' is primitive); an unreadable symbol raises."""
    np = _np()
    cell = _check_cell(cell)
    T = Matrix3(_PRIM_EXACT[_centring(sg_hm, cell)])
    Tf = np.array([[float(x) for x in row] for row in T.rows])
    G0 = np.array(UnitCell(*cell).metric_tensor(), float)
    from .metric import params_from_metric
    prim = tuple(params_from_metric((Tf.T @ G0 @ Tf).tolist()))
    B = T @ selling_matrix(prim)
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


def _ball_superbases(G, eps, max_ball=MAX_BALL):
    """every superbase within eps of obtuse (exact, Lemma A6): yields (key, S, distance)."""
    np = _np()
    V = _ball_vectors(G, eps)
    if max_ball is not None and len(V) > max_ball:
        raise ValueError(f"the ball of Lemma A6 holds {len(V)} lattice vectors at strain {eps:.3f} (limit "
                         f"{max_ball}): the cell is too anisotropic for the exact enumeration at this band; "
                         "use a smaller band, or pass max_ball=None and expect a long run")
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
    _selfmaps: dict = field(default_factory=dict, repr=False)   # (accept, proper) -> self-map candidates
    _table: dict = field(default_factory=dict, repr=False)      # the member table of the reindexing workflow

    def __len__(self):
        return len(self.members)

    @property
    def tier(self):          # deprecated name, one release
        return self.layer


def widened_closure(cell, sg_hm=None, eps=None, certify=True, max_ball=MAX_BALL) -> WidenedClosure:
    """S_eps of the cell, exactly, and the adjacency layer that holds it.

    eps must be below 1/4. With ``certify`` (default) the ball of Lemma A6 is
    enumerated, which is exact for every such eps; ``layer`` then records the smallest
    adjacency layer that contains the whole set (1: the 127 superbases of L1, 2: the 2755
    of L2, 3: neither). Without ``certify`` only L1 is filtered (fast, 127 depths) and
    ``layer`` is 0: complete only if eps is below the cell's adjacency radius delta_1,
    which is not checked (Remark A8). The default ``eps`` is ``band_for(DEFAULT_ACCEPT)``,
    which certifies the default price band of :func:`reindex`. ``max_ball`` caps the ball
    of Lemma A6 (see :data:`MAX_BALL`)."""
    if eps is None:
        eps = band_for(DEFAULT_ACCEPT)
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
    for k, S, d in _ball_superbases(G, eps, max_ball):
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
    price: float             # geodesic distance between G_N and s P^T G_R P after the best scale s
    scale: float             # s minimising the price; sqrt(s) is the fitted change of edge lengths
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
    """linearised mismatch, || G_new^-1/2 (s G_pred - G_new) G_new^-1/2 ||_F, with s the
    scale minimising it when ``fit_scale``; returns (value, s). The strain norm of a metric
    perturbation; prices use :func:`_price`."""
    np = _np()
    w, V = np.linalg.eigh(G_new)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    Y = Gm @ G_pred @ Gm
    s = float(np.trace(Y) / np.sum(Y * Y)) if fit_scale else 1.0
    return float(np.linalg.norm(s * Y - np.eye(3))), s


def _price(G_new, G_pred, fit_scale=True):
    """geodesic price of G_pred against G_new: with l the logs of the eigenvalues of
    G_new^-1/2 G_pred G_new^-1/2, ||l - mean(l)|| (``fit_scale``) or ||l||; returns
    (price, s) with s G_pred the scaled prediction, s = exp(-mean(l))."""
    np = _np()
    w, V = np.linalg.eigh(G_new)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    lam = np.linalg.eigvalsh(Gm @ G_pred @ Gm)
    lv = np.log(lam)
    m = float(lv.mean()) if fit_scale else 0.0
    return float(np.linalg.norm(lv - m)), math.exp(-m)


def reindex(reference, new_cell, new_sg_hm="reference", accept=None, proper=True, fit_scale=True, eps=None,
            max_scale_change=0.1):
    """Candidate operators from the reference to a new cell, ranked by strain price.

    ``reference`` is a :class:`WidenedClosure` (from :func:`widened_closure`). The new
    cell is reduced once. Returns (candidates, layer): every candidate whose price,
    measured at the new cell's metric after the best common scale, is at most
    ``accept``; each P maps the reference's input basis to the new cell's input basis,
    ``h_N = P^T h_R``. The price is the geodesic distance between the new metric and
    the predicted one after the best common scale (:func:`_price`). The default
    ``accept`` is the largest band the reference certifies, ``log(1 + ref.eps)``
    (Lemma A7; see :func:`band_for` and :func:`accept_for`). A candidate whose fitted
    scale changes the edge lengths by more than ``max_scale_change`` (default 10 %) is
    rejected; ``None`` keeps every scale. If the two primitive cells differ in volume by
    more than that scale limit allows ((1 + max_scale_change)^3, or 1.6 without a limit),
    a ValueError names the likely cause (a wrong centring symbol, or a sub- or
    superlattice). An obverse/reverse relation keeps the volume and is not detected. The new cell is read with
    the reference's centring unless ``new_sg_hm`` is given (``None`` for primitive). Candidates of equal
    price are listed nearest the identity first.
    ``proper=True`` (default) keeps only candidates of positive determinant (+1, or +1/k from a
    centred reference's conventional basis): two right-handed
    bases are related by a proper operator, and a left-handed basis is never valid
    output. The choice is a convention: cell parameters carry no hand, so an inverted
    geometry has to be caught before the cell reaches this function. ``proper=None``
    keeps both signs, ``False`` only the improper ones (for analysing hand inversions).
    ``eps`` is a deprecated alias of ``accept``."""
    return _reindex_core(reference, new_cell, new_sg_hm, accept, proper, fit_scale, eps, max_scale_change,
                         trace=False)[:2]


def _reindex_direct(reference, new_cell, new_sg_hm="reference", accept=None, proper=True, fit_scale=True, eps=None,
            max_scale_change=0.1):
    """The direct form of :func:`reindex`: the exact operator P is built and priced for every
    member of the reference and every one of its 48 relabellings. Kept to check the
    workflow, which gives the same candidates (supporting information, Section S10)."""
    np = _np()
    ref = reference
    if eps is not None:
        import warnings
        warnings.warn("reindex(eps=...) is a deprecated alias of accept= (a price, not the widening strain)",
                      DeprecationWarning, stacklevel=2)
    if accept is None:
        accept = eps if eps is not None else accept_for(ref.eps)
    if accept > accept_for(ref.eps) * (1 + 1e-9):
        raise ValueError(f"accept={accept:.4f} needs the reference widened at band_for(accept)="
                         f"{band_for(accept):.4f}; it was widened at {ref.eps:.4f}")
    if new_sg_hm == "reference":
        new_sg_hm = ref.reduced.sg_hm
    new = reduce_cell(new_cell, new_sg_hm)
    ratio = math.sqrt(float(np.linalg.det(new.G)) / float(np.linalg.det(ref.reduced.G)))
    vmax = 1.6 if max_scale_change is None else (1 + max_scale_change) ** 3
    vmin = 1 / 1.6 if max_scale_change is None else (1 - max_scale_change) ** 3
    if not vmin < ratio < vmax:
        raise ValueError(f"the primitive cells differ in volume by a factor {ratio:.2f} (allowed {vmax:.2f}): "
                         "no operator within the scale limit relates them; check the centring symbols, "
                         "or look for a sub- or superlattice (an obverse/reverse relation of rhombohedral "
                         "lattices keeps the volume and is not detected from cell parameters)")
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
            det = 1 if P.det() > 0 else -1          # rational (det +-1/k) for a centred reference
            if proper is True and det != 1:
                continue
            if proper is False and det != -1:
                continue
            Pf = np.array([[float(x) for x in row] for row in P.rows])
            G_pred = Pf.T @ G_R_in @ Pf
            price, s = _price(G_N_in, G_pred, fit_scale)
            if price > accept:
                continue
            if max_scale_change is not None and abs(math.sqrt(s) - 1) > max_scale_change:
                continue
            k = tuple(P.rows)
            if k in seen:
                continue
            seen.add(k)
            out.append(Candidate(P, price, s, m.depth, det))
    if not out:
        import warnings
        warnings.warn("no candidate within the band: the cells may belong to different lattices, or a centring "
                      "symbol may be wrong (a C cell read as I, or the reverse, has the same volume)", stacklevel=2)
    eye = Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out.sort(key=lambda c: (round(c.price, 12), sum(abs(float(x - y)) for r1, r2 in zip(c.operator.rows, eye.rows)
                                                    for x, y in zip(r1, r2))))    # ties: nearest the identity first
    return out, ref.layer


# ---- the reindexing workflow (supporting information, Section S10) ------------------
@dataclass
class ReindexWorkflow:
    """Every step of one reindexing, as :func:`reindex_workflow` returns it.

    The reference's members are tabulated once as metrics; the new cell is compared with
    them in its own reduced frame, where the two input bases drop out; one scale serves
    every candidate; a necessary condition on the ten dot products of the new cell's
    reduced superbase discards most orderings before any eigenvalue is computed; and the
    exact operator is built only for the candidates within the band."""
    accept: float                 # the price band
    eta: float                    # the strain exp(accept) - 1 of Lemma A7
    layer: int                    # the reference's adjacency layer
    reference_conorms: object     # (6,) conorms of the reference's reduced superbase
    new_conorms: object           # (6,) conorms of the new cell's reduced superbase
    basis_reference: Matrix3      # B_R, reduced basis of the reference in its input basis
    basis_new: Matrix3            # B_N, reduced basis of the new cell in its input basis
    n_members: int                # members of the reference's widened set
    n_orderings: int              # 24 ordered bases per member, the distinct metrics
    n_signed: int                 # ordered bases with the sign the determinant rule selects
    scale: float                  # the common fitted scale s (1 without the scale fit)
    edge_scale: float             # sqrt(s), the fitted change of edge lengths
    scale_ok: bool                # whether sqrt(s) is within the scale limit
    bounds: object                # (4,4) eta |u_i| |u_j| for the new superbase vectors u
    n_prefilter: int              # orderings that pass the necessary condition
    n_priced: int                 # eigenvalue problems solved
    n_within: int                 # orderings priced within the band
    n_candidates: int             # distinct exact operators returned
    priced: list                  # (member, ordering, sign, price) for every priced ordering
    candidates: list              # the Candidate list that reindex returns
    seconds: dict                 # time spent in each step


def _ordered_bases(S):
    """the 24 ordered bases of a superbase (three of its four vectors, in order) as integer
    rows; with a common sign of +1 or -1 they are its 48 relabellings, and both signs give
    the same metric."""
    np = _np()
    out = []
    for d in range(4):
        rest = [i for i in range(4) if i != d]
        for perm in itertools.permutations(rest):
            out.append(np.asarray(S)[list(perm)])
    return np.array(out)


def _member_table(ref, G_R):
    """the reference's member table, computed once: for each member and each of its 24
    ordered bases B (rows, in the reference's reduced basis), the integer matrix, its
    determinant and the metric B G_R B^T that it predicts for the new cell's reduced basis."""
    np = _np()
    key = "workflow"
    if key not in ref._table:
        B = np.array([_ordered_bases(np.rint(np.asarray(m.superbase)).astype(int)) for m in ref.members])
        B = B.reshape(len(ref.members), 24, 3, 3).astype(int)
        det = np.rint(np.linalg.det(B.astype(float))).astype(int)
        if not np.all(np.abs(det) == 1):
            raise RuntimeError("a member's ordered basis is not unimodular")
        ref._table[key] = {"B": B, "det": det}
    tab = ref._table[key]
    Bf = tab["B"].astype(float)
    tab["G"] = np.einsum("kois,st,kojt->koij", Bf, G_R, Bf)
    return tab


def _reindex_core(ref, new_cell, new_sg_hm, accept, proper, fit_scale, eps, max_scale_change, trace):
    import time
    np = _np()
    t0 = time.perf_counter()
    seconds = {}
    if eps is not None:
        import warnings
        warnings.warn("reindex(eps=...) is a deprecated alias of accept= (a price, not the widening strain)",
                      DeprecationWarning, stacklevel=3)
    if accept is None:
        accept = eps if eps is not None else accept_for(ref.eps)
    if accept > accept_for(ref.eps) * (1 + 1e-9):
        raise ValueError(f"accept={accept:.4f} needs the reference widened at band_for(accept)="
                         f"{band_for(accept):.4f}; it was widened at {ref.eps:.4f}")
    if new_sg_hm == "reference":
        new_sg_hm = ref.reduced.sg_hm
    # step 1: reduce the new cell once
    new = reduce_cell(new_cell, new_sg_hm)
    ratio = math.sqrt(float(np.linalg.det(new.G)) / float(np.linalg.det(ref.reduced.G)))
    vmax = 1.6 if max_scale_change is None else (1 + max_scale_change) ** 3
    vmin = 1 / 1.6 if max_scale_change is None else (1 - max_scale_change) ** 3
    if not vmin < ratio < vmax:
        raise ValueError(f"the primitive cells differ in volume by a factor {ratio:.2f} (allowed {vmax:.2f}): "
                         "no operator within the scale limit relates them; check the centring symbols, "
                         "or look for a sub- or superlattice (an obverse/reverse relation of rhombohedral "
                         "lattices keeps the volume and is not detected from cell parameters)")
    seconds["reduce"] = time.perf_counter() - t0
    # step 2: the two reduced metrics, from the input cells (the input bases drop out of every price)
    t = time.perf_counter()
    BRf = np.array([[float(x) for x in row] for row in ref.reduced.basis.rows])
    BNf = np.array([[float(x) for x in row] for row in new.basis.rows])
    G_R = BRf.T @ np.array(UnitCell(*ref.reduced.cell).metric_tensor(), float) @ BRf
    G_N = BNf.T @ np.array(UnitCell(*new.cell).metric_tensor(), float) @ BNf
    tab = _member_table(ref, G_R)
    nm = len(ref.members)
    seconds["table"] = time.perf_counter() - t
    # step 3: the determinant rule selects the sign of each ordered basis
    t = time.perf_counter()
    sR = 1 if ref.reduced.basis.det() > 0 else -1
    sN = 1 if new.basis.det() > 0 else -1
    want = {True: (1,), False: (-1,), None: (1, -1)}[proper]
    # an ordered basis with determinant d and sign c gives det(P) of sign c^3 d sR sN = c d sR sN
    signs = [(c, (c * tab["det"] * sR * sN)) for c in ((1, -1))]
    # step 4: one scale for every candidate, since every predicted metric has det G_R
    lam0 = np.linalg.eigvalsh(G_N)
    Nh = np.linalg.eigh(G_N)[1] @ np.diag(lam0 ** -0.5) @ np.linalg.eigh(G_N)[1].T
    s_fit = (np.linalg.det(G_N) / np.linalg.det(G_R)) ** (1 / 3) if fit_scale else 1.0
    edge = math.sqrt(s_fit)
    scale_ok = max_scale_change is None or abs(edge - 1) <= max_scale_change
    # step 5: the necessary condition of Lemma A7(i) and Proposition A4 on the superbase
    S0 = np.array(_S0, float)
    eta = math.expm1(accept)
    V = S0 @ G_N @ S0.T                                   # dot products of the new superbase
    nrm = np.sqrt(np.diag(V))
    bounds = eta * np.outer(nrm, nrm)
    Vk = np.einsum("ia,koab,jb->koij", S0, tab["G"], S0)  # predicted dot products, unscaled
    tol = 1e-9 * float(np.abs(V).max())
    ok = np.all(np.abs(s_fit * Vk - V) <= bounds * (1 + 1e-12) + tol, axis=(2, 3))
    seconds["prefilter"] = time.perf_counter() - t
    # step 6: price the orderings that pass, in one batch
    t = time.perf_counter()
    idx = np.argwhere(ok)
    priced = []
    if len(idx):
        M = Nh @ tab["G"][idx[:, 0], idx[:, 1]] @ Nh
        lv = np.log(np.linalg.eigvalsh(M))
        mean = lv.mean(axis=1) if fit_scale else np.zeros(len(lv))
        prices = np.linalg.norm(lv - mean[:, None], axis=1)
        scales = np.exp(-mean)
        price_of = {(int(k), int(o)): (float(p), float(sc)) for (k, o), p, sc in zip(idx, prices, scales)}
    else:
        price_of = {}
    seconds["price"] = time.perf_counter() - t
    # step 7: the exact operator, only within the band, in the order of the direct form
    t = time.perf_counter()
    BN_inv = new.basis.inverse()
    out, seen, n_within = [], set(), 0
    for k in range(nm):
        for o in range(24):
            if (k, o) not in price_of:
                continue
            price, sc = price_of[(k, o)]
            for c, dsign in signs:
                if int(dsign[k, o]) not in want:
                    continue
                if trace:
                    priced.append((k, o, c, price))
                if price > accept:
                    continue
                if not scale_ok:
                    continue
                n_within += 1
                Brows = c * tab["B"][k, o]
                Mt = Matrix3([[Fraction(int(Brows[j][i])) for j in range(3)] for i in range(3)])
                P = ref.reduced.basis @ Mt @ BN_inv
                det = 1 if P.det() > 0 else -1
                key = tuple(P.rows)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate(P, price, sc, ref.members[k].depth, det))
    seconds["exact"] = time.perf_counter() - t
    if not out:
        import warnings
        warnings.warn("no candidate within the band: the cells may belong to different lattices, or a centring "
                      "symbol may be wrong (a C cell read as I, or the reverse, has the same volume)", stacklevel=3)
    eye = Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out.sort(key=lambda c: (round(c.price, 12), sum(abs(float(x - y)) for r1, r2 in zip(c.operator.rows, eye.rows)
                                                    for x, y in zip(r1, r2))))    # ties: nearest the identity first
    seconds["total"] = time.perf_counter() - t0
    if not trace:
        return out, ref.layer, None
    n_signed = int(sum(int(np.isin(d, want).sum()) for _, d in signs))
    wf = ReindexWorkflow(
        accept=accept, eta=eta, layer=ref.layer, reference_conorms=conorms(S0, G_R),
        new_conorms=conorms(S0, G_N), basis_reference=ref.reduced.basis, basis_new=new.basis,
        n_members=nm, n_orderings=24 * nm, n_signed=n_signed, scale=float(s_fit), edge_scale=edge,
        scale_ok=scale_ok, bounds=bounds, n_prefilter=int(ok.sum()), n_priced=len(idx), n_within=n_within,
        n_candidates=len(out), priced=priced, candidates=out, seconds=seconds)
    return out, ref.layer, wf


def reindex_workflow(reference, new_cell, new_sg_hm="reference", accept=None, proper=True, fit_scale=True,
                     max_scale_change=0.1):
    """:func:`reindex`, step by step: returns a :class:`ReindexWorkflow` with the
    candidates and every intermediate quantity (supporting information, Section S10)."""
    return _reindex_core(reference, new_cell, new_sg_hm, accept, proper, fit_scale, None, max_scale_change,
                         trace=True)[2]


def price_spectrum(reference, proper=True):
    """Prices of every self-map of the reference within its band (the operators H_eps
    that carry the reference onto itself), sorted; the gaps in this list are where a
    band gives a group (Proposition A10)."""
    return sorted(c.price for c in _self_maps(reference, accept_for(reference.eps), proper))


def _self_maps(ref, accept, proper):
    """The reference's self-maps within ``accept`` (cached on the widened closure)."""
    key = (round(accept, 12), proper)
    if key not in ref._selfmaps:
        ref._selfmaps[key] = reindex(ref, ref.reduced.cell, ref.reduced.sg_hm, accept=accept, proper=proper)[0]
    return ref._selfmaps[key]


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
    proper_group_order: int = 0  # |L+|: the proper rotations of L, the group acting when proper=True

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


def indexing_classes(reference, new_cell, space_group_key, new_sg_hm="reference", accept=None,
                     proper=True, anomalous=False, eps=None):
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
    indexing (an acentric, non-chiral group; for a chiral group, with proper operators,
    the count does not change). ``space_group_key`` is the reference's space group in its
    input setting; the new cell is read with the reference's centring unless ``new_sg_hm`` is given
    (``None`` for primitive), as in :func:`reindex`.
    Only proper candidates are used by default (``proper=True``)."""
    ref = reference
    if eps is not None:
        import warnings
        warnings.warn("indexing_classes(eps=...) is a deprecated alias of accept= (a price)",
                      DeprecationWarning, stacklevel=2)
    if accept is None:
        accept = eps if eps is not None else accept_for(ref.eps)
    L = _symmetry_matrices(space_group_key, anomalous)          # validate the group first
    if new_sg_hm == "reference":
        new_sg_hm = ref.reduced.sg_hm
    cands, tier = reindex(ref, new_cell, new_sg_hm, accept=accept, proper=proper)
    selfc = _self_maps(ref, accept, proper)
    H = {c.operator.rows for c in selfc}
    closed = all((a.operator @ b.operator).rows in H for a in selfc for b in selfc)
    laue_in = all(l.rows in H for l in L if proper is not True or l.det() == 1)
    if not laue_in:
        import warnings
        warnings.warn(f"the reference cell does not have the lattice symmetry of {space_group_key!r} within the "
                      "band: check the space group", stacklevel=2)
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
    return IndexingResult(classes, tier, len(H), len(L), closed, laue_in, anomalous,
                          sum(1 for l in L if l.det() == 1))


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


def strain_from_esds(cell, esds, sg_hm=None):
    """root-mean-square strain (norm of equation (2)) of a cell's metric implied by
    independent standard uncertainties ``esds`` = (sa, sb, sc, salpha, sbeta, sgamma), in
    A and degrees, propagated to first order through G(a, b, c, alpha, beta, gamma).
    This is the measurement error alone. For macromolecular crystals the mismatch
    between data sets of one crystal form is dominated by non-isomorphism (a few per cent
    in strain), which sets the reindexing band; the esd route suits small-molecule cells."""
    np = _np()
    cell = [float(x) for x in cell]
    p0 = np.array(cell)
    G0 = np.array(UnitCell(*cell).metric_tensor(), float)
    w, V = np.linalg.eigh(G0)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    var = 0.0
    for k, sk in enumerate(esds):
        h = 1e-6 * (1.0 if k < 3 else 1.0)
        p = p0.copy(); p[k] += h
        dG = (np.array(UnitCell(*p).metric_tensor(), float) - G0) / h
        X = Gm @ dG @ Gm
        var += float(np.sum(X * X)) * float(sk) ** 2
    return math.sqrt(var)


def sort_spectrum(cell, sg_hm=None):
    """{sort: strain distance} from the cell to each of Delone's 24 sorts, minimised over the
    frames of the universal operator set (manuscript, 'Distance to each sort').

    A distance below the cell's adjacency radius delta_1 (:func:`adjacency_radius`) is
    exact; one above it is an upper bound over the frames used, and the true distance is
    at least delta_1. :func:`certified_sort_spectrum` returns delta_1 with the spectrum."""
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
        # exact constrained projection (non-negative letters): the optimum is the smallest
        # residual over the active sets whose unconstrained solution is feasible
        best = np.sqrt(np.maximum(np.einsum("nk,nkl,nl->n", Q, M, Q), 0))
        for r in range(1, len(classes) + 1):
            for act in itertools.combinations(range(len(classes)), r):
                Aa = A[:, list(act)]
                y = np.linalg.solve(np.einsum("ka,nkl,lb->nab", Aa, M, Aa),
                                    np.einsum("ka,nkl,nl->na", Aa, M, Q)[..., None])[..., 0]
                Rr = y @ Aa.T - Q
                res = np.sqrt(np.maximum(np.einsum("nk,nkl,nl->n", Rr, M, Rr), 0))
                best = np.where((y >= -1e-12).all(1), np.minimum(best, res), best)
        out[name] = float(best.min())
    return out


__all__ = [
    "LEPAGE_STRAIN", "LEPAGE_PRICE", "BRAVAIS", "band_for", "accept_for", "price_spectrum", "strain_from_esds",
    "certified_sort_spectrum", "ReducedCell", "reduce_cell", "conorms", "distance_to_obtuse",
    "adjacency_radius", "Member", "WidenedClosure", "widened_closure", "Candidate", "reindex", "reindex_workflow", "ReindexWorkflow",
    "IndexingClass", "IndexingResult", "indexing_classes", "sort_spectrum", "universal_operators",
    "DEFAULT_ACCEPT", "MAX_BALL",
]


def certified_sort_spectrum(cell, sg_hm=None):
    """(spectrum, delta_1): the sort spectrum and the radius below which its values are
    exact. A sort with distance d > delta_1 is reported as an upper bound; its true
    distance lies in [delta_1, d]."""
    red = cell if isinstance(cell, ReducedCell) else reduce_cell(cell, sg_hm)
    return sort_spectrum(red), adjacency_radius(red.G)
