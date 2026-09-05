"""
Lattice (metric) symmetry determination -- Le Page (1982) / Lebedev et al. (2006).

Given a unit cell, find the point group of its lattice (the holohedry): the set
of rotations that leave the metric tensor invariant. Following Zwart,
Grosse-Kunstleve & Adams, "Exploring Metric Symmetry" (2006):

  * Lebedev et al. (2006): every symmetry operation of a reduced cell is one of
    the 480 integer matrices with entries in {-1,0,1}, det = +1, whose powers
    also stay within {-1,0,1}. Exactly 81 of these are two-folds (M != I,
    M^2 = I). Enumerating and closing the accepted two-folds yields the full
    lattice symmetry -- this replaces Le Page's expensive trigonometric search.
  * Le Page (1982): a candidate two-fold with direct-space axis u and
    reciprocal-space axis h* is a true metric symmetry when u and h* are
    parallel; the misfit angle (the "Le Page delta", in degrees) measures how
    far the cell is from having that symmetry.

This module is numeric (it works on a real metric tensor) but the accepted
operations are the exact integer matrices, so the resulting group can be handed
straight back to the exact symmetry algebra (e.g. characterised with a Hall
symbol + change of basis via agentsg.setting).

Both the 480-matrix set and the 81 two-folds are *computed* on import, not
tabulated.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from itertools import product
from math import acos, degrees, sqrt
from typing import Sequence

from .linalg import Matrix3, Vector3, IDENTITY3
from .symmetry_op import SymmetryOp
from .group import close_group, point_group


# --- the Lebedev set, computed once ---------------------------------------
def _mat_pow_in_set(M, max_order=6):
    """True iff M, M^2, ... stay within {-1,0,1} and reach the identity."""
    n = len(M)
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

    for _ in range(max_order):
        P = mul(P, M)
        if any(abs(P[i][j]) > 1 for i in range(n) for j in range(n)):
            return False
        if all(P[i][j] == (1 if i == j else 0) for i in range(n) for j in range(n)):
            return True
    return False


def _det3_int(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def _lebedev_matrices():
    out = []
    for e in product((-1, 0, 1), repeat=9):
        M = [list(e[0:3]), list(e[3:6]), list(e[6:9])]
        if _det3_int(M) != 1:
            continue
        if _mat_pow_in_set(M):
            out.append(M)
    return out


def _matmul_int(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


LEBEDEV_MATRICES = _lebedev_matrices()                       # 480
_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
TWO_FOLD_MATRICES = [M for M in LEBEDEV_MATRICES
                     if M != _I and _matmul_int(M, M) == _I]   # 81


# --- axis directions (Grosse-Kunstleve 1999): the invariant direction of a rotation
def _two_fold_axis_direct(M):
    """Direct-space axis of a two-fold: the +1 eigenvector, as an integer triple.

    For a two-fold, M + I projects onto the axis (since M v = v <=> (M+I)v = 2v
    while vectors perpendicular to the axis map to 0 under M + I). We read the
    axis off the columns of M + I."""
    MI = [[M[i][j] + (1 if i == j else 0) for j in range(3)] for i in range(3)]
    # the axis is any nonzero column of M+I (they are all parallel to the axis)
    for j in range(3):
        col = [MI[0][j], MI[1][j], MI[2][j]]
        if any(col):
            g = _igcd3(col)
            return tuple(c // g for c in col)
    # 180-degree rotation with M+I singular in all columns shouldn't happen
    return (0, 0, 1)


def _igcd3(v):
    from math import gcd
    g = 0
    for x in v:
        g = gcd(g, abs(int(x)))
    return g or 1


def _reciprocal_axis(M):
    """Reciprocal-space axis of the two-fold = direct axis of the transpose."""
    Mt = [[M[j][i] for j in range(3)] for i in range(3)]
    return _two_fold_axis_direct(Mt)


# precompute (matrix, direct axis u, reciprocal axis h) for the 81 two-folds
_TWO_FOLD_TABLE = [(M, _two_fold_axis_direct(M), _reciprocal_axis(M))
                   for M in TWO_FOLD_MATRICES]


# --- Le Page delta ---------------------------------------------------------
def _metric_tensor(cell):
    from .cell.metric import metric_tensor
    return metric_tensor(cell)


def _inv3(G):
    d = (G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
         - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
         + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]))
    cof = [
        [(G[1][1] * G[2][2] - G[1][2] * G[2][1]), -(G[1][0] * G[2][2] - G[1][2] * G[2][0]), (G[1][0] * G[2][1] - G[1][1] * G[2][0])],
        [-(G[0][1] * G[2][2] - G[0][2] * G[2][1]), (G[0][0] * G[2][2] - G[0][2] * G[2][0]), -(G[0][0] * G[2][1] - G[0][1] * G[2][0])],
        [(G[0][1] * G[1][2] - G[0][2] * G[1][1]), -(G[0][0] * G[1][2] - G[0][2] * G[1][0]), (G[0][0] * G[1][1] - G[0][1] * G[1][0])],
    ]
    adj = [[cof[j][i] for j in range(3)] for i in range(3)]
    return [[adj[i][j] / d for j in range(3)] for i in range(3)]


def le_page_delta(cell, u, h):
    """Angle (degrees) between the direct-space axis u and reciprocal-space axis h.

    u is measured with the direct metric G, h with the reciprocal metric G*.
    The physical vectors are t = sum u_i a_i (direct) and n = sum h_i a*_i
    (reciprocal normal). The angle between them is the Le Page delta; it is zero
    exactly when the two-fold is a metric symmetry of the cell."""
    G = _metric_tensor(cell)
    Gs = _inv3(G)
    u = [float(x) for x in u]
    h = [float(x) for x in h]
    # t in cartesian has squared length u^T G u; n has h^T G* h.
    # cos(angle) = (u . h) / (|t| |n|), where u.h is the plain dot product
    # because a_i . a*_j = delta_ij  => t.n = sum_i u_i h_i.
    tn = sum(u[i] * h[i] for i in range(3))
    t2 = sum(u[i] * G[i][j] * u[j] for i in range(3) for j in range(3))
    n2 = sum(h[i] * Gs[i][j] * h[j] for i in range(3) for j in range(3))
    if t2 <= 0 or n2 <= 0:
        return 90.0
    c = tn / sqrt(t2 * n2)
    c = max(-1.0, min(1.0, c))
    return degrees(acos(abs(c)))


# --- Kurlin (root-invariant) distance to a two-fold --------------------------
def _params_from_metric(G):
    """Cell parameters from a metric tensor (angles in degrees)."""
    from .cell.metric import params_from_metric
    return params_from_metric(G)


def _reynolds_two_fold(G, M):
    """Reynolds average of G over {I, M}: (G + Mᵀ G M) / 2."""
    MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)]
           for i in range(3)]
    Gp = [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)]
          for i in range(3)]
    return [[0.5 * (G[i][j] + Gp[i][j]) for j in range(3)] for i in range(3)]


def kurlin_distance_to_two_fold(cell, M) -> float:
    """Kurlin root-invariant distance from ``cell`` to its {I, M}-symmetrisation.

    The metric is Reynolds-averaged under the two-fold ``M``, converted back to
    cell parameters, and compared via :func:`agentsg.cell.rootform.root_distance`.
    Zero (to numerical noise) iff ``M`` is an exact metric automorphism.
    Units: Ångström (same as the root invariant).
    """
    from .cell.rootform import root_distance
    G = _metric_tensor(cell)
    Gs = _reynolds_two_fold(G, M)
    return root_distance(tuple(float(x) for x in cell), _params_from_metric(Gs))


class TwoFoldScore:
    """One Lebedev two-fold with Le Page δ and Kurlin root distance."""
    __slots__ = ("matrix", "direct_axis", "reciprocal_axis",
                 "le_page_delta", "kurlin_distance")

    def __init__(self, matrix, direct_axis, reciprocal_axis,
                 le_page_delta, kurlin_distance):
        self.matrix = matrix
        self.direct_axis = direct_axis
        self.reciprocal_axis = reciprocal_axis
        self.le_page_delta = le_page_delta
        self.kurlin_distance = kurlin_distance

    def __repr__(self):
        return (f"TwoFoldScore(u={self.direct_axis}, "
                f"delta={self.le_page_delta:.4g}°, "
                f"kurlin={self.kurlin_distance:.4g})")


def evaluate_two_folds(cell, *, sort_by: str = "le_page") -> list[TwoFoldScore]:
    """Score all 81 Lebedev two-folds (Le Page δ + Kurlin distance).

    Does not apply a tolerance gate — use this to inspect the full spectrum.
    ``sort_by`` is ``'le_page'`` or ``'kurlin'``.
    """
    if sort_by not in ("le_page", "kurlin"):
        raise ValueError("sort_by must be 'le_page' or 'kurlin'")
    scores = []
    for M, u, h in _TWO_FOLD_TABLE:
        scores.append(TwoFoldScore(
            M, u, h,
            le_page_delta(cell, u, h),
            kurlin_distance_to_two_fold(cell, M),
        ))
    key = ((lambda s: s.le_page_delta) if sort_by == "le_page"
           else (lambda s: s.kurlin_distance))
    scores.sort(key=key)
    return scores


# --- main entry point ------------------------------------------------------
class LatticeSymmetry:
    """Result of a lattice-symmetry determination.

    ``crystal_system`` is normally a holohedry name (``"triclinic"``,
    ``"monoclinic"``, …) when the closed group order matches a known
    centrosymmetric lattice point group.  If the order is not one of those
    seven values, it is a sentinel string ``"order-N"`` (e.g. ``"order-6"``)
    rather than a crystal-system name — see :data:`_ORDER_TO_SYSTEM`.
    """
    __slots__ = ("operations", "order", "crystal_system", "two_folds",
                 "deltas", "two_fold_scores")

    def __init__(self, operations, crystal_system, two_folds, two_fold_scores):
        self.operations = operations
        self.order = len(operations)
        self.crystal_system = crystal_system
        self.two_folds = two_folds
        self.two_fold_scores = two_fold_scores
        # Backward-compatible: (M, u, h, le_page_delta, kurlin_distance)
        self.deltas = [
            (s.matrix, s.direct_axis, s.reciprocal_axis,
             s.le_page_delta, s.kurlin_distance)
            for s in two_fold_scores
        ]

    def __repr__(self):
        return (f"LatticeSymmetry(order={self.order}, "
                f"crystal_system={self.crystal_system!r}, "
                f"n_two_folds={len(self.two_folds)})")


# Known centrosymmetric lattice holohedry order -> crystal-system name.
# Orders outside this map yield the sentinel ``"order-N"`` instead.
_ORDER_TO_SYSTEM = {
    2: "triclinic", 4: "monoclinic", 8: "orthorhombic",
    16: "tetragonal", 12: "trigonal", 24: "hexagonal", 48: "cubic",
}


def lattice_symmetry(cell, max_delta: float = 3.0,
                     length_tol_pct: float = 2.0) -> LatticeSymmetry:
    """Determine the metric (lattice) symmetry of a unit cell.

    Parameters
    ----------
    cell : (a, b, c, alpha, beta, gamma), angles in degrees. Best supplied as a
        reduced (Niggli) cell -- the 480-matrix argument is proven for reduced
        cells -- but any cell works.
    max_delta : angular tolerance in degrees on the Le Page delta; two-folds
        with delta <= max_delta are accepted. Acceptance remains Le-Page-gated;
        each accepted two-fold also carries its Kurlin root-invariant distance
        to the {I, M}-symmetrised metric (see :class:`TwoFoldScore`).
    length_tol_pct : maximum percent length change allowed under MᵀGM vs G.
        Le Page is purely angular; without this gate a few-percent edge mismatch
        can still look "tetragonal". Default 2% (same scale as
        :func:`tolerance_metric_symmetry`).

    Returns a :class:`LatticeSymmetry` with the closed operation set (exact
    integer rotations as SymmetryOp with zero translation), the holohedry order,
    the crystal system (or ``"order-N"`` sentinel when the closed order is not
    a known holohedry — see :data:`_ORDER_TO_SYSTEM`), and the accepted
    two-folds with Le Page / Kurlin scores.
    """
    G = _metric_tensor(cell)
    ref = _cell_params(G)
    accepted = []
    scores = []
    for M, u, h in _TWO_FOLD_TABLE:
        d = le_page_delta(cell, u, h)
        if d > max_delta:
            continue
        MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)]
               for i in range(3)]
        Gp = [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)]
              for i in range(3)]
        try:
            p = _cell_params(Gp)
            dl = max(abs(p[i] - ref[i]) / ref[i] * 100.0 for i in range(3))
        except (ValueError, ZeroDivisionError):
            continue
        if dl > length_tol_pct:
            continue
        accepted.append(M)
        scores.append(TwoFoldScore(
            M, u, h, d, kurlin_distance_to_two_fold(cell, M),
        ))
    # The lattice holohedry always contains the inversion centre: every lattice
    # is centrosymmetric (t and -t are both lattice vectors). The Lebedev set is
    # proper rotations (det = +1) only, so we seed the closure with -I to recover
    # the full centrosymmetric point group.
    gens = [SymmetryOp(IDENTITY3, Vector3((0, 0, 0))),
            SymmetryOp(Matrix3([[Fr(-1), Fr(0), Fr(0)],
                                [Fr(0), Fr(-1), Fr(0)],
                                [Fr(0), Fr(0), Fr(-1)]]), Vector3((0, 0, 0)))]
    for M in accepted:
        gens.append(SymmetryOp(Matrix3([[Fr(M[i][j]) for j in range(3)] for i in range(3)]),
                               Vector3((0, 0, 0))))
    ops = close_group(gens)
    order = len(ops)
    system = _ORDER_TO_SYSTEM.get(order, f"order-{order}")
    scores.sort(key=lambda s: (s.le_page_delta, s.kurlin_distance))
    return LatticeSymmetry(ops, system, accepted, scores)


# --- tolerance metric-automorphism group (for cell comparison / reindexing) ---
def _cell_params(G):
    from .cell.metric import params_from_metric
    return params_from_metric(G)


def tolerance_metric_symmetry(cell, length_tol_pct: float = 2.0,
                              angle_tol_deg: float = 2.0):
    """Metric-automorphism group of a cell within a (length, angle) tolerance.

    Returns the set of integer rotations M (the Lebedev proper rotations closed
    with the inversion centre) for which M^T G M has cell parameters within
    ``length_tol_pct`` (edges, percent) and ``angle_tol_deg`` (angles, degrees)
    of the original cell G.

    Unlike :func:`lattice_symmetry` (which uses Le Page's purely *angular*
    delta), this accepts operators that become symmetries under a *length*
    near-degeneracy too -- e.g. the a~b swap that Niggli reduction flips between,
    or a beta~90 pseudo-orthorhombic two-fold. This is the right object to
    quotient for cell comparison and indexing-ambiguity resolution: it contains
    the exact holohedry, the pseudo-symmetry operators, and the cell-choice /
    reduction-instability transforms in one finite, tolerance-defined set.
    """
    G = _metric_tensor(cell)
    ref = _cell_params(G)
    accepted = []
    for M in LEBEDEV_MATRICES:
        MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        Gp = [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        try:
            p = _cell_params(Gp)
        except (ValueError, ZeroDivisionError):
            continue
        dl = max(abs(p[i] - ref[i]) / ref[i] * 100.0 for i in range(3))
        da = max(abs(p[3 + i] - ref[3 + i]) for i in range(3))
        if dl <= length_tol_pct and da <= angle_tol_deg:
            accepted.append(M)
    # every lattice is centrosymmetric: seed with identity and -I, then close
    gens = [SymmetryOp(IDENTITY3, Vector3((0, 0, 0))),
            SymmetryOp(Matrix3([[Fr(-1), Fr(0), Fr(0)],
                                [Fr(0), Fr(-1), Fr(0)],
                                [Fr(0), Fr(0), Fr(-1)]]), Vector3((0, 0, 0)))]
    for M in accepted:
        gens.append(SymmetryOp(Matrix3([[Fr(M[i][j]) for j in range(3)] for i in range(3)]),
                               Vector3((0, 0, 0))))
    return close_group(gens)
