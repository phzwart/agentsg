"""
G6 / S6 lattice embeddings and boundary-aware distances.

This module embeds a unit cell as a point in a six-dimensional space so that the
*space of lattices* becomes a continuous manifold rather than a discrete list of
Bravais types. It provides the two distance primitives that a "continuous
structural state space" view of crystallography rests on:

  1. cell <-> cell distance          -- a metric on the lattice manifold, robust
                                        to cell choice and to the Niggli
                                        reduction-flip discontinuity.
  2. cell <-> symmetry-subspace       -- a *continuous* measure of how much
     distance                          symmetry a lattice has, replacing the
                                        binary "is it tetragonal? yes/no".

Two embeddings (Andrews & Bernstein):

  * G6  (Niggli):  g = (a.a, b.b, c.c, 2 b.c, 2 a.c, 2 a.b)
                   -- the metric tensor as a vector. Andrews & Bernstein (1988,
                      2014); the basis for NCDist / BGAOL / SAUC.
  * S6  (Selling): s = (b.c, a.c, a.b, a.d, b.d, c.d),  d = -(a+b+c)
                   -- the Selling scalars from Delaunay reduction; all <= 0 when
                      reduced. Cleaner boundary structure and faster distances
                      (Andrews, Bernstein & Sauter 2019, "Selling reduction
                      versus Niggli reduction").

Boundary-aware distance. The reduction boundaries (e.g. b = c, an angle passing
through 90 deg) are where two Niggli-reduced representations of nearby lattices
sit far apart in raw G6 -- the reduction flip. The boundary transformations that
relate them are the reduction operations and, importantly, they are NOT
isometric, so a plain Euclidean G6 distance is wrong near a boundary. The fix
(Andrews & Bernstein 2014, NCDist) is to minimise the distance over the orbit of
boundary transforms. This module implements that as an exact minimisation over a
bounded orbit of integer unimodular transforms (entries in {-1,0,1}), which
contains the reduction-flip / cell-choice transforms and removes the
discontinuity. The full NCDist "Follower" additionally iterates across
successive boundaries for arbitrarily separated database cells; for resolving
the reduction flip and for local manifold geometry the bounded orbit is exact.

Everything is dependency-free and uses float metric arithmetic (distances are
inherently numeric); the transform orbit itself is exact integer algebra.
"""
from __future__ import annotations
from functools import lru_cache
from math import sqrt

from .metric import UnitCell
from .reduction import niggli_reduce


# ---- embeddings -------------------------------------------------------------
def g6_from_metric(G):
    """G6 vector (a.a, b.b, c.c, 2b.c, 2a.c, 2a.b) from a 3x3 metric tensor."""
    return (G[0][0], G[1][1], G[2][2], 2.0 * G[1][2], 2.0 * G[0][2], 2.0 * G[0][1])


def s6_from_metric(G):
    """S6 Selling scalars (b.c, a.c, a.b, a.d, b.d, c.d), d = -(a+b+c)."""
    g11, g22, g33 = G[0][0], G[1][1], G[2][2]
    g23, g13, g12 = G[1][2], G[0][2], G[0][1]
    return (g23, g13, g12,
            -(g11 + g12 + g13),      # a.d
            -(g12 + g22 + g23),      # b.d
            -(g13 + g23 + g33))      # c.d


def g6(cell):
    """G6 vector of a unit cell (a,b,c,alpha,beta,gamma), angles in degrees."""
    return g6_from_metric(UnitCell(*cell).metric_tensor())


def s6(cell):
    """S6 (Selling) vector of a unit cell."""
    return s6_from_metric(UnitCell(*cell).metric_tensor())


def _euclid(u, v):
    return sqrt(sum((u[i] - v[i]) ** 2 for i in range(len(u))))


# ---- bounded orbit of integer unimodular transforms ------------------------
@lru_cache(maxsize=1)
def _unimodular_pm1():
    """All integer 3x3 matrices with entries in {-1,0,1} and det = +/-1.

    This finite set contains the reduction-flip / cell-choice boundary
    transforms; minimising G6 distance over it removes the reduction-flip
    discontinuity. Cached (computed once).
    """
    from itertools import product
    mats = []
    for vals in product((-1, 0, 1), repeat=9):
        m = (vals[0:3], vals[3:6], vals[6:9])
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        if det in (1, -1):
            mats.append(m)
    return tuple(mats)


def _transform_metric(G, M):
    """M^T G M for integer matrix M (columns = new basis)."""
    MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


# ---- cell <-> cell distance -------------------------------------------------
def g6_distance(cell_A, cell_B, boundary_aware=True):
    """Distance between two lattices in G6 (units: Angstrom^2).

    With ``boundary_aware`` (default), both cells are Niggli-reduced and the
    distance is minimised over the bounded orbit of {-1,0,1} unimodular
    transforms of A -- this is robust to cell choice and continuous across the
    reduction-flip boundary. With ``boundary_aware=False`` it is the raw
    Euclidean distance between the reduced G6 vectors (which JUMPS at a
    reduction boundary -- provided for comparison / to show the discontinuity).
    """
    rA, _ = niggli_reduce(*cell_A)
    rB, _ = niggli_reduce(*cell_B)
    gB = g6(rB)
    GA = UnitCell(*rA).metric_tensor()
    if not boundary_aware:
        return _euclid(g6_from_metric(GA), gB)
    best = None
    for M in _unimodular_pm1():
        d = _euclid(g6_from_metric(_transform_metric(GA, M)), gB)
        if best is None or d < best:
            best = d
    return best


def s6_distance(cell_A, cell_B, boundary_aware=True):
    """Distance between two lattices in S6 (Selling) space (units: Angstrom^2)."""
    rA, _ = niggli_reduce(*cell_A)
    rB, _ = niggli_reduce(*cell_B)
    sB = s6(rB)
    GA = UnitCell(*rA).metric_tensor()
    if not boundary_aware:
        return _euclid(s6_from_metric(GA), sB)
    best = None
    for M in _unimodular_pm1():
        d = _euclid(s6_from_metric(_transform_metric(GA, M)), sB)
        if best is None or d < best:
            best = d
    return best


# ---- cell <-> symmetry-subspace distance (continuous symmetry) -------------
def _symmetrize_metric(G, rot_rows):
    """Reynolds average (1/|G|) sum_i W_i^T G W_i over integer rotations."""
    n = len(rot_rows)
    acc = [[0.0] * 3 for _ in range(3)]
    for W in rot_rows:
        WtG = [[sum(W[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        Gp = [[sum(WtG[i][k] * W[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                acc[i][j] += Gp[i][j] / n
    return acc


def _op_rows(point_group_ops):
    rows = []
    for op in point_group_ops:
        W = op.W if hasattr(op, "W") else op
        R = W.rows if hasattr(W, "rows") else W
        rows.append([[float(R[i][j]) for j in range(3)] for i in range(3)])
    return rows


def _cell_from_metric(G):
    """Cell parameters (a,b,c,α,β,γ) from a metric tensor."""
    from .metric import cell_from_metric
    return cell_from_metric(G)


def distance_to_symmetry(cell, point_group_ops):
    """Continuous G6 distance from a cell to the subspace fixed by a point group.

    ``point_group_ops`` is an iterable of SymmetryOp or Matrix3 (rotation parts).
    Returns ||g6(G) - g6(G_symmetrized)||, where G_symmetrized is the Reynolds
    average of the metric over the group -- the nearest metric that is exactly
    invariant under that symmetry. Zero means the lattice already has the
    symmetry; the value grows smoothly as the lattice is distorted away from it.

    This is BGAOL's "distance to a Bravais-lattice subspace" (Andrews &
    Bernstein 2014) and it is the continuous replacement for a binary
    "does this lattice have symmetry X?" test: symmetry becomes a smooth field
    over the lattice manifold, not a yes/no with a tolerance cliff.
    """
    G = UnitCell(*cell).metric_tensor()
    Gs = _symmetrize_metric(G, _op_rows(point_group_ops))
    return _euclid(g6_from_metric(G), g6_from_metric(Gs))


def kurlin_distance_to_symmetry(cell, point_group_ops):
    """Kurlin root-invariant distance from a cell to a point-group subspace.

    Reynolds-averages the metric under ``point_group_ops``, then returns
    :func:`root_distance` between the original cell and that symmetrised cell
    (Ångström). Zero iff the lattice already has the symmetry.
    """
    from .rootform import root_distance
    G = UnitCell(*cell).metric_tensor()
    Gs = _symmetrize_metric(G, _op_rows(point_group_ops))
    return root_distance(tuple(float(x) for x in cell), _cell_from_metric(Gs))


def symmetry_deficiency_spectrum(cell, tol_ops_by_system):
    """Map {system_name: point_group_ops} -> {system_name: distance_to_symmetry}.

    A convenience wrapper: the continuous 'how close is this lattice to each
    candidate holohedry' spectrum, in G6 Angstrom^2 units. Smallest non-trivial
    entry is the nearest higher-symmetry lattice.
    """
    return {name: distance_to_symmetry(cell, ops)
            for name, ops in tol_ops_by_system.items()}


def kurlin_deficiency_spectrum(cell, tol_ops_by_system):
    """Like :func:`symmetry_deficiency_spectrum` but with Kurlin root distances."""
    return {name: kurlin_distance_to_symmetry(cell, ops)
            for name, ops in tol_ops_by_system.items()}
