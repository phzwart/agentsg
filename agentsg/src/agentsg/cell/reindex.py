"""
Reindexing between two settings of the same lattice, and twin-law cosets.

The root invariant (:mod:`agentsg.cell.rootform`) is an *identity* test: it
certifies that two cells describe the same lattice, but it deliberately discards
*which* change of basis relates them. This module recovers that relation.

The key fact is that a reindexing operator is NEVER unique. If ``P`` maps setting
A onto setting B of the same lattice, then so does ``P . h`` for every ``h`` in
the lattice symmetry group ``H`` = {integer unimodular M : M^T G_A M = G_A}. The
complete answer is therefore a **coset** ``P . H``, not a single matrix:

  * :func:`reindexing_operators` returns the whole coset -- every integer
    operator that reindexes A to B.

  * :func:`reindexing_operator` returns one canonical representative (useful when
    any valid reindexing will do).

Twinning is exactly the situation where the coset matters. On a lattice whose
holohedry ``H`` is larger than the crystal's Laue group ``L`` (merohedry or
pseudo-merohedry), the distinct cosets of ``L`` in ``H`` are the twin domains,
and their representatives are the **twin laws**:

  * :func:`twin_laws` returns coset representatives of the Laue group in the
    lattice symmetry group -- the identity plus one operator per twin domain.

All operators are exact integer 3x3 matrices. The lattice symmetry group is the
tolerance-aware metric-automorphism group
(:func:`agentsg.lattice_symmetry.tolerance_metric_symmetry`), so pseudo-merohedral
twinning (e.g. beta ~ 90 deg) is captured, not only exact merohedry.

Reference for the reconstruction step: V. Kurlin, "A complete isometry
classification of 3-dimensional lattices", arXiv:2201.10543 (2022), Lemma 6.2
(a superbase can be reconstructed from the root invariant, up to isometry).
"""
from __future__ import annotations

from .metric import UnitCell, params_from_metric


def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _transp(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))


def _int_inv(M):
    """Inverse of a unimodular integer matrix, returned as an integer matrix."""
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
           - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    if det not in (1, -1):
        raise ValueError(f"matrix is not unimodular (det={det})")
    cof = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            a, b = [r for r in range(3) if r != i], [c for c in range(3) if c != j]
            minor = (M[a[0]][b[0]] * M[a[1]][b[1]]
                     - M[a[0]][b[1]] * M[a[1]][b[0]])
            cof[j][i] = ((-1) ** (i + j)) * minor       # transpose while building
    return tuple(tuple(cof[i][j] * det for j in range(3)) for i in range(3))


def _lattice_symmetry_matrices(cell, length_tol_pct, angle_tol_deg):
    """Integer rotation matrices of the (tolerance) lattice symmetry group H."""
    from ..lattice_symmetry import tolerance_metric_symmetry
    ops = tolerance_metric_symmetry(cell, length_tol_pct=length_tol_pct,
                                    angle_tol_deg=angle_tol_deg)
    mats = []
    for op in ops:
        W = op.W.rows
        mats.append(tuple(tuple(int(round(float(W[i][j]))) for j in range(3))
                          for i in range(3)))
    return mats


def _cell_close(p, q, length_tol_pct, angle_tol_deg):
    """True if two cell-parameter tuples agree within (length%, angle deg)."""
    dl = max(abs(p[i] - q[i]) / q[i] * 100.0 for i in range(3))
    da = max(abs(p[3 + i] - q[3 + i]) for i in range(3))
    return dl <= length_tol_pct and da <= angle_tol_deg


def _find_base_reindex(cell_A, cell_B, length_tol_pct, angle_tol_deg):
    """Find one integer P with P^T G_A P == G_B (the SPECIFIC setting B).

    Matching is on the metric of B, NOT merely on 'same lattice as B' -- a
    reindexing must reproduce B's actual cell parameters (up to orientation),
    otherwise identity would spuriously 'reindex' A onto every setting of its own
    lattice. Searches the unimodular det=+-1 set.
    """
    from .g6 import _unimodular_pm1
    GA = UnitCell(*cell_A).metric_tensor()

    lt = max(length_tol_pct, 1e-6)
    at = max(angle_tol_deg, 1e-6)
    for P in _unimodular_pm1():
        if _cell_close(params_from_metric(_transform_metric(GA, P)), cell_B, lt, at):
            return P
    return None


def _transform_metric(G, M):
    MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def reindexing_operators(cell_A, cell_B, length_tol_pct=2.0, angle_tol_deg=2.0):
    """Return the complete coset of integer operators reindexing A onto B.

    If A and B are the same lattice, returns every integer unimodular P with
    P^T G_A P isometric to G_B -- the coset ``P0 . H`` where ``P0`` is any one
    solution and ``H`` is the lattice symmetry group of A. Returns an empty list
    if A and B are not the same lattice within tolerance.
    """
    P0 = _find_base_reindex(cell_A, cell_B, length_tol_pct, angle_tol_deg)
    if P0 is None:
        return []
    H = _lattice_symmetry_matrices(cell_A, length_tol_pct, angle_tol_deg)
    coset = {_matmul(P0, h) for h in H}
    return sorted(coset)


def reindexing_operator(cell_A, cell_B, length_tol_pct=2.0, angle_tol_deg=2.0):
    """Return one integer operator reindexing A onto B (or None).

    Any valid reindexing; use :func:`reindexing_operators` for the full coset.
    """
    ops = reindexing_operators(cell_A, cell_B, length_tol_pct, angle_tol_deg)
    return ops[0] if ops else None


def twin_laws(space_group_key, cell, length_tol_pct=2.0, angle_tol_deg=2.0):
    """Return twin-law coset representatives for a crystal on a given lattice.

    The lattice symmetry group ``H`` (holohedry, tolerance-aware) contains the
    crystal Laue group ``L`` (point group + inversion). The cosets of ``L`` in
    ``H`` are the twin domains; this returns one integer operator per coset --
    the identity for the untwinned domain, plus one twin law per extra domain.
    The number of cosets (the twin index) is ``|H| / |L|``: 1 means no
    (pseudo)merohedral twinning is possible on this lattice, >1 means it is.

    Captures pseudo-merohedry as well as exact merohedry, because ``H`` is the
    tolerance metric-automorphism group.
    """
    from ..space_groups import space_group
    from ..group import point_group

    sg = space_group(space_group_key)
    # crystal point-group rotations as integer matrices (point_group returns the
    # set of distinct rotation parts W directly, as Matrix3)
    crystal = set()
    for W in point_group(sg.operations()):
        rows = W.rows
        crystal.add(tuple(tuple(int(round(float(rows[i][j]))) for j in range(3))
                          for i in range(3)))
    negI = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    laue = set(crystal) | {_matmul(negI, M) for M in crystal}

    H = set(_lattice_symmetry_matrices(cell, length_tol_pct, angle_tol_deg))
    laue_in_H = laue & H
    if not laue_in_H:
        laue_in_H = {((1, 0, 0), (0, 1, 0), (0, 0, 1))}

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    reps, covered = [], set()
    # enumerate the identity's coset first, so identity is rep[0] (untwinned)
    for M in [identity] + sorted(H):
        if M not in H:
            continue
        coset = frozenset(_matmul(M, h) for h in laue_in_H)
        if coset not in covered:
            covered.add(coset)
            reps.append(M)
    return reps
