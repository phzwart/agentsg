"""
Exact rational linear solvers used across the symmetry algebra.

Public home for reduced-row-echelon form and affine solves over
``fractions.Fraction``. Callers that previously imported private
``_rref`` / ``_solve_affine`` from ``wyckoff`` should import from here.
"""
from __future__ import annotations
from fractions import Fraction as Fr

from .linalg import Matrix3, Vector3


def rref(A: list[list[Fr]], b: list[Fr]):
    """Reduced row echelon of [A|b] over the rationals; returns (R, c, pivots)."""
    A = [row[:] for row in A]
    b = b[:]
    n = len(A)
    m = len(A[0]) if n else 0
    pivots = []
    r = 0
    for col in range(m):
        piv = None
        for i in range(r, n):
            if A[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        b[r], b[piv] = b[piv], b[r]
        inv = A[r][col]
        A[r] = [v / inv for v in A[r]]
        b[r] = b[r] / inv
        for i in range(n):
            if i != r and A[i][col] != 0:
                f = A[i][col]
                A[i] = [av - f * rv for av, rv in zip(A[i], A[r])]
                b[i] = b[i] - f * b[r]
        pivots.append(col)
        r += 1
        if r == n:
            break
    return A, b, pivots


def solve_affine(M: Matrix3, rhs: Vector3):
    """Solve M x = rhs over the rationals.

    Returns ``(particular, nullspace_basis)`` or ``None`` if inconsistent.
    Basis vectors span the free (nullspace) directions of the solution.
    """
    A = [[M.rows[i][j] for j in range(3)] for i in range(3)]
    b = [rhs.v[i] for i in range(3)]
    R, c, pivots = rref(A, b)
    for i in range(3):
        if all(R[i][j] == 0 for j in range(3)) and c[i] != 0:
            return None
    pivot_set = set(pivots)
    free = [j for j in range(3) if j not in pivot_set]
    part = [Fr(0)] * 3
    for ri, col in enumerate(pivots):
        part[col] = c[ri]
    particular = Vector3(part)
    basis = []
    for f in free:
        vec = [Fr(0)] * 3
        vec[f] = Fr(1)
        for ri, col in enumerate(pivots):
            vec[col] = -R[ri][f]
        basis.append(Vector3(vec))
    return particular, basis


# Back-compat private names (used by wyckoff aliases and any lingering callers).
_rref = rref
_solve_affine = solve_affine
