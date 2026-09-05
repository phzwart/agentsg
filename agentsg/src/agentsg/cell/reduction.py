"""
Niggli cell reduction -- the stabilised algorithm of Grosse-Kunstleve, Sauter &
Adams, *Acta Cryst.* A60, 1-6 (2004).

The classic Krivy-Gruber (1976) steps are numerically fragile on near-degenerate
cells: exact-equality comparisons in the presence of floating-point error can
cause the reduction to cycle or terminate at a non-reduced cell. The 2004 paper
fixes this by (a) using a relative tolerance ``eps`` for all comparisons, and
(b) a specific comparison ordering. This module follows that formulation.

Input/output are metric-tensor scalars (A, B, C, xi, eta, zeta) where

    A = a.a,  B = b.b,  C = c.c,
    xi = 2 b.c,  eta = 2 a.c,  zeta = 2 a.b

i.e. G = [[A, zeta/2, eta/2], [zeta/2, B, xi/2], [eta/2, xi/2, C]].
We also track the 3x3 integer change-of-basis matrix so the reduced cell can be
related back to the input.

Every reduction step is paired with an integer matrix whose action on the metric
exactly reproduces that step's scalar update, so the accumulated change of basis
``M`` satisfies

    M^T G_original M == G_reduced

to machine precision. A post-condition assertion enforces this on every return.
"""
from __future__ import annotations
from math import sqrt, cos, radians, degrees, acos


def _params_to_scalars(a, b, c, al, be, ga):
    """Convert cell parameters to scalar products (A, B, C, xi, eta, zeta)."""
    A = a * a
    B = b * b
    C = c * c
    xi = 2 * b * c * cos(radians(al))
    eta = 2 * a * c * cos(radians(be))
    zeta = 2 * a * b * cos(radians(ga))
    return A, B, C, xi, eta, zeta


def _scalars_to_params(A, B, C, xi, eta, zeta):
    """Convert scalar products (A, B, C, xi, eta, zeta) back to cell parameters."""
    a = sqrt(A); b = sqrt(B); c = sqrt(C)
    al = degrees(acos(max(-1.0, min(1.0, xi / (2 * b * c)))))
    be = degrees(acos(max(-1.0, min(1.0, eta / (2 * a * c)))))
    ga = degrees(acos(max(-1.0, min(1.0, zeta / (2 * a * b)))))
    return a, b, c, al, be, ga


def _matmul(P, Q):
    """Multiply two 3x3 matrices represented as nested lists."""
    return [[sum(P[i][k] * Q[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _transform_metric(G, M):
    """G' = M^T G M for 3x3 lists."""
    # PtG[r][b] = sum_k M[k][r] * G[k][b]
    PtG = [[sum(M[k][r] * G[k][b] for k in range(3)) for b in range(3)]
           for r in range(3)]
    return [[sum(PtG[r][k] * M[k][b] for k in range(3)) for b in range(3)]
            for r in range(3)]


def _gram_from_params(a, b, c, alpha, beta, gamma):
    """Construct 3x3 Gram matrix from unit-cell parameters."""
    A, B, C, xi, eta, zeta = _params_to_scalars(a, b, c, alpha, beta, gamma)
    return [
        [A, zeta / 2.0, eta / 2.0],
        [zeta / 2.0, B, xi / 2.0],
        [eta / 2.0, xi / 2.0, C],
    ]


def _check_cob_invariant(G_orig, M, reduced, tol_rel=1e-6):
    """Assert M^T G_orig M == gram(reduced). Raises AssertionError on failure."""
    G_pred = _transform_metric(G_orig, M)
    G_red = _gram_from_params(*reduced)
    scale = max(abs(G_red[i][j]) for i in range(3) for j in range(3)) or 1.0
    err = max(abs(G_pred[i][j] - G_red[i][j])
              for i in range(3) for j in range(3))
    if err > tol_rel * scale:
        raise AssertionError(
            f"Niggli CoB invariant failed: max |M^T G M - G_red| = {err} "
            f"(tol {tol_rel * scale}); M={M}, reduced={reduced}"
        )


def _sign_matrix(l, m, nn):
    """Determinant-+1 diagonal integer matrix for the Krivy-Gruber sign step.

    Given the signs (l, m, nn) of (xi, eta, zeta), return a diagonal matrix
    T = diag(i, j, k) with det = +1 such that after ``G <- T^T G T`` the three
    off-diagonals are all >= 0 (type I, when ``l*m*nn > 0``) or all <= 0
    (type II, otherwise). For a diagonal T the off-diagonals transform as
    xi -> (j*k) xi, eta -> (i*k) eta, zeta -> (i*j) zeta, so a single sign per
    axis does not suffice; the correct even-parity flip is found by enumeration
    of the four det-+1 diagonals.
    """
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    type_I = (l * m * nn > 0)
    best = None
    for i in (1, -1):
        for j in (1, -1):
            for k in (1, -1):
                if i * j * k != 1:
                    continue
                nx, ne, nz = j * k * l, i * k * m, i * j * nn
                ok = (nx >= 0 and ne >= 0 and nz >= 0) if type_I else \
                     (nx <= 0 and ne <= 0 and nz <= 0)
                if ok:
                    T = [[i, 0, 0], [0, j, 0], [0, 0, k]]
                    if T == identity:
                        return identity
                    if best is None:
                        best = T
    return best if best is not None else identity


def niggli_gk(cell, eps_rel: float = 1e-9, max_iter: int = 1000):
    """Niggli-reduce a unit cell, tracking an exact change of basis.

    A reimplementation of the Niggli reduction following the stabilised algorithm
    of Grosse-Kunstleve, Sauter & Adams (Acta Cryst. A60, 1-6, 2004): the
    Krivy-Gruber (1976) reduction steps are applied with a relative epsilon
    derived from the cell scale so that boundary cases are handled robustly.

    Every reduction step is paired with an integer matrix whose action on the
    metric tensor exactly reproduces that step's scalar update. The returned
    change of basis therefore satisfies

        M^T @ G_original @ M == G_reduced

    to machine precision. A post-condition assertion enforces this on every
    return.

    Parameters
    ----------
    cell : tuple
        (a, b, c, alpha, beta, gamma), lengths in Angstrom and angles in degrees.
    eps_rel : float
        Relative tolerance; comparisons use ``eps = eps_rel * (A*B*C)**(1/3)``.
    max_iter : int
        Iteration cap; a RuntimeError is raised if it is exceeded.

    Returns
    -------
    (reduced_params, M) : tuple
        ``reduced_params`` is the reduced (a, b, c, alpha, beta, gamma);
        ``M`` is a 3x3 integer list whose columns are the reduced basis vectors
        expressed in the original basis. ``det(M)`` is in ``{+1, -1}``.
    """
    if len(cell) != 6:
        raise ValueError("cell must be (a, b, c, alpha, beta, gamma)")
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    if min(a, b, c) <= 0:
        raise ValueError(f"cell edge lengths must be positive, got a,b,c={a,b,c}")
    for name, ang in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if not (0.0 < ang < 180.0):
            raise ValueError(f"cell angle {name} must be in (0, 180) degrees, got {ang}")
    # Degenerate parallelepiped: volume ∝ sqrt(1-cos²…) → non-positive.
    try:
        G_orig = _gram_from_params(a, b, c, alpha, beta, gamma)
        vol2 = (
            G_orig[0][0] * (G_orig[1][1] * G_orig[2][2] - G_orig[1][2] ** 2)
            - G_orig[0][1] * (G_orig[0][1] * G_orig[2][2] - G_orig[0][2] * G_orig[1][2])
            + G_orig[0][2] * (G_orig[0][1] * G_orig[1][2] - G_orig[0][2] * G_orig[1][1])
        )
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"degenerate cell {cell!r}") from exc
    if vol2 <= 0:
        raise ValueError(f"degenerate cell (non-positive volume) {cell!r}")

    A, B, C, xi, eta, zeta = _params_to_scalars(a, b, c, alpha, beta, gamma)
    eps = eps_rel * (A * B * C) ** (1.0 / 3.0)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def _apply(T):
        """Accumulate change-of-basis matrix M <- M @ T."""
        nonlocal M
        M = _matmul(M, T)

    def gt(x, y):
        """Tolerance-guarded x > y."""
        return x > y + eps

    def lt(x, y):
        """Tolerance-guarded x < y."""
        return x < y - eps

    def eq(x, y):
        """Tolerance-guarded equality |x - y| <= eps."""
        return abs(x - y) <= eps

    def sign(x):
        """Tolerance-guarded signum (-1, 0, or 1)."""
        if gt(x, 0):
            return 1
        if lt(x, 0):
            return -1
        return 0

    n = 0
    while True:
        n += 1
        if n > max_iter:
            raise RuntimeError("Niggli reduction did not converge")

        # Step 1: order A <= B (swap a, b)
        if gt(A, B) or (eq(A, B) and gt(abs(xi), abs(eta))):
            A, B = B, A
            xi, eta = eta, xi
            _apply([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        # Step 2: order B <= C (swap b, c)
        if gt(B, C) or (eq(B, C) and gt(abs(eta), abs(zeta))):
            B, C = C, B
            eta, zeta = zeta, eta
            _apply([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            continue
        # Step 3/4: sign normalisation of (xi, eta, zeta)
        T = _sign_matrix(sign(xi), sign(eta), sign(zeta))
        if T != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            i, j, k = T[0][0], T[1][1], T[2][2]
            xi *= j * k
            eta *= i * k
            zeta *= i * j
            _apply(T)
        # Step 5: reduce xi (c -> c - s*b)
        if gt(abs(xi), B) or (eq(xi, B) and lt(2 * eta, zeta)) or (eq(xi, -B) and lt(zeta, 0)):
            s = 1 if xi > 0 else -1
            C = B + C - xi * s
            eta = eta - zeta * s
            xi = xi - 2 * B * s
            _apply([[1, 0, 0], [0, 1, -s], [0, 0, 1]])
            continue
        # Step 6: reduce eta (c -> c - s*a)
        if gt(abs(eta), A) or (eq(eta, A) and lt(2 * xi, zeta)) or (eq(eta, -A) and lt(zeta, 0)):
            s = 1 if eta > 0 else -1
            C = A + C - eta * s
            xi = xi - zeta * s
            eta = eta - 2 * A * s
            _apply([[1, 0, -s], [0, 1, 0], [0, 0, 1]])
            continue
        # Step 7: reduce zeta (b -> b - s*a)
        if gt(abs(zeta), A) or (eq(zeta, A) and lt(2 * xi, eta)) or (eq(zeta, -A) and lt(eta, 0)):
            s = 1 if zeta > 0 else -1
            B = A + B - zeta * s
            xi = xi - eta * s
            zeta = zeta - 2 * A * s
            _apply([[1, -s, 0], [0, 1, 0], [0, 0, 1]])
            continue
        # Step 8: final (c -> a + b + c)
        if lt(xi + eta + zeta + A + B, 0) or (
            eq(xi + eta + zeta + A + B, 0) and gt(2 * (A + eta) + zeta, 0)
        ):
            C = A + B + C + xi + eta + zeta
            xi = 2 * B + xi + zeta
            eta = 2 * A + eta + zeta
            _apply([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
            continue
        break

    reduced = _scalars_to_params(A, B, C, xi, eta, zeta)
    _check_cob_invariant(G_orig, M, reduced)
    return reduced, M


def niggli_reduce(a, b, c, alpha, beta, gamma, eps_rel: float = 1e-9,
                  max_iter: int = 1000):
    """Niggli-reduce a unit cell.

    Returns ``(reduced_params, change_of_basis)`` where ``reduced_params`` is the
    6-tuple (a, b, c, alpha, beta, gamma) of the reduced cell and
    ``change_of_basis`` is the integer 3x3 matrix M such that the reduced basis
    vectors are (old basis) @ M, i.e.

        M^T G_original M == G_reduced .

    ``det(M)`` is in ``{+1, -1}``. A negative determinant is an
    orientation-reversing (improper) change of basis; it still describes the
    same lattice metric. Callers that need a proper rotation should check the
    sign.

    Implements the stabilised algorithm (Grosse-Kunstleve/Sauter/Adams 2004).
    This is a thin wrapper around :func:`niggli_gk` (same implementation, same
    CoB invariant); prefer ``niggli_gk(cell)`` when you already have a 6-tuple.
    """
    return niggli_gk((a, b, c, alpha, beta, gamma),
                     eps_rel=eps_rel, max_iter=max_iter)
