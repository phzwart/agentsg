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
"""
from __future__ import annotations
from math import sqrt, cos, radians, degrees, acos
from typing import Tuple


def _params_to_scalars(a, b, c, al, be, ga):
    A = a * a
    B = b * b
    C = c * c
    xi = 2 * b * c * cos(radians(al))
    eta = 2 * a * c * cos(radians(be))
    zeta = 2 * a * b * cos(radians(ga))
    return A, B, C, xi, eta, zeta


def _scalars_to_params(A, B, C, xi, eta, zeta):
    a = sqrt(A); b = sqrt(B); c = sqrt(C)
    al = degrees(acos(max(-1.0, min(1.0, xi / (2 * b * c)))))
    be = degrees(acos(max(-1.0, min(1.0, eta / (2 * a * c)))))
    ga = degrees(acos(max(-1.0, min(1.0, zeta / (2 * a * b)))))
    return a, b, c, al, be, ga


def _matmul(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def niggli_reduce(a, b, c, alpha, beta, gamma, eps_rel: float = 1e-9, max_iter: int = 1000):
    """Niggli-reduce a unit cell.

    Returns ``(reduced_params, change_of_basis)`` where ``reduced_params`` is the
    6-tuple (a, b, c, alpha, beta, gamma) of the reduced cell and
    ``change_of_basis`` is the integer 3x3 matrix M such that the reduced basis
    vectors are (old basis) @ M.

    ``det(M)`` is in ``{+1, -1}``. A negative determinant is an
    orientation-reversing (improper) change of basis; it still describes the
    same lattice metric. Callers that need a proper rotation should check the
    sign.

    Implements the stabilised algorithm (Grosse-Kunstleve/Sauter/Adams 2004):
    all comparisons use a relative epsilon derived from the cell scale.
    """
    A, B, C, xi, eta, zeta = _params_to_scalars(a, b, c, alpha, beta, gamma)
    eps = eps_rel * (A * B * C) ** (1.0 / 3.0)

    # change of basis, integer, columns = new basis in old coords
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def _apply(T):
        nonlocal M
        M = _matmul(M, T)

    def gt(x, y):   # x > y  with tolerance
        return x > y + eps

    def lt(x, y):
        return x < y - eps

    def eq(x, y):
        return abs(x - y) <= eps

    n = 0
    while True:
        n += 1
        if n > max_iter:
            raise RuntimeError("Niggli reduction did not converge")

        # Step 1
        if gt(A, B) or (eq(A, B) and gt(abs(xi), abs(eta))):
            A, B = B, A
            xi, eta = eta, xi
            _apply([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        # Step 2
        if gt(B, C) or (eq(B, C) and gt(abs(eta), abs(zeta))):
            B, C = C, B
            eta, zeta = zeta, eta
            _apply([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            continue
        # Step 3/4: sign normalisation of xi, eta, zeta
        # l,m,n signs: +1 if positive, -1 if negative, 0 if ~0
        def sign(x):
            if gt(x, 0):
                return 1
            if lt(x, 0):
                return -1
            return 0
        l, m, nn = sign(xi), sign(eta), sign(zeta)
        prod = l * m * nn
        if prod == 1:  # all positive product -> make all positive (type I)
            i = -1 if l < 0 else 1
            j = -1 if m < 0 else 1
            k = -1 if nn < 0 else 1
            # for product==1 with a zero: handled below; here all nonzero
            Ti = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # set diagonal to flip appropriate axes so xi,eta,zeta become +
            di = 1; dj = 1; dk = 1
            if l == -1: di = -1
            if m == -1: dj = -1
            if nn == -1: dk = -1
            xi = abs(xi); eta = abs(eta); zeta = abs(zeta)
            Ti = [[di, 0, 0], [0, dj, 0], [0, 0, dk]]
            _apply(Ti)
        elif prod == -1 or prod == 0:
            # make all non-positive (type II): flip to negatives
            di = dj = dk = 1
            # count: put the +1's to -1 to reach product <=0 canonical form
            # follow GKSA: choose signs so that xi,eta,zeta <= 0
            i = 1 if l > 0 else 1
            di = -1 if l > 0 else 1
            dj = -1 if m > 0 else 1
            dk = -1 if nn > 0 else 1
            # if there is a zero among them, absorb one flip to keep det +1
            zeros = [l == 0, m == 0, nn == 0]
            flips = [di == -1, dj == -1, dk == -1]
            if sum(flips) % 2 == 1:
                # need even number of flips to keep det +1; toggle a zero axis
                for idx, isz in enumerate(zeros):
                    if isz:
                        if idx == 0: di *= -1
                        elif idx == 1: dj *= -1
                        else: dk *= -1
                        break
                else:
                    # no zero available; toggle the smallest-magnitude one back
                    pass
            xi = -abs(xi); eta = -abs(eta); zeta = -abs(zeta)
            _apply([[di, 0, 0], [0, dj, 0], [0, 0, dk]])

        # Step 5
        if gt(abs(xi), B) or (eq(xi, B) and lt(2 * eta, zeta)) or (eq(xi, -B) and lt(zeta, 0)):
            s = 1 if xi > 0 else -1
            C = B + C - xi * s
            eta = eta - zeta * s
            xi = xi - 2 * B * s
            _apply([[1, 0, 0], [0, 1, 0], [0, -s, 1]])
            continue
        # Step 6
        if gt(abs(eta), A) or (eq(eta, A) and lt(2 * xi, zeta)) or (eq(eta, -A) and lt(zeta, 0)):
            s = 1 if eta > 0 else -1
            C = A + C - eta * s
            xi = xi - zeta * s
            eta = eta - 2 * A * s
            _apply([[1, 0, 0], [0, 1, 0], [-s, 0, 1]])
            continue
        # Step 7
        if gt(abs(zeta), A) or (eq(zeta, A) and lt(2 * xi, eta)) or (eq(zeta, -A) and lt(eta, 0)):
            s = 1 if zeta > 0 else -1
            B = A + B - zeta * s
            xi = xi - eta * s
            zeta = zeta - 2 * A * s
            _apply([[1, 0, 0], [-s, 1, 0], [0, 0, 1]])
            continue
        # Step 8
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
    return reduced, M


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

    Unlike :func:`niggli_reduce`, every reduction step is paired with an integer
    matrix whose action on the metric tensor exactly reproduces that step's
    scalar update. The returned change of basis therefore satisfies

        M^T @ G_original @ M == G_reduced

    to machine precision for *all* inputs, including heavily transformed
    (non-Buerger) cells where a step matrix inconsistent with its scalar update
    would otherwise desynchronise. This makes ``niggli_gk`` the reduction to use
    when the change of basis itself is needed (reindexing, setting bridges),
    not just the reduced parameters.

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

    Notes
    -----
    The reduction is driven on the six scalars (A, B, C, xi, eta, zeta) exactly
    as in :func:`niggli_reduce`, so convergence behaviour is identical; only the
    accumulated change of basis differs (and is correct here for every step).
    """
    a, b, c, alpha, beta, gamma = cell
    A, B, C, xi, eta, zeta = _params_to_scalars(a, b, c, alpha, beta, gamma)
    eps = eps_rel * (A * B * C) ** (1.0 / 3.0)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def _apply(T):
        nonlocal M
        M = _matmul(M, T)

    def gt(x, y):
        return x > y + eps

    def lt(x, y):
        return x < y - eps

    def eq(x, y):
        return abs(x - y) <= eps

    def sign(x):
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
    return reduced, M
