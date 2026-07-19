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
