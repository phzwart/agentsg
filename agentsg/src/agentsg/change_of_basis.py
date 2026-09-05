"""
Change-of-basis / reindexing operator (P, p).

CONVENTION (derived once, in comments, and validated by tests/test_change_of_basis.py
against the hexagonal->rhombohedral obverse transform -- rather than assumed):

  - Columns of P give the NEW basis vectors in terms of the OLD ones:
        (a', b', c') = (a, b, c) . P
  - p is the origin of the new setting, expressed in OLD fractional coords.
  - Fractional coordinates transform contravariantly to basis vectors:
        x' = P^-1 (x - p)
  - A symmetry operation (W, w), i.e. x' = W x + w in the OLD system,
    substituting x = P x' + p, becomes in the NEW system:
        W' = P^-1 W P
        w' = P^-1 (W p + w - p)
  - Miller indices are dual to the basis vectors and transform WITH P
    (not its inverse), since h.x must be an invariant phase argument:
        h' = h @ P     (h as a row vector)
    An origin shift p contributes only a phase exp(2 pi i h.p) to
    structure factors; it does not change h itself.
"""
from __future__ import annotations
from .linalg import Matrix3, Vector3
from .symmetry_op import SymmetryOp


class ChangeOfBasis:
    """Exact rational change-of-basis operator (P, p).

    Columns of P express the new basis vectors in terms of the old basis vectors.
    p is the origin shift expressed in old fractional coordinates.
    """

    __slots__ = ("P", "p")

    def __init__(self, P: Matrix3, p: Vector3):
        self.P = P
        self.p = p

    def inverse(self) -> "ChangeOfBasis":
        """Invert the change-of-basis operator: (P^-1, -P^-1 p)."""
        Pinv = self.P.inverse()
        return ChangeOfBasis(Pinv, -(Pinv @ self.p))

    def apply_to_op(self, op: SymmetryOp) -> SymmetryOp:
        """Transform a symmetry operation (W, w) to the new setting."""
        Pinv = self.P.inverse()
        W_new = Pinv @ (op.W @ self.P)
        w_new = Pinv @ ((op.W @ self.p) + op.w - self.p)
        return SymmetryOp(W_new, w_new)

    def apply_to_hkl(self, hkl: Vector3) -> Vector3:
        """Transform Miller indices hkl dual to the basis: h' = h @ P."""
        rows = self.P.rows
        h = hkl.v
        return Vector3(sum(h[i] * rows[i][j] for i in range(3)) for j in range(3))

    def __repr__(self) -> str:
        return f"ChangeOfBasis(P={self.P.rows}, p={self.p.v})"
