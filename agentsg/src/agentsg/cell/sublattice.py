"""
Sublattice generation via the Hermite normal form (HNF).

From Zwart, Grosse-Kunstleve & Adams, "Exploring Metric Symmetry" (2006), sec.
2.4: to enumerate all sublattices that change the (reduced) cell volume by an
integer factor d, it is enough to generate all integer matrices with
determinant d in Hermite normal form -- avoiding the ~2.4e9 brute-force
iterations over a +/-5 element range (Billiet & Rolley-Le Coz 1980;
Rutherford 2006).

We use the upper-triangular HNF

        [ a  b  c ]
    M = [ 0  d  e ]
        [ 0  0  f ]

with a, d, f > 0, a*d*f = index. The *columns* of M are the sublattice basis
vectors in the old basis (matching agentsg's change-of-basis convention, where
columns of P are the new basis vectors). For this column convention the unique
canonical form reduces each above-diagonal entry modulo its ROW pivot:
0 <= b < a, 0 <= c < a, 0 <= e < d. The number of such matrices for index n is
sum_{a d f = n} a^2 * d (OEIS A001001), which this module reproduces exactly,
with one distinct matrix per sublattice.

A sublattice matrix M acts on a unit cell through its metric tensor as
G' = M^T G M (new basis vectors are integer combinations of the old, with the
columns of M giving the new vectors) -- consistent with agentsg's change-of-basis
convention. Applying M enlarges the cell volume by det(M) = index.
"""
from __future__ import annotations
from typing import Iterator, Sequence


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def diagonal_triples(index: int) -> Iterator[tuple[int, int, int]]:
    """All ordered positive triples (a, d, f) with a*d*f = index (trial division)."""
    for a in _divisors(index):
        m = index // a
        for d in _divisors(m):
            f = m // d
            yield (a, d, f)


def generate_sublattices(index: int) -> list[list[list[int]]]:
    """All index-``index`` sublattice matrices of Z^3 in Hermite normal form.

    Returns a list of integer 3x3 matrices (nested lists), each upper triangular
    with positive diagonal whose product is ``index`` and off-diagonal entries
    reduced modulo the relevant diagonal element.
    """
    if index < 1:
        raise ValueError("index must be a positive integer")
    mats = []
    for (a, d, f) in diagonal_triples(index):
        for b in range(a):
            for c in range(a):
                for e in range(d):
                    mats.append([[a, b, c],
                                 [0, d, e],
                                 [0, 0, f]])
    return mats


def sublattice_count(index: int) -> int:
    """Closed-form count of index-``index`` sublattices: sum_{a d f = n} a^2 d."""
    return sum(a * a * d for (a, d, f) in diagonal_triples(index))


def is_hermite_normal_form(M: Sequence[Sequence[int]]) -> bool:
    """True iff M is a valid upper-triangular HNF (positive diagonal, off-diagonal
    entries reduced modulo their column's diagonal)."""
    a, d, f = M[0][0], M[1][1], M[2][2]
    if a <= 0 or d <= 0 or f <= 0:
        return False
    if M[1][0] != 0 or M[2][0] != 0 or M[2][1] != 0:
        return False
    # column convention: above-diagonal entries reduced modulo the ROW pivot
    if not (0 <= M[0][1] < a):
        return False
    if not (0 <= M[0][2] < a):
        return False
    if not (0 <= M[1][2] < d):
        return False
    return True


def apply_to_cell(cell, M):
    """Apply a sublattice matrix M to a unit cell via G' = M^T G M.

    ``cell`` is (a, b, c, alpha, beta, gamma) in degrees; returns the enlarged
    cell parameters. Volume scales by det(M) = index.
    """
    from .metric import UnitCell
    from math import sqrt, degrees, acos
    G = UnitCell(*cell).metric_tensor()
    # G' = M^T G M
    MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    Gp = [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    a = sqrt(Gp[0][0]); b = sqrt(Gp[1][1]); c = sqrt(Gp[2][2])
    al = degrees(acos(max(-1.0, min(1.0, Gp[1][2] / (b * c)))))
    be = degrees(acos(max(-1.0, min(1.0, Gp[0][2] / (a * c)))))
    ga = degrees(acos(max(-1.0, min(1.0, Gp[0][1] / (a * b)))))
    return (a, b, c, al, be, ga)
