"""
Wyckoff positions, site symmetry, and orbits -- all *computed* from the exact
operation list, no lookup table.

The site-symmetry group of a point x is the stabiliser
    S(x) = { (W,w) in G : W x + w == x  (mod 1) }.
Its order divides the group order, and the orbit (general/special position)
multiplicity is
    mult(x) = |G| / |S(x)|.

Special-position *loci* are the fixed-point sets of operations: solving
    (W - I) x == -w   (mod 1)
exactly over the rationals gives, for each operation, an affine subspace
(point, line, or plane) of fixed points. Points lying on such a locus have a
larger stabiliser and hence lower multiplicity. The Wyckoff letter (a, b, c...)
attached to each stratum is an ITA historical convention and is deliberately
NOT reproduced here; everything numeric (multiplicity, site-symmetry order and
its operations, locus dimension) is derived.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from itertools import product
from typing import Iterable, Sequence
from .linalg import Matrix3, Vector3, IDENTITY3, frac_mod1
from .symmetry_op import SymmetryOp


def _vec_mod1(v: Vector3) -> Vector3:
    return v.mod1()


def site_symmetry_ops(x: Vector3, operations: Iterable[SymmetryOp]) -> frozenset[SymmetryOp]:
    """Operations (W,w) that fix x modulo an integer lattice translation.

    The returned operations are *reduced to the site*: (W, w + Wx - x) so that
    their translation part is a pure lattice vector at x -- i.e. they form the
    site-symmetry point group acting about x. We return them in their original
    (W,w) form here; use :func:`site_symmetry_point_group` for the rotation
    parts only.
    """
    xr = x.mod1()
    out = set()
    for op in operations:
        img = (op.W @ xr) + op.w
        if (img - xr).mod1() == Vector3((0, 0, 0)):
            out.add(op)
    return frozenset(out)


def site_symmetry_point_group(x: Vector3, operations: Iterable[SymmetryOp]) -> frozenset[Matrix3]:
    """Rotation parts of the site-symmetry group at x -- the site point group."""
    return frozenset(op.W for op in site_symmetry_ops(x, operations))


def site_symmetry_order(x: Vector3, operations: Iterable[SymmetryOp]) -> int:
    """Order of the site-symmetry group at x."""
    return len(site_symmetry_ops(x, operations))


def orbit(x: Vector3, operations: Iterable[SymmetryOp]) -> frozenset[Vector3]:
    """Distinct images of x under the group, reduced into the unit cell."""
    xr = x.mod1()
    return frozenset(((op.W @ xr) + op.w).mod1() for op in operations)


def multiplicity(x: Vector3, operations: Sequence[SymmetryOp]) -> int:
    """Orbit multiplicity of x = |G| / |site-symmetry(x)|."""
    return len(orbit(x, operations))


def general_position_multiplicity(operations: Sequence[SymmetryOp]) -> int:
    """Multiplicity of a general position = order of the space group (in the
    conventional cell, i.e. including centring)."""
    return len(list(operations))


# ---------------------------------------------------------------------------
# Exact fixed-locus solver:  solve  (W - I) x == -w  (mod 1)  over the rationals
# ---------------------------------------------------------------------------

def _rref(A: list[list[Fr]], b: list[Fr]):
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


def _solve_affine(M: Matrix3, rhs: Vector3):
    """Solve M x = rhs over the rationals. Returns (particular, nullspace_basis)
    or None if inconsistent. Basis vectors span the solution space directions."""
    A = [[M.rows[i][j] for j in range(3)] for i in range(3)]
    b = [rhs.v[i] for i in range(3)]
    R, c, pivots = _rref(A, b)
    # consistency: any all-zero row of R with nonzero c is inconsistent
    for i in range(3):
        if all(R[i][j] == 0 for j in range(3)) and c[i] != 0:
            return None
    pivot_set = set(pivots)
    free = [j for j in range(3) if j not in pivot_set]
    # particular solution: free vars = 0
    part = [Fr(0)] * 3
    for ri, col in enumerate(pivots):
        part[col] = c[ri]
    particular = Vector3(part)
    # null space basis
    basis = []
    for f in free:
        vec = [Fr(0)] * 3
        vec[f] = Fr(1)
        for ri, col in enumerate(pivots):
            vec[col] = -R[ri][f]
        basis.append(Vector3(vec))
    return particular, basis


def fixed_locus(op: SymmetryOp, t_range: int = 2):
    """All points in the unit cell fixed by ``op`` modulo the lattice.

    Solves (W - I) x = t - w for each integer lattice vector t in a small box,
    returning a list of affine subspaces (particular_point, basis_vectors),
    with particular points reduced into [0,1). ``basis_vectors`` is empty for an
    isolated fixed point, length 1 for a fixed line, 2 for a fixed plane.
    """
    WmI = Matrix3([[op.W.rows[i][j] - (1 if i == j else 0) for j in range(3)]
                   for i in range(3)])
    results = []
    seen = set()
    for t in product(range(-t_range, t_range + 1), repeat=3):
        rhs = Vector3((Fr(t[0]), Fr(t[1]), Fr(t[2]))) - op.w
        sol = _solve_affine(WmI, rhs)
        if sol is None:
            continue
        particular, basis = sol
        p = particular.mod1()
        # canonical key: reduced particular + sorted basis directions
        key = (p.v, tuple(sorted(tuple(bv.v) for bv in basis)))
        if key in seen:
            continue
        seen.add(key)
        results.append((p, basis))
    return results
