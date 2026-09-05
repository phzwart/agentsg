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

    Returns the stabiliser operators in their *original* (W, w) form — they are
    not re-based to act about x. Use :func:`site_symmetry_point_group` for the
    rotation parts only (the site point group).
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
# Implementations live in rational_solve; keep private aliases for back-compat.
from .rational_solve import rref as _rref, solve_affine as _solve_affine  # noqa: E402


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
