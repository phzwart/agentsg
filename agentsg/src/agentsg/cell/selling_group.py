"""The Selling reduction group as a set of change-of-basis operators.

The Selling/Delaunay reduction acts on the *superbase* of a lattice,

    {v0, v1, v2, v3},   with   v0 = -(v1 + v2 + v3),

a zero-sum configuration of four vectors.  Any relabelling of the four indices
is a symmetry of that configuration and induces an integer, unimodular change of
basis on the actual cell vectors (v1, v2, v3): the new basis is
(v_sigma[1], v_sigma[2], v_sigma[3]) expressed in the old (v1, v2, v3) coords,
using v0 = -(v1+v2+v3) whenever index 0 appears.

The permutation group of four labels is S4, of order 24.  But the superbase is
also *centrosymmetric*: negating every vector,

    {v0, v1, v2, v3}  ->  {-v0, -v1, -v2, -v3},

preserves the zero sum and preserves obtuseness (each product v_i.v_j is
unchanged), yet -I is not one of the 24 relabellings -- because -v_i equals
v0 + v_j + v_k (the sum of the other three), never a single superbase vector.
The full group of change-of-basis operators covering all obtuse superbases of
a *generic* (Voronoi type V1) lattice is therefore

    S4 x {+I, -I},   order 48,

a faithful subgroup of GL(3, Z).  At higher-symmetry Voronoi types V2--V5,
Kurlin's Lemmas 4.2--4.5 add further *non-isometric* obtuse-superbase classes
that are not reached by this single-class group alone; the type-dependent
finite closure is built by :mod:`agentsg.cell.selling_closure`.  This module
exposes the V1 / single-class orbit so callers can enumerate settings of one
obtuse-superbase class, or generate the group from a minimal generating set.

Elements with det = +1 are proper (orientation-preserving) changes of basis;
det = -1 are improper.  A transposition, a 4-cycle, and the inversion -I
generate the whole group; the transposition + 4-cycle alone generate the
order-24 permutation subgroup (``proper_rotations`` / ``selling_group_S4``).

Everything is exact integer arithmetic; no runtime dependencies.
"""
from __future__ import annotations

from itertools import permutations
from typing import Iterable

from ..linalg import Matrix3, Vector3
from ..change_of_basis import ChangeOfBasis

# Superbase vectors in cell coordinates: rows are v0, v1, v2, v3 with
# v0 = -(v1+v2+v3) and (v1, v2, v3) the identity basis.
_SUPERBASE = (
    (-1, -1, -1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

_ZERO = Vector3((0, 0, 0))

# A minimal generating pair for the S4 permutation subgroup on the four
# superbase labels: the transposition (0 1) and the 4-cycle (0 1 2 3).
GENERATOR_PERMS = ((1, 0, 2, 3), (1, 2, 3, 0))

# The identity and the inversion (global superbase negation), as Matrix3.
_IDENTITY = Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
_INVERSION = Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])


def _perm_matrix(sigma) -> Matrix3:
    """Change-of-basis matrix for a permutation ``sigma`` of the four superbase
    labels.  Columns are the new basis vectors (v_sigma[1], v_sigma[2],
    v_sigma[3]) in old (v1, v2, v3) coordinates.
    """
    cols = [_SUPERBASE[sigma[1]], _SUPERBASE[sigma[2]], _SUPERBASE[sigma[3]]]
    # column j, row i  ->  Matrix3 rows[i][j]
    rows = [[cols[j][i] for j in range(3)] for i in range(3)]
    return Matrix3(rows)


def permutation_cob(sigma) -> ChangeOfBasis:
    """The :class:`ChangeOfBasis` for one permutation ``sigma`` of the four
    superbase labels (a 4-tuple that is a permutation of ``(0, 1, 2, 3)``)."""
    if sorted(sigma) != [0, 1, 2, 3]:
        raise ValueError(f"sigma must be a permutation of (0,1,2,3); got {sigma!r}")
    return ChangeOfBasis(_perm_matrix(tuple(sigma)), _ZERO)


def inversion_cob() -> ChangeOfBasis:
    """The inversion -I (global negation of the superbase), as a change of
    basis.  This is the generator that S4 alone lacks; together with the
    permutation generators it produces the full order-48 Selling group."""
    return ChangeOfBasis(_INVERSION, _ZERO)


def selling_generators() -> list[ChangeOfBasis]:
    """Generators of the *full* Selling reduction group (order 48): the
    transposition (0 1), the 4-cycle (0 1 2 3), and the inversion -I.  Expanding
    these by :func:`expand_group` reproduces the whole group."""
    return [permutation_cob(s) for s in GENERATOR_PERMS] + [inversion_cob()]


def selling_generators_S4() -> list[ChangeOfBasis]:
    """Generators of the order-24 permutation subgroup only (no inversion): the
    transposition (0 1) and the 4-cycle (0 1 2 3)."""
    return [permutation_cob(s) for s in GENERATOR_PERMS]


def expand_group(generators: Iterable[ChangeOfBasis],
                 max_order: int = 96) -> list[ChangeOfBasis]:
    """Expand a set of change-of-basis generators to the group they generate,
    by breadth-first closure under matrix multiplication (linear part only;
    all operators here have zero origin shift).

    Returns the distinct group elements as :class:`ChangeOfBasis` objects, with
    the identity first.  Raises ``RuntimeError`` if the closure exceeds
    ``max_order`` (a guard against non-group inputs).
    """
    gens = [g.P for g in generators]
    identity = Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    seen = {identity}
    frontier = [identity]
    while frontier:
        X = frontier.pop()
        for g in gens:
            for Y in (X @ g, g @ X):
                if Y not in seen:
                    seen.add(Y)
                    frontier.append(Y)
        if len(seen) > max_order:
            raise RuntimeError(
                f"closure exceeded max_order={max_order}; check generators")
    # identity first, then a stable order by matrix rows
    ordered = [identity] + sorted((M for M in seen if M != identity),
                                  key=lambda M: M.rows)
    return [ChangeOfBasis(M, _ZERO) for M in ordered]


def selling_group_S4() -> list[ChangeOfBasis]:
    """The order-24 permutation subgroup: the 24 superbase relabellings as
    change-of-basis operators (identity first), *without* the inversion.  These
    are the proper + improper label permutations; use :func:`selling_group` for
    the full order-48 group that also covers the centrosymmetric superbases.
    """
    mats = {}
    for sigma in permutations(range(4)):
        M = _perm_matrix(sigma)
        mats[M] = None                       # dedupe (the rep is faithful, so 24)
    ordered = [_IDENTITY] + sorted((M for M in mats if M != _IDENTITY),
                                   key=lambda M: M.rows)
    return [ChangeOfBasis(M, _ZERO) for M in ordered]


def selling_group() -> list[ChangeOfBasis]:
    """The full Selling reduction group: all 48 change-of-basis operators
    (identity first) covering every obtuse superbase of a generic lattice.

    This is S4 x {+I, -I}: the 24 superbase relabellings and their negations.
    It is the same group :func:`expand_group` produces from
    :func:`selling_generators`, and the two are asserted equal in the tests.
    """
    mats = {}
    for sigma in permutations(range(4)):
        M = _perm_matrix(sigma)
        mats[M] = None
        mats[_INVERSION @ M] = None          # centrosymmetric partner
    ordered = [_IDENTITY] + sorted((M for M in mats if M != _IDENTITY),
                                   key=lambda M: M.rows)
    return [ChangeOfBasis(M, _ZERO) for M in ordered]
