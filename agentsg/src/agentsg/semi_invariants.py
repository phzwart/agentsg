"""
Structure semi-invariants derived from the operator list.

An origin shift o is *allowed* when conjugating the group by (I, o) leaves the
operator set unchanged, which is equivalent to

    (W - I) o  ∈  L   for every rotation W

with L the set of centring translations (exact rational vectors mod 1).
Structure semi-invariants are Miller indices h for which the structure-factor
phase is invariant under every allowed origin shift: h · o ∈ Z.

Continuous freedoms (nullspace of the stacked W−I) become modulus-0
constraints; discrete cyclic freedoms become modulus-m constraints. Both are
obtained with exact ``Fraction`` linear algebra — no fixed twelfths grid.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Fr
from itertools import product
from math import lcm
from typing import Iterable, Sequence

from .linalg import Matrix3, Vector3, IDENTITY3, ZERO3, frac_mod1
from .symmetry_op import SymmetryOp
from .group import point_group, centering_translations, is_systematically_absent
from .wyckoff import _rref


def _WmI(W: Matrix3) -> Matrix3:
    return Matrix3([
        [W.rows[i][j] - (1 if i == j else 0) for j in range(3)]
        for i in range(3)
    ])


def _matrix_order(W: Matrix3) -> int:
    """Smallest k≥1 with W^k = I (crystallographic: k ∈ {1,2,3,4,6})."""
    M = IDENTITY3
    for k in range(1, 7):
        M = W @ M
        if M == IDENTITY3:
            return k
    return 1


def _continuous_kernel(rotations: Sequence[Matrix3]) -> list[Vector3]:
    """Nullspace over Q of the stacked (W − I) matrices.

    This is the *floating-origin* subspace: origin shifts along these
    directions leave every Seitz operator unchanged (P1: all of R³; P2/P21:
    unique axis; P3/P4/P6: c axis; etc.).
    """
    mats = [_WmI(W) for W in rotations if W != IDENTITY3]
    if not mats:
        return [Vector3((1, 0, 0)), Vector3((0, 1, 0)), Vector3((0, 0, 1))]
    A: list[list[Fr]] = []
    b: list[Fr] = []
    for M in mats:
        for i in range(3):
            A.append([M.rows[i][j] for j in range(3)])
            b.append(Fr(0))
    R, c, pivots = _rref(A, b)
    pivot_set = set(pivots)
    free = [j for j in range(3) if j not in pivot_set]
    basis = []
    for f in free:
        vec = [Fr(0)] * 3
        vec[f] = Fr(1)
        for ri, col in enumerate(pivots):
            vec[col] = -R[ri][f]
        basis.append(Vector3(vec))
    return basis


def floating_origin_basis(operations: Iterable[SymmetryOp]) -> tuple[Vector3, ...]:
    """Basis for continuous (floating) origin freedom of ``operations``.

    Empty for groups with a unique origin (e.g. P222, P23). Length 1 for
    unique-axis polar groups (P2, P4, P3, P6, …). Length 3 for P1.
    """
    return tuple(_continuous_kernel(list(point_group(operations))))


def pin_floating_origin(p: Vector3, operations: Iterable[SymmetryOp]) -> Vector3:
    """Canonical gauge: set floating-origin components of ``p`` to zero.

    Crystallographic floating directions are axis-aligned (unique axis or
    all three for P1), so this zeros those coordinates. Non-floating
    (torsion) components are left unchanged.
    """
    coords = list(p.v)
    for v in floating_origin_basis(operations):
        nonzero = [i for i in range(3) if v.v[i] != 0]
        if len(nonzero) == 1 and abs(v.v[nonzero[0]]) == 1:
            coords[nonzero[0]] = Fr(0)
        else:
            # General direction: subtract (p·v / v·v) v over Q.
            vv = v.dot(v)
            if vv != 0:
                scale = p.dot(v) / vv
                coords = [coords[i] - scale * v.v[i] for i in range(3)]
    return Vector3(coords).mod1()


def _in_centering(v: Vector3, centering: Sequence[Vector3]) -> bool:
    vm = v.mod1()
    return any(vm == c for c in centering)


def is_allowed_origin(o: Vector3, operations: Iterable[SymmetryOp]) -> bool:
    """True if conjugating the group by origin o leaves the operator set unchanged."""
    ops = list(operations)
    centering = list(centering_translations(ops))
    for W in point_group(ops):
        if W == IDENTITY3:
            continue
        if not _in_centering(_WmI(W) @ o, centering):
            return False
    return True


def _torsion_denominator(operations: Sequence[SymmetryOp]) -> int:
    """Denominator bound for discrete allowed origins, from the group itself.

    Includes translation denominators *and* crystallographic rotation orders
    (so P3 finds the (1/3, 2/3, 0) Cheshire points; never a fixed STBF).
    """
    d = 1
    for op in operations:
        for x in op.w.v:
            d = lcm(d, x.denominator)
    for W in point_group(operations):
        if W == IDENTITY3:
            continue
        d = lcm(d, _matrix_order(W))
        for i in range(3):
            for j in range(3):
                a = W.rows[i][j] - (Fr(1) if i == j else Fr(0))
                d = lcm(d, a.denominator)
    return max(d, 1)


def _floating_axes(cont: Sequence[Vector3]) -> list[int]:
    """Coordinate axes spanned by continuous kernel basis (when axis-aligned)."""
    axes: set[int] = set()
    for v in cont:
        nonzero = [i for i in range(3) if v.v[i] != 0]
        if len(nonzero) == 1 and abs(v.v[nonzero[0]]) == 1:
            axes.add(nonzero[0])
        elif not nonzero:
            continue
        else:
            # Non-axis-aligned floating direction: treat all nonzero comps as free
            # for sampling (held at 0 when collecting torsion reps).
            axes.update(nonzero)
    return sorted(axes)


def _discrete_allowed_origins(operations: Sequence[SymmetryOp]) -> list[Vector3]:
    """Torsion representatives of allowed origins (floating axes held at 0).

    Floating-origin coordinates are pinned to 0 — they form a continuum and
    must not be discretised. Only the Cheshire / torsion part is sampled, at
    denominators native to the rotation orders and translations.
    """
    ops = list(operations)
    rotations = [W for W in point_group(ops) if W != IDENTITY3]
    if not rotations:
        # P1: every origin is floating — no discrete torsion.
        return [ZERO3]

    cont = _continuous_kernel(list(point_group(ops)))
    float_axes = set(_floating_axes(cont))
    torsion_axes = [i for i in range(3) if i not in float_axes]
    if not torsion_axes:
        # Pure floating (should not happen when rotations exist, but be safe).
        return [ZERO3]

    d = _torsion_denominator(ops)

    found: list[Vector3] = []
    seen: set[tuple] = set()
    ranges = [range(d) if i in torsion_axes else range(1) for i in range(3)]
    for coords in product(*ranges):
        o = Vector3(
            Fr(coords[i], d) if i in torsion_axes else Fr(0) for i in range(3)
        )
        if not is_allowed_origin(o, ops):
            continue
        # Pin floating components (already 0) and store.
        key = o.mod1().v
        if key in seen:
            continue
        seen.add(key)
        found.append(o.mod1())
    return found


# Trial semi-invariant bases — exact order of SgInfo TabTrial_si (sgsi.c).
# First match against the derived (non-absent) property table wins.
_TAB_TRIAL_SI: list[list[tuple[tuple[int, int, int], int]]] = [
    [],
    [((0, 2, -1), 4)],
    [((2, -1, 0), 4)],
    [((-1, 0, 2), 4)],
    [((2, 4, 3), 6)],
    [((4, 3, 2), 6)],
    [((3, 2, 4), 6)],
    [((1, 1, 1), 4)],
    [((1, 1, 1), 2)],
    [((1, 1, 1), 0)],
    [((0, 0, 1), 2)],
    [((0, 1, 0), 2)],
    [((1, 0, 0), 2)],
    [((0, 0, 1), 0)],
    [((0, 1, 0), 0)],
    [((1, 0, 0), 0)],
    [((1, -1, 0), 3), ((0, 0, 1), 0)],
    [((-1, 0, 1), 3), ((0, 1, 0), 0)],
    [((0, 1, -1), 3), ((1, 0, 0), 0)],
    [((0, 1, 1), 4), ((1, 0, 0), 0)],
    [((1, 0, 1), 4), ((0, 1, 0), 0)],
    [((1, 1, 0), 4), ((0, 0, 1), 0)],
    [((1, 0, 0), 2), ((0, 0, 1), 2)],
    [((0, 1, 0), 2), ((0, 0, 1), 2)],
    [((1, 0, 0), 2), ((0, 1, 0), 2)],
    [((1, 1, 0), 2), ((0, 0, 1), 2)],
    [((1, 0, 1), 2), ((0, 1, 0), 2)],
    [((0, 1, 1), 2), ((1, 0, 0), 2)],
    [((1, 0, 0), 2), ((0, 0, 1), 0)],
    [((0, 1, 0), 2), ((0, 0, 1), 0)],
    [((1, 0, 0), 2), ((0, 1, 0), 0)],
    [((1, 0, 0), 0), ((0, 0, 1), 2)],
    [((0, 1, 0), 0), ((0, 0, 1), 2)],
    [((1, 0, 0), 0), ((0, 1, 0), 2)],
    [((1, 1, 0), 2), ((0, 0, 1), 0)],
    [((1, 0, 1), 2), ((0, 1, 0), 0)],
    [((0, 1, 1), 2), ((1, 0, 0), 0)],
    [((1, 0, 0), 0), ((0, 0, 1), 0)],
    [((0, 1, 0), 0), ((0, 0, 1), 0)],
    [((1, 0, 0), 0), ((0, 1, 0), 0)],
    [((1, 0, 0), 2), ((0, 1, 0), 2), ((0, 0, 1), 2)],
    [((1, 0, 0), 0), ((0, 1, 0), 2), ((0, 0, 1), 2)],
    [((1, 0, 0), 2), ((0, 1, 0), 0), ((0, 0, 1), 2)],
    [((1, 0, 0), 2), ((0, 1, 0), 2), ((0, 0, 1), 0)],
    [((1, 0, 0), 2), ((0, 1, 0), 0), ((0, 0, 1), 0)],
    [((1, 0, 0), 0), ((0, 1, 0), 2), ((0, 0, 1), 0)],
    [((1, 0, 0), 0), ((0, 1, 0), 0), ((0, 0, 1), 2)],
    [((1, 0, 0), 0), ((0, 1, 0), 0), ((0, 0, 1), 0)],
    [((-1, 0, 0), 2), ((0, -1, 1), 4), ((0, 1, 1), 4)],
    [((-1, 0, 1), 4), ((0, -1, 0), 2), ((1, 0, 1), 4)],
    [((1, 1, 0), 4), ((1, -1, 0), 4), ((0, 0, -1), 2)],
    [((-1, 1, 1), 4), ((1, -1, 1), 4), ((1, 1, -1), 4)],
    [((0, 1, 1), 4), ((1, 0, 1), 4), ((1, 1, 0), 4)],
    [((-1, 0, 0), 0), ((0, -1, 1), 4), ((0, 1, 1), 4)],
    [((-1, 0, 1), 4), ((0, -1, 0), 0), ((1, 0, 1), 4)],
    [((1, 1, 0), 4), ((1, -1, 0), 4), ((0, 0, -1), 0)],
]


@dataclass(frozen=True)
class SemiInvariant:
    """One structure-semi-invariant vector and its modulus.

    ``modulus == 0`` means the linear form must vanish exactly (continuous
    origin freedom along that direction). Otherwise ``v · h ≡ 0 (mod m)``.
    """

    vector: tuple[int, int, int]
    modulus: int

    def accepts(self, h: int, k: int, l: int) -> bool:
        u = self.vector[0] * h + self.vector[1] * k + self.vector[2] * l
        if self.modulus == 0:
            return u == 0
        return u % self.modulus == 0


def _verify_si(
    h: int, k: int, l: int,
    cont: Sequence[Vector3],
    discrete: Sequence[Vector3],
) -> bool:
    hv = Vector3((h, k, l))
    for v in cont:
        # Continuous freedom: phase must be strictly invariant (h·v = 0).
        if hv.dot(v) != 0:
            return False
    for o in discrete:
        if hv.dot(o).denominator != 1:
            return False
    return True


def _matches_trial(
    trial: list[tuple[tuple[int, int, int], int]],
    h: int, k: int, l: int,
) -> bool:
    for vec, mod in trial:
        u = vec[0] * h + vec[1] * k + vec[2] * l
        if mod == 0:
            if u != 0:
                return False
        elif u % mod != 0:
            return False
    return True


def semi_invariants(operations: Iterable[SymmetryOp]) -> list[SemiInvariant]:
    """Derive structure semi-invariant vectors and moduli from ``operations``."""
    ops = list(operations)
    rotations = list(point_group(ops))
    cont = _continuous_kernel(rotations)
    discrete = _discrete_allowed_origins(ops)

    # Property table over a small hkl box. Like SgInfo, only non-absent
    # reflections participate in trial matching (centring absences can make
    # some parity constraints redundant in the reported basis).
    maxh = 7
    props: dict[tuple[int, int, int], bool | None] = {}
    for h, k, l in product(range(-maxh, maxh + 1), repeat=3):
        if is_systematically_absent(Vector3((h, k, l)), ops):
            props[(h, k, l)] = None
        else:
            props[(h, k, l)] = _verify_si(h, k, l, cont, discrete)

    for trial in _TAB_TRIAL_SI:
        ok = True
        for (h, k, l), is_si in props.items():
            if is_si is None:
                continue
            if _matches_trial(trial, h, k, l) != is_si:
                ok = False
                break
        if ok:
            return [SemiInvariant(vec, mod) for vec, mod in trial]

    # Fallback: express continuous constraints + parity from discrete samples.
    result: list[SemiInvariant] = []
    for v in cont:
        # Map continuous basis vector to an axis-aligned modulus-0 constraint
        # when it is a standard basis vector; otherwise keep the integer direction.
        comps = tuple(int(x) if x.denominator == 1 else 0 for x in v.v)
        if comps != (0, 0, 0) and all(x.denominator == 1 for x in v.v):
            result.append(SemiInvariant(comps, 0))  # type: ignore[arg-type]
        else:
            # Clear fractional direction to integer by clearing denominators.
            dens = [x.denominator for x in v.v]
            D = dens[0]
            for dd in dens[1:]:
                D = lcm(D, dd)
            ivec = tuple(int(x * D) for x in v.v)
            g = abs(ivec[0])
            for t in ivec[1:]:
                g = math_gcd(g, abs(t))
            if g == 0:
                continue
            ivec = tuple(t // g for t in ivec)
            result.append(SemiInvariant(ivec, 0))  # type: ignore[arg-type]

    # Discrete: for each non-zero discrete origin, add constraints h·o ∈ Z.
    for o in discrete:
        if o == ZERO3:
            continue
        dens = [x.denominator for x in o.v]
        D = 1
        for dd in dens:
            D = lcm(D, dd)
        ivec = tuple(int(x * D) for x in o.v)
        g = 0
        for t in ivec:
            g = math_gcd(g, abs(t))
        if g == 0:
            continue
        ivec = tuple(t // g for t in ivec)
        mod = D // g
        if mod > 1:
            result.append(SemiInvariant(ivec, mod))  # type: ignore[arg-type]
    return result


def math_gcd(a: int, b: int) -> int:
    from math import gcd
    return gcd(a, b)


def is_semi_invariant(hkl: Vector3 | tuple, operations: Iterable[SymmetryOp]) -> bool:
    """True if ``hkl`` is a structure semi-invariant for ``operations``."""
    if isinstance(hkl, Vector3):
        h, k, l = (int(hkl.v[0]), int(hkl.v[1]), int(hkl.v[2]))
        if any(x.denominator != 1 for x in hkl.v):
            raise ValueError("hkl must be integral")
    else:
        h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
    return all(si.accepts(h, k, l) for si in semi_invariants(operations))
