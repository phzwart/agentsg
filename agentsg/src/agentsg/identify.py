"""
Space-group identification: closed operations -> IT number / Hall / HM.

Matches the closed operator set against the 230 standard settings in
``space_groups.py``. When the input differs from a standard setting by an
origin shift only, recovers that shift by solving

    (W - I) p = Δw

exactly over the rationals (no fixed translation denominator).

Arbitrary lattice reorientations (unique-axis swaps, etc.) are out of scope
here; use ``SpaceGroupSetting`` when the change of basis is already known.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Fr
from itertools import product
from typing import Iterable

from .linalg import Matrix3, Vector3, IDENTITY3, ZERO3
from .symmetry_op import SymmetryOp
from .group import close_group, point_group, centering_translations
from .change_of_basis import ChangeOfBasis
from .space_groups import SpaceGroup, space_group
from .rational_solve import solve_affine as _solve_affine, rref as _rref
from .semi_invariants import floating_origin_basis, pin_floating_origin


def _WmI(W: Matrix3) -> Matrix3:
    return Matrix3([
        [W.rows[i][j] - (1 if i == j else 0) for j in range(3)]
        for i in range(3)
    ])


def _group_by_W(ops: Iterable[SymmetryOp]) -> dict[Matrix3, list[Vector3]]:
    out: dict[Matrix3, list[Vector3]] = {}
    for op in ops:
        out.setdefault(op.W, []).append(op.w)
    return out


def _shift_translations(ws: list[Vector3], delta: Vector3) -> frozenset[Vector3]:
    return frozenset((w + delta).mod1() for w in ws)


def _delta_relating(ws_in: list[Vector3], ws_std: list[Vector3]) -> Vector3 | None:
    """Find δ (mod 1) with {w + δ} = {w_std}, or None."""
    if len(ws_in) != len(ws_std):
        return None
    target = frozenset(ws_std)
    for w_in in ws_in:
        for w_std in ws_std:
            delta = (w_std - w_in).mod1()
            if _shift_translations(ws_in, delta) == target:
                return delta
    return None


def _conjugate_by_origin(ops: frozenset[SymmetryOp], p: Vector3) -> frozenset[SymmetryOp]:
    cob = ChangeOfBasis(IDENTITY3, p)
    return frozenset(cob.apply_to_op(op) for op in ops)


def _origin_shift_to_standard(
    ops_in: frozenset[SymmetryOp],
    ops_std: frozenset[SymmetryOp],
) -> Vector3 | None:
    """Return p such that conjugating ops_in by (I, p) yields ops_std.

    Solves (W - I) p ≡ δ_W (mod 1) for each rotation W, exactly over Q.
    """
    if ops_in == ops_std:
        return ZERO3
    if len(ops_in) != len(ops_std):
        return None
    if point_group(ops_in) != point_group(ops_std):
        return None
    if centering_translations(ops_in) != centering_translations(ops_std):
        return None

    in_by_W = _group_by_W(ops_in)
    std_by_W = _group_by_W(ops_std)
    Ws = [W for W in in_by_W if W != IDENTITY3]
    if not Ws:
        # P1: every origin is floating — ops are always identical; gauge p = 0.
        return ZERO3 if ops_in == ops_std else None

    deltas: dict[Matrix3, Vector3] = {}
    for W in Ws:
        delta = _delta_relating(in_by_W[W], std_by_W[W])
        if delta is None:
            return None
        deltas[W] = delta

    def _accept(p: Vector3) -> Vector3 | None:
        """Pin floating components, then verify conjugation."""
        pinned = pin_floating_origin(p, ops_std)
        if _conjugate_by_origin(ops_in, pinned) == ops_std:
            return pinned
        # Unpinned may still work if pin moved off a solution coset — try raw.
        if pinned != p.mod1() and _conjugate_by_origin(ops_in, p.mod1()) == ops_std:
            return pin_floating_origin(p.mod1(), ops_std)
        return None

    # Stack (W - I) p = δ_W + n_W and try small integer n_W (mod-1 lift).
    # Free variables of (W−I) are exactly the floating-origin directions; we
    # prefer the gauge with those components zero (via pin_floating_origin).
    W0 = Ws[0]
    M0 = _WmI(W0)
    d0 = deltas[W0]
    candidates: list[Vector3] = []
    for n in product(range(-1, 2), repeat=3):
        rhs = Vector3(d0.v[i] + n[i] for i in range(3))
        sol = _solve_affine(M0, rhs)
        if sol is None:
            continue
        particular, basis = sol
        # Sample free vars at {0, 1/2}; floating dirs will be pinned to 0.
        free_dims = len(basis)
        if free_dims == 0:
            trials = [particular]
        else:
            trials = []
            for bits in product((Fr(0), Fr(1, 2)), repeat=free_dims):
                p = particular
                for b, coeff in zip(basis, bits):
                    p = p + Vector3(coeff * x for x in b.v)
                trials.append(p.mod1())
        for p in trials:
            accepted = _accept(p)
            if accepted is not None:
                return accepted
            candidates.append(p.mod1())

    # If first-W sampling missed (rare), try stacking two Ws when available.
    if len(Ws) >= 2:
        W1 = Ws[1]
        M1 = _WmI(W1)
        d1 = deltas[W1]
        for p0 in candidates[:32]:
            got = (M1 @ p0).mod1()
            if got == d1.mod1():
                accepted = _accept(p0)
                if accepted is not None:
                    return accepted
        for n0 in product(range(-1, 2), repeat=3):
            for n1 in product(range(-1, 2), repeat=3):
                A = (
                    [[M0.rows[i][j] for j in range(3)] for i in range(3)]
                    + [[M1.rows[i][j] for j in range(3)] for i in range(3)]
                )
                b = [d0.v[i] + n0[i] for i in range(3)] + [
                    d1.v[i] + n1[i] for i in range(3)
                ]
                R, c, pivots = _rref(A, b)
                inconsistent = False
                for i in range(len(R)):
                    if all(R[i][j] == 0 for j in range(3)) and c[i] != 0:
                        inconsistent = True
                        break
                if inconsistent:
                    continue
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
                free_dims = len(basis)
                bit_iters = (
                    [()] if free_dims == 0
                    else product((Fr(0), Fr(1, 2)), repeat=free_dims)
                )
                for bits in bit_iters:
                    p = particular
                    for bv, coeff in zip(basis, bits):
                        p = p + Vector3(coeff * x for x in bv.v)
                    accepted = _accept(p)
                    if accepted is not None:
                        return accepted
    return None


@dataclass(frozen=True)
class IdentifyResult:
    """Result of matching operations to a standard-setting space group.

    ``change_of_basis`` maps the input setting to the standard one. Its origin
    shift is in the *pinned* gauge: floating-origin components (unique axis of
    P2/P4/P3/…, or all of P1) are set to zero — those directions are arbitrary
    and reported separately in ``floating_origin``.
    """

    space_group: SpaceGroup
    change_of_basis: ChangeOfBasis  # maps input setting -> standard
    floating_origin: tuple[Vector3, ...] = ()

    @property
    def number(self) -> int:
        return self.space_group.number

    @property
    def hall(self) -> str:
        return self.space_group.hall

    @property
    def hermann_mauguin(self) -> str:
        return self.space_group.hermann_mauguin


_OPS_CACHE: dict[int, frozenset[SymmetryOp]] | None = None
_BY_OPS: dict[frozenset[SymmetryOp], SpaceGroup] | None = None
_BY_FINGERPRINT: dict[tuple[int, frozenset[Matrix3]], list[SpaceGroup]] | None = None


def _ensure_cache() -> None:
    global _OPS_CACHE, _BY_OPS, _BY_FINGERPRINT
    if _OPS_CACHE is not None:
        return
    _OPS_CACHE = {}
    _BY_OPS = {}
    _BY_FINGERPRINT = {}
    for n in range(1, 231):
        sg = space_group(n)
        ops = sg.operations()
        _OPS_CACHE[n] = ops
        _BY_OPS[ops] = sg
        fp = (len(ops), point_group(ops))
        _BY_FINGERPRINT.setdefault(fp, []).append(sg)


def identify_space_group(
    operations: Iterable[SymmetryOp],
) -> IdentifyResult | None:
    """Identify a closed (or generatable) operation set as one of the 230.

    Returns ``IdentifyResult`` with the matched ``SpaceGroup`` and a
    ``ChangeOfBasis`` taking the *input* setting to that standard setting.
    Only origin shifts are recovered; returns ``None`` if no match.
    """
    ops_list = list(operations)
    if not ops_list:
        return None
    ops = frozenset(ops_list)

    _ensure_cache()
    assert _BY_OPS is not None and _BY_FINGERPRINT is not None and _OPS_CACHE is not None

    def _result(sg: SpaceGroup, p: Vector3) -> IdentifyResult:
        ops_sg = _OPS_CACHE[sg.number]
        float_basis = floating_origin_basis(ops_sg)
        pinned = pin_floating_origin(p, ops_sg)
        return IdentifyResult(
            sg, ChangeOfBasis(IDENTITY3, pinned), floating_origin=float_basis
        )

    sg = _BY_OPS.get(ops)
    if sg is not None:
        return _result(sg, ZERO3)

    # Close only if the input was a generator list / incomplete set.
    centering = list(centering_translations(ops)) or [ZERO3]
    closed = close_group(list(ops), centering)
    if closed != ops:
        ops = closed
        sg = _BY_OPS.get(ops)
        if sg is not None:
            return _result(sg, ZERO3)

    fp = (len(ops), point_group(ops))
    for cand in _BY_FINGERPRINT.get(fp, []):
        std = _OPS_CACHE[cand.number]
        p = _origin_shift_to_standard(ops, std)
        if p is not None:
            return _result(cand, p)
    return None


def hall_from_ops(operations: Iterable[SymmetryOp]) -> str:
    """Return the tabulated Hall symbol for the identified standard setting.

    Raises ``ValueError`` if the operations do not match a standard setting
    (up to origin shift).
    """
    result = identify_space_group(operations)
    if result is None:
        raise ValueError(
            "operations do not match a standard-setting space group "
            "(up to origin shift)"
        )
    return result.hall
