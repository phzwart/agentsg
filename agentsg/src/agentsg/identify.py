"""
Space-group identification: closed operations -> IT number / Hall / HM.

Matches the closed operator set against the 230 standard settings in
``space_groups.py`` (Hall-derived generators, closed by composition).

When the input differs from a standard setting by an origin shift only,
recovers that shift by solving ``(W - I) p = Δw`` exactly over the rationals.

When the input is a non-reference crystallographic setting (axis permutation,
diagonal 2-folds written in a tetragonal P cell, etc.), recovers an integer
change of basis ``P`` (entries in ``{-1,0,1}``, ``|det P| ≤ 4``) that conjugates
the point group onto a Hall reference, expands by the centring cosets implied
by ``P`` when ``|det P| ≠ 1``, then solves for the residual origin shift.
The returned ``ChangeOfBasis`` maps the *input* setting to that Hall/ITA
reference (so its inverse is the parenthetical of an extended HM/Hall symbol).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Fr
from itertools import product
from math import lcm
from typing import Iterable

from .linalg import Matrix3, Vector3, IDENTITY3, ZERO3
from .symmetry_op import SymmetryOp
from .group import close_group, point_group, centering_translations
from .change_of_basis import ChangeOfBasis
from .space_groups import SpaceGroup, space_group
from .rational_solve import solve_affine as _solve_affine, rref as _rref
from .semi_invariants import floating_origin_basis, pin_floating_origin
from .setting import _is_crystallographic_W, _lattice_coset_ops


def _WmI(W: Matrix3) -> Matrix3:
    """Compute (W - I) as an exact Matrix3."""
    return Matrix3([
        [W.rows[i][j] - (1 if i == j else 0) for j in range(3)]
        for i in range(3)
    ])


def _group_by_W(ops: Iterable[SymmetryOp]) -> dict[Matrix3, list[Vector3]]:
    """Group translation vectors by their rotation matrix W."""
    out: dict[Matrix3, list[Vector3]] = {}
    for op in ops:
        out.setdefault(op.W, []).append(op.w)
    return out


def _shift_translations(ws: list[Vector3], delta: Vector3) -> frozenset[Vector3]:
    """Shift a collection of translation vectors by delta (mod 1)."""
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
    """Conjugate an operation set by the origin shift (I, p)."""
    cob = ChangeOfBasis(IDENTITY3, p)
    return frozenset(cob.apply_to_op(op) for op in ops)


def _origin_shift_to_standard(
    ops_in: frozenset[SymmetryOp],
    ops_std: frozenset[SymmetryOp],
) -> Vector3 | None:
    """Return p such that conjugating ops_in by (I, p) yields ops_std.

    Solves (W - I) p ≡ δ_W (mod 1) for each rotation W, exactly over Q.
    Free variables are sampled on the grid of denominator D, where D is the
    lcm of all translation denominators appearing in the two operator sets
    (and the δ_W that relate them) — the crystallographic translation lattice,
    not a hard-coded {0, ½} stencil.
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
    # Deterministic order: frozenset iteration order must not affect results.
    Ws = sorted((W for W in in_by_W if W != IDENTITY3), key=lambda W: W.rows)
    if not Ws:
        # P1: every origin is floating — ops are always identical; gauge p = 0.
        return ZERO3 if ops_in == ops_std else None

    deltas: dict[Matrix3, Vector3] = {}
    for W in Ws:
        delta = _delta_relating(in_by_W[W], std_by_W[W])
        if delta is None:
            return None
        deltas[W] = delta

    # Centring makes δ unique only modulo the centering lattice: {w}+δ = {w_std}
    # still holds for δ + c. The equation (W−I)p ≡ δ needs the representative
    # consistent with a single p, so we try the full coset.
    cens = list(centering_translations(ops_std))
    if not cens:
        cens = [ZERO3]

    # Translation-lattice denominator from the operators themselves.
    dens = [1]
    for ops in (ops_in, ops_std):
        for op in ops:
            for wi in op.w.v:
                dens.append(wi.denominator)
    for delta in deltas.values():
        for wi in delta.v:
            dens.append(wi.denominator)
    D = 1
    for d in dens:
        D = lcm(D, d)

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
    for cen in cens:
        delta0 = (d0 + cen).mod1()
        for n in product(range(-1, 2), repeat=3):
            rhs = Vector3(delta0.v[i] + n[i] for i in range(3))
            sol = _solve_affine(M0, rhs)
            if sol is None:
                continue
            particular, basis = sol
            # Denominators in the affine solution can refine the free-var grid.
            dens_sol = [D]
            for vec in (particular, *basis):
                for x in vec.v:
                    dens_sol.append(x.denominator)
            D_sol = 1
            for d in dens_sol:
                D_sol = lcm(D_sol, d)
            grid = tuple(Fr(k, D_sol) for k in range(D_sol))
            free_dims = len(basis)
            if free_dims == 0:
                trials = [particular]
            else:
                trials = []
                for bits in product(grid, repeat=free_dims):
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
        for p0 in candidates[:64]:
            got = (M1 @ p0).mod1()
            # Accept if got is d1 up to centring.
            if any(got == (d1 + c).mod1() for c in cens):
                accepted = _accept(p0)
                if accepted is not None:
                    return accepted
        for cen0 in cens:
            for cen1 in cens:
                delta0 = (d0 + cen0).mod1()
                delta1 = (d1 + cen1).mod1()
                for n0 in product(range(-1, 2), repeat=3):
                    for n1 in product(range(-1, 2), repeat=3):
                        A = (
                            [[M0.rows[i][j] for j in range(3)] for i in range(3)]
                            + [[M1.rows[i][j] for j in range(3)] for i in range(3)]
                        )
                        b = [delta0.v[i] + n0[i] for i in range(3)] + [
                            delta1.v[i] + n1[i] for i in range(3)
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
                        dens_sol = [D]
                        for vec in (particular, *basis):
                            for x in vec.v:
                                dens_sol.append(x.denominator)
                        D_sol = 1
                        for d in dens_sol:
                            D_sol = lcm(D_sol, d)
                        grid = tuple(Fr(k, D_sol) for k in range(D_sol))
                        free_dims = len(basis)
                        bit_iters = (
                            [()] if free_dims == 0
                            else product(grid, repeat=free_dims)
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
        """Space-group number (1-230)."""
        return self.space_group.number

    @property
    def hall(self) -> str:
        """Hall symbol of the matched space group."""
        return self.space_group.hall

    @property
    def hermann_mauguin(self) -> str:
        """Hermann-Mauguin symbol of the matched space group."""
        return self.space_group.hermann_mauguin


_OPS_CACHE: dict[int, frozenset[SymmetryOp]] | None = None
_BY_OPS: dict[frozenset[SymmetryOp], SpaceGroup] | None = None
_BY_FINGERPRINT: dict[tuple[int, frozenset[Matrix3]], list[SpaceGroup]] | None = None
_BY_PG: dict[frozenset[Matrix3], list[SpaceGroup]] | None = None
_COB_MATRICES: list[Matrix3] | None = None


def _ensure_cache() -> None:
    """Precompute standard ops, fingerprints, and integer CoB matrices."""
    global _OPS_CACHE, _BY_OPS, _BY_FINGERPRINT, _BY_PG, _COB_MATRICES
    if _OPS_CACHE is not None:
        return
    _OPS_CACHE = {}
    _BY_OPS = {}
    _BY_FINGERPRINT = {}
    _BY_PG = {}
    for n in range(1, 231):
        sg = space_group(n)
        ops = sg.operations()
        _OPS_CACHE[n] = ops
        _BY_OPS[ops] = sg
        pg = point_group(ops)
        _BY_FINGERPRINT.setdefault((len(ops), pg), []).append(sg)
        _BY_PG.setdefault(pg, []).append(sg)
    # Integer P with entries in {-1,0,1} and |det|≤4: enough for ITA axis
    # permutations and common P↔C / P↔I / P↔F redescriptions. Sorted by
    # |det| so unimodular settings win over centred rewrites when both work.
    mats: list[tuple[int, Matrix3]] = []
    for rows in product(product((-1, 0, 1), repeat=3), repeat=3):
        P = Matrix3(rows)
        d = abs(P.det())
        if d in (1, 2, 3, 4):
            mats.append((int(d), P))
    mats.sort(key=lambda t: t[0])
    _COB_MATRICES = [P for _, P in mats]


def _result_for(sg: SpaceGroup, P: Matrix3, p_in_new: Vector3) -> IdentifyResult:
    """IdentifyResult; ``p_in_new`` is the origin shift in the *reference* frame.

    ``ChangeOfBasis`` stores the origin of the new (reference) setting in *old*
    (input) coordinates, so with linear part ``P`` one has
    ``p_old = P @ p_in_new`` (composition of ``(P,0)`` then ``(I, p_in_new)``).
    Floating-origin components are pinned in the reference frame before that.
    """
    assert _OPS_CACHE is not None
    ops_sg = _OPS_CACHE[sg.number]
    float_basis = floating_origin_basis(ops_sg)
    pinned_new = pin_floating_origin(p_in_new, ops_sg)
    p_old = P @ pinned_new
    return IdentifyResult(
        sg, ChangeOfBasis(P, p_old), floating_origin=float_basis
    )


def _identify_by_origin(
    ops: frozenset[SymmetryOp],
) -> IdentifyResult | None:
    """Match ``ops`` to a Hall reference up to origin shift only."""
    assert _BY_OPS is not None and _BY_FINGERPRINT is not None and _OPS_CACHE is not None
    sg = _BY_OPS.get(ops)
    if sg is not None:
        return _result_for(sg, IDENTITY3, ZERO3)
    fp = (len(ops), point_group(ops))
    for cand in _BY_FINGERPRINT.get(fp, []):
        std = _OPS_CACHE[cand.number]
        p = _origin_shift_to_standard(ops, std)
        if p is not None:
            return _result_for(cand, IDENTITY3, p)
    return None


def _identify_by_cob(
    ops: frozenset[SymmetryOp],
) -> IdentifyResult | None:
    """Match via integer CoB + centring expansion + origin shift.

    For each candidate ``P``, conjugate the input point group. If the image is
    a tabulated Hall point group, transform the operators, expand by the
    lattice cosets of ``P`` (symmetry expansion when ``|det P| ≠ 1``), and
    solve for an origin that lands on that Hall operator set.
    """
    assert _BY_PG is not None and _COB_MATRICES is not None and _OPS_CACHE is not None
    pg = point_group(ops)
    best: IdentifyResult | None = None
    best_key: tuple | None = None
    for P in _COB_MATRICES:
        try:
            Pinv = P.inverse()
        except Exception:
            continue
        mapped_pg = frozenset(Pinv @ (W @ P) for W in pg)
        cands = _BY_PG.get(mapped_pg)
        if not cands:
            continue
        cob0 = ChangeOfBasis(P, ZERO3)
        transformed: list[SymmetryOp] = []
        ok = True
        for op in ops:
            top = cob0.apply_to_op(op)
            if not _is_crystallographic_W(top.W):
                ok = False
                break
            transformed.append(top)
        if not ok:
            continue
        det = int(abs(P.det()))
        seeds = transformed + _lattice_coset_ops(cob0)
        try:
            expanded = close_group(seeds, max_order=max(192, 192 * det))
        except RuntimeError:
            continue
        for cand in cands:
            std = _OPS_CACHE[cand.number]
            if len(expanded) != len(std):
                continue
            p = _origin_shift_to_standard(expanded, std)
            if p is None:
                continue
            hit = _result_for(cand, P, p)
            key = (det, cand.number)
            if best_key is None or key < best_key:
                best = hit
                best_key = key
        # Matrices are sorted by |det|; once we have a hit at this |det|,
        # no larger |det| can improve the primary sort key.
        if best is not None and best_key is not None and best_key[0] == det:
            return best
    return best


def identify_space_group(
    operations: Iterable[SymmetryOp],
) -> IdentifyResult | None:
    """Identify a closed (or generatable) operation set as one of the 230.

    Returns ``IdentifyResult`` with the matched ``SpaceGroup`` (Hall reference)
    and a ``ChangeOfBasis`` taking the *input* setting to that reference.
    Recovers origin shifts and integer axis/centring redescriptions
    (``|det P| ≤ 4``, entries in ``{-1,0,1}``). Returns ``None`` if no match.
    """
    ops_list = list(operations)
    if not ops_list:
        return None
    ops = frozenset(ops_list)

    _ensure_cache()

    # Close only if the input was a generator list / incomplete set.
    centering = list(centering_translations(ops)) or [ZERO3]
    closed = close_group(list(ops), centering)
    ops = closed

    hit = _identify_by_origin(ops)
    if hit is not None:
        return hit
    return _identify_by_cob(ops)


def hall_from_ops(operations: Iterable[SymmetryOp]) -> str:
    """Return the tabulated Hall symbol for the identified reference setting.

    Raises ``ValueError`` if the operations cannot be identified as one of the
    230 (up to integer CoB with ``|det|≤4`` and origin shift).
    """
    result = identify_space_group(operations)
    if result is None:
        raise ValueError(
            "operations do not match a space group up to crystallographic "
            "change of basis to a Hall reference setting"
        )
    return result.hall
