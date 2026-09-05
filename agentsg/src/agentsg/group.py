"""
Build a full space-group operation list from a small generator set, by
closure under composition -- this is the piece that replaces "extensive
tables" with "tabulated generators". Space-group order is bounded (point
group order <= 48, times centering multiplicity <= 4, so <= 192), so this
terminates fast and needs no floating point.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence
from .linalg import Matrix3, Vector3, IDENTITY3, frac_mod1
from .symmetry_op import SymmetryOp

_NEG_I = Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])


def close_group(
    generators: Sequence[SymmetryOp],
    centering_vectors: Sequence[Vector3] = (Vector3((0, 0, 0)),),
    max_order: int = 192,
) -> frozenset[SymmetryOp]:
    """Compute the closed space-group operation set by repeated composition.

    Combines the generator operations with lattice centring vectors until no new
    operations are produced. Raises :class:`RuntimeError` if the order exceeds
    ``max_order`` (default 192, the maximum order of any conventional 3D space group).
    """
    centering_ops = [SymmetryOp(IDENTITY3, v) for v in centering_vectors]
    seeds = list(generators) + centering_ops
    elements: set[SymmetryOp] = {SymmetryOp.identity()}

    changed = True
    while changed:
        changed = False
        for a in list(elements):
            for g in seeds:
                for prod in (a * g, g * a):
                    if prod not in elements:
                        elements.add(prod)
                        changed = True
        if len(elements) > max_order:
            raise RuntimeError(
                f"closure exceeded max_order={max_order}; "
                f"check generators/centering (possible parsing or convention error)"
            )
    return frozenset(elements)


def point_group(operations: Iterable[SymmetryOp]) -> frozenset:
    """The set of distinct rotation parts W, ignoring translations -- this
    *is* the crystallographic point group, no separate table needed."""
    return frozenset(op.W for op in operations)


def centering_translations(operations: Iterable[SymmetryOp]) -> frozenset[Vector3]:
    """The subset of operations with W = identity -- the Bravais centering."""
    return frozenset(op.w for op in operations if op.W == IDENTITY3)


def transform_hkl(hkl: Vector3, W: Matrix3) -> Vector3:
    """Miller indices as a row vector: h' = h @ W."""
    rows = W.rows
    h = hkl.v
    return Vector3(sum(h[i] * rows[i][j] for i in range(3)) for j in range(3))


def phase_shift(hkl: Vector3, op: SymmetryOp) -> Fraction:
    """Phase shift h·w in turns (multiply by 360 for degrees), reduced to [0, 1)."""
    return frac_mod1(hkl.dot(op.w))


def is_systematically_absent(hkl: Vector3, operations: Iterable[SymmetryOp]) -> bool:
    """A reflection h is absent iff some operation (W,w) fixes h in
    reciprocal space (h W = h) but h.w is not integral -- derived directly
    from the operator list, no reflection-condition table needed."""
    for op in operations:
        if transform_hkl(hkl, op.W) == hkl:
            if phase_shift(hkl, op).denominator != 1:
                return True
    return False


def is_centrosymmetric(operations: Iterable[SymmetryOp]) -> bool:
    """True if the point group contains inversion (−I)."""
    return _NEG_I in point_group(operations)


def is_reflection_centric(hkl: Vector3, operations: Iterable[SymmetryOp]) -> bool:
    """True if some operation maps h → −h (phase-restricted / 'centric' reflection)."""
    neg = Vector3((-hkl.v[0], -hkl.v[1], -hkl.v[2]))
    if hkl == neg:  # 000 (and only 000 for integral indices)
        return True
    for op in operations:
        if transform_hkl(hkl, op.W) == neg:
            return True
    return False


@dataclass(frozen=True)
class PhaseRestriction:
    """Phase restriction for one reflection (SgInfo / gemmi semantics).

    ``phase`` is in turns on [0, 1). For a centric reflection the allowed
    phases are ``phase`` and ``phase + 1/2``. ``phase is None`` means
    unrestricted. Systematically absent reflections set ``absent=True``
    (``phase`` is then undefined / None).
    """

    absent: bool
    centric: bool
    phase: Fraction | None


def phase_restriction(hkl: Vector3, operations: Iterable[SymmetryOp]) -> PhaseRestriction:
    """Derive absence / centricity / restricted phase for ``hkl``.

    If some (W, w) maps h → −h, then F(h) = exp(2πi h·w) F(h)*, so the
    restricted phase is (h·w)/2 turns (mod ½). Allowed phases are then
    ``phase`` and ``phase + 1/2``. The location of an inversion centre enters
    through w; there is no special centrosymmetric shortcut to phase 0.
    """
    ops = list(operations)
    neg = Vector3((-hkl.v[0], -hkl.v[1], -hkl.v[2]))
    thr: Fraction | None = None
    mismatch = False
    absent = False

    for op in ops:
        hp = transform_hkl(hkl, op.W)
        ph = phase_shift(hkl, op)
        if hp == hkl:
            if ph.denominator != 1:
                absent = True
        elif hp == neg:
            if thr is None:
                thr = ph
            elif thr != ph:
                mismatch = True

    if mismatch:
        absent = True

    centric = is_reflection_centric(hkl, ops)
    if absent:
        return PhaseRestriction(True, centric, None)

    if thr is not None:
        # F(h) = exp(2πi thr) F(h)*  ⇒  φ ≡ thr/2  (mod 1/2).
        return PhaseRestriction(False, True, frac_mod1(thr / 2))

    # 000 is centric with phase 0 even in P1.
    if centric:
        return PhaseRestriction(False, True, Fraction(0))

    return PhaseRestriction(False, False, None)
