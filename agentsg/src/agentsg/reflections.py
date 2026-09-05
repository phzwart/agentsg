"""
Reflection conditions (systematic absences) derived from the operation list.

A reflection h is *systematically absent* iff some operation (W,w) leaves h
invariant in reciprocal space (h W = h) but gives a non-integral phase h.w.
This is the single rule in :func:`agentsg.group.is_systematically_absent`; here
we *enumerate and classify* the resulting conditions into the familiar
integral/zonal/serial families that appear in International Tables, all derived
-- never looked up.

Classes reported:
  * integral   (hkl): centring conditions, e.g. F -> h+k, h+l, k+l = 2n
  * zonal      (hk0, h0l, 0kl, and the hexagonal hh0l etc.): glide planes
  * serial     (h00, 0k0, 00l): screw axes

For each class we take operators that fix every generic member of the class
(hW = h for all such h). Each such operator contributes the modular condition
h·w ∈ ℤ, rewritten on the free indices of the class. The reported string is a
minimal subset of those constraints that reproduces the absence pattern —
derived only from the operator list, with no form tables or group-specific
branches.

Also provides symmetry-equivalent reflection orbits (SgInfo ``BuildEq_hkl``).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import Sequence
from .linalg import Vector3, frac_mod1
from .symmetry_op import SymmetryOp
from .group import is_systematically_absent, transform_hkl, phase_shift


# reflection classes: name -> function(h,k,l)->bool selecting membership.
# Classes nest (hkl contains 0kl contains 0k0, …); conditions are reported only
# for absences *intrinsic* to a class, not those inherited from a subclass.
_CLASSES = [
    ("hkl", lambda h, k, l: True),
    ("0kl", lambda h, k, l: h == 0),
    ("h0l", lambda h, k, l: k == 0),
    ("hk0", lambda h, k, l: l == 0),
    ("h00", lambda h, k, l: k == 0 and l == 0),
    ("0k0", lambda h, k, l: h == 0 and l == 0),
    ("00l", lambda h, k, l: h == 0 and k == 0),
    ("hhl", lambda h, k, l: h == k),
    ("hh0", lambda h, k, l: h == k and l == 0),
]
_CLASS_SEL = dict(_CLASSES)
# Proper subclasses used to strip inherited absences from parent reporting.
_SUBCLASSES: dict[str, tuple[str, ...]] = {
    "hkl": ("0kl", "h0l", "hk0", "hhl"),
    "0kl": ("0k0", "00l"),
    "h0l": ("h00", "00l"),
    "hk0": ("h00", "0k0", "hh0"),
    "hhl": ("hh0", "00l"),
    "hh0": (),
    "h00": (),
    "0k0": (),
    "00l": (),
}
# Substitute class equalities into a linear form (a,b,c)·(h,k,l).
# hhl/hh0: k=h is folded into the h coefficient so the printed form uses h,l only.
_CLASS_SUBST = {
    "hkl": lambda a, b, c: (a, b, c),
    "0kl": lambda a, b, c: (0, b, c),
    "h0l": lambda a, b, c: (a, 0, c),
    "hk0": lambda a, b, c: (a, b, 0),
    "h00": lambda a, b, c: (a, 0, 0),
    "0k0": lambda a, b, c: (0, b, 0),
    "00l": lambda a, b, c: (0, 0, c),
    "hhl": lambda a, b, c: (a, b, c),  # samples already satisfy h=k
    "hh0": lambda a, b, c: (a, b, 0),
}


def _point_group_orbit(
    hkl: tuple[int, int, int], operations: Sequence[SymmetryOp],
) -> set[tuple[int, int, int]]:
    """Distinct Miller indices produced by acting on hkl with the point-group rotations."""
    v = Vector3(hkl)
    return {
        (int(hp.v[0]), int(hp.v[1]), int(hp.v[2]))
        for op in operations
        for hp in (transform_hkl(v, op.W),)
    }


def _generic_members(
    members: list[tuple[int, int, int]],
    subclass_sels: list,
    operations: Sequence[SymmetryOp],
) -> list[tuple[int, int, int]]:
    """Members whose point-group orbit avoids every proper subclass.

    A zonal screw/glide absence on e.g. hhl contaminates its orbit mates; those
    mates are not intrinsic to the parent class and must not drive its report.
    """
    if not subclass_sels:
        return list(members)
    out = []
    for hkl in members:
        orbit = _point_group_orbit(hkl, operations)
        if any(sel(*mate) for mate in orbit for sel in subclass_sels):
            continue
        out.append(hkl)
    return out


def _reduce_constraint(coeffs: tuple[int, int, int], mod: int):
    """Canonical ((a,b,c), m) with positive leading coeff and content removed."""
    a, b, c = coeffs
    g = reduce(gcd, (a, b, c, mod))
    if g > 1:
        a, b, c, mod = a // g, b // g, c // g, mod // g
    if mod <= 1 or (a, b, c) == (0, 0, 0):
        return None
    if (a, b, c) < (0, 0, 0):
        a, b, c = -a, -b, -c
    return (a, b, c), mod


def _project_constraints(
    constraints: list[tuple[tuple[int, int, int], int]],
    class_name: str,
) -> list[tuple[tuple[int, int, int], int]]:
    """Restrict each constraint to the class's free indices, then merge.

    Same linear form with moduli m1, m2 becomes modulus lcm(m1, m2) — e.g.
    l=2n and l=3n → l=6n — from the definition of simultaneous congruences.
    """
    subst = _CLASS_SUBST[class_name]
    projected: list[tuple[tuple[int, int, int], int]] = []
    for (a, b, c), m in constraints:
        red = _reduce_constraint(subst(a, b, c), m)
        if red is not None:
            projected.append(red)
    # Merge identical forms by lcm of moduli.
    by_form: dict[tuple[int, int, int], int] = {}
    for coeffs, m in projected:
        by_form[coeffs] = lcm(by_form.get(coeffs, 1), m)
    out = []
    for coeffs, m in by_form.items():
        red = _reduce_constraint(coeffs, m)
        if red is not None:
            out.append(red)
    return sorted(out)


def _form_name(a: int, b: int, c: int) -> str:
    """Pretty-print ah+bk+cl from its integer coefficients."""
    parts: list[str] = []
    for coeff, sym in ((a, "h"), (b, "k"), (c, "l")):
        if coeff == 0:
            continue
        if not parts:
            if coeff == 1:
                parts.append(sym)
            elif coeff == -1:
                parts.append(f"-{sym}")
            else:
                parts.append(f"{coeff}{sym}")
        else:
            if coeff == 1:
                parts.append(f"+{sym}")
            elif coeff == -1:
                parts.append(f"-{sym}")
            elif coeff > 0:
                parts.append(f"+{coeff}{sym}")
            else:
                parts.append(f"{coeff}{sym}")
    return "".join(parts) or "0"


def _constraints_from_ops(
    operations: Sequence[SymmetryOp],
    generic: list[tuple[int, int, int]],
) -> list[tuple[tuple[int, int, int], int]]:
    """Modular constraints h·w ∈ ℤ from operators that fix every generic hkl.

    Returns unique ``((a,b,c), m)`` meaning ``a h + b k + c l ≡ 0 (mod m)``.
    """
    out: set[tuple[tuple[int, int, int], int]] = set()
    for op in operations:
        if not all(transform_hkl(Vector3(hkl), op.W) == Vector3(hkl) for hkl in generic):
            continue
        w = op.w.v
        dens = [wi.denominator for wi in w]
        L = reduce(lcm, dens, 1)
        coeffs = tuple(int(wi * L) for wi in w)
        g = reduce(gcd, coeffs + (L,))
        coeffs = tuple(c // g for c in coeffs)
        mod = L // g
        if mod <= 1:
            continue
        # Drop identically-zero conditions (always true).
        if all(c % mod == 0 for c in coeffs):
            continue
        # Canonicalise sign: leading nonzero coeff > 0.
        a, b, c = coeffs
        if (a, b, c) < (0, 0, 0):
            a, b, c = -a, -b, -c
        out.add(((a, b, c), mod))
    return sorted(out)


def _points_satisfying(
    work: list[tuple[int, int, int]],
    constraints: Sequence[tuple[tuple[int, int, int], int]],
) -> set[tuple[int, int, int]]:
    """Subset of reflections in work that satisfy all given congruences."""
    out = set()
    for hkl in work:
        h, k, l = hkl
        if all((a * h + b * k + c * l) % m == 0 for (a, b, c), m in constraints):
            out.add(hkl)
    return out


def _minimise_constraints(
    work: list[tuple[int, int, int]],
    constraints: list[tuple[tuple[int, int, int], int]],
) -> list[tuple[tuple[int, int, int], int]]:
    """Drop redundant constraints (same satisfying set on ``work``)."""
    if not constraints:
        return []
    target = _points_satisfying(work, constraints)
    minimal = list(constraints)
    for c in sorted(
        constraints,
        key=lambda t: (-abs(t[0][0]) - abs(t[0][1]) - abs(t[0][2]), -t[1]),
    ):
        trial = [g for g in minimal if g is not c]
        if trial and _points_satisfying(work, trial) == target:
            minimal = trial
    return minimal


def _format_constraints(constraints: list[tuple[tuple[int, int, int], int]]) -> str:
    """Format integer linear congruences into an ITA condition string."""
    if not constraints:
        return ""
    by_mod: dict[int, list[tuple[int, int, int]]] = {}
    for coeffs, m in constraints:
        by_mod.setdefault(m, []).append(coeffs)
    parts = []
    for m in sorted(by_mod):
        names = [_form_name(*coeffs) for coeffs in by_mod[m]]
        parts.append(", ".join(names) + f" = {m}n")
    return "; ".join(parts)


def _condition_for_class(
    operations: Sequence[SymmetryOp],
    generic: list[tuple[int, int, int]],
    class_name: str,
) -> str | None:
    """Return the class-wide (necessary) condition, or None.

    Operators that fix every generic member contribute h·w ∈ ℤ. That condition
    is necessary for presence but need not be sufficient: individual reflections
    in the class may still be absent under operators that fix only them (those
    absences belong to more special classes / positions). We therefore report
    the operator-derived necessary constraints, not a partition of all absences.
    """
    if not generic:
        return None

    absent_flag = {
        hkl: is_systematically_absent(Vector3(hkl), operations) for hkl in generic
    }
    allowed = [hkl for hkl in generic if not absent_flag[hkl]]
    if not allowed:
        return "all absent"

    raw = _constraints_from_ops(operations, generic)
    constraints = _project_constraints(raw, class_name)
    if not constraints:
        return None

    # Necessary: every allowed reflection satisfies the constraints.
    if any(
        any((a * h + b * k + c * l) % m != 0 for (a, b, c), m in constraints)
        for (h, k, l) in allowed
    ):
        constraints = raw  # projection lost content; use unprojected
        if any(
            any((a * h + b * k + c * l) % m != 0 for (a, b, c), m in constraints)
            for (h, k, l) in allowed
        ):
            return "restricted"

    minimal = _minimise_constraints(generic, constraints)
    return _format_constraints(minimal) if minimal else None


def reflection_conditions(operations: Sequence[SymmetryOp],
                          ita_classes: bool = True) -> dict[str, str]:
    """ITA-style general reflection conditions, derived as sublattices.

    Delegates to :func:`agentsg.reflection_lattice.reflection_conditions`:
    reflection classes are the stabiliser strata of reciprocal space and the
    condition on each is the dual of the translation lattice generated by the
    stabiliser's operations -- exact, with no hkl enumeration. See that module
    for the derivation. The sampling implementation is kept as
    :func:`reflection_conditions_grid` for comparison.
    """
    from .reflection_lattice import reflection_conditions as _lattice
    return _lattice(operations, ita_classes=ita_classes)


def reflection_conditions_grid(operations: Sequence[SymmetryOp], rng: int = 6) -> dict[str, str]:
    """Reflection conditions by enumerating an hkl block (legacy).

    Returns a dict {class_name: condition_string} for every class in the fixed
    list ``_CLASSES`` that carries a non-trivial condition, found by testing
    every hkl in ``[-rng, rng]^3`` with :func:`is_systematically_absent`.
    Superseded by the sublattice derivation in :mod:`agentsg.reflection_lattice`.
    """
    conditions: dict[str, str] = {}
    # generate members per class within a bounded block
    block = [(h, k, l)
             for h in range(-rng, rng + 1)
             for k in range(-rng, rng + 1)
             for l in range(-rng, rng + 1)
             if not (h == 0 and k == 0 and l == 0)]
    for name, sel in _CLASSES:
        members = [hkl for hkl in block if sel(*hkl)]
        if not members:
            continue
        sub_sels = [_CLASS_SEL[s] for s in _SUBCLASSES[name]]
        generic = _generic_members(members, sub_sels, operations)
        cond = _condition_for_class(operations, generic, name)
        if cond is not None:
            conditions[name] = cond
    return conditions


def _as_int_hkl(hkl: Vector3) -> tuple[int, int, int]:
    """Coerce Vector3 to an integer triple (h, k, l)."""
    if any(x.denominator != 1 for x in hkl.v):
        raise ValueError("hkl must be integral")
    return int(hkl.v[0]), int(hkl.v[1]), int(hkl.v[2])


def epsilon_factor(hkl: Vector3, operations: Sequence[SymmetryOp]) -> int:
    """Number of symmetry operations that leave ``hkl`` invariant (hW = h).

    Matches gemmi's ``GroupOps.epsilon_factor``. Related to multiplicity by
    ``|G| = multiplicity × epsilon``.
    """
    return sum(1 for op in operations if transform_hkl(hkl, op.W) == hkl)


def reflection_multiplicity(hkl: Vector3, operations: Sequence[SymmetryOp]) -> int:
    """Geometric multiplicity: number of distinct symmetry-equivalent ``hkl``.

    Equals ``|{ h W : (W,w) ∈ G }|`` and, by orbit–stabiliser,
    ``|G| / epsilon_factor(hkl, G)``. For a general reflection this is the
    point-group order (centring cancels in the ratio).
    """
    ops = list(operations)
    if not ops:
        return 0
    eps = epsilon_factor(hkl, ops)
    if eps == 0:
        raise RuntimeError("epsilon_factor vanished — inconsistent operator list")
    return len(ops) // eps


def laue_multiplicity(hkl: Vector3, operations: Sequence[SymmetryOp]) -> int:
    """Multiplicity under the Laue group (point group closed by inversion).

    Counts Friedel mates as equivalent for intensity purposes. For centrosymmetric
    groups this equals :func:`reflection_multiplicity`; for acentric groups it is
    larger when ``−h`` is not already in the crystal point-group orbit.
    """
    h0 = _as_int_hkl(hkl)
    if h0 == (0, 0, 0):
        return 1
    seen: set[tuple[int, int, int]] = set()
    for op in operations:
        ht = _as_int_hkl(transform_hkl(hkl, op.W))
        seen.add(ht)
        seen.add((-ht[0], -ht[1], -ht[2]))
    return len(seen)


@dataclass(frozen=True)
class EquivalentHKL:
    """Symmetry-equivalent Miller indices for one reflection.

    ``hkls`` lists one representative from each Friedel pair ``{h, −h}``
    (SgInfo-style). ``multiplicity`` is the geometric orbit size
    ``|G|/ε``; ``laue_multiplicity`` counts the Laue-expanded set (always
    ``2·N`` for nonzero reflections in this listing). ``epsilon`` is the
    stabiliser order.
    """

    hkls: tuple[tuple[int, int, int], ...]
    phases: tuple[Fraction, ...]
    multiplicity: int
    epsilon: int
    laue_multiplicity: int

    @property
    def N(self) -> int:
        """Number of symmetry mates in the orbit."""
        return len(self.hkls)


def equivalent_reflections(
    hkl: Vector3,
    operations: Sequence[SymmetryOp],
) -> EquivalentHKL:
    """Build the Friedel-unique orbit of ``hkl`` under the space group."""
    ops = list(operations)
    h0 = _as_int_hkl(hkl)
    eps = epsilon_factor(hkl, ops)
    mult = reflection_multiplicity(hkl, ops)
    if h0 == (0, 0, 0):
        return EquivalentHKL(
            ((0, 0, 0),), (Fraction(0),),
            multiplicity=1, epsilon=eps, laue_multiplicity=1,
        )

    # One representative op per rotation (phase from that op).
    by_W: dict = {}
    for op in ops:
        by_W.setdefault(op.W, op)

    seen: set[tuple[int, int, int]] = set()
    hkls: list[tuple[int, int, int]] = []
    phases: list[Fraction] = []

    for W, op in by_W.items():
        hp = transform_hkl(hkl, W)
        ht = _as_int_hkl(hp)
        mh = (-ht[0], -ht[1], -ht[2])
        if ht in seen or mh in seen:
            continue
        seen.add(ht)
        hkls.append(ht)
        phases.append(phase_shift(hkl, op))

    # Re-base phases so the input (or its first stored mate) is 0.
    if h0 in hkls:
        i0 = hkls.index(h0)
    else:
        i0 = 0
    base = phases[i0]
    phases = [frac_mod1(p - base) for p in phases]
    # Rotate lists so input hkl is first when present.
    if i0 != 0 and h0 in hkls:
        hkls = [hkls[i0]] + hkls[:i0] + hkls[i0 + 1 :]
        phases = [phases[i0]] + phases[:i0] + phases[i0 + 1 :]

    n = len(hkls)
    return EquivalentHKL(
        tuple(hkls), tuple(phases),
        multiplicity=mult, epsilon=eps, laue_multiplicity=2 * n,
    )


def are_equivalent_reflections(
    h1: Vector3,
    h2: Vector3,
    operations: Sequence[SymmetryOp],
) -> bool:
    """True if ``h2`` is in the symmetry orbit of ``h1`` (including Friedel)."""
    t2 = _as_int_hkl(h2)
    mt2 = (-t2[0], -t2[1], -t2[2])
    eq = equivalent_reflections(h1, operations)
    return t2 in eq.hkls or mt2 in eq.hkls
