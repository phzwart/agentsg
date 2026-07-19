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

For each class we scan a bounded reflection block, find the sublattice of
*allowed* reflections, and express it as the condition string that is satisfied
by exactly the non-absent reflections in that class.

Also provides symmetry-equivalent reflection orbits (SgInfo ``BuildEq_hkl``).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
from .linalg import Vector3, frac_mod1
from .symmetry_op import SymmetryOp
from .group import is_systematically_absent, transform_hkl, phase_shift


# reflection classes: name -> function(h,k,l)->bool selecting membership,
# and the free indices that vary within the class.
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


def _condition_for_class(members: list[tuple[int, int, int]],
                         operations: Sequence[SymmetryOp]) -> str | None:
    """Given all reflections in a class (excluding 000), return a human-readable
    condition on the *allowed* ones, or None if there is no restriction."""
    allowed = [hkl for hkl in members
               if not is_systematically_absent(Vector3(hkl), operations)]
    if len(allowed) == len(members):
        return None  # no condition: every reflection in the class is allowed
    if not allowed:
        return "all absent"
    # Find the common divisor pattern of the allowed reflections. For the usual
    # crystallographic conditions the allowed set is exactly those hkl with some
    # integer linear form == 0 (mod n). We detect the simplest such form by
    # testing candidate linear combinations of h,k,l against a modulus.
    forms = {
        "h": (1, 0, 0), "k": (0, 1, 0), "l": (0, 0, 1),
        "h+k": (1, 1, 0), "h+l": (1, 0, 1), "k+l": (0, 1, 1),
        "h+k+l": (1, 1, 1), "-h+k+l": (-1, 1, 1), "h-k+l": (1, -1, 1),
        "h+k-l": (1, 1, -1), "2h+k": (2, 1, 0), "h+2k": (1, 2, 0),
    }
    absent_flag = {hkl: is_systematically_absent(Vector3(hkl), operations)
                   for hkl in members}

    # (1) single form  ==  0 (mod n)
    best = None
    for name, (a, b, c) in forms.items():
        vals = {a * h + b * k + c * l for (h, k, l) in allowed}
        for n in (2, 3, 4):
            if all(v % n == 0 for v in vals):
                ok = all(((a * h + b * k + c * l) % n == 0) == (not absent_flag[(h, k, l)])
                         for (h, k, l) in members)
                if ok:
                    cond = f"{name} = {n}n"
                    if best is None or len(cond) < len(best):
                        best = cond
        if best:
            break
    if best:
        return best

    # (2) conjunction of forms all == 0 (mod n) -- centring-type conditions,
    #     e.g. F-lattice hkl: h+k, h+l, k+l = 2n.
    conj_forms = {
        "h+k": (1, 1, 0), "h+l": (1, 0, 1), "k+l": (0, 1, 1),
        "h+k+l": (1, 1, 1),
    }
    for n in (2, 3, 4):
        active = []
        for nm, (a, b, c) in conj_forms.items():
            if all((a * h + b * k + c * l) % n == 0 for (h, k, l) in allowed):
                active.append((nm, (a, b, c)))
        if not active:
            continue

        def _partition(forms_subset):
            return {hkl: all((a * hkl[0] + b * hkl[1] + c * hkl[2]) % n == 0
                             for _, (a, b, c) in forms_subset)
                    for hkl in members}

        target = {hkl: not absent_flag[hkl] for hkl in members}
        if _partition(active) != target:
            continue
        # greedy minimal cover: drop forms whose removal doesn't change the partition
        minimal = list(active)
        for f in list(minimal):
            trial = [g for g in minimal if g is not f]
            if trial and _partition(trial) == target:
                minimal = trial
        return ", ".join(nm for nm, _ in minimal) + f" = {n}n"

    return "restricted"


def reflection_conditions(operations: Sequence[SymmetryOp], rng: int = 6) -> dict[str, str]:
    """Derive the general reflection conditions for a space group.

    Returns a dict {class_name: condition_string} for every reflection class
    that carries a non-trivial condition (classes with no restriction are
    omitted). Purely derived from the operation list.
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
        cond = _condition_for_class(members, operations)
        if cond is not None:
            conditions[name] = cond
    return conditions


def _as_int_hkl(hkl: Vector3) -> tuple[int, int, int]:
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
