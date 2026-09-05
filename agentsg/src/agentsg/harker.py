"""
Harker sections / lines from space-group operators (exact rational algebra).

For a Seitz operator ``(W|w)``, self-Patterson vectors between an atom at ``x``
and its mate ``W x + w`` are

    u = x − (W x + w) = (I − W) x − w.

As ``x`` varies these fill the affine flat ``im(I−W) − w``. Equivalently, every
left-null vector ``n`` of ``(I−W)`` gives a constant linear constraint

    n · u ≡ −n · w   (mod 1),

which is the Harker section (plane), line, or (rarely) point. Pure translations
(``W = I``) and full-rank maps (e.g. inversion, which fills Patterson space)
produce no reduced locus and are skipped.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Fr
from math import gcd
from typing import Iterable, Sequence

from .linalg import Matrix3, Vector3, IDENTITY3, frac_mod1
from .symmetry_op import SymmetryOp
from .rational_solve import solve_affine as _solve_affine, rref as _rref


def _ImW(W: Matrix3) -> Matrix3:
    """Compute (I - W) as an exact Matrix3."""
    return Matrix3([
        [(1 if i == j else 0) - W.rows[i][j] for j in range(3)]
        for i in range(3)
    ])


def _normalize_constraint(n: Vector3, c: Fr) -> tuple[tuple[Fr, Fr, Fr], Fr]:
    """Primitive integer direction + constant in [0, 1). Flip sign for canonical."""
    vals = list(n.v)
    # Scale so components are integers with gcd 1
    from math import lcm
    D = 1
    for v in vals:
        D = lcm(D, v.denominator)
    ints = [int(v * D) for v in vals]
    g = abs(ints[0])
    for a in ints[1:]:
        g = gcd(g, abs(a))
    if g == 0:
        g = 1
    ints = [a // g for a in ints]
    # Canonical sign: first nonzero component > 0
    for a in ints:
        if a != 0:
            if a < 0:
                ints = [-x for x in ints]
                c = -c
            break
    nn = tuple(Fr(a) for a in ints)
    # Scale the constant by the same factor used to clear denominators and
    # divide out gcd.  With n_orig · u ≡ c (mod 1), we set
    #   n' = (D/g) n_orig  (primitive integer components),
    # so the congruent constant is c' ≡ (D/g) c (mod 1).
    scale = Fr(D, g)
    c2 = frac_mod1(c * scale)
    return nn, c2


@dataclass(frozen=True)
class HarkerConstraint:
    """Linear constraint ``n · u ≡ c (mod 1)`` on Patterson coordinates."""

    normal: Vector3
    constant: Fr

    def satisfied(self, u: Vector3 | Sequence, tol: Fr | None = None) -> bool:
        """True if Patterson vector ``u`` satisfies ``n · u ≡ c (mod 1)``."""
        if isinstance(u, Vector3):
            uv = u
        else:
            uv = Vector3(u)
        val = frac_mod1(self.normal.dot(uv) - self.constant)
        if tol is None:
            return val == 0
        # numeric fallback unused in exact path
        return min(val, 1 - val) <= tol

    def __str__(self) -> str:
        parts = []
        labels = ("u", "v", "w")
        for lab, a in zip(labels, self.normal.v):
            if a == 0:
                continue
            if a == 1:
                parts.append(lab)
            elif a == -1:
                parts.append(f"-{lab}")
            else:
                parts.append(f"{a}*{lab}")
        left = " + ".join(parts).replace("+ -", "- ")
        return f"{left} = {self.constant}"


@dataclass(frozen=True)
class HarkerLocus:
    """A distinct Harker plane / line / point (deduped across operators)."""

    constraints: tuple[HarkerConstraint, ...]
    operations: tuple[SymmetryOp, ...]
    rank: int  # rank(I−W); 2→plane, 1→line, 0→point (degenerate)

    @property
    def kind(self) -> str:
        """Geometric locus dimension category: 'section' (plane), 'line', or 'point'."""
        if self.rank == 2:
            return "section"
        if self.rank == 1:
            return "line"
        if self.rank == 0:
            return "point"
        return "unknown"

    def contains(self, u: Vector3 | Sequence) -> bool:
        """True if Patterson vector ``u`` satisfies all defining constraints of this locus."""
        return all(c.satisfied(u) for c in self.constraints)

    def __str__(self) -> str:
        if not self.constraints:
            return f"HarkerLocus({self.kind})"
        return "; ".join(str(c) for c in self.constraints)


def harker_vector(op: SymmetryOp, x: Vector3 | Sequence) -> Vector3:
    """Self-Patterson vector ``u = x − op(x)`` for site ``x``."""
    if not isinstance(x, Vector3):
        x = Vector3(x)
    return x - ((op.W @ x) + op.w)


def site_from_harker(
    op: SymmetryOp, u: Vector3 | Sequence,
) -> tuple[Vector3, tuple[Vector3, ...]] | None:
    """Solve ``(I−W) x = u + w`` for a site giving Harker vector ``u``.

    Returns ``(particular, nullspace_basis)`` or ``None`` if inconsistent.
    Free directions are the axis of the symmetry element (when present).
    """
    if not isinstance(u, Vector3):
        u = Vector3(u)
    ImW = _ImW(op.W)
    return _solve_affine(ImW, u + op.w)


def _left_nullspace(M: Matrix3) -> list[Vector3]:
    """Basis for ``{ n | nᵀ M = 0 }`` over Q."""
    sol = _solve_affine(M.transpose(), Vector3((0, 0, 0)))
    if sol is None:
        return []
    _part, basis = sol
    return list(basis)


def _matrix_rank(M: Matrix3) -> int:
    """Exact rank of a 3x3 matrix over the rationals."""
    A = [[M.rows[i][j] for j in range(3)] for i in range(3)]
    b = [Fr(0), Fr(0), Fr(0)]
    _R, _c, pivots = _rref(A, b)
    return len(pivots)


def _locus_key(constraints: Sequence[HarkerConstraint]) -> tuple:
    """Canonical hashable key for a set of Harker constraints."""
    return tuple(sorted(
        (c.normal.v, c.constant) for c in constraints
    ))


def harker_sections(
    operations: Iterable[SymmetryOp],
) -> list[HarkerLocus]:
    """Derive distinct Harker loci from a closed (or generating) op list.

    Identity and pure translations are ignored. Operators with
    ``rank(I−W) = 3`` (e.g. inversion) fill Patterson space and yield no
    reduced section. Results are deduplicated by constraint set; each locus
    lists the operators that generate it.
    """
    buckets: dict[tuple, list[tuple[tuple[HarkerConstraint, ...], int, SymmetryOp]]] = {}
    for op in operations:
        if op.W == IDENTITY3:
            continue
        ImW = _ImW(op.W)
        rank = _matrix_rank(ImW)
        if rank == 3:
            continue  # no reduced Harker locus
        null = _left_nullspace(ImW)
        cons = []
        for n in null:
            raw_c = -n.dot(op.w)
            nn, cc = _normalize_constraint(n, raw_c)
            cons.append(HarkerConstraint(Vector3(nn), cc))
        # Drop linearly dependent duplicates after normalisation
        cons = _unique_constraints(cons)
        key = _locus_key(cons)
        buckets.setdefault(key, []).append((tuple(cons), rank, op))

    loci: list[HarkerLocus] = []
    for items in buckets.values():
        cons = items[0][0]
        rank = items[0][1]
        ops = tuple(it[2] for it in items)
        loci.append(HarkerLocus(cons, ops, rank))

    # Stable order: planes first, then lines, then by string form
    kind_order = {"section": 0, "line": 1, "point": 2, "unknown": 3}
    loci.sort(key=lambda L: (kind_order.get(L.kind, 9), str(L)))
    return loci


def _unique_constraints(cons: Sequence[HarkerConstraint]) -> list[HarkerConstraint]:
    """Deduplicate Harker constraints preserving first appearance order."""
    seen = set()
    out = []
    for c in cons:
        key = (c.normal.v, c.constant)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
