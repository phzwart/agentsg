"""
Euclidean normalizer of a space group, derived from its operators.

The Euclidean normalizer ``N_E(G)`` is the set of isometries ``(M, m)`` that
conjugate the space group to itself:

    (M, m) (W, w) (M, m)^{-1}  ∈  G    for every (W, w) in G.

Only the translation part of that normalizer lived here before: the allowed
origin shifts ``T'`` from :func:`agentsg.semi_invariants.origin_lattice`. This
module adds the linear part. ``M`` runs through the metric holohedry ``H`` of
a supplied cell, or, when no cell is given, through the holohedry of the
group's own lattice type. An ``M`` is kept only when both of the following
hold.

**Point-group test.** ``M P M^{-1} = P`` as sets of rotation parts. Failure
means ``M`` changes ``G``. The recorded reason is ``point_group``.

**Translation test.** For every rotation ``W'`` that labels a block of the
stacked matrix ``A`` already factored by :func:`origin_lattice` as
``U A V = D``, let ``W`` be the preimage ``M^{-1} W' M`` and let ``w``,
``w'`` be the translations ``G`` pairs with ``W`` and ``W'``. The shift
``m`` must solve

    (W' − I) m  ≡  M w − w'    (mod L)

for every ``W'`` at once. In the primitive basis of
:func:`agentsg.semi_invariants._primitive_basis` the left-hand side is that
same matrix ``A``; the factorization is reused and not recomputed per ``M``.
With ``b`` the stacked right-hand sides, the system is solvable exactly when
``(U b)_i`` is an integer on every row whose invariant factor is 0, including
every row past the rank. A particular solution has ``y_i = (U b)_i / d_i``
on the nonzero factors, ``y_i = 0`` on floating directions, and ``m = V y``
pulled back to conventional coordinates and reduced modulo ``T'``.

Inversion (det −1) maps a chiral molecule to its mirror image. It is a
normalizer element only when the translation test accepts it; it is not a
re-indexing of the same enantiomer. The affine normalizer is not computed,
and nothing is read from a tabulated list of normalizers.

The gates are :data:`agentsg.tolerances.METRIC_LENGTH_TOL_PCT` and
:data:`agentsg.tolerances.METRIC_ANGLE_TOL_DEG`.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction as Fr
from pathlib import Path
from typing import Iterable, Sequence

from .group import point_group
from .lattice_symmetry import tolerance_metric_symmetry
from .linalg import Matrix3, Vector3
from .semi_invariants import OriginLattice, origin_lattice
from .space_groups import SpaceGroup, space_group
from .symmetry_op import SymmetryOp
from .tolerances import METRIC_ANGLE_TOL_DEG, METRIC_LENGTH_TOL_PCT


def _gate_quotes() -> tuple[tuple[str, str], ...]:
    """Sentences that follow the metric-gate assignments in ``tolerances.py``."""
    import agentsg.tolerances as tolerances

    tree = ast.parse(Path(tolerances.__file__).read_text(encoding="utf-8"))
    want = ("METRIC_LENGTH_TOL_PCT", "METRIC_ANGLE_TOL_DEG")
    found: dict[str, str] = {}
    body = tree.body
    for i, node in enumerate(body):
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            if name not in want or i + 1 >= len(body):
                continue
            nxt = body[i + 1]
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                found[name] = nxt.value.value
    missing = [name for name in want if name not in found]
    if missing:
        raise RuntimeError(f"tolerances.py quotes missing for {missing}")
    return tuple((name, found[name]) for name in want)


_GATE_QUOTES = _gate_quotes()


def _mat_key(M: Matrix3) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in M.rows)


def _key_matrix(key: tuple[tuple[int, ...], ...]) -> Matrix3:
    return Matrix3([[Fr(x) for x in row] for row in key])


def _in_lattice(v: Vector3, lat: OriginLattice) -> bool:
    """True when ``v`` lies in the translation lattice ``L``."""
    c = lat._Bt_inv @ v
    return all(x.denominator == 1 for x in c.v)


def _unit_interval(x: Fr) -> Fr:
    """Fractional part in ``[0, 1)``."""
    return x - Fr(x.numerator // x.denominator)


def _ops_and_system(sg):
    """Closed operators, crystal system, and a display record."""
    if isinstance(sg, SpaceGroup):
        return list(sg.operations()), sg.crystal_system, sg
    if hasattr(sg, "operations") and hasattr(sg, "base"):
        system = getattr(sg, "crystal_system", None) or sg.base.crystal_system
        return list(sg.operations()), system, sg
    rec = space_group(sg)
    return list(rec.operations()), rec.crystal_system, rec


def _holohedry_matrices(cell, length_tol_pct: float, angle_tol_deg: float) -> list[Matrix3]:
    ops = tolerance_metric_symmetry(
        cell, length_tol_pct=length_tol_pct, angle_tol_deg=angle_tol_deg,
    )
    seen: set[tuple[tuple[int, ...], ...]] = set()
    out: list[Matrix3] = []
    for op in ops:
        key = _mat_key(op.W)
        if key not in seen:
            seen.add(key)
            out.append(_key_matrix(key))
    return out


def _generic_cell(crystal_system: str) -> tuple[float, float, float, float, float, float]:
    """A cell whose metric symmetry is exactly that crystal system's holohedry.

    Edges and angles are far enough apart that the default length and angle
    gates do not admit an extra operator.
    """
    system = (crystal_system or "").lower()
    if system.startswith("triclinic"):
        return (12.0, 15.0, 18.0, 75.0, 100.0, 115.0)
    if system.startswith("monoclinic"):
        return (12.0, 15.0, 18.0, 90.0, 105.0, 90.0)
    if system.startswith("orthorhombic"):
        return (12.0, 15.0, 18.0, 90.0, 90.0, 90.0)
    if system.startswith("tetragonal"):
        return (12.0, 12.0, 18.0, 90.0, 90.0, 90.0)
    if system.startswith("trigonal") or system.startswith("hexagonal"):
        return (12.0, 12.0, 18.0, 90.0, 90.0, 120.0)
    if system.startswith("cubic"):
        return (12.0, 12.0, 12.0, 90.0, 90.0, 90.0)
    raise ValueError(f"unknown crystal system {crystal_system!r}")


def _preserves_lattice(M: Matrix3, lat: OriginLattice) -> bool:
    return all(_in_lattice(M @ c, lat) for c in lat.centering)


def _lattice_type_holohedry(
    crystal_system: str,
    lat: OriginLattice,
    length_tol_pct: float,
    angle_tol_deg: float,
) -> list[Matrix3]:
    """Holohedry of the Bravais lattice type, in the conventional basis.

    Built from a generic cell of ``crystal_system``, then restricted to the
    matrices that preserve the centring lattice. That restriction is what
    makes a rhombohedral lattice on hexagonal axes smaller than 6/mmm.
    """
    raw = _holohedry_matrices(_generic_cell(crystal_system), length_tol_pct, angle_tol_deg)
    return [M for M in raw if _preserves_lattice(M, lat)]


def _translation_of(ops: Iterable[SymmetryOp]) -> dict[Matrix3, Vector3]:
    """One translation per rotation. The others differ from it by ``L``."""
    out: dict[Matrix3, Vector3] = {}
    for op in sorted(ops, key=lambda item: (item.W.rows, item.w.v)):
        out.setdefault(op.W, op.w)
    return out


def _solve_translation(
    lat: OriginLattice,
    M: Matrix3,
    w_of: dict[Matrix3, Vector3],
) -> Vector3 | None:
    """Particular ``m`` modulo ``T'``, or ``None`` when no solution exists.

    Rows of ``A`` are labeled by ``lat._stack_W`` in that order. The
    right-hand side for each label ``W'`` uses the preimage rotation
    ``M^{-1} W' M``.
    """
    if not _preserves_lattice(M, lat):
        return None
    Minv = M.inverse()
    b: list[Fr] = []
    for Wp in lat._stack_W:
        W = Minv @ Wp @ M
        rhs = (M @ w_of[W]) - w_of[Wp]
        b.extend((lat._Bt_inv @ rhs).v)
    if len(b) != len(lat._U):
        raise RuntimeError("Smith factor U does not match the stacked (W − I) blocks")
    ub: list[Fr] = []
    for row in lat._U:
        acc = Fr(0)
        for u, x in zip(row, b):
            acc += u * x
        ub.append(acc)
    # ``_s`` is padded to length 3. Rows of ``U`` beyond that have factor 0.
    y: list[Fr] = [Fr(0), Fr(0), Fr(0)]
    n_factor = len(lat._s)
    for i, value in enumerate(ub):
        factor = lat._s[i] if i < n_factor else 0
        if factor == 0:
            if value.denominator != 1:
                return None
            continue
        y[i] = _unit_interval(value) / factor
    return lat.reduce(lat._from_y(y))


@dataclass(frozen=True)
class EuclideanNormalizer:
    """Euclidean normalizer ``N_E(G)`` inside the metric holohedry.

    ``linear_reps`` is every accepted ``(M, m, det)``. ``m`` is one
    particular solution, reduced modulo ``T'``. Inversion (det −1) maps a
    chiral molecule to its mirror image.

    ``rejected`` lists holohedry matrices that are not normalizer elements,
    each with reason ``point_group`` or ``translation``.

    ``metric_specialized`` is true when some accepted ``M`` lies outside the
    holohedry of the group's own lattice type, so the cell's gate is what
    admitted it.

    ``coset_reps`` multiplies those linear elements by the discrete allowed
    origins. That product is the finite part of ``N_E(G)/G`` once the
    continuous floating origin is set aside. ``index`` is the size of the
    product.
    """

    linear_reps: tuple[tuple[Matrix3, Vector3, int], ...]
    origin_lattice: OriginLattice
    rejected: tuple[tuple[Matrix3, str], ...]
    metric_specialized: bool
    length_tol_pct: float
    angle_tol_deg: float
    gate_quotes: tuple[tuple[str, str], ...] = _GATE_QUOTES

    def coset_reps(self) -> list[tuple[Matrix3, Vector3, int]]:
        """Finite part of ``N_E/G``: each linear element times each discrete origin."""
        origins = self.origin_lattice.discrete_origins()
        out: list[tuple[Matrix3, Vector3, int]] = []
        for M, m, det in self.linear_reps:
            for origin in origins:
                out.append((M, self.origin_lattice.reduce(m + origin), det))
        return out

    def proper_coset_reps(self) -> list[tuple[Matrix3, Vector3, int]]:
        """Coset representatives with ``det == +1``.

        The det −1 representatives are the ones that invert a chiral molecule.
        """
        return [rep for rep in self.coset_reps() if rep[2] == 1]

    def index(self) -> int:
        """``|N_E/G|``, finite part (floating directions excluded)."""
        return len(self.linear_reps) * self.origin_lattice.n_alternative_origins

    @property
    def gates(self) -> dict[str, object]:
        """Numeric gates actually used, and the sentences from ``tolerances.py``."""
        return {
            "length_tol_pct": self.length_tol_pct,
            "angle_tol_deg": self.angle_tol_deg,
            "quotes": {name: text for name, text in self.gate_quotes},
        }


def euclidean_normalizer(
    sg,
    cell: Sequence[float] | None = None,
    *,
    length_tol_pct: float = METRIC_LENGTH_TOL_PCT,
    angle_tol_deg: float = METRIC_ANGLE_TOL_DEG,
) -> EuclideanNormalizer:
    """Compute ``N_E(G)`` from the operators of ``sg``.

    ``sg`` is an IT number, Hermann–Mauguin or Hall symbol, or a
    :class:`~agentsg.space_groups.SpaceGroup`. ``cell`` is
    ``(a, b, c, alpha, beta, gamma)``. Without it, ``H`` is the holohedry of
    the group's lattice type. Inversion (det −1) maps a chiral molecule to
    its mirror image.
    """
    ops, system, _rec = _ops_and_system(sg)
    lat = origin_lattice(ops)
    type_H = _lattice_type_holohedry(system, lat, length_tol_pct, angle_tol_deg)
    if cell is None:
        holos = type_H
    else:
        if len(cell) != 6:
            raise ValueError("cell must be (a, b, c, alpha, beta, gamma)")
        holos = _holohedry_matrices(tuple(float(x) for x in cell), length_tol_pct, angle_tol_deg)
    type_keys = {_mat_key(M) for M in type_H}
    P = point_group(ops)
    w_of = _translation_of(ops)
    accepted: list[tuple[Matrix3, Vector3, int]] = []
    rejected: list[tuple[Matrix3, str]] = []
    specialized = False
    for M in sorted(holos, key=_mat_key):
        conjugated = {M @ W @ M.inverse() for W in P}
        if conjugated != P:
            rejected.append((M, "point_group"))
            continue
        m = _solve_translation(lat, M, w_of)
        if m is None:
            rejected.append((M, "translation"))
            continue
        det = int(M.det())
        accepted.append((M, m, det))
        if _mat_key(M) not in type_keys:
            specialized = True
    return EuclideanNormalizer(
        linear_reps=tuple(accepted),
        origin_lattice=lat,
        rejected=tuple(rejected),
        metric_specialized=specialized,
        length_tol_pct=float(length_tol_pct),
        angle_tol_deg=float(angle_tol_deg),
    )


def conjugates_into_group(
    ops: Sequence[SymmetryOp],
    M: Matrix3,
    m: Vector3,
    lat: OriginLattice | None = None,
) -> bool:
    """True when conjugation by ``(M, m)`` maps ``ops`` onto itself modulo ``L``.

    This is the direct operator-set test. The Smith solve is accepted only
    when it agrees with this check.
    """
    if lat is None:
        lat = origin_lattice(ops)
    Minv = M.inverse()
    by_W: dict[Matrix3, list[Vector3]] = {}
    for op in ops:
        by_W.setdefault(op.W, []).append(op.w)
    for op in ops:
        Wp = M @ op.W @ Minv
        w_new = (M @ op.w) + m - (Wp @ m)
        bucket = by_W.get(Wp)
        if not bucket:
            return False
        if not any(_in_lattice(w_new - w, lat) for w in bucket):
            return False
    return True
