"""
Asymmetric units in real and reciprocal space.

* ``ReciprocalAsu`` — CCP4 / cctbx / gemmi reflection ASU (condition strings).
* ``DirectAsuBrick`` — conventional axis-aligned real-space ASU brick.
* ``DirichletAsu`` — metric Voronoi / Dirichlet fundamental domain, with a
  sphericity / inertia-ellipsoid optimiser over allowed origin gauges.

Brick and reciprocal tables live in ``asu_data`` (gemmi-verified literals).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from math import acos, cos, pi, sqrt
from typing import Sequence

from .linalg import Vector3, ZERO3, IDENTITY3, frac_mod1, Matrix3
from .symmetry_op import SymmetryOp
from .group import transform_hkl, point_group
from .space_groups import SpaceGroup, space_group
from .semi_invariants import (
    floating_origin_basis,
    discrete_allowed_origins,
    is_allowed_origin,
)
from . import asu_data

_NEG_I = Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])


# ---------------------------------------------------------------------------
# Space-group key helpers
# ---------------------------------------------------------------------------

def _sg_number(key) -> int:
    if isinstance(key, int):
        return key
    if isinstance(key, SpaceGroup):
        return key.number
    return space_group(key).number


def laue_class(key) -> str:
    """Laue-class symbol for a space group (e.g. ``'2/m'``, ``'m-3m'``)."""
    n = _sg_number(key)
    return asu_data.LAUE_CLASS[n]


# ---------------------------------------------------------------------------
# Reciprocal ASU — indices match asu_data.RECIPROCAL_CONDITIONS order
# ---------------------------------------------------------------------------

def _recip_in(h: int, k: int, l: int, cond_idx: int) -> bool:
    """Evaluate CCP4/gemmi reciprocal-ASU condition by table index."""
    if cond_idx == 0:  # -1
        return l > 0 or (l == 0 and (h > 0 or (h == 0 and k >= 0)))
    if cond_idx == 1:  # 2/m
        return k >= 0 and (l > 0 or (l == 0 and h >= 0))
    if cond_idx == 2:  # mmm
        return h >= 0 and k >= 0 and l >= 0
    if cond_idx == 3:  # 4/m, 6/m
        return l >= 0 and ((h >= 0 and k > 0) or (h == 0 and k == 0))
    if cond_idx == 4:  # 4/mmm, 6/mmm
        return h >= k and k >= 0 and l >= 0
    if cond_idx == 5:  # -3
        return (h >= 0 and k > 0) or (h == 0 and k == 0 and l >= 0)
    if cond_idx == 6:  # -3m (variant B)
        return h >= k and k >= 0 and (k > 0 or l >= 0)
    if cond_idx == 7:  # -3m (variant A)
        return h >= k and k >= 0 and (h > k or l >= 0)
    if cond_idx == 8:  # m-3
        return h >= 0 and ((l >= h and k > h) or (l == h and k == h))
    if cond_idx == 9:  # m-3m
        return k >= l and l >= h and h >= 0
    raise ValueError(f"unknown reciprocal condition index {cond_idx}")


@dataclass(frozen=True)
class ReciprocalAsu:
    """CCP4 / cctbx / gemmi reciprocal-space asymmetric unit."""

    number: int
    _cond_idx: int

    @classmethod
    def from_space_group(cls, key) -> "ReciprocalAsu":
        n = _sg_number(key)
        return cls(n, asu_data.RECIPROCAL_CONDITION_INDEX[n - 1])

    @property
    def condition_str(self) -> str:
        return asu_data.RECIPROCAL_CONDITIONS[self._cond_idx]

    def is_in(self, hkl: Sequence[int] | Vector3) -> bool:
        if isinstance(hkl, Vector3):
            h, k, l = int(hkl.v[0]), int(hkl.v[1]), int(hkl.v[2])
        else:
            h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
        return _recip_in(h, k, l, self._cond_idx)

    def to_asu(
        self,
        hkl: Sequence[int] | Vector3,
        operations: Sequence[SymmetryOp],
    ) -> tuple[tuple[int, int, int], int]:
        """Map ``hkl`` into the ASU.

        Returns ``(hkl_asu, isym)`` where ``isym`` follows the MTZ convention
        relative to the given ``operations`` order: for 0-based index ``i``,
        ``2*i+1`` means a direct image and ``2*i+2`` a Friedel image.
        """
        if isinstance(hkl, Vector3):
            h0 = Vector3(hkl.v)
        else:
            h0 = Vector3((int(hkl[0]), int(hkl[1]), int(hkl[2])))
        for i, op in enumerate(operations):
            ht = transform_hkl(h0, op.W)
            t = (int(ht.v[0]), int(ht.v[1]), int(ht.v[2]))
            if self.is_in(t):
                return t, 2 * i + 1
            mt = (-t[0], -t[1], -t[2])
            if self.is_in(mt):
                return mt, 2 * i + 2
        raise RuntimeError(
            f"no ASU image of {tuple(int(x) for x in h0.v)} under the given operations"
        )


# ---------------------------------------------------------------------------
# Direct-space ASU brick
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisBound:
    """Bound on one fractional coordinate: ``0 <= x </<= hi``."""

    closed_hi: bool
    hi: Fraction

    def contains(self, x: Fraction) -> bool:
        if x < 0:
            return False
        if self.closed_hi:
            return x <= self.hi
        return x < self.hi


@dataclass(frozen=True)
class DirectAsuBrick:
    """Conventional axis-aligned real-space ASU brick (CCP4 / gemmi)."""

    number: int
    x: AxisBound
    y: AxisBound
    z: AxisBound
    _string: str

    @classmethod
    def from_space_group(cls, key) -> "DirectAsuBrick":
        n = _sg_number(key)
        idx = asu_data.ASU_BRICK_INDEX[n - 1]
        xc, xh, yc, yh, zc, zh = asu_data.ASU_BRICK_BOUNDS[idx]
        return cls(
            n,
            AxisBound(xc, xh), AxisBound(yc, yh), AxisBound(zc, zh),
            asu_data.ASU_BRICK_STRINGS[idx],
        )

    def __str__(self) -> str:
        return self._string

    @property
    def bounds(self) -> tuple[AxisBound, AxisBound, AxisBound]:
        return self.x, self.y, self.z

    def contains(self, xyz: Sequence | Vector3) -> bool:
        if isinstance(xyz, Vector3):
            x, y, z = xyz.v
        else:
            x, y, z = Fraction(xyz[0]), Fraction(xyz[1]), Fraction(xyz[2])
        x, y, z = frac_mod1(x), frac_mod1(y), frac_mod1(z)
        return self.x.contains(x) and self.y.contains(y) and self.z.contains(z)


# ---------------------------------------------------------------------------
# Dirichlet / Voronoi ASU (metric)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HalfSpace:
    """Cartesian half-space ``n · x <= c`` (outward normal ``n``)."""

    normal: tuple[float, float, float]
    offset: float


def _wrap_half(t: float) -> float:
    """Reduce a fractional delta into (-0.5, 0.5]."""
    t = t - int(t)  # toward zero-ish; then fix
    t = t % 1.0
    if t > 0.5:
        t -= 1.0
    elif t <= -0.5:
        t += 1.0
    return t


@dataclass
class DirichletAsu:
    """Dirichlet / Voronoi ASU: points closer to ``origin_frac`` than any orbit mate.

    Membership is the orbit-distance test (works for special-position seeds).
    ``facets`` are perpendicular bisectors of the seed orbit (geometric
    Voronoi cell); they may be incomplete when the seed is symmetry-fixed.
    """

    facets: list[HalfSpace]
    cell: object  # UnitCell
    space: str  # "direct" | "reciprocal"
    origin_frac: Vector3
    operations: list[SymmetryOp]
    _seed_cart: list[float]
    _G: list[list[float]]  # metric (direct G or reciprocal G*)
    _origin_f: tuple[float, float, float]
    _ops_f: list[tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]]

    def contains_cart(self, cart: Sequence[float], tol: float = 1e-9) -> bool:
        for f in self.facets:
            if (
                f.normal[0] * cart[0]
                + f.normal[1] * cart[1]
                + f.normal[2] * cart[2]
                > f.offset + tol
            ):
                return False
        return True

    def _frac_to_cart(self, fv: Sequence[float]) -> list[float]:
        if self.space == "direct":
            return self.cell.orthogonalize(fv)
        Mr = self.cell.reciprocal().orthogonalization_matrix()
        return [
            Mr[0][0] * fv[0] + Mr[0][1] * fv[1] + Mr[0][2] * fv[2],
            Mr[1][0] * fv[0] + Mr[1][1] * fv[1] + Mr[1][2] * fv[2],
            Mr[2][0] * fv[0] + Mr[2][1] * fv[1] + Mr[2][2] * fv[2],
        ]

    def contains(self, frac: Sequence | Vector3, tol: float = 1e-8) -> bool:
        """Unit-cell ASU membership (alias of :meth:`is_asu_representative`)."""
        return self.is_asu_representative(frac, tol=tol)

    def contains_frac(self, frac: Sequence | Vector3, tol: float = 1e-8) -> bool:
        return self.contains(frac, tol=tol)

    def is_asu_representative(
        self, frac: Sequence | Vector3, tol: float = 1e-8
    ) -> bool:
        """True if ``frac`` is the closest orbit mate to the seed (lex-max on ties)."""
        if isinstance(frac, Vector3):
            xf = (
                float(frac.v[0]) % 1.0,
                float(frac.v[1]) % 1.0,
                float(frac.v[2]) % 1.0,
            )
        else:
            xf = (float(frac[0]) % 1.0, float(frac[1]) % 1.0, float(frac[2]) % 1.0)
        # Python's % for negatives is fine for fractions in [0,1)
        xf = (xf[0] % 1.0, xf[1] % 1.0, xf[2] % 1.0)
        d_self = self._dist2_to_seed(xf)
        x0, x1, x2 = xf
        for W, w in self._ops_f:
            yf = (
                (W[0][0] * x0 + W[0][1] * x1 + W[0][2] * x2 + w[0]) % 1.0,
                (W[1][0] * x0 + W[1][1] * x1 + W[1][2] * x2 + w[1]) % 1.0,
                (W[2][0] * x0 + W[2][1] * x1 + W[2][2] * x2 + w[2]) % 1.0,
            )
            if yf == xf:
                continue
            dy = self._dist2_to_seed(yf)
            if dy + tol < d_self:
                return False
            # Equidistant mates (e.g. inversion through the seed): keep lex-max.
            if dy <= d_self + tol and yf > xf:
                return False
        return True

    def _dist2_to_seed(self, frac: tuple[float, float, float]) -> float:
        """Squared Cartesian distance to the seed under the minimum-image convention."""
        v0 = _wrap_half(frac[0] - self._origin_f[0])
        v1 = _wrap_half(frac[1] - self._origin_f[1])
        v2 = _wrap_half(frac[2] - self._origin_f[2])
        G = self._G
        return (
            v0 * (G[0][0] * v0 + G[0][1] * v1 + G[0][2] * v2)
            + v1 * (G[1][0] * v0 + G[1][1] * v1 + G[1][2] * v2)
            + v2 * (G[2][0] * v0 + G[2][1] * v1 + G[2][2] * v2)
        )

    def sample_points(self, n: int = 2000, seed: int = 0) -> list[list[float]]:
        """MC samples of ASU points in the unit cell (Cartesian)."""
        import random
        rng = random.Random(seed)
        pts: list[list[float]] = []
        budget = n * max(20, len(self.operations) * 3)
        tries = 0
        while len(pts) < n and tries < budget:
            tries += 1
            frac = [rng.random(), rng.random(), rng.random()]
            if self.is_asu_representative(frac):
                pts.append(self._frac_to_cart(frac))
        return pts

    def volume_fraction(self, n: int = 8000, seed: int = 0) -> float:
        """Fraction of the unit cell in the ASU (≈ ``1/|G|``)."""
        import random
        rng = random.Random(seed)
        hit = 0
        for _ in range(n):
            frac = [rng.random(), rng.random(), rng.random()]
            if self.is_asu_representative(frac):
                hit += 1
        return hit / n

    def inertia_eigenvalues(
        self, n: int = 2000, seed: int = 0
    ) -> tuple[float, float, float]:
        """Eigenvalues of the Cartesian second-moment tensor of the ASU."""
        pts = self.sample_points(n=n, seed=seed)
        if len(pts) < 10:
            return (0.0, 0.0, 0.0)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        I = [[0.0] * 3 for _ in range(3)]
        for p in pts:
            d = [p[0] - cx, p[1] - cy, p[2] - cz]
            for i in range(3):
                for j in range(3):
                    I[i][j] += d[i] * d[j]
        return _eigh3_sorted(I)

    def sphericity(self, n: int = 2000, seed: int = 0) -> float:
        """Inertia-ellipsoid sphericity ``λ_min / λ_max`` in [0, 1] (1 = sphere)."""
        ev = self.inertia_eigenvalues(n=n, seed=seed)
        if ev[2] <= 0:
            return 0.0
        return ev[0] / ev[2]

    def ellipsoid_score(self, n: int = 2000, seed: int = 0) -> float:
        """Alias of :meth:`sphericity` (axis ratio of the inertia ellipsoid)."""
        return self.sphericity(n=n, seed=seed)

    def isoperimetric_quotient(self, n: int = 2000, seed: int = 0) -> float:
        """``36π V² / A³`` proxy via inertia sphericity (1 = sphere)."""
        return self.sphericity(n=n, seed=seed)


def _eigh3_sorted(A: list[list[float]]) -> tuple[float, float, float]:
    """Eigenvalues of a symmetric 3×3, ascending."""
    a, b, c = A[0][0], A[1][1], A[2][2]
    d, e, f = A[0][1], A[0][2], A[1][2]
    I1 = a + b + c
    I2 = a * b + a * c + b * c - d * d - e * e - f * f
    I3 = a * b * c + 2 * d * e * f - a * f * f - b * e * e - c * d * d
    ev = _cubic_roots(1.0, -I1, I2, -I3)
    ev.sort()
    return ev[0], ev[1], ev[2]


def _cubic_roots(a: float, b: float, c: float, d: float) -> list[float]:
    """Real roots of aλ³ + bλ² + cλ + d = 0 (expects three real for SPD).

    Uses the trigonometric form of the depressed cubic. For near-degenerate
    SPD inputs, float noise can push the discriminant slightly positive (or
    ``p`` slightly non-negative); those cases are clamped onto the three-real
    branch rather than the one-real Cardano path, which would otherwise return
    three identical bogus eigenvalues.
    """
    b, c, d = b / a, c / a, d / a
    p = c - b * b / 3.0
    q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d
    if p >= 0.0:
        # Triple root of the depressed cubic (all eigenvalues equal after shift).
        ys = [0.0, 0.0, 0.0]
    else:
        r = 2.0 * sqrt(-p / 3.0)
        denom = sqrt((-p / 3.0) ** 3)
        arg = 0.0 if denom == 0.0 else (-q / 2.0) / denom
        # Clamp acos domain when disc > 0 from float noise.
        arg = max(-1.0, min(1.0, arg))
        phi = acos(arg)
        ys = [
            r * cos(phi / 3.0),
            r * cos((phi + 2.0 * pi) / 3.0),
            r * cos((phi + 4.0 * pi) / 3.0),
        ]
    shift = b / 3.0
    return [y - shift for y in ys]


def _laue_ops(operations: Sequence[SymmetryOp]) -> list[SymmetryOp]:
    mats = set(point_group(operations))
    if _NEG_I not in mats:
        mats |= {_NEG_I @ W for W in list(mats)}
    return [SymmetryOp(W, ZERO3) for W in mats]


def build_dirichlet_asu(
    operations: Sequence[SymmetryOp],
    cell,
    *,
    space: str = "direct",
    origin_frac: Vector3 | None = None,
    max_images: int = 80,
) -> DirichletAsu:
    """Build a Dirichlet ASU from symmetry images of ``origin_frac``.

    Half-spaces are perpendicular bisectors between the seed and each distinct
    Cartesian image under the group (direct: full ops; reciprocal: Laue).
    """
    if space not in ("direct", "reciprocal"):
        raise ValueError("space must be 'direct' or 'reciprocal'")
    origin = origin_frac if origin_frac is not None else ZERO3
    ops = list(operations)
    use_ops = _laue_ops(ops) if space == "reciprocal" else ops

    def to_cart(frac_v: Vector3) -> list[float]:
        fv = [float(x) for x in frac_v.v]
        if space == "direct":
            return cell.orthogonalize(fv)
        Mr = cell.reciprocal().orthogonalization_matrix()
        return [
            Mr[0][0] * fv[0] + Mr[0][1] * fv[1] + Mr[0][2] * fv[2],
            Mr[1][0] * fv[0] + Mr[1][1] * fv[1] + Mr[1][2] * fv[2],
            Mr[2][0] * fv[0] + Mr[2][1] * fv[1] + Mr[2][2] * fv[2],
        ]

    seed = to_cart(origin)
    images: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    # Always include near-neighbour lattice images of the seed
    lattice_ops = list(use_ops) + [SymmetryOp(IDENTITY3, ZERO3)]
    for op in lattice_ops:
        img_frac = (op.W @ origin) + op.w
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    shifted = Vector3((
                        img_frac.v[0] + dx,
                        img_frac.v[1] + dy,
                        img_frac.v[2] + dz,
                    ))
                    cart = to_cart(shifted)
                    key = tuple(round(c, 8) for c in cart)
                    if key in seen:
                        continue
                    if all(abs(cart[i] - seed[i]) < 1e-10 for i in range(3)):
                        continue
                    seen.add(key)
                    images.append(cart)
                    if len(images) >= max_images:
                        break
                if len(images) >= max_images:
                    break
            if len(images) >= max_images:
                break
        if len(images) >= max_images:
            break

    facets: list[HalfSpace] = []
    for img in images:
        mid = [(seed[i] + img[i]) / 2.0 for i in range(3)]
        nrm = [img[i] - seed[i] for i in range(3)]
        length = sqrt(sum(t * t for t in nrm))
        if length < 1e-14:
            continue
        nrm = [t / length for t in nrm]
        offset = nrm[0] * mid[0] + nrm[1] * mid[1] + nrm[2] * mid[2]
        facets.append(HalfSpace(tuple(nrm), offset))

    if space == "direct":
        G = cell.metric_tensor()
    else:
        G = cell.reciprocal_metric_tensor()

    ops_f = []
    for op in use_ops:
        Wf = tuple(tuple(float(x) for x in row) for row in op.W.rows)
        wf = (float(op.w.v[0]), float(op.w.v[1]), float(op.w.v[2]))
        ops_f.append((Wf, wf))

    return DirichletAsu(
        facets=facets, cell=cell, space=space,
        origin_frac=origin, operations=list(use_ops),
        _seed_cart=seed, _G=G,
        _origin_f=(float(origin.v[0]), float(origin.v[1]), float(origin.v[2])),
        _ops_f=ops_f,
    )


@dataclass(frozen=True)
class OptimizedAsu:
    asu: DirichletAsu
    score: float
    origin_shift: Vector3
    metrics: dict


def optimize_asu(
    operations: Sequence[SymmetryOp],
    cell,
    *,
    space: str = "direct",
    score: str = "sphericity",
    n_sample: int = 3000,
) -> OptimizedAsu:
    """Search allowed origin gauges for the most spherical Dirichlet ASU.

    Candidates: identity, floating-axis samples, and Cheshire torsion origins
    from :mod:`agentsg.semi_invariants`.
    """
    if score not in ("sphericity", "ellipsoid"):
        raise ValueError("score must be 'sphericity' or 'ellipsoid'")
    ops = list(operations)
    candidates = [ZERO3]
    for v in floating_origin_basis(ops):
        for t in (Fraction(0), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            candidates.append(Vector3(t * x for x in v.v))
    for o in discrete_allowed_origins(ops):
        candidates.append(o)
        for v in floating_origin_basis(ops):
            candidates.append(Vector3(
                o.v[i] + Fraction(1, 4) * v.v[i] for i in range(3)
            ))

    uniq: list[Vector3] = []
    seen: set = set()
    for o in candidates:
        key = o.mod1().v
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o.mod1())

    best: OptimizedAsu | None = None
    for o in uniq:
        asu = build_dirichlet_asu(ops, cell, space=space, origin_frac=o)
        sc = asu.sphericity(n=n_sample)
        metrics = {
            "sphericity": sc,
            "volume_fraction": asu.volume_fraction(n=n_sample),
            "inertia": asu.inertia_eigenvalues(n=n_sample),
            "allowed_origin": is_allowed_origin(o, ops),
        }
        if best is None or sc > best.score:
            best = OptimizedAsu(asu, sc, o, metrics)
    assert best is not None
    return best
