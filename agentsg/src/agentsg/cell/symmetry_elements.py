"""Exact classification of crystallographic symmetry elements.

An operation ``x -> W x + w`` is reduced modulo the full translation lattice
(primitive translations and centring). The intrinsic (screw or glide) part is
the shortest representative in the fixed space of ``W``. A zero intrinsic part
is a plain rotation or mirror. Locations are exact rationals, and each line or
plane is reported at the point of that locus nearest the origin.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def _F(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x).limit_denominator(10_000)


def _frac3(v) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(_F(x) for x in v)


def _int_matrix(W) -> tuple[tuple[int, int, int], ...]:
    arr = np.asarray(W)
    rows = []
    for i in range(3):
        rows.append(tuple(int(round(float(arr[i, j]))) for j in range(3)))
    return tuple(rows)


def _matvec(W, v):
    return tuple(sum(W[i][j] * v[j] for j in range(3)) for i in range(3))


def _mul(A, B):
    """3x3 times 3x3, rows of the product."""
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(A[i][k] * B[k][j] for k in range(3)))
        out.append(tuple(row))
    return tuple(out)


def _eye():
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _det(W) -> int:
    a, b, c = W
    return int(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _trace(W) -> int:
    return int(W[0][0] + W[1][1] + W[2][2])


def _dot(a, b) -> Fraction:
    return sum((a[i] * b[i] for i in range(3)), Fraction(0))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def _scale(s, v):
    return tuple(s * v[i] for i in range(3))


def _is_zero(v) -> bool:
    return all(x == 0 for x in v)


def _primitive_int(v) -> tuple[int, int, int]:
    """Fraction vector -> primitive integer vector, first nonzero component > 0."""
    den = 1
    comps = tuple(_F(x) for x in v)
    for x in comps:
        den = math.lcm(den, x.denominator)
    ints = [int(x * den) for x in comps]
    g = 0
    for x in ints:
        g = math.gcd(g, abs(x))
    if g == 0:
        return (0, 0, 0)
    ints = [x // g for x in ints]
    for x in ints:
        if x:
            if x < 0:
                ints = [-a for a in ints]
            break
    return (ints[0], ints[1], ints[2])


def _row_reduce(rows, extra_cols):
    """Gaussian elimination over Fractions.

    ``rows`` is a list of rows of length 3 + extra_cols. Returns the reduced
    matrix and a dict mapping pivot column -> row index.
    """
    M = [list(r) for r in rows]
    n = len(M)
    width = len(M[0]) if M else 0
    pivots = {}
    row = 0
    for col in range(3):
        sel = None
        for i in range(row, n):
            if M[i][col] != 0:
                sel = i
                break
        if sel is None:
            continue
        M[row], M[sel] = M[sel], M[row]
        piv = M[row][col]
        M[row] = [v / piv for v in M[row]]
        for i in range(n):
            if i == row or M[i][col] == 0:
                continue
            f = M[i][col]
            M[i] = [M[i][j] - f * M[row][j] for j in range(width)]
        pivots[col] = row
        row += 1
    return M, pivots, row


def _integer_kernel(A) -> list[tuple[int, int, int]]:
    """Primitive integer basis of the kernel of a 3x3 integer matrix."""
    rows = [[Fraction(x) for x in row] for row in A]
    M, pivots, _ = _row_reduce(rows, 0)
    free = [c for c in range(3) if c not in pivots]
    basis = []
    for fcol in free:
        v = [Fraction(0), Fraction(0), Fraction(0)]
        v[fcol] = Fraction(1)
        for col, r in pivots.items():
            v[col] = -M[r][fcol]
        basis.append(_primitive_int(v))
    return basis


def _particular(A, b):
    """One solution of A x = b. Free variables are 0. None if inconsistent."""
    rows = [[Fraction(A[i][j]) for j in range(3)] + [Fraction(b[i])] for i in range(3)]
    M, pivots, rank = _row_reduce(rows, 1)
    for i in range(rank, 3):
        if M[i][3] != 0:
            return None
    x = [Fraction(0), Fraction(0), Fraction(0)]
    for col, r in pivots.items():
        x[col] = M[r][3]
    return tuple(x)


def _frac_mod1(x: Fraction) -> Fraction:
    x = _F(x)
    return Fraction(x.numerator % x.denominator, x.denominator)


def _mod1(v):
    return tuple(_frac_mod1(x) for x in v)


def translation_lattice(centring) -> list[tuple[Fraction, Fraction, Fraction]]:
    """Lattice vectors in a neighbourhood of the origin, including centring."""
    cents = [_frac3(c) for c in centring] if centring else [(Fraction(0),) * 3]
    if not any(_is_zero(c) for c in cents):
        cents = [(Fraction(0),) * 3] + cents
    seen = set()
    out = []
    for c in cents:
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    v = (c[0] + i, c[1] + j, c[2] + k)
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
    return out


class Lattice:
    """Translation lattice with the fixed-space vectors of each ``W`` cached.

    Building those vectors is a scan over every lattice vector. A centred
    group asks for the same ``W`` thousands of times while it tiles the cell,
    so the scan runs once per matrix.
    """

    def __init__(self, vecs):
        self.vecs = tuple(_frac3(v) for v in vecs)
        self._fixed = {}
        self._short = {}

    def fixed(self, W):
        hit = self._fixed.get(W)
        if hit is None:
            hit = tuple(L for L in self.vecs if _matvec(W, L) == L)
            self._fixed[W] = hit
        return hit

    def shortest_along(self, axis):
        hit = self._short.get(axis)
        if hit is None:
            hit = _shortest_along(axis, self.vecs)
            self._short[axis] = hit
        return hit


_PRIMITIVE = None


def primitive_lattice() -> Lattice:
    """The primitive translation lattice, cached: origin only, no centring vectors."""
    global _PRIMITIVE
    if _PRIMITIVE is None:
        _PRIMITIVE = Lattice(translation_lattice([(0, 0, 0)]))
    return _PRIMITIVE


def _as_lattice(lattice) -> Lattice:
    if isinstance(lattice, Lattice):
        return lattice
    if lattice is None:
        return primitive_lattice()
    return Lattice(lattice)


def _reduce_intrinsic(t, W, lattice: Lattice):
    """Shortest representative of ``t`` modulo lattice vectors fixed by ``W``.

    Equal-length candidates keep the one nearest the unreduced translation, so
    a diamond glide and its neighbour keep opposite senses. A lattice vector
    still reduces to 0, which is strictly shorter than any glide.
    """
    t = _frac3(t)
    best = t
    best_len = _dot(t, t)
    best_dist = Fraction(0)
    for L in lattice.fixed(W):
        cand = _sub(t, L)
        ln = _dot(cand, cand)
        delta = _sub(cand, t)
        dist = _dot(delta, delta)
        if (
            ln < best_len
            or (ln == best_len and dist < best_dist)
            or (ln == best_len and dist == best_dist and cand < best)
        ):
            best = cand
            best_len = ln
            best_dist = dist
    return best


def _orient_axis(W, axis):
    """Flip ``axis`` so ``W`` is a right-handed rotation about it.

    A 2-fold has no preferred sense; the caller keeps the primitive sign.
    """
    a = axis
    abs_a = [abs(x) for x in a]
    p_idx = abs_a.index(min(abs_a))
    p = [0, 0, 0]
    p[p_idx] = 1
    Wp = _matvec(W, p)
    d = (
        a[0] * (p[1] * Wp[2] - p[2] * Wp[1])
        - a[1] * (p[0] * Wp[2] - p[2] * Wp[0])
        + a[2] * (p[0] * Wp[1] - p[1] * Wp[0])
    )
    if d < 0:
        return (-a[0], -a[1], -a[2])
    return a


def _shortest_along(axis, lattice):
    """Shortest lattice vector pointing the same way as ``axis``."""
    best = None
    best_len = None
    for L in lattice:
        if _is_zero(L):
            continue
        if _cross(L, axis) != (0, 0, 0):
            continue
        if _dot(L, axis) <= 0:
            continue
        ln = _dot(L, L)
        if best is None or ln < best_len or (ln == best_len and L < best):
            best = L
            best_len = ln
    return best


def _screw_index(intr, period, n) -> int:
    """Screw index k of an intrinsic translation along an n-fold axis.

    k/n is the fraction of the axis period, folded into 0..n-1. Returns 0
    when the intrinsic part is a lattice vector (a pure rotation).
    """
    if period is None or _is_zero(intr):
        return 0
    coef = _dot(intr, period) / _dot(period, period)
    k = coef * n
    if k.denominator != 1:
        k = Fraction(int(round(float(k))))
    return int(k) % n


def _canonical_line(loc, axis):
    """Point of the line nearest the origin, inside the unit cell."""
    loc = _frac3(loc)
    den = 1
    for x in loc:
        den = math.lcm(den, x.denominator)
    for a in axis:
        if a:
            den = math.lcm(den, 1)
    best = None
    best_key = None
    for j in range(den):
        t = Fraction(j, den)
        p = _mod1(tuple(loc[i] + t * axis[i] for i in range(3)))
        d2 = _dot(p, p)
        key = (d2, p)
        if best_key is None or key < best_key:
            best_key = key
            best = p
    return best


def _canonical_plane(loc, normal):
    """Point of the plane nearest the origin, folded into the unit cell.

    Planes that differ by a lattice translation (the cell boundary counted
    twice) share this point.
    """
    loc = _frac3(loc)
    n = normal
    g = 0
    for a in n:
        g = math.gcd(g, abs(a))
    g = g or 1
    c = _dot(n, loc)
    # c modulo g, in [0, g)
    q = c / g
    c = c - g * (q.numerator // q.denominator)
    nn = sum(a * a for a in n)
    closest = tuple(c * n[i] / nn for i in range(3))
    return _mod1(closest)


def _glide_symbol(t, W, hexagonal=False, normal=None) -> str:
    """ITA glide letter for a reduced intrinsic translation lying in the plane.

    ``m`` when the translation is a lattice vector. ``a``/``b``/``c`` for half
    of one cell axis. ``n`` for half a face or body diagonal on a coordinate
    plane, and for a body-diagonal glide (a horizontal part and a ``c`` part)
    outside the hexagonal and trigonal families. ``d`` for a quarter diagonal.
    ``g`` is what remains: diagonal glides of tetragonal mirrors and the
    hexagonal in-plane glides.
    """
    t = _frac3(t)
    if _is_zero(t):
        return "m"
    doubled = tuple(2 * x for x in t)
    if not all(x.denominator == 1 for x in doubled):
        quad = tuple(4 * x for x in t)
        if all(x.denominator == 1 for x in quad):
            return "d"
        return "g"
    iv = tuple(int(x) for x in doubled)
    nz = [i for i in range(3) if iv[i] != 0]
    if len(nz) == 1:
        return "abc"[nz[0]]
    diagonal_w = all(W[i][j] == 0 for i in range(3) for j in range(3) if i != j)
    if diagonal_w:
        return "n"
    # Half a face diagonal of a coordinate plane is n in every family,
    # whether or not one leg is c. A diagonal mirror (normal not along one
    # cell axis) keeps g: that is the g of P4mm, not the n of Pm-3m.
    if (not hexagonal and len(nz) == 2
            and all(abs(iv[i]) == 1 for i in nz) and normal is not None):
        nnz = [i for i, comp in enumerate(normal) if comp]
        if len(nnz) == 1 and nnz[0] not in nz:
            return "n"
    if iv[2] != 0 and not hexagonal:
        return "n"
    # Hexagonal c-glides carry c/2 plus an in-plane half. The skewed basis
    # makes that look like a diagonal g; the ITA letter is still c.
    # A glide with no c component (the in-plane g of P3m1) stays g.
    if iv[2] != 0 and hexagonal:
        return "c"
    return "g"


_ROT_BY_TRACE = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}
_INV_BY_TRACE = {-3: -1, -2: -6, -1: -4, 0: -3, 1: -2}


def _intrinsic(W, w, n):
    acc = (Fraction(0), Fraction(0), Fraction(0))
    P = _eye()
    w = _frac3(w)
    for _ in range(n):
        acc = _add(acc, _matvec(P, w))
        P = _mul(P, W)
    return tuple(x / n for x in acc)


def _as_numpy_el(el):
    """Float copies of the exact fields for the renderer."""
    out = dict(el)
    if el.get("axis_exact") is not None:
        out["axis"] = np.array(el["axis_exact"], dtype=float)
    else:
        out["axis"] = None
    if el.get("location_exact") is not None:
        out["location"] = np.array(el["location_exact"], dtype=float)
    else:
        out["location"] = None
    out["intrinsic"] = np.array(el["intrinsic_exact"], dtype=float)
    if el.get("W_exact") is not None:
        out["W"] = np.array(el["W_exact"], dtype=float)
    return out


def classify_element(W, w, lattice=None, hexagonal=False):
    """Classify ``(W, w)`` as an ITA symmetry element.

    ``lattice`` is the translation lattice (primitive cell plus centring).
    When it is omitted the classification is modulo the primitive lattice
    only. ``hexagonal`` selects the trigonal/hexagonal glide letters (``g``
    rather than ``n`` for a non-axial glide with a ``c`` component).

    The returned dict carries numpy ``axis`` / ``location`` / ``intrinsic``
    for the renderer and exact ``*_exact`` tuples.
    """
    Wi = _int_matrix(W)
    wf = _frac3(w)
    lattice = _as_lattice(lattice)
    det = _det(Wi)
    tr = _trace(Wi)

    def pack(kind, order, symbol, axis, loc, intr):
        return _as_numpy_el({
            "type": kind,
            "order": order,
            "symbol": symbol,
            "axis_exact": axis,
            "location_exact": loc,
            "intrinsic_exact": intr,
            "W_exact": Wi,
        })

    if Wi == _eye():
        wr = _mod1(wf)
        if _is_zero(wr):
            return pack("identity", 1, "1", None, None, (Fraction(0),) * 3)
        return pack("translation", 1, "t", None, None, wr)

    if det == 1:
        n = _ROT_BY_TRACE.get(tr)
        if not n:
            n = _order_of(Wi)
        intr_raw = _intrinsic(Wi, wf, n)
        intr = _reduce_intrinsic(intr_raw, Wi, lattice)
        ker = _integer_kernel([[Wi[i][j] - (1 if i == j else 0) for j in range(3)]
                               for i in range(3)])
        axis = ker[0] if ker else (0, 0, 1)
        if n > 2:
            axis = _orient_axis(Wi, axis)
        period = lattice.shortest_along(axis)
        k = _screw_index(intr, period, n)
        # Reducing the intrinsic part by a fixed lattice vector does not move
        # the axis. The location part is taken from the unreduced split.
        w_loc = _sub(wf, intr_raw)
        loc = _particular(
            [[Wi[i][j] - (1 if i == j else 0) for j in range(3)] for i in range(3)],
            tuple(-x for x in w_loc),
        )
        if loc is None:
            loc = (Fraction(0), Fraction(0), Fraction(0))
        loc = _canonical_line(loc, axis)
        if k:
            return pack("screw", n, f"{n}_{k}", axis, loc, intr)
        return pack("rotation", n, str(n), axis, loc, (Fraction(0),) * 3)

    kind = _INV_BY_TRACE.get(tr)
    if kind == -1:
        loc = _mod1(tuple(x / 2 for x in wf))
        return pack("inversion", 2, "-1", None, loc, (Fraction(0),) * 3)
    if kind == -2:
        intr_raw = _intrinsic(Wi, wf, 2)
        intr = _reduce_intrinsic(intr_raw, Wi, lattice)
        w_loc = _sub(wf, intr_raw)
        # The plane normal is a left eigenvector of W (kernel of Wᵀ+I).
        # The right kernel of W+I is the direction the matrix reverses. In a
        # skewed hexagonal basis those two vectors differ, and the right one
        # drops the half-offset plane onto the mirror through the origin.
        WT = tuple(tuple(Wi[j][i] for j in range(3)) for i in range(3))
        ker = _integer_kernel([[WT[i][j] + (1 if i == j else 0) for j in range(3)]
                               for i in range(3)])
        normal = ker[0] if ker else (0, 0, 1)
        gname = _glide_symbol(intr, Wi, hexagonal=hexagonal, normal=normal)
        loc = _particular(
            [[Wi[i][j] - (1 if i == j else 0) for j in range(3)] for i in range(3)],
            tuple(-x for x in w_loc),
        )
        if loc is None:
            loc = (Fraction(0), Fraction(0), Fraction(0))
        loc = _canonical_plane(loc, normal)
        if gname == "m":
            el = pack("mirror", 2, "m", normal, loc, (Fraction(0),) * 3)
            # Centring can make an n the same plane as an m. Keep the letter
            # so the JSON is complete; the plate does not draw it twice.
            raw_name = _glide_symbol(
                _mod1(intr_raw), Wi, hexagonal=hexagonal, normal=normal)
            if raw_name == "n":
                el["_coincident_symbol"] = "n"
                el["_coincident_intrinsic"] = _mod1(intr_raw)
        else:
            el = pack("glide", 2, gname, normal, loc, intr)
        # Unreduced in-plane part, folded into the primitive cell. Two of
        # these on one plane, along different axes, are an e glide.
        el["intrinsic_raw_exact"] = _mod1(intr_raw)
        return el

    n = {-3: 3, -4: 4, -6: 6}.get(kind) or _order_of(Wi)
    # Axis of the proper part R = -W.
    R = tuple(tuple(-x for x in row) for row in Wi)
    ker = _integer_kernel([[R[i][j] - (1 if i == j else 0) for j in range(3)]
                           for i in range(3)])
    axis = ker[0] if ker else (0, 0, 1)
    if n > 2:
        axis = _orient_axis(R, axis)
    loc = _particular(
        [[Wi[i][j] - (1 if i == j else 0) for j in range(3)] for i in range(3)],
        tuple(-x for x in wf),
    )
    if loc is None:
        loc = (Fraction(0), Fraction(0), Fraction(0))
    loc = _mod1(loc)
    return pack("rotoinversion", n, str(kind), axis, loc, (Fraction(0),) * 3)


def _order_of(W, max_n=6) -> int:
    P = _eye()
    for n in range(1, max_n + 1):
        P = _mul(P, W)
        if P == _eye():
            return n
    return 0


# Public alias used by the diagram module and the tests.
_glide_name = _glide_symbol
