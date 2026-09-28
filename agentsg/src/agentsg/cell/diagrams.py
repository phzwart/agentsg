"""ITA-style space-group diagrams (general-position and symmetry-element).

Renders the classic *International Tables for Crystallography* Volume A diagrams
from the package's own derived symmetry operations -- no tabulated diagram data.
Two diagram kinds:

* :func:`general_position_diagram` -- the equivalent-points diagram: a general
  point projected through every space-group operation, drawn with the ITA glyph
  convention (open circle; exact heights ``+``, ``-``, ``1/2+``, ``1/3-`` ...;
  a comma for points related by an operation of the opposite handedness).
* :func:`symmetry_element_diagram` -- the symmetry-element diagram (built in a
  later step): axes, planes and inversion centres drawn with ITA graphical
  symbols, classified from each operation's (W, w).

Drawing needs matplotlib; it is imported lazily inside the functions so the
package runtime stays dependency-free.

ITA drawing convention used here: projection down **c** by default, origin at
the upper-left, **a** pointing down the page, **b** pointing right.

Everything on the plate is derived from the operator set:

* the **cell frame** (rectangle, square, 120-degree rhombus, or oblique
  parallelogram) comes from the metric constraints ``W^T g W = g`` that the
  in-plane parts of the operations impose on the projected 2-D metric
  (:func:`cell_frame`);
* **heights** of the general-position points are the exact rational
  translations ``t`` in ``z' = +/-z + t`` read off each operation, printed the
  ITA way (``+``, ``-``, ``1/2+``, ``1/3-``, ...) -- never a sign guessed from
  a floating-point ``z`` (:func:`height_label`);
* **screw senses** (3_1 vs 3_2, 4_1 vs 4_3, 6_1..6_5) are measured against an
  axis oriented by the right-hand rule from ``W`` itself, so an operation and
  its inverse name the same element (:func:`classify_element`).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..space_groups import space_group


def _wmatrix(op):
    """(W, w) as float ndarray / vector from a SymmetryOp."""
    W = np.array([[float(x) for x in row] for row in op.W.rows])
    w = np.array([float(x) for x in op.w.v])
    return W, w


def _sg_ops(sg):
    """List of (W, w, xyz) for a SpaceGroup.

    ``SpaceGroup.operations()`` already includes the centring copies (the total
    equals distinct-rotation count times the centring multiplicity), so no
    separate centring expansion is needed.
    """
    out = []
    for op in sg.operations():
        W, w = _wmatrix(op)
        out.append((W, w, op.as_xyz()))
    return out


def _resolve_sg(sg):
    """Accept a SpaceGroup, a SpaceGroupSetting, an int number, or an HM/Hall
    string. Objects that already expose ``operations()`` pass through -- this
    is what lets a non-standard setting (base group + change of basis) be drawn
    with exactly the same code as a standard group."""
    if hasattr(sg, "operations"):
        return sg
    return space_group(sg)


def _sg_label(sg):
    """(number, name) for the title, robust to SpaceGroup vs SpaceGroupSetting.

    A setting has no space-group number of its own (it may not even be one of
    the 230 standard settings); we show its base number and the full
    'HM (cob)' string it prints as."""
    num = getattr(sg, "number", None)
    if num is None and hasattr(sg, "base"):
        num = getattr(sg.base, "number", None)
    name = getattr(sg, "hermann_mauguin", None) or str(sg)
    return num, name


def _sg_order(sg):
    """Return the order of the space group object or setting."""
    o = getattr(sg, "order", None)
    return o() if callable(o) else (o if o is not None else len(_sg_ops(sg)))


# projection -> (perm, down_label, right_label, depth_label)
# perm reorders an (a,b,c) vector so [0]=vertical(down), [1]=horizontal(right),
# [2]=depth(out of page). The default 'c' is the standard ITA projection.
_PROJ = {
    "c": ((0, 1, 2), "a", "b", "c"),
    "a": ((1, 2, 0), "b", "c", "a"),
    "b": ((2, 0, 1), "c", "a", "b"),
}


def _cell_edge_sense(rd):
    """Flip a projected glide onto the positive cell edges.

    ``rd`` is fractional (right, down). Reversing the whole vector when its
    components sum to less than zero points an axial glide along +a or +c
    and a diagonal glide along a+c. An a−c glide stays on that diagonal.
    """
    v = np.asarray(rd, float).copy()
    if float(v[0] + v[1]) < 0.0:
        v = -v
    return v


def _perm_vec(v, perm):
    """Reorder a 3-vector by perm (returns None passthrough)."""
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    return v[list(perm)]


def _perm_mat(W, perm):
    """Conjugate a 3x3 matrix into the permuted (down, right, depth) frame."""
    idx = list(perm)
    return np.asarray(W, dtype=float)[np.ix_(idx, idx)]


def _sg_ops_exact(sg):
    """List of (W_rows, w) with integer rows and ``Fraction`` translations."""
    out = []
    for op in sg.operations():
        W = tuple(tuple(int(x) for x in row) for row in op.W.rows)
        w = tuple(Fraction(x) for x in op.w.v)
        out.append((W, w))
    return out


# --- cell frame from the metric constraints -----------------------------------

_OBLIQUE_DEG = 100.0   # drawing angle for a cell whose in-plane angle is free


def cell_frame(sg, projection="c"):
    """Shape of the projected unit cell, derived from the operations.

    The in-plane parts ``W2`` of every operation that maps the projection axis
    onto itself constrain the projected 2-D metric ``g`` through
    ``W2^T g W2 = g``. Counting the free parameters of that linear system gives
    the frame:

    ======  ===========================  =========================
    free    forced                       frame
    ======  ===========================  =========================
    3       nothing                      oblique parallelogram
    2       ``g12 = 0``                  rectangle
    1       ``g11 = g22``, ``g12 = 0``   square (4-fold present)
    1       ``g11 = g22 = -2 g12``       120-degree rhombus (3/6-fold)
    ======  ===========================  =========================

    Returns a dict with ``angle`` (degrees between the down- and right-going
    cell edges), ``kind`` (``'oblique'``, ``'rect'``, ``'square'``,
    ``'hex'``), ``n_free`` and ``matrix`` -- the 2x2 map from fractional
    ``(right, down)`` to plot ``(x, y)`` coordinates (``y`` grows downward).
    """
    sg = _resolve_sg(sg)
    perm = _PROJ[projection][0]
    # Solve the full 3-D metric constraint W^T G W = G (so a cubic 3-fold,
    # oblique to the page, still ties the in-plane lengths together), then
    # read the projected 2x2 block off the solution space.
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]   # G unknowns
    col = {p: i for i, p in enumerate(pairs)}

    def gidx(i, j):
        """Map symmetric matrix indices (i, j) to 6D column vector index."""
        return col[(i, j) if i <= j else (j, i)]

    rows = []
    for W, _, _ in _sg_ops(sg):
        # (W^T G W)_{ij} = sum_ab W_ai W_bj G_ab  ; minus G_ij = 0
        for i, j in pairs:
            row = np.zeros(6)
            for a in range(3):
                for b in range(3):
                    if W[a, i] and W[b, j]:
                        row[gidx(a, b)] += W[a, i] * W[b, j]
            row[gidx(i, j)] -= 1.0
            rows.append(row)
    A = np.array(rows) if rows else np.zeros((1, 6))
    _, s, vt = np.linalg.svd(A)
    rank = int(np.sum(s > 1e-9))
    N = vt[rank:].T                       # null-space basis, 6 x n_free
    n_free = N.shape[1]

    def forced(f):
        """Is the linear functional f (on G) zero on every admissible G?"""
        return np.allclose(f @ N, 0, atol=1e-9)

    d, r = perm[0], perm[1]               # down- and right-going axes
    e = np.eye(6)
    equal = forced(e[gidx(d, d)] - e[gidx(r, r)])
    right = forced(e[gidx(d, r)])
    hexag = equal and forced(e[gidx(d, r)] + 0.5 * e[gidx(d, d)])
    if hexag:
        angle, kind = 120.0, "hex"
    elif right and equal:
        angle, kind = 90.0, "square"
    elif right:
        angle, kind = 90.0, "rect"
    else:
        angle, kind = _OBLIQUE_DEG, "oblique"
    th = np.radians(angle)
    # columns: right-edge (b) -> (1, 0); down-edge (a) -> (cos th, sin th)
    M = np.array([[1.0, np.cos(th)], [0.0, np.sin(th)]])
    return {"angle": angle, "kind": kind, "n_free": n_free, "matrix": M}


class _Frame:
    """Fractional (right, down) <-> plot (x, y) mapping for one cell frame."""

    def __init__(self, info):
        self.M = info["matrix"]
        self.angle = info["angle"]
        self.kind = info["kind"]
        self.corners = [self.pt((v, u)) for v, u in
                        ((0, 0), (1, 0), (1, 1), (0, 1))]

    def pt(self, rd):
        """fractional (right, down) -> plot (x, y)."""
        v = self.M @ np.asarray(rd, float)
        return (float(v[0]), float(v[1]))

    def vec(self, rd):
        """fractional direction (right, down) -> plot direction (unit)."""
        v = self.M @ np.asarray(rd, float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def limits(self, pad=0.15):
        """Compute (x_limits, y_limits) with padding for the cell bounding box."""
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return (min(xs) - pad, max(xs) + pad), (max(ys) + pad, min(ys) - pad)

    def draw_cell(self, ax, lw=1.2):
        """Draw the projected unit cell boundary polygon onto matplotlib ax."""
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(self.corners, closed=True, fill=False, ec="k",
                             lw=lw, zorder=2))

    def label_axes(self, ax, dlab, rlab, off=0.07):
        """Axis letters just past the b (right) and a (down) corners."""
        bx, by = self.corners[1]
        ax_, ay_ = self.corners[3]
        ax.annotate(rlab, xy=(bx + off, by - off), fontsize=7,
                    ha="center", va="center")
        ax.annotate(dlab, xy=(ax_ - off, ay_ + off), fontsize=7,
                    ha="center", va="center")


# --- exact heights -------------------------------------------------------------

_SUBDIGIT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_UNICODE_FRAC = {
    Fraction(1, 2): "½", Fraction(1, 3): "⅓", Fraction(2, 3): "⅔",
    Fraction(1, 4): "¼", Fraction(3, 4): "¾", Fraction(1, 6): "⅙",
    Fraction(5, 6): "⅚", Fraction(1, 8): "⅛", Fraction(3, 8): "⅜",
    Fraction(5, 8): "⅝", Fraction(7, 8): "⅞",
}


def frac_label(t):
    """``Fraction`` in [0, 1) -> ITA-style string ('' for 0, '½', '1/12', ...)."""
    t = Fraction(t) % 1
    if t == 0:
        return ""
    return _UNICODE_FRAC.get(t, f"{t.numerator}/{t.denominator}")


def _height_caption(heights):
    """ITA height note for one trace.

    Height 0 is left blank. A lattice copy that only repeats its partner is
    left implicit: 1/2 with 0, and 3/4 with 1/4. A trace that carries 1/4
    and 3/4 is therefore labelled just ``¼``.
    """
    vals = []
    for h in heights:
        f = Fraction(float(h)).limit_denominator(24) % 1
        if f not in vals:
            vals.append(f)
    vals.sort()
    if 0 in vals and Fraction(1, 2) in vals:
        vals = [v for v in vals if v != Fraction(1, 2)]
    if Fraction(1, 4) in vals and Fraction(3, 4) in vals:
        vals = [v for v in vals if v != Fraction(3, 4)]
    nonzero = [v for v in vals if v != 0]
    if not nonzero:
        return ""
    return " ".join(frac_label(v) for v in nonzero)


def height_label(coef, t, coord="z"):
    """ITA height string for an image whose depth is ``coef * coord + t``.

    ``coef = +1`` gives ``'+'``, ``'½+'``, ...; ``coef = -1`` gives ``'−'``,
    ``'½−'`` (i.e. 1/2 - z). When the depth is another coordinate (e.g. after
    a cubic 3-fold) that coordinate is spelled out: ``'½+x'``, ``'−y'``.
    """
    s = frac_label(t) + ("+" if coef > 0 else "−")
    return s if coord == "z" else s + coord


def general_position_images(sg, projection="c"):
    """Exact ITA description of every general-position image.

    For each operation returns ``(coef, t, coord, W, w)``: the projected depth
    of the image of ``(x, y, z)`` is ``coef * coord + t`` with ``coef`` +1/-1,
    ``coord`` the letter ``'z'`` when the operation keeps the projection axis
    (the usual case) or the letter of the coordinate it maps onto the depth
    (cubic 3-folds), and ``t`` an exact ``Fraction`` in [0, 1).
    """
    sg = _resolve_sg(sg)
    perm = _PROJ[projection][0]
    depth = perm[2]
    out = []
    for W, w in _sg_ops_exact(sg):
        row = W[depth]
        t = w[depth] % 1
        nz = [j for j in range(3) if row[j] != 0]
        if len(nz) == 1:
            j = nz[0]
            coord = "z" if j == depth else "xyz"[j]
            out.append((row[j], t, coord, W, w))
        else:
            # depth is a combination of coordinates (e.g. a hexagonal group
            # projected along a or b -- not an ITA plate, but drawable):
            # spell the expression out with coef +1.
            expr = ""
            for j in nz:
                sgn = "+" if row[j] > 0 else "−"
                expr += (sgn if expr or sgn == "−" else "") + "xyz"[j]
            out.append((1, t, expr, W, w))
    return out


# --- ITA graphical symbols -------------------------------------------------
#
# Rotation points (axis perpendicular to the page): filled polygons.
#   2 -> filled lens (pointed oval), 3 -> filled triangle, 4 -> filled square,
#   6 -> filled hexagon.  Screw axes carry "tails"; rotoinversions are open with
#   a small open circle at the inversion point.  In-plane axes are drawn as full
#   or half arrows lying in the page.
#
# Planes perpendicular to the page are drawn as lines:
#   m -> bold solid, glide a/b/c -> dashed, n -> dot-dash, d -> dotted with arrow.
# Inversion centre -> small open circle.

_POLY_SIDES = {2: None, 3: 3, 4: 4, 6: 6}  # 2 handled as lens


def _clip_line_to_box(c, d, lo=0.0, hi=1.0):
    """Clip the infinite line through point c with direction d to the unit box
    [lo,hi]^2. Returns (p0, p1) endpoints on the box boundary."""
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) or 1.0)
    ts = []
    for axis in (0, 1):
        if abs(d[axis]) > 1e-9:
            for bound in (lo, hi):
                t = (bound - c[axis]) / d[axis]
                p = c + t * d
                other = 1 - axis
                if lo - 1e-6 <= p[other] <= hi + 1e-6:
                    ts.append(t)
    if len(ts) < 2:
        return None, None
    return c + min(ts) * d, c + max(ts) * d


def _draw_inplane_arrowhead(ax, tip, d, full=True, size=0.07):
    """ITA in-plane axis arrowhead at ``tip`` pointing along unit vector ``d``.

    full=True -> a solid filled triangular head marking a pure 2-fold rotation;
    full=False -> a half head (one side of the triangle only) marking a 2_1
    screw. Rendered as filled polygons so both read cleanly at cell scale.
    """
    import matplotlib.pyplot as plt
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) or 1.0)
    tip = np.asarray(tip, float)
    perp = np.array([-d[1], d[0]])
    back = tip - d * size
    w = size * 0.5
    if full:
        poly = [tip, back + perp * w, back - perp * w]
    else:
        # half arrow: a single barb on ONE side of the shaft. Apex at the tip,
        # barb corner out to one side, and the third vertex ON the shaft axis
        # at the base -- so the triangle's straight edge lies exactly along the
        # line and no shaft shows past it.
        poly = [tip, back + perp * w, back]
    ax.add_patch(plt.Polygon(poly, closed=True, facecolor="k",
                             edgecolor="k", lw=0.5, zorder=5))


def _line_lattice_segments(c, d):
    """Clipped pieces of the line through ``c`` and its translates by -1, 0, +1.

    ``c`` and ``d`` are fractional (right, down). Each translate is clipped to
    the unit cell. A diagonal through the origin only meets the anti-diagonal
    after a shift by one cell edge; axis-aligned edge lines pick up the
    opposite edge the same way. Degenerate corner touches are dropped, and
    identical clips are returned once.
    """
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    seen = set()
    out = []
    for i in (-1.0, 0.0, 1.0):
        for j in (-1.0, 0.0, 1.0):
            p0, p1 = _clip_line_to_box(c + np.array([i, j]), d)
            if p0 is None or np.linalg.norm(p1 - p0) < 1e-3:
                continue
            key = tuple(sorted((
                tuple(np.round(p0, 4)), tuple(np.round(p1, 4)))))
            if key in seen:
                continue
            seen.add(key)
            out.append((p0, p1))
    return out


def _segment_on_edge(p0, p1, tol=1e-3):
    """True when a fractional segment lies on x=0, x=1, y=0 or y=1."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    for axis in (0, 1):
        if abs(p0[axis] - p1[axis]) < tol and min(abs(p0[axis]), abs(p0[axis] - 1.0)) < tol:
            return True
    return False


def _axis_page_points(el, perm):
    """Projected (right, down) points where an inclined axis meets the cell.

    The stored location is one point of the line. Stepping by 1/order along
    the axis visits the other heights in the same cell, so a 3-fold drawn at
    (1/3, 1/3) also appears at the image (2/3, 2/3).
    """
    loc_e = el.get("location_exact")
    ax_e = el.get("axis_exact")
    if loc_e is None or ax_e is None:
        loc = np.asarray(el["location"], float)
        v = _perm_vec(loc, perm)
        return [((float(v[1] % 1.0), float(v[0] % 1.0)),
                 float(v[2] % 1.0))]
    loc = [Fraction(c) for c in loc_e]
    axis = [int(a) for a in ax_e]
    n = max(int(el.get("order") or 1), 1)
    out = []
    seen = set()
    for j in range(n):
        p = [(loc[i] + Fraction(j, n) * axis[i]) % 1 for i in range(3)]
        v = _perm_vec([float(c) for c in p], perm)
        rd = (float(v[1] % 1.0), float(v[0] % 1.0))
        key = (round(rd[0], 3), round(rd[1], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append((rd, float(v[2] % 1.0)))
    return out


def _glyph_halo():
    """White stroke behind a glyph so a grid line does not eat the ink."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=3.2, foreground="white")]


def _y_increases_down(ax) -> bool:
    y0, y1 = ax.get_ylim()
    return y0 > y1


def _as_on_plate(ax, xy, pt):
    """Mirror a glyph point so an upright axis matches the plate (y down)."""
    pt = np.asarray(pt, float)
    if _y_increases_down(ax):
        return pt
    out = pt.copy()
    out[1] = 2.0 * float(np.asarray(xy, float)[1]) - float(pt[1])
    return out


def _ink(ax, xs, ys, **kw):
    line, = ax.plot(xs, ys, **kw)
    line.set_path_effects(_glyph_halo())
    return line


def _draw_lens(ax, xy, size, angle=0.0, **kw):
    """ITA 2-fold glyph: one filled lens, long axis along x before rotation.

    The outline is the pair of circular arcs that bound the intersection of
    two circles stacked on the y axis. That is a single pointed oval. A polar
    radius ``a + b|cos t|`` instead draws two discs touching at the origin.
    """
    import math
    from matplotlib.patches import Polygon
    half_w = float(size)
    half_h = float(size) * 0.46
    cy = (half_w ** 2 / half_h - half_h) / 2.0
    radius = half_h + cy
    a = math.atan2(cy, half_w)
    upper = np.linspace(a, math.pi - a, 36)
    lower = np.linspace(-a, -(math.pi - a), 36)
    arc_hi = np.column_stack([
        radius * np.cos(upper), -cy + radius * np.sin(upper),
    ])
    arc_lo = np.column_stack([
        radius * np.cos(lower), cy + radius * np.sin(lower),
    ])
    # Top arc right-to-left, then the bottom arc reversed so the outline is
    # one simple loop. Concatenating both arcs in the same direction leaves
    # a diameter chord and a bowtie, which fills as two lobes.
    pts = np.vstack([arc_hi, arc_lo[::-1]])
    ca, sa = math.cos(angle), math.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]])
    pts = pts @ rot.T + np.asarray(xy, float)
    if not _y_increases_down(ax):
        pts = np.column_stack([
            pts[:, 0],
            2.0 * float(xy[1]) - pts[:, 1],
        ])
    patch = Polygon(pts, closed=True, **kw)
    patch.set_path_effects(_glyph_halo())
    ax.add_patch(patch)


def _draw_regular_polygon(ax, xy, n, size, filled=True, **kw):
    """Draw a regular polygon with n vertices centered at xy."""
    from matplotlib.patches import RegularPolygon
    orient = np.pi / n
    if not _y_increases_down(ax):
        orient = -orient
    patch = RegularPolygon(xy, numVertices=n, radius=size,
                           orientation=orient, fill=filled, **kw)
    patch.set_path_effects(_glyph_halo())
    ax.add_patch(patch)


def _draw_screw_tails(ax, xy, order, k, size):
    """ITA screw 'pinwheel' tails.

    Each of the ``order`` arms gets a tail bent tangentially. The bend sense and
    magnitude encode the screw component k: a right-handed screw (k < n/2) bends
    one way, a left-handed one (k > n/2) the mirror, and the neutral screw
    (k == n/2, e.g. 4_2, 6_3, 2_1) draws straight radial tails. The bend angle is
    proportional to (n/2 - k), so 6_1..6_5 are five visually distinct glyphs.
    """
    n = order
    handed = (n / 2.0) - k           # >0 right, <0 left, 0 neutral
    L = size * 1.7
    # The hook DIRECTION (sign of `handed`) encodes handedness and its LENGTH
    # encodes the screw magnitude |handed|. A solid base length keeps even the
    # smallest chiral hook (|handed|=1, e.g. 6_2) clearly visible, while the
    # magnitude term separates 6_1 (|handed|=2) from 6_2 (|handed|=1) and
    # 6_5 from 6_4. k=n/2 (2_1, 4_2, 6_3) has handed=0 -> no hook, plainly
    # distinct even at icon scale.
    for i in range(n):
        a = 2 * np.pi * i / n
        rad = np.array([np.cos(a), np.sin(a)])
        base = _as_on_plate(ax, xy, np.array(xy) + rad * size)
        tip = _as_on_plate(ax, xy, np.array(xy) + rad * (size + L))
        _ink(ax, [base[0], tip[0]], [base[1], tip[1]], color="k",
             lw=1.0, zorder=7.5, solid_capstyle="round")
        if abs(handed) > 1e-6:
            hook_len = L * (0.35 + 0.32 * abs(handed))
            tang = np.array([-rad[1], rad[0]]) * np.sign(handed)
            flag = _as_on_plate(
                ax, xy,
                np.array(xy) + rad * (size + L * 0.78) + tang * hook_len)
        else:
            # 4_2 and 6_3: a short flag, alternating, so the glyph is not a
            # plain polygon and not a chiral screw.
            sign = 1.0 if i % 2 == 0 else -1.0
            tang = np.array([-rad[1], rad[0]]) * sign
            flag = _as_on_plate(
                ax, xy, np.array(xy) + rad * (size + L) + tang * (L * 0.55))
        _ink(ax, [tip[0], flag[0]], [tip[1], flag[1]], color="k",
             lw=1.0, zorder=7.5, solid_capstyle="round")


def draw_axis_symbol(ax, xy, order, screw_k=0, rotoinv=False, size=0.035):
    """Draw a rotation/screw/rotoinversion axis symbol perpendicular to the page
    at fractional position ``xy`` (b-right/a-down mapping done by the caller).

    ``screw_k`` is the full screw index (1..order-1); 0 means a pure rotation.
    The distinct 2_1 / 3_1 / 3_2 / 4_1 / 4_2 / 4_3 / 6_1..6_5 tail conventions
    are produced by :func:`_draw_screw_tails`.
    """
    fc = "white" if rotoinv else "k"
    ec = "k"
    # order-2 screw (2_1) gets tails PERPENDICULAR to the lens long axis so
    # they are visible past the pointed oval; other screws use the pinwheel.
    if screw_k and order != 2:
        _draw_screw_tails(ax, xy, order, screw_k, size)
    if order == 2 and not rotoinv:
        _draw_lens(ax, xy, size * 1.3, fc=fc, ec=ec, lw=1.0, zorder=7.5)
        if screw_k:
            # Hooked tails, not straight ticks. The hook bends back toward
            # the lens so a 2_1 reads as a screw even on a cell edge.
            # Above the plane traces (zorder 6.5) so a vertical glide cannot
            # paint the hook out.
            for s in (1.0, -1.0):
                stem = np.array([xy[0], xy[1] + s * size * 2.15])
                hook = np.array([xy[0] + size * 0.9, xy[1] + s * size * 1.55])
                stem = _as_on_plate(ax, xy, stem)
                hook = _as_on_plate(ax, xy, hook)
                _ink(ax, [xy[0], stem[0], hook[0]],
                     [xy[1], stem[1], hook[1]],
                     color="k", lw=1.2, zorder=7.5, solid_capstyle="round")
    elif order in (3, 4, 6):
        _draw_regular_polygon(ax, xy, order, size, filled=not rotoinv,
                              fc=fc, ec=ec, lw=1.0, zorder=7.5)
    elif rotoinv:
        _draw_regular_polygon(ax, xy, abs(order), size, filled=False,
                              ec=ec, lw=1.0, zorder=7.5)
    # -6 is an open hexagon. The inversion dot belongs on -3 and -4 only.
    if rotoinv and order != 6:
        dot, = ax.plot(xy[0], xy[1], "o", ms=3, mfc="white", mec="k",
                       mew=0.8, zorder=8)
        dot.set_path_effects(_glyph_halo())


def _draw_axis_inversion(ax, xy, radius):
    """Open circle in the middle of a lens: the −1 of a 2/m or 2₁/m symbol."""
    from matplotlib.patches import Circle
    dot = Circle(tuple(xy), radius, facecolor="white", edgecolor="k",
                 lw=0.8, zorder=9)
    ax.add_patch(dot)


def _draw_combined_axis(ax, xy, max_rot, rot_k, roto, size=0.035,
                        inversion=False):
    """Draw ONE ITA glyph for all c-axis axes coincident at ``xy``.

    ``max_rot`` is the highest pure-rotation order present (0 if none),
    ``rot_k`` its screw index, ``roto`` the rotoinversion order (0 if none).
    When a rotation and a rotoinversion share the site (e.g. 4 and -4 in
    4/mmm), the filled rotation glyph is drawn first and the rotoinversion is
    marked by an open square outline + centre dot on top, so both are legible
    rather than one white glyph erasing the other.

    ``inversion`` draws the open centre of 2/m or 2₁/m: a filled lens (hooked
    when the axis is a screw) with a small open circle in the middle.
    """
    if max_rot >= 2:
        draw_axis_symbol(ax, xy, max_rot, screw_k=rot_k, rotoinv=False,
                         size=size)
    elif roto:
        draw_axis_symbol(ax, xy, roto, rotoinv=True, size=size)
    if roto and max_rot >= 2:
        # overlay the rotoinversion marker: open polygon outline + centre dot,
        # slightly larger so it frames the filled rotation glyph
        _draw_regular_polygon(ax, xy, abs(roto), size * 1.35, filled=False,
                              ec="k", lw=1.0, zorder=8)
        if roto != 6:
            dot, = ax.plot(xy[0], xy[1], "o", ms=3, mfc="white", mec="k",
                           mew=0.8, zorder=8.5)
            dot.set_path_effects(_glyph_halo())
    # A −1 on this axis. The rotoinversion glyph already has a centre dot.
    if inversion and not (roto and roto != 6):
        _draw_axis_inversion(ax, xy, size * 0.36)


def draw_parallel_plane_symbol(ax, name, corner=(0.06, 0.06), size=0.11,
                               glide_dir=None):
    """ITA symbol for a plane PARALLEL to the page (normal perpendicular to the
    projection): a right-angle bracket in a cell corner, drawn with SOLID legs.

    Following the ITA convention the leg style does not encode the glide type;
    the ARROW does. A mirror (``m``) has no arrow. A glide carries an arrow
    along the in-plane component of its glide vector:

    - axial glide (a, b, c): a full arrowhead (``-|>``) -- half-lattice glide;
    - diagonal glide (n): a half/open arrowhead (``->``) -- the (a+b)/2-type
      diagonal glide;
    - diamond glide (d): an open arrowhead with a barbed 1/4 tail (``-|>`` on a
      thinner shaft) -- the quarter glide.

    ``glide_dir`` is the projected in-plane glide direction (2-vector) in the
    SAME data frame the caller draws in (x=right, y in the axis' own sense); if
    ``None`` a 45-degree fallback is used. The glyph is rendered so it looks
    identical ON SCREEN whether the axis y runs up (legend) or down (element
    diagram): the corner sits at top-left with legs to the right and downward,
    and the arrow points outward (up / up-and-left) from the corner.

    ``name`` is the plane symbol ('m', 'a', 'b', 'c', 'n', 'd', ...).
    """
    import numpy as _np
    x0, y0 = corner
    # Detect axis y-orientation so the glyph reads the same on screen either
    # way. ys = +1 when data-y increases downward (inverted axis, the element
    # diagram); -1 when it increases upward (normal axis, the legend). "Down on
    # screen" is then y0 + ys*size in data coords.
    ylo, yhi = ax.get_ylim()
    ys = 1.0 if ylo > yhi else -1.0
    # right-angle bracket: corner at top-left on screen, legs right and DOWN.
    ax.plot([x0, x0 + size], [y0, y0], color="k", lw=1.6, zorder=6)
    ax.plot([x0, x0], [y0, y0 + ys * size], color="k", lw=1.6, zorder=6)
    if name == "m":
        return
    # glide_dir is in the caller's data frame (x=right, y in the axis' own
    # sense). On an upright axis the data-y component is flipped so the arrow
    # points the same way on screen as it does on the plate, where y runs down.
    # The sign of the reduced glide is kept.
    if glide_dir is None:
        du = _np.array([1.0, -ys])
    else:
        gd = _np.asarray(glide_dir, float)
        du = _np.array([gd[0], ys * gd[1]])
    nrm = _np.hypot(*du)
    du = du / nrm if nrm > 1e-9 else _np.array([1.0, -ys])
    base = _np.array([x0 + size * 0.25, y0 + ys * size * 0.25])
    tip = base + du * size * 0.85
    astyle = "-|>" if name in ("a", "b", "c", "d") else "->"
    shaft_lw = 1.0 if name != "d" else 0.8
    ax.annotate("", xy=tuple(tip), xytext=tuple(base),
                arrowprops=dict(arrowstyle=astyle, color="k", lw=shaft_lw),
                zorder=6)
    if name == "d":
        # d-glide: quarter-glide tick across the shaft midpoint
        mid = 0.5 * (base + tip)
        perp = _np.array([-du[1], du[0]]) * size * 0.12
        ax.plot([mid[0] - perp[0], mid[0] + perp[0]],
                [mid[1] - perp[1], mid[1] + perp[1]],
                color="k", lw=1.0, zorder=6)


def draw_inversion(ax, xy, size=0.012):
    """Draw an ITA inversion centre symbol (small open circle) at xy."""
    ax.plot(xy[0], xy[1], "o", ms=4, mfc="white", mec="k", mew=1.0, zorder=6)


# glide-plane line styles (plane perpendicular to page -> a line in the page)
_PLANE_STYLE = {
    "m": dict(ls="-", lw=2.2, color="k"),
    "a": dict(ls=(0, (6, 3)), lw=1.3, color="k"),          # dashed
    "b": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
    "c": dict(ls=(0, (1, 1.6)), lw=1.4, color="k"),        # dotted
    "n": dict(ls=(0, (6, 2, 1, 2)), lw=1.3, color="k"),    # dash-dot
    "d": dict(ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.3, color="k"),  # dash-dot-dot
    "e": dict(ls=(0, (4, 1.5, 1, 1.5, 1, 1.5)), lw=1.3, color="k"),
    "g": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
}


def draw_plane_symbol(ax, p0, p1, name, glide_dir=None):
    """Draw a plane (perpendicular to the page) as a styled line from p0 to p1."""
    if abs(p1[0] - p0[0]) < 1e-6 and abs(p1[1] - p0[1]) < 1e-6:
        return                        # degenerate (line only touches a corner)
    style = _PLANE_STYLE.get(name, _PLANE_STYLE["g"])
    # Planes are drawn after in-plane axes and above them, so a glide that
    # shares a trace with an axis keeps its own dash pattern.
    z = 6 if name == "m" else 6.5
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], zorder=z, **style)
    if name == "d":
        # The arrow follows the reduced glide's component along the trace,
        # including its sign, so neighbouring diamond glides point opposite ways.
        import numpy as _np
        p0a, p1a = _np.asarray(p0, float), _np.asarray(p1, float)
        along = p1a - p0a
        L = _np.hypot(*along)
        if L > 1e-6:
            u = along / L
            if glide_dir is not None:
                comp = float(_np.dot(_np.asarray(glide_dir, float), u))
                if abs(comp) > 1e-8:
                    u = u if comp > 0 else -u
            mid = 0.5 * (p0a + p1a)
            ax.annotate("", xy=mid + u * 0.12, xytext=mid - u * 0.12,
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.0),
                        zorder=z + 0.1)


def symbol_legend(ax=None):
    """Render every ITA glyph this module draws, with a label, on one axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_aspect("equal")
    # rotation / screw / rotoinversion axes
    items = [
        ("2", 2, 0, False), ("2\u2081", 2, 1, False),
        ("3", 3, 0, False), ("3\u2081", 3, 1, False), ("3\u2082", 3, 2, False),
        ("4", 4, 0, False), ("4\u2081", 4, 1, False), ("4\u2082", 4, 2, False),
        ("4\u2083", 4, 3, False),
        ("6", 6, 0, False), ("6\u2081", 6, 1, False), ("6\u2082", 6, 2, False),
        ("6\u2083", 6, 3, False), ("6\u2084", 6, 4, False),
        ("6\u2085", 6, 5, False),
        ("-3", 3, 0, True), ("-4", 4, 0, True), ("-6", 6, 0, True),
    ]
    ax.set_xlim(0, 6.4); ax.set_ylim(0, 5.4)
    for i, (lbl, order, k, ro) in enumerate(items):
        x = 0.6 + (i % 6) * 1.0
        y = 4.8 - (i // 6) * 0.95
        draw_axis_symbol(ax, (x, y), order, screw_k=k, rotoinv=ro, size=0.11)
        ax.text(x, y - 0.34, lbl, ha="center", fontsize=8)
    ax.text(0.1, 5.25, "Axes (⊥ page):", fontsize=8, style="italic")
    # inversion centre
    draw_inversion(ax, (0.7, 2.0), size=0.05)
    ax.text(0.7, 1.7, "\u22121", ha="center", fontsize=8)
    ax.text(0.1, 2.35, "Inversion:", fontsize=8, style="italic")
    # planes
    ax.text(0.1, 1.2, "Planes (⊥ page):", fontsize=8, style="italic")
    for i, name in enumerate(["m", "a", "n", "d"]):
        x = 0.6 + i * 1.4
        draw_plane_symbol(ax, (x, 0.55), (x + 1.0, 0.55), name)
        ax.text(x + 0.5, 0.3, name, ha="center", fontsize=8)
    ax.set_title("ITA graphical symbols rendered by agentsg.cell.diagrams",
                 fontsize=9)
    return ax


_BGP_CACHE = {}


def _projection_maps(sg, projection):
    """In-plane affine maps of every operation: image (right, down) = A x + t."""
    perm = _PROJ[projection][0]
    As, ts = [], []
    for W, w, _ in _sg_ops(sg):
        W = np.asarray(W, float)
        w = np.asarray(w, float)
        As.append(np.stack([W[perm[1]], W[perm[0]]]))
        ts.append(np.array([w[perm[1]], w[perm[0]]]))
    return np.stack(As), np.stack(ts)


def _map_signature(A, t):
    return (tuple(np.round(A.ravel(), 6)), tuple(np.round(np.asarray(t) % 1.0, 6)))


def _projected_images(x, As, ts):
    imgs = np.einsum("nij,j->ni", As, np.asarray(x, float)) + ts
    return imgs - np.floor(imgs)


def _min_projected_separation(imgs, same):
    """Smallest torus distance between images that are not forced to coincide."""
    d = imgs[:, None, :] - imgs[None, :, :]
    d = d - np.round(d)
    dist = np.linalg.norm(d, axis=-1)
    dist = np.where(same, np.inf, dist)
    np.fill_diagonal(dist, np.inf)
    if not np.isfinite(dist).any():
        return 1.0  # P1, or every image is a projection duplicate
    return float(dist.min())


def _trace_clearance(imgs, traces):
    """Smallest distance from any projected image to a symmetry-element trace."""
    if not traces or len(imgs) == 0:
        return 1.0
    best = np.full(len(imgs), 1.0)
    for kind, c, n in traces:
        if kind == "point":
            d = imgs - c
            d -= np.round(d)
            best = np.minimum(best, np.linalg.norm(d, axis=1))
        else:
            target = (imgs - c) @ n
            offs = [i * n[0] + j * n[1]
                    for i in range(-2, 3) for j in range(-2, 3)]
            delta = np.abs(target[:, None] - np.asarray(offs)[None, :])
            best = np.minimum(best, delta.min(axis=1))
    return float(best.min())


def _element_traces(sg, projection):
    """Projected traces: points for axes perpendicular to the page, lines otherwise.

    A plane parallel to the page fills the whole drawing, so it is not a trace
    the representative has to avoid.
    """
    perm = _PROJ[projection][0]
    traces = []
    for el in _element_copies(sg):
        loc = np.asarray(el["location"], float)
        axis = el.get("axis")
        p = _perm_vec(loc, perm)
        xy = np.array([p[1] % 1.0, p[0] % 1.0])
        if axis is None:
            traces.append(("point", xy, None))
            continue
        d = _perm_vec(np.asarray(axis, float), perm)
        d2 = np.array([d[1], d[0]], float)
        along = abs(d[2]) > 0.85 * (np.linalg.norm(d) or 1.0)
        if along or np.linalg.norm(d2) < 1e-8:
            # Axis straight out of the page, or a plane parallel to it.
            if el["type"] in ("mirror", "glide") and not along:
                continue
            traces.append(("point", xy, None))
            continue
        n = np.array([-d2[1], d2[0]], float)
        n = n / (np.linalg.norm(n) or 1.0)
        traces.append(("line", xy, n))
    return traces


def _edge_distance(imgs):
    return float(np.minimum(imgs, 1.0 - imgs).min())


def best_general_point(sg, n_grid=5, projection="c"):
    """A general position whose projected orbit stays readable.

    The seed is chosen so that images which are not forced to share a
    projected point stay apart, and so that no image sits on a projected
    symmetry element or on the cell boundary. Pairs that a mirror parallel
    to the page superposes are left coincident: the diagram draws them as
    one split circle.

    Results are cached per space group and projection.
    """
    sg = _resolve_sg(sg)
    ck = (str(_sg_label(sg)), projection, n_grid)
    if ck in _BGP_CACHE:
        return _BGP_CACHE[ck]
    As, ts = _projection_maps(sg, projection)
    sigs = [_map_signature(A, t) for A, t in zip(As, ts)]
    same = np.array([[a == b for b in sigs] for a in sigs], dtype=bool)
    traces = _element_traces(sg, projection)

    def quality(x):
        imgs = _projected_images(x, As, ts)
        sep = _min_projected_separation(imgs, same)
        edge = _edge_distance(imgs)
        elem = _trace_clearance(imgs, traces)
        # Distance to the nearest quarter-cell line (1/4, 1/2, 3/4). A seed
        # on that grid (P222 at 1/4, 1/4) is legal but not generic.
        quarters = np.array([0.25, 0.5, 0.75])
        off_q = float(np.abs(imgs[..., None] - quarters).min())
        # A seed with equal in-plane coordinates (P222 at x = y) draws a
        # square. Prefer a rectangular orbit, the way P2_12_12 already does.
        u, v = imgs[:, 0], imgs[:, 1]
        d_eq = np.minimum(np.abs(u - v), 1.0 - np.abs(u - v))
        d_neg = np.minimum(np.abs(u + v), np.abs(u + v - 1.0))
        off_sq = float(np.minimum(d_eq, d_neg).min())
        return (sep >= 0.02, min(elem, 0.04), min(edge, 0.04),
                off_q >= 0.04, min(off_sq, 0.08), min(edge, 0.12), sep)

    candidates = [(0.11, 0.19, 0.07), (0.13, 0.17, 0.23),
                  (0.07, 0.31, 0.13), (0.19, 0.11, 0.29),
                  (0.23, 0.37, 0.11)]
    grid = np.linspace(0.08, 0.42, n_grid)
    for xi in grid:
        for yi in grid:
            for zi in grid:
                candidates.append((float(xi), float(yi), float(zi)))
    best = np.array(candidates[0], float)
    best_q = (-1.0, -1.0, -1.0, False, -1.0, -1.0, -1.0)
    for c in candidates:
        q = quality(c)
        if q > best_q:
            best_q = q
            best = np.array(c, float)
    # A short walk off the current point, in case it still clips a trace.
    step = 0.02
    for _ in range(24):
        improved = False
        for delta in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                      (0, 0, step), (0, 0, -step)):
            cand = (best + delta) % 1.0
            if np.any(cand < 0.04) or np.any(cand > 0.96):
                continue
            q = quality(cand)
            if q > best_q:
                best_q = q
                best = cand
                improved = True
        if not improved:
            break
    result = tuple(np.round(np.asarray(best, float) % 1.0, 4))
    _BGP_CACHE[ck] = result
    return result


def _draw_split_circle(ax, center, members, radius=0.042):
    """One circle split by a vertical diameter, for two heights at one spot.

    The positive-sense label sits to the left of the circle and the other to
    the right. A comma, marking the opposite hand, leads the right-hand label.
    """
    from matplotlib.patches import Circle
    cx, cy = center
    ax.add_patch(Circle((cx, cy), radius, facecolor="white", edgecolor="k",
                        lw=1.0, zorder=3))
    ax.plot([cx, cx], [cy - radius, cy + radius], color="k", lw=0.7, zorder=4)
    left, right = members
    if right[0][1] > left[0][1]:
        left, right = right, left
    right_txt = right[1]
    if left[2] < 0 or right[2] < 0:
        right_txt = "," + right_txt
    ax.text(cx - radius - 0.012, cy, left[1], fontsize=6, ha="right",
            va="center", zorder=5)
    ax.text(cx + radius + 0.012, cy, right_txt, fontsize=6, ha="left",
            va="center", zorder=5)


def general_position_diagram(sg, ax=None, point=None,
                             show_title=True, projection="c"):
    """Draw the ITA general-position (equivalent-points) diagram.

    Parameters
    ----------
    sg : SpaceGroup | int | str
        A space group, its number (1..230), or an HM/Hall symbol.
    ax : matplotlib Axes, optional
        Target axes; a new figure+axes is created if omitted.
    point : (x, y, z), optional
        The general position to replicate. Defaults to
        :func:`best_general_point` -- the centre of the largest sphere
        inscribed in the asymmetric unit, which maximises separation of the
        equivalent points so overlaps occur only where symmetry requires them.
    show_title : bool
        Whether to stamp the "#N  HM" header.
    projection : {'c', 'a', 'b'}
        Which axis points out of the page (default 'c', the standard ITA view).

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    sg = _resolve_sg(sg)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    if point is None:
        point = best_general_point(sg, projection=projection)

    perm, dlab, rlab, _ = _PROJ[projection]
    frame = _Frame(cell_frame(sg, projection))
    x0 = np.array(point, dtype=float)

    frame.draw_cell(ax)

    # collect distinct points first, then group by (x,y) so coincident
    # projections (different heights) can be drawn as ITA split circles.
    # The height label is exact: coef * z + t read off the operation.
    seen = set()
    pts = []   # (right, down, label, sortkey, det)
    for coef, t, coord, Wi, wi in general_position_images(sg, projection):
        W = np.array(Wi, float)
        w = np.array([float(x) for x in wi])
        det = round(np.linalg.det(W))
        base = _perm_vec(W @ x0 + w, perm)  # [0]=down, [1]=right, [2]=depth
        label = height_label(coef, t, coord)
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                p = base + np.array([tx, ty, 0.0])
                if not (-0.03 <= p[0] <= 1.03 and -0.03 <= p[1] <= 1.03):
                    continue
                key = (round(p[0], 3), round(p[1], 3), label, det)
                if key in seen:
                    continue
                seen.add(key)
                pts.append((p[1], p[0], label, (float(t), coef, coord), det))

    # group by shared projected position
    groups = {}
    for px, py, label, sk, det in pts:
        gk = (round(px, 3), round(py, 3))
        groups.setdefault(gk, []).append((sk, label, det))

    for (px, py), members in groups.items():
        members = sorted(members)
        cx, cy = frame.pt((px, py))
        if len(members) == 2:
            # ITA split circle: one disk, vertical diameter, the two heights
            # in the two halves, comma in the right half.
            _draw_split_circle(ax, (cx, cy), members)
            continue
        m = len(members)
        span = 0.05 * (m - 1)
        for j, (_, label, det) in enumerate(members):
            dv = -span / 2 + j * 0.05 if m > 1 else 0.0
            ox, oy = frame.pt((px + dv, py))
            ax.plot(ox, oy, "o", ms=9, mfc="white", mec="k", mew=1.0,
                    zorder=3)
            ax.text(ox + 0.03, oy - 0.03, label,
                    fontsize=6, zorder=4, ha="left", va="center")
            if det < 0:
                ax.text(ox, oy, ",", fontsize=8, zorder=4,
                        ha="center", va="center")

    # a down, b right, origin top-left
    xl, yl = frame.limits(0.12)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.axis("off")
    frame.label_axes(ax, dlab, rlab)
    if show_title:
        num, name = _sg_label(sg)
        pfx = f"#{num}  " if num is not None else ""
        ax.set_title(f"{pfx}{name}", fontsize=8)
    return ax


def _hexagonal_family(sg):
    """Trigonal and hexagonal groups use the glide letter g, not n."""
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    return str(system or "").lower() in ("trigonal", "hexagonal")


def _element_lattice(sg):
    from .symmetry_elements import Lattice, translation_lattice
    return Lattice(translation_lattice(_centring_translations(sg)))


def classify_element(W, w, lattice=None, hexagonal=False):
    """Classify a symmetry operation (W, w) as an ITA symmetry element.

    See :func:`agentsg.cell.symmetry_elements.classify_element`. ``lattice``
    defaults to the primitive translations; pass the centring lattice when the
    operation came from a centred group.
    """
    from .symmetry_elements import classify_element as _classify
    return _classify(W, w, lattice=lattice, hexagonal=hexagonal)


def classify_space_group(sg):
    """Classify every operation of a space group. Returns a list of element
    dicts (see :func:`classify_element`), skipping the identity."""
    from .symmetry_elements import classify_element as _classify
    sg = _resolve_sg(sg)
    lattice = _element_lattice(sg)
    hexagonal = _hexagonal_family(sg)
    out = []
    for W, w, xyz in _sg_ops(sg):
        el = _classify(W, w, lattice=lattice, hexagonal=hexagonal)
        el["xyz"] = xyz
        if el["type"] != "identity":
            out.append(el)
    return out


def _dir_class(vec, tol=0.2):
    """Classify a direction as 'c' (perpendicular to the page), 'ab' (in the
    page), or 'gen' (inclined)."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    v = v / n
    if abs(v[2]) > 1 - tol:
        return "c"
    if abs(v[2]) < tol:
        return "ab"
    return "gen"


def _distinct_glide_axes(raws):
    """True when one plane carries two axial half-glides along different axes.

    That pair is an e glide. A hexagonal g, whose translation has two
    nonzero components, is not one of those halves, so it stays g.
    """
    axes = set()
    half = Fraction(1, 2)
    for raw in raws:
        if not raw:
            continue
        nonzero = [i for i, comp in enumerate(raw) if comp != 0]
        if len(nonzero) == 1 and abs(raw[nonzero[0]]) == half:
            axes.add(nonzero[0])
    return len(axes) >= 2


def _axis_sign_key(axis):
    if axis is None:
        return None
    signed = list(axis)
    for comp in signed:
        if comp:
            if comp < 0:
                signed = [-c for c in signed]
            break
    return tuple(signed)


def _mark_contained_axes(elements):
    """Mark the 3-fold and 2-fold that sit inside a 6-fold at the same locus.

    They stay in the element list (the 3_1 inside 6_1 distinguishes the
    enantiomorph) and are tagged ``contained_in`` so the plate draws the
    6-fold glyph once.
    """
    groups = {}
    for el in elements:
        if el["type"] not in ("rotation", "screw"):
            continue
        key = (_axis_sign_key(el.get("axis_exact")), el.get("location_exact"))
        groups.setdefault(key, []).append(el)
    for group in groups.values():
        host = next((el for el in group if el["order"] == 6), None)
        if host is None:
            continue
        for el in group:
            if el["order"] in (2, 3):
                el["contained_in"] = host["symbol"]
    return elements


def _element_copies(sg):
    """Enumerate every symmetry element across the full cell.

    Combining an operation with a lattice translation relocates it. The
    intrinsic part is reduced modulo the full lattice, including centring, and
    each line or plane is kept once, at the point of the locus nearest the
    origin.
    """
    from .symmetry_elements import classify_element as _classify
    sg = _resolve_sg(sg)
    lattice = _element_lattice(sg)
    hexagonal = _hexagonal_family(sg)
    ops = _sg_ops(sg)
    # operations() already includes one copy of each centring translate, so the
    # integer window below is enough to reach every parallel element.
    seen = set()
    out = []
    planes = {}
    Ls = [(i, j, k)
          for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for W, w, _ in ops:
        w0 = tuple(Fraction(x).limit_denominator(10_000) for x in w)
        for L in Ls:
            shift = (w0[0] + L[0], w0[1] + L[1], w0[2] + L[2])
            el = _classify(W, shift, lattice=lattice, hexagonal=hexagonal)
            if el["type"] in ("identity", "translation"):
                continue
            loc = el.get("location_exact")
            if loc is None:
                continue
            axis = el.get("axis_exact")
            if el["type"] == "glide":
                raw = el.get("intrinsic_raw_exact")
                prev = planes.get((axis, loc))
                if prev is not None:
                    raws = prev["_glide_raws"]
                    if raw is not None and raw not in raws:
                        raws.append(raw)
                        if _distinct_glide_axes(raws):
                            prev["symbol"] = "e"
                            prev["type"] = "glide"
                    continue
                el["_glide_raws"] = [] if raw is None else [raw]
                planes[(axis, loc)] = el
            key = (el["type"], el["symbol"], axis, loc)
            if key in seen:
                continue
            seen.add(key)
            out.append(el)
    return _mark_contained_axes(out)




def _centring_translations(sg):
    """Centring translations (incl. origin), derived from the operation set.

    A pure lattice translation appears as an operation with identity rotation
    and a non-integer translation. Reading them from ``operations()`` works for
    both a standard SpaceGroup and a non-standard SpaceGroupSetting (where a
    det!=1 change of basis surfaces new centring vectors), so we do not rely on
    the Hermann-Mauguin lattice letter."""
    cvs = [np.zeros(3)]
    seen = {(0.0, 0.0, 0.0)}
    for W, w, _ in _sg_ops(sg):
        if np.allclose(W, np.eye(3)):
            v = np.array(w, float) % 1.0
            key = tuple(np.round(v, 3))
            if key not in seen:
                seen.add(key)
                cvs.append(v)
    return cvs


def _draw_centring_markers(ax, sg, perm, frame=None):
    """Explicitly indicate pure lattice (centring) translations.

    Standard ITA does not give a pure lattice translation its own glyph -- the
    centring is implied by the lattice letter and by every element being
    repeated at the centring-shifted position. For non-standard settings (where
    a det!=1 change of basis surfaces centring that the symbol does not name)
    an explicit marker is clearer: each non-trivial centring lattice node is
    marked with a red dot and its fractional coordinates.
    """
    cvs = _centring_translations(sg)
    n = 0
    for v in cvs:
        if np.allclose(np.asarray(v) % 1.0, 0):
            continue
        vp = _perm_vec(v, perm)             # [0]=down(a'), [1]=right(b')
        x, y = vp[1] % 1.0, vp[0] % 1.0
        if frame is not None:
            x, y = frame.pt((x, y))
        # red dot at the centring node, on top of whatever glyph sits there
        ax.plot(x, y, "o", ms=6, mfc="red", mec="red", zorder=7)
        # label the fractional vector
        frac = "(" + ",".join(_frac_str(c) for c in v) + ")"
        ax.text(x + 0.03, y + 0.05, frac, fontsize=6, color="red",
                ha="left", va="top", zorder=7)
        n += 1
    return n


def _frac_str(x, tol=1e-3):
    """Short fraction string for a small rational-ish float (0, 1/2, 1/3...)."""
    for den in (1, 2, 3, 4, 6):
        num = round(x * den)
        if abs(x - num / den) < tol:
            if num == 0:
                return "0"
            return f"{num}" if den == 1 else f"{num}/{den}"
    return f"{x:.2f}"


def symmetry_element_diagram(sg, ax=None, show_title=True, projection="c",
                             full_cell=True, show_general_positions=False,
                             show_centring=False):
    """Draw the ITA symmetry-element diagram.

    Parameters
    ----------
    sg : SpaceGroup | int | str
    ax : matplotlib Axes, optional
    show_title : bool
    projection : {'c', 'a', 'b'}
        Axis pointing out of the page (default 'c').
    full_cell : bool
        If True (default) draw every element copy across the whole cell
        (parallel axes at 0 and 1/2, etc.); if False draw one representative
        per distinct operation.
    show_general_positions : bool
        Overlay the general-position points (grey) on the element diagram. Off
        by default -- the ITA keeps the two diagrams separate because the
        overlay is busy for high-symmetry groups.
    show_centring : bool
        Explicitly mark pure lattice (centring) translations with a labelled
        vector from the origin and an open square at the centring node. Off by
        default (matching the ITA, which leaves centring implicit); most useful
        for non-standard settings where a det!=1 change of basis surfaces
        centring the symbol does not name.

    Axes ⊥ page are point glyphs; two-folds are arrowed lines along their
    *projected* trace (in-plane axes and body-diagonal axes after an F/I/R
    primitive CoB). Planes ⊥ page are styled lines; inversion centres are
    small open circles. Screw and glide elements already encode their
    translation (screw tails / dashed glide lines); pure lattice translations
    are shown only with show_centring. Cubic 3-folds are drawn as inclined
    triangles where they meet the page. In-plane 4-folds use the 4-fold
    glyph. Inclined 2-folds use a small lens along the projected axis.
    """
    import matplotlib.pyplot as plt

    sg = _resolve_sg(sg)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    perm, dlab, rlab, _ = _PROJ[projection]
    frame = _Frame(cell_frame(sg, projection))
    frame.draw_cell(ax)

    if show_centring:
        _draw_centring_markers(ax, sg, perm, frame)

    if show_general_positions:
        general_position_diagram(sg, ax=ax, show_title=False,
                                 projection=projection)
        # dim the overlaid circles
        for ln in ax.lines:
            ln.set_alpha(0.25)

    els = _element_copies(sg) if full_cell else classify_space_group(sg)
    els = [el for el in els if "contained_in" not in el]
    omitted = 0

    # Establish the inverted y-axis (a' down the page, ITA convention) BEFORE
    # drawing, so orientation-aware glyphs (the parallel-plane corner bracket)
    # detect the correct screen sense at draw time.
    xl, yl = frame.limits(0.24)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    def P(fr):
        """fractional 3-vector -> fractional (right, down) in the cell."""
        v = _perm_vec(fr, perm)  # [0]=down(a'), [1]=right(b')
        return (v[1] % 1.0, v[0] % 1.0)

    def dcls(axis):
        """Classify direction of axis after permutation."""
        return _dir_class(_perm_vec(axis, perm)) if axis is not None else None

    def frac_dir(axis):
        """fractional 3-vector -> fractional in-plane direction (right, down)."""
        v = _perm_vec(axis, perm)
        return np.array([v[1], v[0]])

    def inplane_fixed_dir(W):
        """Direction (right, down), in fractional coordinates, of the line in
        which a plane perpendicular to the page cuts the page: the in-plane
        +1 eigenvector of W. Taking the perpendicular of the plotted normal
        is only right for an orthogonal frame; this is right for all."""
        A = _perm_mat(W, perm)[:2, :2] - np.eye(2)   # acts on (down, right)
        _, s, vt = np.linalg.svd(A)
        v = vt[-1]                                     # null vector (down,right)
        return np.array([v[1], v[0]])

    def _trace_id(el):
        loc = np.asarray(el["location"], float)
        xy = np.array(P(loc), float)
        axis = el.get("axis")
        if axis is None:
            return ("point", tuple(np.round(xy, 3)))
        d = np.asarray(frac_dir(axis), float)
        nrm = np.linalg.norm(d)
        if el.get("type") in ("mirror", "glide") and nrm < 1e-9:
            # Normal along the projection axis: the plane is parallel to the
            # page. Its height belongs on the corner bracket, not on whatever
            # point symbol happens to share the plane's canonical location.
            depth = round(float(loc[perm[2]]) % 1.0, 3)
            return ("parallel", el.get("symbol"), depth)
        if nrm < 1e-9:
            return ("point", tuple(np.round(xy, 3)))
        d = d / nrm
        if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        if el.get("type") in ("mirror", "glide"):
            # ``axis`` is the plane normal, so the offset is along it.
            off = round(float(np.dot(xy, d)), 3)
            return ("plane", tuple(np.round(d, 3)), off)
        normal = np.array([-d[1], d[0]])
        off = round(float(np.dot(xy, normal)), 3)
        return ("line", tuple(np.round(d, 3)), off)

    heights = {}
    for el in els:
        depth = float(np.asarray(el["location"], float)[perm[2]]) % 1.0
        heights.setdefault(_trace_id(el), []).append(depth)
    captions = {key: _height_caption(vals) for key, vals in heights.items()}
    labelled = set()

    def edge_copies(xy, tol=1e-3):
        """Replicate a point glyph onto the opposite edge/corner: a glyph on
        x=0 also belongs at x=1, on y=0 also at y=1, and the origin at all
        four corners -- the standard ITA boundary duplication."""
        x, y = xy
        xs = [x] + ([1.0] if abs(x) < tol else
                    ([0.0] if abs(x - 1.0) < tol else []))
        ys = [y] + ([1.0] if abs(y) < tol else
                    ([0.0] if abs(y - 1.0) < tol else []))
        return [frame.pt((cx, cy)) for cx in xs for cy in ys]

    def draw_line_family(c, d, draw):
        """Draw the line through ``c`` and its translates by one cell along a and b.

        Each piece is clipped to the unit cell. ``draw`` receives plot-space
        endpoints and the plot-space direction.
        """
        for p0, p1 in _line_lattice_segments(c, d):
            draw(np.array(frame.pt(p0)), np.array(frame.pt(p1)),
                 frame.vec(d))

    # Pre-pass: collect c-axis rotation/rotoinversion elements by projected
    # site so coincident axes (e.g. 4, -4 and 2 all at the origin in 4/mmm)
    # become ONE combined ITA glyph instead of overdrawing each other (a white
    # -4 square painted over a black 4 square leaves only an outline).
    c_sites = {}
    for el in els:
        if el["type"] in ("rotation", "screw", "rotoinversion") \
                and el["axis"] is not None and dcls(el["axis"]) == "c":
            key = tuple(np.round(P(el["location"]), 3))
            s = c_sites.setdefault(key, {"max_rot": 0, "rot_k": 0,
                                         "roto": 0, "inv": False})
            if el["type"] == "rotoinversion":
                s["roto"] = max(s["roto"], el["order"])
            else:
                k = 0
                if el["type"] == "screw" and "_" in el["symbol"]:
                    k = int(el["symbol"].split("_")[1])
                if el["order"] > s["max_rot"]:
                    s["max_rot"] = el["order"]
                    s["rot_k"] = k
    for el in els:
        if el["type"] == "inversion":
            key = tuple(np.round(P(el["location"]), 3))
            if key in c_sites:
                c_sites[key]["inv"] = True
    inplane_axes = {}
    inclined2 = []
    inclined3 = []
    inplane_higher = []

    parallel_planes_drawn = set()   # planes parallel to the page (corner glyph)
    for el in els:
        t = el["type"]
        loc = el["location"]
        if t == "inversion":
            key = tuple(np.round(P(loc), 3))
            # A −1 on a perpendicular axis is the open circle of 2/m or 2₁/m.
            if key in c_sites and c_sites[key].get("inv"):
                continue
            cap = captions.get(_trace_id(el), "")
            for xy in edge_copies(P(loc)):
                draw_inversion(ax, xy)
                if cap and _trace_id(el) not in labelled:
                    labelled.add(_trace_id(el))
                    ax.text(xy[0] + 0.04, xy[1] - 0.03, cap, fontsize=6,
                            zorder=8, ha="left", va="top")
            continue
        if t in ("rotation", "screw", "rotoinversion"):
            dc = dcls(el["axis"])
            if dc == "c":
                continue   # handled by the combined-glyph pass after the lines
            d = frac_dir(el["axis"])
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-9:
                omitted += 1
                continue
            d = d / nrm
            if el["order"] == 2 and dc == "ab":
                slot = inplane_axes.setdefault(_trace_id(el), {
                    "full": False, "half": False, "loc": P(loc), "d": d,
                    "el": el,
                })
                if t == "rotation":
                    slot["full"] = True
                else:
                    slot["half"] = True
            elif el["order"] == 2 and dc == "gen":
                inclined2.append(el)
            elif el["order"] == 3 and dc == "gen":
                inclined3.append(el)
            elif el["order"] in (4, 6) and dc == "ab":
                inplane_higher.append(el)
            elif t == "rotoinversion" and el["order"] == 3 and dc == "gen":
                inclined3.append(el)
            else:
                omitted += 1
            continue
        if t in ("mirror", "glide"):
            dc = dcls(el["axis"])
            if dc == "ab":
                d = inplane_fixed_dir(el["W"])
                d = d / (np.linalg.norm(d) or 1.0)
                sym = el["symbol"]
                intr = el.get("intrinsic_exact")
                if sym == "g" and intr is not None:
                    depth = abs(float(intr[perm[2]])) % 1.0
                    if 0.05 < depth < 0.95:
                        sym = "n"
                gdir = None
                if sym == "d":
                    raw = el.get("intrinsic")
                    if raw is not None:
                        gdir = frame.vec(frac_dir(np.asarray(raw, float)))
                draw_line_family(
                    P(loc), d,
                    lambda p0, p1, _dp, _s=sym, _g=gdir:
                        draw_plane_symbol(ax, p0, p1, _s, glide_dir=_g))
                cap = captions.get(_trace_id(el), "")
                if cap and _trace_id(el) not in labelled:
                    labelled.add(_trace_id(el))
                    xy = frame.pt(P(loc))
                    ax.text(xy[0] + 0.04, xy[1] - 0.04, cap, fontsize=6,
                            zorder=8, ha="left", va="top")
            elif dc == "c":
                # plane PARALLEL to the page (normal along the projection axis):
                # ITA draws a right-angle bracket in a cell corner, once per
                # distinct plane symbol. The glide type is read off the arrow
                # (the in-plane projection of the glide vector), not the legs.
                if el["symbol"] not in parallel_planes_drawn:
                    slot = len(parallel_planes_drawn)
                    gdir = None
                    intr = el.get("intrinsic")
                    if intr is not None:
                        # frame.vec returns a plot-space direction in the SAME data frame
                        # the symbol draws in (y increases downward), so pass it
                        # straight through -- no sign flip.
                        gdir = frame.vec(_cell_edge_sense(
                            frac_dir(np.asarray(intr, float))))
                    # Low in the cell, clear of the origin, the 2₁ on y=0,
                    # and the lens near (1/2, 1/4).
                    corner = (0.10 + 0.36 * slot, 0.68)
                    draw_parallel_plane_symbol(
                        ax, el["symbol"],
                        corner=corner,
                        size=0.16, glide_dir=gdir)
                    parallel_planes_drawn.add(el["symbol"])
                    cap = _height_caption(
                        float(np.asarray(other["location"], float)[perm[2]]) % 1.0
                        for other in els
                        if other["type"] in ("mirror", "glide")
                        and other["symbol"] == el["symbol"]
                        and dcls(other["axis"]) == "c")
                    if cap:
                        ax.text(corner[0] + 0.18, corner[1], cap, fontsize=6,
                                zorder=8, ha="left", va="center")
            else:
                omitted += 1
            continue

    def draw_axis_line(p0, p1, full, half, key, on_edge):
        """Arrowheads at both ends of an in-plane axis.

        An axis that lies on the cell edge is a short stub just outside the
        frame, so it clears the corner 2-fold. A diagonal or hexagonal axis
        is the whole clipped trace, so its direction stays visible, with the
        heads just past the boundary.
        """
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if on_edge:
            head_size = 0.055
            shaft_len = 0.03
            clear = 0.06
            mid = 0.5 * (p0 + p1)
            for end in (p0, p1):
                outward = end - mid
                nrm = np.linalg.norm(outward) or 1.0
                outward = outward / nrm
                tip = end + outward * (clear + shaft_len + head_size)
                if full and half:
                    _draw_inplane_arrowhead(
                        ax, tip, outward, full=True, size=head_size)
                    _draw_inplane_arrowhead(
                        ax, tip + outward * (head_size * 0.55), outward,
                        full=False, size=head_size)
                else:
                    _draw_inplane_arrowhead(
                        ax, tip, outward, full=bool(full), size=head_size)
                shaft_end = tip - outward * head_size
                shaft_in = shaft_end - outward * shaft_len
                ax.plot([shaft_in[0], shaft_end[0]],
                        [shaft_in[1], shaft_end[1]],
                        color="k", lw=1.2, zorder=3)
        else:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color="k", lw=1.2, zorder=3)
            head_size = 0.045
            for end, other in ((p0, p1), (p1, p0)):
                outward = end - other
                nrm = np.linalg.norm(outward) or 1.0
                outward = outward / nrm
                tip = end + outward * 0.012
                if full and half:
                    perp = np.array([-outward[1], outward[0]])
                    _draw_inplane_arrowhead(
                        ax, tip + perp * 0.02, outward, full=True, size=head_size)
                    _draw_inplane_arrowhead(
                        ax, tip - perp * 0.02, outward, full=False, size=head_size)
                else:
                    _draw_inplane_arrowhead(
                        ax, tip, outward, full=bool(full), size=head_size)
        cap = captions.get(key, "")
        if cap and key not in labelled:
            labelled.add(key)
            midpt = 0.5 * (p0 + p1)
            ax.text(midpt[0] + 0.04, midpt[1] + 0.03, cap,
                    fontsize=7, color="k", zorder=8, ha="left", va="center")

    for key, slot in inplane_axes.items():
        direction = np.asarray(slot["d"], float)
        for fp0, fp1 in _line_lattice_segments(slot["loc"], direction):
            # A trace on the cell edge stays a stub outside the frame. Any
            # trace that crosses the interior is the clipped line itself.
            draw_axis_line(
                frame.pt(fp0), frame.pt(fp1),
                slot["full"], slot["half"], key,
                _segment_on_edge(fp0, fp1))

    def _screw_k(el):
        if el["type"] == "screw" and "_" in el["symbol"]:
            return int(el["symbol"].split("_")[1])
        return 0

    def _draw_inclined_family(elements, kind):
        """One glyph at every in-cell projection of each inclined axis.

        The stored location is a single reference point. The other heights
        of the same axis land on the 2-fold images of that point, so a symbol
        at (1/3, 1/3) is repeated at (2/3, 2/3). Distinct symbols that share a
        projected point are stepped apart, and the heights are printed once.
        """
        sites = {}
        for el in elements:
            k = _screw_k(el)
            roto = el["type"] == "rotoinversion"
            dvec = frac_dir(el["axis"])
            dvec = dvec / (np.linalg.norm(dvec) or 1.0)
            direction = tuple(np.round(frame.vec(dvec), 3))
            for rd, depth in _axis_page_points(el, perm):
                key = (round(rd[0], 3), round(rd[1], 3))
                slot = sites.setdefault(
                    key, {"depths": [], "glyphs": [], "rd": rd})
                slot["depths"].append(depth)
                # A triangle does not show the axis direction, so the eight
                # <111> axes that share a projected point are one glyph.
                # A lens is rotated onto its axis, so the direction stays.
                mark = (k, roto) if kind != "lens" else (k, roto, direction)
                if mark not in slot["glyphs"]:
                    slot["glyphs"].append(mark)
        for key, slot in sites.items():
            cap = _height_caption(slot["depths"])
            glyphs = slot["glyphs"]
            n = len(glyphs)
            for xy in edge_copies(slot["rd"]):
                for i, mark in enumerate(glyphs):
                    shift = (i - (n - 1) / 2.0) * 0.05
                    pos = (xy[0] + shift, xy[1])
                    if kind == "lens":
                        k, roto, direction = mark
                        direction = np.asarray(direction, float)
                        ang = math.atan2(float(direction[1]), float(direction[0]))
                        _draw_lens(ax, pos, 0.028, angle=ang, fc="k", ec="k",
                                   lw=0.8, zorder=5)
                        if k:
                            tang = np.array([-math.sin(ang), math.cos(ang)])
                            tip = np.asarray(pos, float) + direction * 0.05
                            hook = tip + tang * 0.03
                            _ink(ax, [pos[0], tip[0], hook[0]],
                                 [pos[1], tip[1], hook[1]],
                                 color="k", lw=1.0, zorder=5)
                    else:
                        k, roto = mark
                        draw_axis_symbol(
                            ax, pos, 3, screw_k=k, rotoinv=roto, size=0.028)
                if cap and ("inclined", key) not in labelled:
                    labelled.add(("inclined", key))
                    ax.text(xy[0] + 0.05, xy[1] + 0.03, cap, fontsize=7,
                            zorder=8, ha="left", va="center")

    _draw_inclined_family(inclined2, "lens")
    _draw_inclined_family(inclined3, "tri")

    for el in inplane_higher:
        for xy in edge_copies(P(el["location"])):
            draw_axis_symbol(
                ax, xy, el["order"], screw_k=_screw_k(el),
                rotoinv=(el["type"] == "rotoinversion"), size=0.03)

    for key, s in c_sites.items():
        cap = captions.get(("point", tuple(np.round(key, 3))), "")
        for xy in edge_copies(key):
            _draw_combined_axis(ax, xy, s["max_rot"], s["rot_k"], s["roto"],
                                inversion=s.get("inv", False))
            if cap and ("point", tuple(np.round(key, 3))) not in labelled:
                labelled.add(("point", tuple(np.round(key, 3))))
                ax.text(xy[0] + 0.05, xy[1] + 0.02, cap, fontsize=7,
                        zorder=8, ha="left", va="center")

    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal"); ax.axis("off")
    frame.label_axes(ax, dlab, rlab, off=0.1)
    if show_title:
        extra = f"  (+{omitted} oblique)" if omitted else ""
        num, name = _sg_label(sg)
        pfx = f"#{num}  " if num is not None else ""
        ax.set_title(f"{pfx}{name}{extra}", fontsize=8)
    return ax


def element_legend(sg, ax=None, projection="c"):
    """Legend of only the symmetry elements that actually occur in ``sg``.

    Unlike :func:`symbol_legend` (which shows the full glyph alphabet), this
    inspects the group and draws just the glyphs present: the axes ⊥ page, the
    in-plane axes (with the ITA full-head=rotation / half-head=screw
    distinction), the plane types, and the inversion centre -- plus the height
    and handedness conventions for the general-position points. Works for a
    SpaceGroup or a SpaceGroupSetting.
    """
    import matplotlib.pyplot as plt
    sg = _resolve_sg(sg)
    if ax is None:
        _, ax = plt.subplots(figsize=(2.8, 3.4))
    ax.set_xlim(0, 4.3)
    ax.axis("off")

    # Inventory from the SAME full-cell reconstruction the element diagram
    # draws, so lattice-generated screws (e.g. the diagonal 2_1 in a symmorphic
    # tetragonal group) are listed rather than only the base coset set.
    els = [el for el in _element_copies(sg) if "contained_in" not in el]
    perm = _PROJ[projection][0]
    frame = _Frame(cell_frame(sg, projection))
    perp = {}      # order -> set of screw_k for axes ⊥ page
    inplane = {"rot": False, "screw": False, "both": False}
    inclined_2 = {"rot": False, "screw": False}
    inclined_3 = set()
    inplane_n = set()
    planes = set()          # planes ⊥ page (drawn as lines)
    par_planes = {}         # planes ∥ page -> projected glide dir (bracket)
    has_inv = False
    two_xy = set()
    inv_xy = set()

    def _proj_xy(el):
        loc = el.get("location")
        if loc is None:
            return None
        v = _perm_vec(np.asarray(loc, float), perm)
        return (round(float(v[1]) % 1.0, 3), round(float(v[0]) % 1.0, 3))

    for el in els:
        t, sym, axis = el["type"], el["symbol"], el["axis"]
        # classify relative to the projection axis (permute first), so the
        # legend matches what the element diagram draws for that projection.
        dc = _dir_class(_perm_vec(axis, perm)) if axis is not None else None
        if t == "inversion":
            has_inv = True
            xy = _proj_xy(el)
            if xy is not None:
                inv_xy.add(xy)
        elif t in ("rotation", "screw", "rotoinversion"):
            k = 0
            if t == "screw" and "_" in sym:
                k = int(sym.split("_")[1])
            if dc == "c":
                perp.setdefault(el["order"], set()).add(
                    (k, t == "rotoinversion"))
                if t == "rotation" and el["order"] == 2:
                    xy = _proj_xy(el)
                    if xy is not None:
                        two_xy.add(xy)
            elif el["order"] == 2 and dc == "ab":
                if t == "rotation":
                    inplane["rot"] = True
                else:
                    inplane["screw"] = True
            elif el["order"] == 2 and dc == "gen":
                if t == "rotation":
                    inclined_2["rot"] = True
                else:
                    inclined_2["screw"] = True
            elif el["order"] == 3 and dc == "gen":
                inclined_3.add((k, t == "rotoinversion"))
            elif el["order"] in (4, 6) and dc == "ab":
                inplane_n.add((el["order"], k, t == "rotoinversion"))
        elif t in ("mirror", "glide"):
            if dc == "ab":
                planes.add(sym)
            elif dc == "c":
                gdir = None
                intr = el.get("intrinsic")
                if intr is not None:
                    pv = _perm_vec(np.asarray(intr, float), perm)
                    # same cell frame as the element diagram, so the arrow
                    # is tilted identically in an oblique projection
                    gdir = frame.vec(_cell_edge_sense(np.array([pv[1], pv[0]])))
                par_planes.setdefault(sym, gdir)
    # A trace that carries both a 2 and a 2_1 is drawn with both heads.
    inplane["both"] = inplane["rot"] and inplane["screw"]

    def _axis_lbl(order, k, ro):
        if ro:
            return "\u2212" + str(order)
        if k:
            return str(order) + str(k).translate(_SUBDIGIT)
        return str(order)

    y_top = 9.0
    y = y_top - 0.4
    ax.set_title("elements present", fontsize=9, pad=2)
    if perp:
        ax.text(0.15, y, "Axes ⊥ page:", fontsize=8, style="italic")
        y -= 1.05
        for order in sorted(perp):
            for k, ro in sorted(perp[order]):
                draw_axis_symbol(ax, (0.55, y), order, screw_k=k,
                                 rotoinv=ro, size=0.15)
                ax.text(1.15, y, _axis_lbl(order, k, ro), fontsize=8,
                        va="center")
                y -= 0.95
        y -= 0.2
    if inplane["rot"] or inplane["screw"]:
        ax.text(0.05, y, "Axes in ab-plane:", fontsize=8, style="italic")
        y -= 1.0
        if inplane["rot"]:
            ax.plot([0.15, 0.95], [y, y], "k-", lw=1.3)
            _draw_inplane_arrowhead(ax, (1.0, y), (1, 0), full=True, size=0.16)
            ax.text(1.3, y, "2  (full head)", fontsize=8, va="center")
            y -= 0.9
        if inplane["screw"]:
            ax.plot([0.15, 0.95], [y, y], "k-", lw=1.3)
            _draw_inplane_arrowhead(ax, (1.0, y), (1, 0), full=False, size=0.16)
            ax.text(1.3, y, "2\u2081  (half head)", fontsize=8, va="center")
            y -= 0.9
        y -= 0.3
    if inplane_n:
        ax.text(0.15, y, "Axes in the page, order > 2:", fontsize=8,
                style="italic")
        y -= 1.0
        for order, k, ro in sorted(inplane_n):
            draw_axis_symbol(ax, (0.55, y), order, screw_k=k, rotoinv=ro,
                             size=0.15)
            ax.text(1.15, y, _axis_lbl(order, k, ro), fontsize=8, va="center")
            y -= 0.95
        y -= 0.2
    if inclined_3:
        ax.text(0.15, y, "Inclined 3-fold:", fontsize=8, style="italic")
        y -= 1.0
        for k, ro in sorted(inclined_3):
            draw_axis_symbol(ax, (0.55, y), 3, screw_k=k, rotoinv=ro, size=0.15)
            ax.text(1.15, y, _axis_lbl(3, k, ro), fontsize=8, va="center")
            y -= 0.95
        y -= 0.2
    if inclined_2["rot"] or inclined_2["screw"]:
        ax.text(0.15, y, "Inclined 2-fold:", fontsize=8, style="italic")
        y -= 0.9
        _draw_lens(ax, (0.55, y), 0.12, fc="k", ec="k", lw=0.8, zorder=5)
        ax.text(1.15, y, "2", fontsize=8, va="center")
        y -= 0.9
    if planes:
        ax.text(0.05, y, "Planes ⊥ page:", fontsize=8, style="italic")
        y -= 0.9
        for name in sorted(planes):
            draw_plane_symbol(ax, (0.15, y), (1.05, y), name)
            ax.text(1.3, y, name, fontsize=8, va="center")
            y -= 0.8
        y -= 0.3
    if par_planes:
        ax.text(0.05, y, "Planes ∥ page:", fontsize=8, style="italic")
        y -= 1.0
        for name in sorted(par_planes):
            # draw_parallel_plane_symbol normalises for axis orientation, so the
            # glyph looks identical to the element diagram; pass the stored
            # (right, down) direction straight through.
            draw_parallel_plane_symbol(ax, name, corner=(0.25, y - 0.18),
                                       size=0.5, glide_dir=par_planes[name])
            ax.text(1.3, y, f"{name}  (corner bracket)", fontsize=8,
                    va="center")
            y -= 1.0
        y -= 0.2
    if has_inv:
        ax.text(0.05, y, "Inversion:", fontsize=8, style="italic")
        y -= 0.9
        draw_inversion(ax, (0.55, y), size=0.05)
        ax.text(1.15, y, "\u22121", fontsize=8, va="center")
        y -= 1.0
    if two_xy & inv_xy:
        # A 2-fold and an inversion at one projected site (2/m). The plate
        # draws the lens and the dot together; the legend shows that glyph.
        ax.text(0.2, y, "2/m:", fontsize=8, style="italic")
        y -= 0.95
        _draw_lens(ax, (0.7, y), 0.14, fc="k", ec="k", lw=1.0, zorder=5)
        ax.plot(0.7, y, "o", ms=4, mfc="white", mec="k", mew=0.8, zorder=6)
        ax.text(1.15, y, "2/m", fontsize=8, va="center")
        y -= 1.0
    # points convention. The circle is a data-sized patch so it stays inside
    # the axes; a marker of fixed point size drawn on the last row was clipped
    # by the axes boundary.
    depth = {"a": "x", "b": "y", "c": "z"}[projection]
    ax.text(0.2, y, "Points:", fontsize=8, style="italic")
    y -= 1.15
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0.7, y), 0.16, facecolor="white", edgecolor="k",
                        lw=1.0, zorder=3))
    ax.text(0.7, y, "+", fontsize=7, ha="center", va="center", zorder=4)
    ax.text(1.2, y, f"+ / \u2212 : {depth} / \u2212{depth};  \u00bd+ : {depth}+\u00bd  \u2026",
            fontsize=7.5, va="center")
    # Fixed limits with a pad, and aspect adjustable="box", so equal aspect
    # does not pull the circle or the first row onto the axes spine.
    ax.set_xlim(-0.45, 5.0)
    ax.set_ylim(y - 0.6, y_top)
    ax.set_aspect("equal", adjustable="box")
    return ax


def ita_plate(sg, figsize=None, legend=False, show_centring=False,
              projection="c"):
    """Render the classic ITA pairing: general-position diagram (left) and
    symmetry-element diagram (right), with a header. Returns the Figure.

    Parameters
    ----------
    sg : SpaceGroup | SpaceGroupSetting | int | str
    figsize : (w, h), optional
        Defaults to (6.6, 3.8), or (10.2, 5.6) when ``legend=True`` so the
        legend's last row is not clipped.
    legend : bool
        Append a third panel listing only the elements present in this group
        (see :func:`element_legend`).
    show_centring : bool
        Mark pure lattice (centring) translations on the element diagram (a red
        node + labelled vector); useful for non-standard settings.
    projection : {'c', 'a', 'b'}
        Projection axis. For monoclinic groups the ITA standard plate is the
        unique-axis-b projection (``projection='b'``), where a c-glide plane
        lies parallel to the page and shows its glide-direction arrow.
    """
    import matplotlib.pyplot as plt
    sg = _resolve_sg(sg)
    if figsize is None:
        figsize = (10.2, 5.6) if legend else (6.6, 3.8)
    ncol = 3 if legend else 2
    ratios = [1, 1, 1.05] if legend else [1, 1]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, ncol, width_ratios=ratios, wspace=0.22)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1])
    general_position_diagram(sg, ax=axL, show_title=False,
                             projection=projection)
    symmetry_element_diagram(sg, ax=axR, show_title=False,
                             show_centring=show_centring,
                             projection=projection)
    axL.set_title("general positions", fontsize=8)
    axR.set_title("symmetry elements", fontsize=8)
    if legend:
        element_legend(sg, ax=fig.add_subplot(gs[2]), projection=projection)
    order = _sg_order(sg)
    num, name = _sg_label(sg)
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    pfx = f"#{num}   " if num is not None else ""
    extra = f"({system}, order {order})" if system else f"(order {order})"
    fig.suptitle(f"{pfx}{name}   {extra}", fontsize=9)
    # tight_layout fights the legend's equal aspect and crops its last row.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04,
                        wspace=0.28)
    return fig


def general_position_multiplicity(sg):
    """Number of distinct general-position points in one cell (== group order
    including centring)."""
    sg = _resolve_sg(sg)
    ops = _sg_ops(sg)
    x0 = np.array([0.13, 0.08, 0.20])
    pts = set()
    for W, w, _ in ops:
        p = (W @ x0 + w) % 1.0
        pts.add((round(p[0], 4), round(p[1], 4), round(p[2], 4)))
    return len(pts)
