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
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return (min(xs) - pad, max(xs) + pad), (max(ys) + pad, min(ys) - pad)

    def draw_cell(self, ax, lw=1.2):
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


def _line_edge_copies(c, nrm, tol=1e-3):
    """A line whose location lies on a cell edge (its normal-offset ~0 or ~1)
    is duplicated onto the opposite edge, matching the ITA boundary drawing.
    ``nrm`` is the in-plane normal to the line direction. Returns the list of
    line-location points (the original plus any +/-1 shift along nrm that lands
    the line on the opposite boundary of the unit square)."""
    # Only axis-aligned lines get the opposite-edge copy: a unit normal along
    # x or y steps exactly one cell, mapping an edge line onto the opposite
    # edge. For a DIAGONAL normal the unit-normal step is not a lattice vector
    # (it would land the copy at x+y=sqrt(2), off the lattice); the diagonal
    # family is already positioned by the (W, w+L) lattice reconstruction, so
    # no extra edge copy is added here.
    axis_aligned = abs(nrm[0]) < tol or abs(nrm[1]) < tol
    off = float(np.dot(c, nrm))       # signed offset of the line along nrm
    shifts = [0.0]
    if axis_aligned:
        if abs(off) < tol:
            shifts.append(1.0)
        elif abs(off - 1.0) < tol:
            shifts.append(-1.0)
    return [c + s * nrm for s in shifts]


def _draw_lens(ax, xy, size, angle=0.0, **kw):
    """Filled pointed-oval (2-fold) glyph."""
    from matplotlib.patches import Polygon
    t = np.linspace(0, 2 * np.pi, 60)
    r = size * (0.35 + 0.65 * np.abs(np.cos(t)))  # pointed oval
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    pts = pts @ R.T + np.array(xy)
    ax.add_patch(Polygon(pts, closed=True, **kw))


def _draw_regular_polygon(ax, xy, n, size, filled=True, **kw):
    from matplotlib.patches import RegularPolygon
    ax.add_patch(RegularPolygon(xy, numVertices=n, radius=size,
                                orientation=np.pi / n,
                                fill=filled, **kw))


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
        base = np.array(xy) + rad * size
        tip = base + rad * L
        ax.plot([base[0], tip[0]], [base[1], tip[1]], "-", color="k",
                lw=1.0, zorder=4)
        if abs(handed) > 1e-6:
            hook_len = L * (0.35 + 0.32 * abs(handed))
            tang = np.array([-rad[1], rad[0]]) * np.sign(handed)
            flag = tip + tang * hook_len - rad * (L * 0.22)
            ax.plot([tip[0], flag[0]], [tip[1], flag[1]], "-", color="k",
                    lw=1.0, zorder=4)


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
        _draw_lens(ax, xy, size * 1.3, fc=fc, ec=ec, lw=1.0, zorder=5)
        if screw_k:
            # two tails at +/-90 deg to the lens axis (which lies along x)
            for s in (1, -1):
                ax.plot([xy[0], xy[0]],
                        [xy[1] + s * size * 1.3, xy[1] + s * size * 2.6],
                        "-", color="k", lw=1.2, zorder=4)
    elif order in (3, 4, 6):
        _draw_regular_polygon(ax, xy, order, size, filled=not rotoinv,
                              fc=fc, ec=ec, lw=1.0, zorder=5)
    elif rotoinv:
        _draw_regular_polygon(ax, xy, abs(order), size, filled=False,
                              ec=ec, lw=1.0, zorder=5)
    if rotoinv:
        ax.plot(xy[0], xy[1], "o", ms=3, mfc="white", mec="k", mew=0.8,
                zorder=6)


def _draw_combined_axis(ax, xy, max_rot, rot_k, roto, size=0.035):
    """Draw ONE ITA glyph for all c-axis axes coincident at ``xy``.

    ``max_rot`` is the highest pure-rotation order present (0 if none),
    ``rot_k`` its screw index, ``roto`` the rotoinversion order (0 if none).
    When a rotation and a rotoinversion share the site (e.g. 4 and -4 in
    4/mmm), the filled rotation glyph is drawn first and the rotoinversion is
    marked by an open square outline + centre dot on top, so both are legible
    rather than one white glyph erasing the other.
    """
    if max_rot >= 2:
        draw_axis_symbol(ax, xy, max_rot, screw_k=rot_k, rotoinv=False,
                         size=size)
    elif roto:
        # rotoinversion only (no pure rotation of that order): draw it directly
        draw_axis_symbol(ax, xy, roto, rotoinv=True, size=size)
        return
    if roto and max_rot >= 2:
        # overlay the rotoinversion marker: open polygon outline + centre dot,
        # slightly larger so it frames the filled rotation glyph
        _draw_regular_polygon(ax, xy, abs(roto), size * 1.35, filled=False,
                              ec="k", lw=1.0, zorder=6)
        ax.plot(xy[0], xy[1], "o", ms=3, mfc="white", mec="k", mew=0.8,
                zorder=7)


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
    # Arrow along the in-plane glide direction. Interpret glide_dir in SCREEN
    # terms (x=right, y=up-on-screen) so the arrow is consistent across axes;
    # map its screen-up component to data via -ys.
    if glide_dir is None:
        su = _np.array([1.0, 1.0])            # up-and-right fallback (screen)
    else:
        gd = _np.asarray(glide_dir, float)
        # incoming gd is in data coords of the caller; convert its y to screen-up
        su = _np.array([gd[0], -ys * gd[1]])
    nrm = _np.hypot(*su)
    su = su / nrm if nrm > 1e-9 else _np.array([1.0, 1.0]) / _np.sqrt(2)
    # A glide direction is an undirected axis (the operation equals its
    # reverse), so canonicalise to one screen sense for a consistent glyph:
    # always point to screen-up; if it is purely horizontal, point right; the
    # sign of the two components is chosen independently so the diagonal glide
    # reads the same way regardless of axis orientation.
    if abs(su[1]) < 1e-9:
        su[0] = abs(su[0])            # purely horizontal -> point right
    else:
        if su[1] < 0:
            su[1] = -su[1]            # force screen-up
        su[0] = abs(su[0])            # diagonal -> up-and-right, consistently
    # convert screen direction back to data coords for plotting
    du = _np.array([su[0], -ys * su[1]])
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
    ax.plot(xy[0], xy[1], "o", ms=4, mfc="white", mec="k", mew=1.0, zorder=6)


# glide-plane line styles (plane perpendicular to page -> a line in the page)
_PLANE_STYLE = {
    "m": dict(ls="-", lw=2.0, color="k"),
    "a": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
    "b": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
    "c": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
    "n": dict(ls=(0, (6, 2, 1, 2)), lw=1.3, color="k"),
    "d": dict(ls=(0, (1, 2)), lw=1.4, color="k"),
    "g": dict(ls=(0, (6, 3)), lw=1.3, color="k"),   # ITA: dashed, like a/b/c
}


def draw_plane_symbol(ax, p0, p1, name):
    """Draw a plane (perpendicular to the page) as a styled line from p0 to p1."""
    if abs(p1[0] - p0[0]) < 1e-6 and abs(p1[1] - p0[1]) < 1e-6:
        return                        # degenerate (line only touches a corner)
    style = _PLANE_STYLE.get(name, _PLANE_STYLE["g"])
    # solid mirrors sit at the base layer; patterned glide planes draw a touch
    # higher so their dash/dot signature is not chopped where bold lines cross.
    z = 4 if name == "m" else 4.5
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], zorder=z, **style)
    if name == "d":
        # ITA marks a d-glide (diamond glide) line with an arrow showing the
        # 1/4 glide direction along the line, plus a small perpendicular tick.
        import numpy as _np
        p0a, p1a = _np.asarray(p0, float), _np.asarray(p1, float)
        d = p1a - p0a
        L = _np.hypot(*d)
        if L > 1e-6:
            u = d / L
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


def best_general_point(sg, n_grid=12, refine=3, interior_wt=0.03):
    """Center of the largest sphere inscribed in the asymmetric unit.

    Returns the fractional point whose minimum distance to its own symmetry
    images (modulo the lattice) is maximal, with a light preference for cell
    interior over the walls. Using this as the general position maximises the
    separation of the equivalent points in the diagram, so points overlap only
    where the symmetry *requires* a projection coincidence (handled by ITA
    split circles) rather than by accident of a hand-picked point.

    Results are cached per space-group number for the default parameters.
    """
    sg = _resolve_sg(sg)
    ck = (str(_sg_label(sg)), n_grid, refine, interior_wt)
    if ck in _BGP_CACHE:
        return _BGP_CACHE[ck]
    ops = [(W, w) for W, w, _ in _sg_ops(sg)]

    def score(x):
        x = np.asarray(x, float)
        dmin = np.inf
        for W, w in ops:
            if np.allclose(W, np.eye(3)) and np.allclose(w % 1, 0):
                continue
            d = (W @ x + w) - x
            dd = d - np.round(d)
            dmin = min(dmin, np.linalg.norm(dd))
        if not np.isfinite(dmin):
            dmin = 0.87   # P1: no non-trivial images
        interior = min(np.min(x % 1.0), np.min(1 - x % 1.0))
        return dmin + interior_wt * interior

    best = None
    bestd = -1.0
    grid = np.linspace(0.05, 0.95, n_grid)
    for xi in grid:
        for yi in grid:
            for zi in grid:
                s = score((xi, yi, zi))
                if s > bestd:
                    bestd = s
                    best = np.array([xi, yi, zi])
    step = 1.0 / n_grid
    for _ in range(refine):
        step *= 0.4
        improved = True
        while improved:
            improved = False
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    for dz in (-step, 0, step):
                        cand = best + [dx, dy, dz]
                        s = score(cand)
                        if s > bestd + 1e-9:
                            bestd = s
                            best = cand
                            improved = True
    result = tuple(np.round(best % 1.0, 4))
    _BGP_CACHE[ck] = result
    return result


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
        point = best_general_point(sg)

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
        m = len(members)
        # spread coincident circles along the b direction so each is legible
        span = 0.05 * (m - 1)
        for j, (_, label, det) in enumerate(sorted(members)):
            dv = -span / 2 + j * 0.05 if m > 1 else 0.0
            cx, cy = frame.pt((px + dv, py))
            ax.plot(cx, cy, "o", ms=9, mfc="white", mec="k", mew=1.0,
                    zorder=3)
            ax.text(cx + 0.03, cy - 0.03, label,
                    fontsize=6, zorder=4, ha="left", va="center")
            if det < 0:
                ax.text(cx, cy, ",", fontsize=8, zorder=4,
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


def _matrix_order(W, max_n=6):
    """Smallest n>=1 with W**n == I."""
    W = np.asarray(W, float)
    P = np.eye(W.shape[0])
    for n in range(1, max_n + 1):
        P = P @ W
        if np.allclose(P, np.eye(W.shape[0]), atol=1e-6):
            return n
    return 0


def _axis_direction(W, proper):
    """Rotation axis (proper) or mirror-plane normal (improper) as an integer-ish
    direction. For proper rotation it is the +1 eigenvector of W; for a mirror it
    is the -1 eigenvector of W."""
    target = 1.0 if proper else -1.0
    # for rotoinversions the "axis" is the +1 eigenvector of -W (proper part)
    M = W if proper else -W
    vals, vecs = np.linalg.eig(M)
    for i, lam in enumerate(vals):
        if abs(lam.real - 1.0) < 1e-6 and abs(lam.imag) < 1e-6:
            v = vecs[:, i].real
            v = v / (np.max(np.abs(v)) or 1.0)
            # snap to small integers
            vi = np.round(v * 12) / 12
            return vi
    return None


def _orient_axis(W, axis):
    """Flip ``axis`` if needed so that ``W`` is a right-handed rotation about it.

    ``np.linalg.eig`` returns the +1 eigenvector with an arbitrary sign; the
    screw index only makes sense against the sense of rotation. For a rotation
    by 0 < theta < pi about the unit vector ``n`` the triple
    ``(n, p, W p)`` is right-handed for any ``p`` not parallel to ``n`` --
    ``det[n, p, W p] > 0`` -- and a crystallographic basis is right-handed, so
    the sign of that determinant in fractional coordinates is the sign in
    Cartesian ones. (theta = pi, the 2-fold, leaves the sign undefined; 2_1 has
    k = n/2 so it does not matter.)
    """
    a = np.asarray(axis, float)
    # a transverse probe: whichever unit vector is least parallel to the axis
    p = np.eye(3)[int(np.argmin(np.abs(a)))]
    d = np.linalg.det(np.column_stack([a, p, W @ p]))
    return -a if d < -1e-9 else a


def _intrinsic_translation(W, w, n):
    """Screw/glide (intrinsic) part = (1/n) sum_{k=0}^{n-1} W^k w."""
    acc = np.zeros(3)
    P = np.eye(3)
    for _ in range(n):
        acc += P @ w
        P = P @ W
    return acc / n


def _location_point(W, w_loc):
    """A representative point on the element: solve (W - I) x = -w_loc
    (least squares; the element is the fixed locus of x -> W x + w_loc)."""
    A = W - np.eye(3)
    x, *_ = np.linalg.lstsq(A, -w_loc, rcond=None)
    return x % 1.0


_ROT_BY_TRACE = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}       # det +1
_INV_BY_TRACE = {-3: -1, -2: -6, -1: -4, 0: -3, 1: -2}  # det -1 (-2 == m)


def _glide_name(t, W=None):
    """Name a glide plane from its intrinsic translation ``t`` (fractional).

    The glide vector is half a lattice vector lying in the plane, defined
    modulo lattice vectors in the plane. ``v = 2 t`` is that lattice vector.

    * ``v = 0`` -> ``m``;
    * ``v`` along a single cell axis that lies in the plane -> ``a``/``b``/``c``;
    * ``v`` with two or three components, plane a coordinate plane (``W`` is
      diagonal) -> ``n`` (diagonal glide);
    * ``v`` with two or three components in a plane that is NOT a coordinate
      plane (hexagonal groups: planes perpendicular to <100> / <210> contain
      a+2b, 2a+b, a-b) -> ``g``, the generic ITA glide;
    * quarter translations -> ``d``.

    Reduction is only applied along cell axes that lie in the plane
    (``W e_i = e_i``); reducing every component mod 1 -- the old behaviour --
    turned the hexagonal glide ``1/2 (a + 2b)`` into a spurious ``a``.
    """
    t = np.asarray(t, float)
    v = 2.0 * t
    if W is not None:
        W = np.asarray(W, float)
        diag = np.allclose(W, np.diag(np.diag(W)), atol=1e-6)
        inplane_axes = [i for i in range(3)
                        if np.allclose(W[:, i], np.eye(3)[i], atol=1e-6)]
    else:
        diag = True
        inplane_axes = [0, 1, 2]
    if np.allclose(v, np.round(v), atol=0.08):
        v = np.round(v).astype(int)
        for i in inplane_axes:
            v[i] %= 2
        nz = [i for i in range(3) if v[i] != 0]
        if not nz:
            return "m"
        if len(nz) == 1 and nz[0] in inplane_axes:
            return "abc"[nz[0]]
        return "n" if diag else "g"
    v4 = 4.0 * t
    if np.allclose(v4, np.round(v4), atol=0.16):
        return "d"
    return "g"  # unclassified glide


def classify_element(W, w):
    """Classify a symmetry operation (W, w) as an ITA symmetry element.

    Returns a dict with:
      ``type``      -- 'rotation' | 'screw' | 'mirror' | 'glide' |
                        'inversion' | 'rotoinversion' | 'identity' |
                        'translation'
      ``order``     -- rotation order n (2,3,4,6); the |n| of a rotoinversion;
                        1 for identity/translation
      ``symbol``    -- ITA-ish label ('2', '2_1', '3', '4_2', 'm', 'c', 'n',
                        'd', '-1', '-3', '-4', '-6')
      ``axis``      -- rotation axis (rotations/screws/rotoinversions) or
                        plane normal (mirror/glide) as a direction vector, or None
      ``location``  -- a fractional point lying on the element (or the inversion
                        centre), or None
      ``intrinsic`` -- the screw/glide translation vector (0 for symmorphic)
    """
    W = np.asarray(W, dtype=float)
    w = np.asarray(w, dtype=float)
    det = round(np.linalg.det(W))
    tr = round(np.trace(W))

    # pure lattice translation / identity
    if np.allclose(W, np.eye(3), atol=1e-6):
        if np.allclose(w % 1.0, 0, atol=1e-6):
            return {"type": "identity", "order": 1, "symbol": "1",
                    "axis": None, "location": None,
                    "intrinsic": np.zeros(3)}
        return {"type": "translation", "order": 1, "symbol": "t",
                "axis": None, "location": None, "intrinsic": w % 1.0}

    if det == 1:
        n = _ROT_BY_TRACE.get(tr, _matrix_order(W))
        intr = _intrinsic_translation(W, w, n)
        w_loc = w - intr
        loc = _location_point(W, w_loc)
        # screw if intrinsic translation is nonzero along the axis
        screw = not np.allclose(intr - np.round(intr), 0, atol=0.05)
        axis = _axis_direction(W, proper=True)
        if n > 2:
            axis = _orient_axis(W, axis)
        if screw:
            # screw index k: intrinsic translation as a fraction k/n of the
            # lattice repeat along the ORIENTED axis (axis has max |comp| = 1,
            # i.e. it is the shortest lattice vector along the axis for a
            # primitive lattice), measured with the rotation sense so that an
            # operation and its inverse give the same k.
            k = int(round(n * float(np.dot(intr, axis)) /
                          float(np.dot(axis, axis)))) % n
            sym = f"{n}_{k}" if k else str(n)
            return {"type": "screw" if k else "rotation", "order": n,
                    "symbol": sym, "axis": axis, "location": loc,
                    "intrinsic": intr, "W": W}
        return {"type": "rotation", "order": n, "symbol": str(n),
                "axis": axis, "location": loc, "intrinsic": np.zeros(3),
                "W": W}

    # det == -1 : improper
    kind = _INV_BY_TRACE.get(tr)
    if kind == -1:  # inversion centre
        loc = (w / 2.0) % 1.0
        return {"type": "inversion", "order": 2, "symbol": "-1",
                "axis": None, "location": loc, "intrinsic": np.zeros(3)}
    if kind == -2:  # mirror or glide (n=2 for W)
        intr = _intrinsic_translation(W, w, 2)  # in-plane glide part
        w_loc = w - intr
        loc = _location_point(W, w_loc)
        normal = _axis_direction(W, proper=False)
        gname = _glide_name(intr, W)
        if gname == "m":
            return {"type": "mirror", "order": 2, "symbol": "m",
                    "axis": normal, "location": loc, "intrinsic": np.zeros(3),
                    "W": W}
        return {"type": "glide", "order": 2, "symbol": gname,
                "axis": normal, "location": loc, "intrinsic": intr, "W": W}
    # rotoinversion -3, -4, -6
    n = {-3: 3, -4: 4, -6: 6}.get(kind, _matrix_order(W))
    axis = _axis_direction(W, proper=False)
    loc = _location_point(W, w)
    return {"type": "rotoinversion", "order": n, "symbol": str(kind),
            "axis": axis, "location": loc, "intrinsic": np.zeros(3)}


def classify_space_group(sg):
    """Classify every operation of a space group. Returns a list of element
    dicts (see :func:`classify_element`), skipping the identity."""
    sg = _resolve_sg(sg)
    out = []
    for W, w, xyz in _sg_ops(sg):
        el = classify_element(W, w)
        el["xyz"] = xyz
        if el["type"] != "identity":
            out.append(el)
    return out


def _dir_class(vec, tol=0.2):
    """Classify a direction as 'c' (⊥ page, along c), 'ab' (in page), or 'gen'."""
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


def _element_copies(sg):
    """Enumerate every symmetry element across the full cell.

    A single coset operation (W, w) generates a family of parallel elements
    inside one cell: combining it with a lattice (or centring) translation L
    relocates the element (the classic result that parallel 2-folds sit at
    x=0 and x=1/2). We therefore reclassify (W, w+L) for L over the integer
    lattice and centring translations, and collect the distinct in-cell
    locations for each element.
    """
    ops = _sg_ops(sg)
    cvs = _centring_translations(sg)
    seen = set()
    out = []
    Ls = [np.array([i, j, k], float)
          for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for W, w, _ in ops:
        for cv in cvs:
            for L in Ls:
                el = classify_element(W, w + L + cv)
                if el["type"] in ("identity", "translation"):
                    continue
                loc = el["location"]
                if loc is None:
                    continue
                # canonical key: type/symbol/axis-dir + location mod 1,
                # rounded so translated duplicates collapse
                axis = el["axis"]
                ak = tuple(np.round(np.asarray(axis), 2)) if axis is not None \
                    else None
                lk = tuple(np.round(np.asarray(loc) % 1.0, 3))
                key = (el["type"], el["symbol"], ak, lk)
                if key in seen:
                    continue
                seen.add(key)
                out.append(el)
    return out


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

    Axes ⊥ page are point glyphs; in-plane two-folds are arrowed lines; planes
    ⊥ page are styled lines; inversion centres are small open circles. Screw
    and glide elements already encode their translation (screw tails / dashed
    glide lines); pure lattice translations are shown only with show_centring.
    Elements oblique to the projection (e.g. cubic body-diagonal axes) are
    counted and reported in the title, not drawn.
    """
    import matplotlib.pyplot as plt

    sg = _resolve_sg(sg)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    perm, dlab, rlab, _ = _PROJ[projection]
    frame = _Frame(cell_frame(sg, projection))
    frame.draw_cell(ax)

    n_centring = 0
    if show_centring:
        n_centring = _draw_centring_markers(ax, sg, perm, frame)

    if show_general_positions:
        general_position_diagram(sg, ax=ax, show_title=False,
                                 projection=projection)
        # dim the overlaid circles
        for ln in ax.lines:
            ln.set_alpha(0.25)

    els = _element_copies(sg) if full_cell else classify_space_group(sg)
    omitted = 0

    # Establish the inverted y-axis (a' down the page, ITA convention) BEFORE
    # drawing, so orientation-aware glyphs (the parallel-plane corner bracket)
    # detect the correct screen sense at draw time.
    xl, yl = frame.limits(0.18)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    def P(fr):
        """fractional 3-vector -> fractional (right, down) in the cell."""
        v = _perm_vec(fr, perm)  # [0]=down(a'), [1]=right(b')
        return (v[1] % 1.0, v[0] % 1.0)

    def dcls(axis):
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
        """Clip the family of lines through fractional point ``c`` along
        fractional direction ``d`` (both (right, down)) to the cell, add the
        opposite-edge copy, and hand plot-space endpoints to ``draw``."""
        nn = np.array([-d[1], d[0]])
        for cc in _line_edge_copies(np.asarray(c, float), nn):
            p0, p1 = _clip_line_to_box(cc, d)
            if p0 is None or np.linalg.norm(p1 - p0) < 1e-6:
                continue
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
                                         "roto": 0})
            if el["type"] == "rotoinversion":
                s["roto"] = max(s["roto"], el["order"])
            else:
                k = 0
                if el["type"] == "screw" and "_" in el["symbol"]:
                    k = int(el["symbol"].split("_")[1])
                if el["order"] > s["max_rot"]:
                    s["max_rot"] = el["order"]
                    s["rot_k"] = k
    for key, s in c_sites.items():
        for xy in edge_copies(key):
            _draw_combined_axis(ax, xy, s["max_rot"], s["rot_k"], s["roto"])

    parallel_planes_drawn = set()   # planes parallel to the page (corner glyph)
    for el in els:
        t = el["type"]
        loc = el["location"]
        if t == "inversion":
            for xy in edge_copies(P(loc)):
                draw_inversion(ax, xy)
            continue
        if t in ("rotation", "screw", "rotoinversion"):
            dc = dcls(el["axis"])
            if dc == "c":
                continue   # handled by the combined-glyph pre-pass above
            elif dc == "ab" and el["order"] == 2:
                d = frac_dir(el["axis"])
                d = d / (np.linalg.norm(d) or 1.0)

                def draw_axis_line(p0, p1, dp, _full=(t == "rotation")):
                    # ITA convention for an axis lying in the plane of the
                    # page: a SOLID line; a pure 2-fold carries a FULL
                    # (two-barbed) arrowhead, a 2_1 screw a HALF (one-barbed)
                    # arrowhead. The head shape -- not the line style -- is
                    # what distinguishes them (dashes are reserved for planes).
                    # The arrowhead sits JUST OUTSIDE the boundary on the exit
                    # side, as the ITA does.
                    exit_pt = p1 if np.dot(p1 - p0, dp) > 0 else p0
                    start = p0 if exit_pt is p1 else p1
                    head_size = 0.07
                    base = exit_pt + dp * 0.06         # just outside the cell
                    tip = base + dp * head_size         # apex beyond the base
                    ax.plot([start[0], base[0]], [start[1], base[1]],
                            color="k", lw=1.3, zorder=4)
                    _draw_inplane_arrowhead(ax, tip, dp, full=_full,
                                            size=head_size)

                draw_line_family(P(loc), d, draw_axis_line)
            else:
                omitted += 1
            continue
        if t in ("mirror", "glide"):
            dc = dcls(el["axis"])
            if dc == "ab":
                d = inplane_fixed_dir(el["W"])
                d = d / (np.linalg.norm(d) or 1.0)
                sym = el["symbol"]
                draw_line_family(
                    P(loc), d,
                    lambda p0, p1, _dp, _s=sym: draw_plane_symbol(ax, p0, p1, _s))
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
                        gdir = frame.vec(frac_dir(np.asarray(intr, float)))
                    draw_parallel_plane_symbol(
                        ax, el["symbol"],
                        corner=(0.08 + 0.34 * slot, 0.08),
                        size=0.16, glide_dir=gdir)
                    parallel_planes_drawn.add(el["symbol"])
            else:
                omitted += 1
            continue

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
    els = _element_copies(sg)
    perm = _PROJ[projection][0]
    frame = _Frame(cell_frame(sg, projection))
    perp = {}      # order -> set of screw_k for axes ⊥ page
    inplane = {"rot": False, "screw": False}
    planes = set()          # planes ⊥ page (drawn as lines)
    par_planes = {}         # planes ∥ page -> projected glide dir (bracket)
    has_inv = False
    for el in els:
        t, sym, axis = el["type"], el["symbol"], el["axis"]
        # classify relative to the projection axis (permute first), so the
        # legend matches what the element diagram draws for that projection.
        dc = _dir_class(_perm_vec(axis, perm)) if axis is not None else None
        if t == "inversion":
            has_inv = True
        elif t in ("rotation", "screw", "rotoinversion"):
            if dc == "c":
                k = 0
                if t == "screw" and "_" in sym:
                    k = int(sym.split("_")[1])
                perp.setdefault(el["order"], set()).add(
                    (k, t == "rotoinversion"))
            elif dc == "ab" and el["order"] == 2:
                inplane["rot" if t == "rotation" else "screw"] = True
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
                    gdir = frame.vec(np.array([pv[1], pv[0]]))
                par_planes.setdefault(sym, gdir)

    y = 9.3
    ax.set_title("elements present", fontsize=9)
    if perp:
        ax.text(0.05, y, "Axes ⊥ page:", fontsize=8, style="italic")
        y -= 1.05
        for order in sorted(perp):
            for k, ro in sorted(perp[order]):
                draw_axis_symbol(ax, (0.45, y), order, screw_k=k,
                                 rotoinv=ro, size=0.15)
                lbl = ("-" if ro else "") + str(order)
                if k:
                    lbl = f"{order}_{k}"
                ax.text(1.0, y, f"{lbl}", fontsize=8, va="center")
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
        draw_inversion(ax, (0.45, y), size=0.05)
        ax.text(1.0, y, "\u22121", fontsize=8, va="center")
        y -= 1.0
    # points convention
    ax.text(0.05, y, "Points:", fontsize=8, style="italic")
    y -= 0.9
    ax.plot(0.45, y, "o", ms=8, mfc="white", mec="k", mew=1.0)
    ax.text(0.6, y + 0.12, "+", fontsize=7)
    ax.text(1.0, y, "+ / \u2212 : z / \u2212z;  \u00bd+ : z+\u00bd  \u2026",
            fontsize=8, va="center")
    # Fit the y-range to the content so element-rich groups (many rows) do not
    # push later entries below the axis and off the canvas.
    ax.set_ylim(y - 0.6, 10)
    # Equal aspect so the round/polygon glyphs (2-fold lens, 4-fold square,
    # -4 outline) are not horizontally stretched by the tall, narrow panel.
    # adjustable="datalim" keeps the panel box (set by the caller / gridspec)
    # and widens the x-range instead of shrinking the drawing.
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def ita_plate(sg, figsize=None, legend=False, show_centring=False,
              projection="c"):
    """Render the classic ITA pairing: general-position diagram (left) and
    symmetry-element diagram (right), with a header. Returns the Figure.

    Parameters
    ----------
    sg : SpaceGroup | SpaceGroupSetting | int | str
    figsize : (w, h), optional
        Defaults to (6.6, 3.4), or (9.4, 3.6) when ``legend=True``.
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
        figsize = (9.4, 3.6) if legend else (6.6, 3.4)
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
    fig.suptitle(f"{pfx}{name}   {extra}", fontsize=9, y=1.02)
    fig.tight_layout()
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
