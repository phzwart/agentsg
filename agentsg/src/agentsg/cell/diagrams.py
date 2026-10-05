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


# Pre-2016 Hermann–Mauguin symbols that the 2016 edition writes with an e glide.
_HM_2016 = {
    39: "A e m 2",
    41: "A e a 2",
    64: "C m c e",
    67: "C m m e",
    68: "C c c e",
}


def _setting_row(sg):
    """ITA setting row for a group or a transformed setting, or None."""
    from ..ita_settings import lookup_setting, match_ops
    if hasattr(sg, "base") and hasattr(sg, "operations"):
        try:
            return match_ops(sg.base.number, sg.operations())
        except Exception:
            return None
    hall = getattr(sg, "hall", None)
    if hall:
        return lookup_setting(hall)
    return None


def _ita_names(sg):
    """(classic symbol, 2016 symbol) of this group or setting.

    The basis change stays out of the name. Origin choice 2 and rhombohedral
    axes are written out. The 2016 name renames an e glide on this setting.
    """
    from ..ita_settings import display_hm
    row = _setting_row(sg)
    if row is not None:
        classic = display_hm(row[1], row[3], "")
        modern = display_hm(row[1], row[3], row[4])
        return classic, modern
    raw = getattr(sg, "hermann_mauguin", None) or str(sg)
    num = getattr(sg, "number", None)
    if num is None and hasattr(sg, "base"):
        num = getattr(sg.base, "number", None)
    return raw, _HM_2016.get(num, raw)


def _sg_label(sg):
    """(number, name) for the title.

    The title is the ITA symbol of this setting, with an e glide written the
    2016 way. The change of basis is not repeated in the title.
    """
    num = getattr(sg, "number", None)
    if num is None and hasattr(sg, "base"):
        num = getattr(sg.base, "number", None)
    _classic, modern = _ita_names(sg)
    return num, modern


def _spaced_hm(name):
    """ITA symbol with spaces kept and screw indices as subscripts."""
    return " ".join(_subscript_token(tok) for tok in str(name).split())


def _subscript_token(token):
    chars = []
    for i, ch in enumerate(token):
        prev = token[i - 1] if i else ""
        if ch.isdigit() and prev.isdigit():
            chars.append(ch.translate(_SUBDIGIT))
        else:
            chars.append(ch)
    return "".join(chars)


def _display_hm(name):
    """Compact a spaced HM symbol and print screw indices as subscripts."""
    text = str(name)
    if "(" in text:
        head, _, tail = text.partition("(")
        return _display_hm(head.strip()) + " (" + tail
    out = []
    for token in text.split():
        chars = []
        for i, ch in enumerate(token):
            prev = token[i - 1] if i else ""
            if ch.isdigit() and prev.isdigit():
                chars.append(ch.translate(_SUBDIGIT))
            else:
                chars.append(ch)
        out.append("".join(chars))
    return "".join(out)


def _panel_change_of_basis(projection):
    """Right-handed cob whose new axes are the panel's down, right, and depth.

    Columns are those three old axes. One in-plane sign is flipped when the
    permutation is left-handed, so ``det = +1``. The spelling uses a, b, c.
    """
    from fractions import Fraction as Fr
    from ..change_of_basis import ChangeOfBasis
    from ..linalg import Matrix3, Vector3
    from ..setting import format_cob
    down, right, depth = _PROJ[projection][0]
    axes = (
        (Fr(1), Fr(0), Fr(0)),
        (Fr(0), Fr(1), Fr(0)),
        (Fr(0), Fr(0), Fr(1)),
    )
    cols = [axes[down], axes[right], axes[depth]]

    def _matrix(columns):
        return Matrix3([[columns[j][i] for j in range(3)] for i in range(3)])

    if _matrix(cols).det() < 0:
        cols[1] = tuple(-c for c in cols[1])
    cob = ChangeOfBasis(_matrix(cols), Vector3((0, 0, 0)))
    return cob, format_cob(cob, letters="abc")


def _panel_transformed_ops(sg, cob):
    """Operators and centring translations of ``sg`` in the panel basis."""
    from ..group import close_group
    from ..setting import SpaceGroupSetting
    if hasattr(sg, "base") and hasattr(sg, "cob"):
        transformed = [cob.apply_to_op(op) for op in sg.operations()]
        return close_group(transformed)
    return SpaceGroupSetting(sg, cob).operations()


def orthorhombic_panel(sg, projection):
    """ITA setting of an orthorhombic group drawn along ``projection``.

    Returns ``(title, cob_text, operators, setting_row)``. The title is the
    tabulated Hermann–Mauguin symbol of the transformed group, not a shuffle
    of the original letters.
    """
    from ..ita_settings import display_hm, match_ops
    cob, cob_text = _panel_change_of_basis(projection)
    ops = _panel_transformed_ops(sg, cob)
    number = _sg_number(sg)
    row = match_ops(number, ops)
    if row is None:
        raise ValueError(
            f"no ITA setting of No. {number} matches projection {projection}"
        )
    hm = display_hm(row[1], row[3], row[4])
    title = f"along {projection}: {_spaced_hm(hm)} {cob_text}"
    return title, cob_text, ops, row


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


def _monoclinic_angle(sg):
    """Page angle for an oblique monoclinic frame.

    Symmetry fixes the angle to be neither 90° nor 120°, not its value.
    The three unique-axis cell choices are drawn with different angles so
    the plates are not the same parallelogram.
    """
    try:
        classic, _modern = _ita_names(sg)
    except Exception:
        return _OBLIQUE_DEG
    text = str(classic).replace(" ", "")
    if "21/n" in text or "2/n" in text:
        return 80.0
    if "21/a" in text or "2/a" in text:
        return 112.0
    if "21/b" in text or "2/b" in text:
        return 105.0
    return _OBLIQUE_DEG


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
        angle, kind = _monoclinic_angle(sg), "oblique"
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


def _caption_xy(xy, frame, dx=0.11, dy=-0.08):
    """A height note clear of its glyph and inside the cell.

    A ¼ sitting on a filled square is clipped down to a mark like ⁷, and a
    note on the cell edge loses its first character.
    """
    x, y = float(xy[0]), float(xy[1])
    xs = [c[0] for c in frame.corners]
    ys = [c[1] for c in frame.corners]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    if x > hi_x - 0.16:
        dx = -abs(dx)
    elif x < lo_x + 0.16:
        dx = abs(dx)
    if y + dy < lo_y + 0.04:
        dy = abs(dy)
    elif y + dy > hi_y - 0.04:
        dy = -abs(dy)
    return x + dx, y + dy


def _on_cell_edge(p0, p1, frame, tol=0.04):
    """True when both endpoints lie on one edge of the projected cell."""
    corners = list(frame.corners)
    edges = list(zip(corners, corners[1:] + corners[:1]))

    def near(pt, a, b):
        ab = np.asarray(b, float) - np.asarray(a, float)
        ap = np.asarray(pt, float) - np.asarray(a, float)
        length2 = float(np.dot(ab, ab)) or 1.0
        t = float(np.dot(ap, ab) / length2)
        if t < -0.08 or t > 1.08:
            return False
        return float(np.linalg.norm(ap - t * ab)) < tol

    return any(near(p0, a, b) and near(p1, a, b) for a, b in edges)


def _draw_rhombo_outline(ax, frame):
    """Rhombohedral cell in the hexagonal projection, drawn over the hex cell."""
    from ..setting import parse_cob
    cob = parse_cob("((2a+b+c)/3,(-a+b+c)/3,(-a-2b+c)/3)")
    cols = []
    for j in range(3):
        old = [cob.P.rows[i][j] for i in range(3)]
        # projection c: right = b, down = a
        cols.append((float(old[1]), float(old[0])))
    origin = frame.pt((0.0, 0.0))
    verts = [origin]
    # The eight corners, in plot space. Draw each edge of the parallelepiped.
    corners = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                rd = (i * cols[0][0] + j * cols[1][0] + k * cols[2][0],
                      i * cols[0][1] + j * cols[1][1] + k * cols[2][1])
                corners.append(frame.pt(rd))
    steps = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    pts = {(i, j, k): corners[i * 4 + j * 2 + k]
           for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for i, j, k in pts:
        here = pts[(i, j, k)]
        for di, dj, dk in steps:
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni, nj, nk) in pts:
                there = pts[(ni, nj, nk)]
                ax.plot([here[0], there[0]], [here[1], there[1]],
                        color="0.35", lw=0.8, zorder=3)
    return verts


def _is_rhombo_axes(sg):
    row = _setting_row(sg)
    return row is not None and row[3] == "R"


# --- exact heights -------------------------------------------------------------

_SUBDIGIT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_UNICODE_FRAC = {
    Fraction(1, 2): "½", Fraction(1, 3): "⅓", Fraction(2, 3): "⅔",
    Fraction(1, 4): "¼", Fraction(3, 4): "¾", Fraction(1, 5): "⅕",
    Fraction(2, 5): "⅖", Fraction(3, 5): "⅗", Fraction(4, 5): "⅘",
    Fraction(1, 6): "⅙", Fraction(5, 6): "⅚", Fraction(1, 8): "⅛",
    Fraction(3, 8): "⅜", Fraction(5, 8): "⅝", Fraction(7, 8): "⅞",
}
_SUPERDIGIT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def frac_label(t):
    """``Fraction`` in [0, 1) -> one fraction glyph ('' for 0, '½', '⁵⁄₁₂')."""
    t = Fraction(t) % 1
    if t == 0:
        return ""
    hit = _UNICODE_FRAC.get(t)
    if hit:
        return hit
    num = str(t.numerator).translate(_SUPERDIGIT)
    den = str(t.denominator).translate(_SUBDIGIT)
    return f"{num}⁄{den}"


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
    # A height and the copy half a cell above it are one note. Keep the
    # representative below 1/2 (0 rather than 1/2, 1/6 rather than 2/3,
    # 5/12 rather than 11/12) and do not print both.
    kept = []
    present = set(vals)
    for v in vals:
        partner = (v + Fraction(1, 2)) % 1
        if partner in present and partner < v:
            continue
        if partner in present and partner == v:
            continue
        kept.append(v)
    vals = kept
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
#   m -> bold solid, glide a/b/c -> dashed, n and d -> dash-dot, e -> dash-dot-dot.
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
    """Where an inclined axis meets the drawing plane, as (right, down) and depth.

    ITA marks an inclined axis by the dot where it cuts the plane of the
    diagram at height 0 (Volume A, §1.4.7). One axis therefore contributes one
    point. The stored location is some other point of the same line; stepping
    along it would pile a symbol at every z = 1/3 and z = 2/3 crossing.
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
    depth_i = perm[2]
    ad = axis[depth_i]
    if ad == 0:
        p = loc
    else:
        t = -loc[depth_i] / ad
        p = [(loc[i] + t * axis[i]) % 1 for i in range(3)]
    v = _perm_vec([float(c) for c in p], perm)
    rd = (float(v[1] % 1.0), float(v[0] % 1.0))
    return [(rd, float(v[2] % 1.0))]


def _same_trace(origin, direction, other_origin, other_direction, tol=0.02):
    """True when two fractional lines are the same trace in the page."""
    d = np.asarray(direction, float)
    e = np.asarray(other_direction, float)
    dn = np.linalg.norm(d) or 1.0
    en = np.linalg.norm(e) or 1.0
    d, e = d / dn, e / en
    if abs(d[0] * e[1] - d[1] * e[0]) > 1e-3:
        return False
    delta = np.asarray(origin, float) - np.asarray(other_origin, float)
    return abs(delta[0] * e[1] - delta[1] * e[0]) < tol


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
    # Proportional to the polygon, so a dense cubic plate keeps each hook on
    # its own glyph and the legend copies still stop short of the caption.
    L = size * 1.65
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
             lw=1.0, zorder=8.5, solid_capstyle="round")
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
             lw=1.0, zorder=8.5, solid_capstyle="round")


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
    # they are visible past the pointed oval; other screws use the pinwheel,
    # drawn after the polygon so the glyph does not cover the hooks.
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
    if screw_k and order != 2:
        _draw_screw_tails(ax, xy, order, screw_k, size)
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


def _draw_bar4(ax, xy, size):
    """ITA −4: an open square with a lens inscribed in it."""
    _draw_regular_polygon(ax, xy, 4, size, filled=False,
                          ec="k", lw=1.0, zorder=7.5)
    _draw_lens(ax, xy, size * 0.42, angle=0.0, fc="none", ec="k",
               lw=0.9, zorder=7.6)


def _draw_combined_axis(ax, xy, max_rot, rot_k, roto, size=0.035,
                        inversion=False):
    """Draw ONE ITA glyph for all c-axis axes coincident at ``xy``.

    ``max_rot`` is the highest pure-rotation order present (0 if none),
    ``rot_k`` its screw index, ``roto`` the rotoinversion order (0 if none).
    A −6 with no 6-fold on the same line is the open hexagon. 4/m and 6/m
    are the filled rotation, screw tails included, plus the open −1 circle.
    An open −4 on top of the filled square erases it, so it is not drawn.

    ``inversion`` draws that open circle: the −1 of 2/m, 4/m and 6/m.
    """
    if roto == 6 and max_rot < 6:
        # A −6, including the one whose proper 3 lies on the same line.
        # An open hexagon, the same glyph the legend draws. A filled
        # triangle here reads as a 3.
        draw_axis_symbol(ax, xy, 6, rotoinv=True, size=size)
    elif max_rot >= 2:
        # 4/m and 6/m are the filled rotation (screw tails included) with
        # the −1 circle in the middle. An open −4 painted on top erases the 4.
        draw_axis_symbol(ax, xy, max_rot, screw_k=rot_k, rotoinv=False,
                         size=size)
        if inversion:
            _draw_axis_inversion(ax, xy, size * 0.48)
    elif roto == 4:
        _draw_bar4(ax, xy, size)
    elif roto:
        draw_axis_symbol(ax, xy, roto, rotoinv=True, size=size)
    elif inversion:
        _draw_axis_inversion(ax, xy, size * 0.48)


def draw_parallel_plane_symbol(ax, name, corner=(0.06, 0.06), size=0.11,
                               glide_dir=None, glide_dirs=None):
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
    # One arrow, or both axial halves of an e glide.
    if glide_dirs:
        directions = list(glide_dirs)
    elif glide_dir is not None:
        directions = [glide_dir]
    else:
        directions = [None]
    for gd in directions:
        # glide_dir is in the caller's data frame (x=right, y in the axis' own
        # sense). On an upright axis the data-y component is flipped so the
        # arrow points the same way on screen as it does on the plate.
        if gd is None:
            du = _np.array([1.0, -ys])
        else:
            gd = _np.asarray(gd, float)
            du = _np.array([gd[0], ys * gd[1]])
        nrm = _np.hypot(*du)
        du = du / nrm if nrm > 1e-9 else _np.array([1.0, -ys])
        base = _np.array([x0 + size * 0.25, y0 + ys * size * 0.25])
        tip = base + du * size * 0.85
        astyle = "-|>" if name in ("a", "b", "c", "d", "e") else "->"
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


def _separate_overlapping_labels(ax, rounds=8):
    """Nudge text labels whose boxes overlap."""
    fig = ax.figure
    if fig is None or not ax.texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text().strip()]
    for _ in range(rounds):
        fig.canvas.draw()
        boxes = []
        for t in texts:
            try:
                boxes.append(t.get_window_extent(renderer).expanded(1.05, 1.15))
            except Exception:
                boxes.append(None)
        moved = False
        for i, bi in enumerate(boxes):
            if bi is None:
                continue
            for j in range(i + 1, len(boxes)):
                bj = boxes[j]
                if bj is None or not bi.overlaps(bj):
                    continue
                xi, yi = texts[i].get_position()
                x, y = texts[j].get_position()
                # Step away from the label it sits on, so a crossing does not
                # march every note in the same direction.
                sx = 0.06 if x >= xi else -0.06
                sy = 0.04 if y >= yi else -0.04
                texts[j].set_position((x + sx, y + sy))
                moved = True
                break
        if not moved:
            break


def draw_inversion(ax, xy, size=0.012):
    """Draw an ITA inversion centre symbol (small open circle) at xy.

    ``size`` is a data radius. The plate uses a fixed marker so the circle
    stays readable inside a small glyph. A large radius (the legend) is a
    patch, so the hole scales with the square around it.
    """
    if size >= 0.05:
        from matplotlib.patches import Circle
        ax.add_patch(Circle(xy, size, facecolor="white", edgecolor="k",
                            lw=1.0, zorder=8))
        return
    ax.plot(xy[0], xy[1], "o", ms=4, mfc="white", mec="k", mew=1.0, zorder=8)


# glide-plane line styles (plane perpendicular to page -> a line in the page)
_PLANE_STYLE = {
    "m": dict(ls="-", lw=2.2, color="k"),
    "a": dict(ls=(0, (6, 3)), lw=1.3, color="k"),          # dashed
    "b": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
    "c": dict(ls=(0, (1, 1.6)), lw=1.4, color="k"),        # dotted
    "n": dict(ls=(0, (6, 2, 1, 2)), lw=1.3, color="k"),    # dash-dot
    "d": dict(ls=(0, (6, 2, 1, 2)), lw=1.3, color="k"),    # dash-dot, plus arrow
    "e": dict(ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.3, color="k"),  # dash-dot-dot
    "g": dict(ls=(0, (6, 3)), lw=1.3, color="k"),
}


def draw_plane_symbol(ax, p0, p1, name, glide_dir=None, erase_under=False):
    """Draw a plane (perpendicular to the page) as a styled line from p0 to p1."""
    if abs(p1[0] - p0[0]) < 1e-6 and abs(p1[1] - p0[1]) < 1e-6:
        return                        # degenerate (line only touches a corner)
    style = _PLANE_STYLE.get(name, _PLANE_STYLE["g"])
    # Planes are drawn after in-plane axes and above them, so a glide that
    # shares a trace with an axis keeps its own dash pattern.
    z = 6 if name == "m" else 6.5
    if erase_under and name != "m":
        # The cell outline is a solid stroke on this same edge. A white
        # line under the dashes keeps the outline from filling the gaps.
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=2.6,
                solid_capstyle="butt", zorder=z - 0.2)
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


def _draw_split_circle(ax, center, members, radius=0.042, fontsize=6,
                       draw_labels=True):
    """One circle split by a vertical diameter, for two heights at one spot.

    The positive-sense label sits to the left of the circle and the other to
    the right. A comma, marking the opposite hand, leads the right-hand label.
    """
    from matplotlib.patches import Circle
    cx, cy = center
    ax.add_patch(Circle((cx, cy), radius, facecolor="white", edgecolor="k",
                        lw=1.0, zorder=3))
    ax.plot([cx, cx], [cy - radius, cy + radius], color="k", lw=0.7, zorder=4)
    if not draw_labels:
        return
    left, right = members
    if right[0][1] > left[0][1]:
        left, right = right, left
    def _with_comma(text, member):
        # member is (sortkey, label, comma?). The comma marks the enantiomorph,
        # which is not always the height written with a minus.
        return ("," + text) if len(member) > 2 and member[2] else text
    ax.text(cx - radius - 0.012, cy, _with_comma(left[1], left),
            fontsize=fontsize, ha="right", va="center", zorder=5)
    ax.text(cx + radius + 0.012, cy, _with_comma(right[1], right),
            fontsize=fontsize, ha="left", va="center", zorder=5)


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

    if _crystal_system(sg) == "cubic":
        drawn = _draw_cubic_polyhedra(ax, sg, frame, perm)
        xs = [c[0] for c in frame.corners] + [p[0] for p in drawn]
        ys = [c[1] for c in frame.corners] + [p[1] for p in drawn]
        pad = 0.08
        xl = (min(xs) - pad, max(xs) + pad)
        yl = (max(ys) + pad, min(ys) - pad)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.axis("off")
        frame.label_axes(ax, dlab, rlab)
        if show_title:
            num2, name = _sg_label(sg)
            pfx = f"#{num2}  " if num2 is not None else ""
            ax.set_title(f"{pfx}{_display_hm(name)}", fontsize=8)
        return ax

    # Half-cell lines. The cubic polyhedra above are not given this grid.
    for u in (0.5,):
        for a, b in (((u, 0.0), (u, 1.0)), ((0.0, u), (1.0, u))):
            p0, p1 = frame.pt(a), frame.pt(b)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.75", lw=0.4,
                    zorder=0)

    # collect distinct points first, then group by (x,y) so coincident
    # projections (different heights) can be drawn as ITA split circles.
    # The height label is exact: coef * z + t read off the operation.
    seen = set()
    pts = []   # (right, down, label, sortkey, det)
    for coef, t, coord, Wi, wi in general_position_images(sg, projection):
        W = np.array(Wi, float)
        w = np.array([float(x) for x in wi])
        det = round(np.linalg.det(W))
        # The comma marks the opposite hand, including the mirror that
        # lies parallel to the page.
        comma = det < 0
        base = _perm_vec(W @ x0 + w, perm)  # [0]=down, [1]=right, [2]=depth
        label = height_label(coef, t, coord)
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                p = base + np.array([tx, ty, 0.0])
                # Images ITA draws just outside the cell, not a full extra cell.
                if not (-0.16 <= p[0] <= 1.16 and -0.16 <= p[1] <= 1.16):
                    continue
                key = (round(p[0], 3), round(p[1], 3), label, comma)
                if key in seen:
                    continue
                seen.add(key)
                pts.append((p[1], p[0], label, (float(t), coef, coord), comma))

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
            if det:
                ax.text(ox, oy, ",", fontsize=8, zorder=4,
                        ha="center", va="center")

    # a down, b right, origin top-left. The pad keeps the images that sit
    # just outside the cell inside the axes.
    xl, yl = frame.limits(0.22)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.axis("off")
    frame.label_axes(ax, dlab, rlab)
    if show_title:
        num, name = _sg_label(sg)
        pfx = f"#{num}  " if num is not None else ""
        ax.set_title(f"{pfx}{_display_hm(name)}", fontsize=8)
    return ax


def _crystal_system(sg):
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    return str(system or "").lower()


def _sg_number(sg):
    n = getattr(sg, "number", None)
    if n is None and hasattr(sg, "base"):
        n = getattr(sg.base, "number", None)
    return n


def _hexagonal_family(sg):
    """Trigonal and hexagonal groups use the glide letter g, not n."""
    return _crystal_system(sg) in ("trigonal", "hexagonal")


def _stabilizer(ops, centre, tol=1e-5):
    """Operations that fix ``centre`` (torus distance)."""
    H = []
    c = np.asarray(centre, float)
    for W, w, det in ops:
        img = np.asarray(W, float) @ c + np.asarray(w, float) - c
        img = img - np.round(img)
        if float(np.linalg.norm(img)) < tol:
            H.append((np.asarray(W, float), np.asarray(w, float), det))
    return H


def _highest_site(ops):
    """The point fixed by the largest subgroup, nearest the origin on a tie."""
    coords = (0.0, 0.125, 0.25, 1.0 / 3.0, 0.375, 0.5, 0.625, 2.0 / 3.0,
              0.75, 0.875)
    best, best_key = np.zeros(3), (-1, 0.0)
    for x in coords:
        for y in coords:
            for z in coords:
                c = np.array([x, y, z])
                n = len(_stabilizer(ops, c))
                d = float(np.linalg.norm(np.minimum(c, 1.0 - c)))
                key = (n, -d)
                if key > best_key:
                    best_key, best = key, c
    return best


def _polyhedron_seed(sg):
    """A general point about 0.09 from the highest-symmetry site.

    The site-group orbit of a point this close is a compact polyhedron, about
    a quarter of the gap between neighbouring centres, instead of a hull that
    reaches across the cell and buries the corner labels.
    """
    c = _highest_site(_sg_ops(sg))
    offset = np.array([0.072, 0.048, 0.034])
    return (np.asarray(c, float) + offset) % 1.0


def _polyhedron_sets(sg, point, projection="c"):
    """One polyhedron per image of the highest-symmetry site.

    Each vertex is a general-position image, kept with its centre (within half
    a cell) so the hull is not torn by folding coordinates one at a time.
    Vertices carry the ITA height label and the sign of the operation.
    """
    ops = _sg_ops(sg)
    c = _highest_site(ops)
    centres, seen_c = [], set()
    for W, w, _det in ops:
        img = (np.asarray(W, float) @ c + np.asarray(w, float)) % 1.0
        key = tuple(np.round(img, 5))
        if key not in seen_c:
            seen_c.add(key)
            centres.append(img)
    groups = [[] for _ in centres]
    x = np.asarray(point, float)
    for coef, t, coord, Wi, wi in general_position_images(sg, projection):
        W = np.asarray(Wi, float)
        w = np.array([float(v) for v in wi])
        xyz = W @ x + w
        best_i, best_r, best_d = 0, xyz - centres[0], 1e9
        for i, centre in enumerate(centres):
            r = xyz - centre
            r = r - np.round(r)
            dist = float(np.dot(r, r))
            if dist < best_d:
                best_i, best_r, best_d = i, r, dist
        det = int(round(np.linalg.det(W)))
        groups[best_i].append((
            centres[best_i] + best_r,
            height_label(coef, t, coord),
            det,
        ))
    return list(zip(centres, groups))


def _edge_translates(centre, perm):
    """Lattice shifts that repeat a boundary centre onto the far cell edge.

    A centre at x = 0 is also drawn at x = 1, and the same for y. The depth
    axis is not repeated: that coordinate is the height, not a page edge.
    """
    options = [[0.0], [0.0], [0.0]]
    for axis in (perm[0], perm[1]):
        folded = float(centre[axis]) % 1.0
        if min(folded, 1.0 - folded) < 1e-3:
            options[axis] = [0.0, 1.0]
    shifts = []
    for a in options[0]:
        for b in options[1]:
            for c in options[2]:
                shifts.append(np.array([a, b, c], float))
    return shifts


def _hull_polygon(pts2):
    """2D convex hull as an (n, 2) array, or None when the points are flat."""
    from scipy.spatial import ConvexHull
    arr = np.unique(np.round(np.asarray(pts2, float), 5), axis=0)
    if len(arr) < 3:
        return None
    try:
        hull = ConvexHull(arr)
    except Exception:
        return None
    return arr[hull.vertices]


def _draw_polyhedron_faces(ax, pts2, fc, zorder=2):
    from matplotlib.patches import Polygon
    hull = _hull_polygon(pts2)
    if hull is None:
        if len(pts2) >= 2:
            ax.plot([p[0] for p in pts2], [p[1] for p in pts2],
                    color="k", lw=0.6, zorder=zorder + 1)
        return
    ax.add_patch(Polygon(hull, closed=True, facecolor=fc, edgecolor="k",
                         lw=0.6, zorder=zorder))


def _draw_position_mark(ax, xy, det):
    """Open circle at a general position, with a comma for the opposite hand."""
    ax.plot(xy[0], xy[1], "o", ms=4.5, mfc="white", mec="k", mew=0.6, zorder=4)
    if det < 0:
        ax.text(xy[0], xy[1], ",", fontsize=6, zorder=5, ha="center", va="center")


def _corner_buckets(marks, tol=0.016):
    """Group corners that project on top of each other.

    Only a true overlap is merged. A wider tolerance glued a whole corner
    into one circle carrying every label. Two heights at one point still
    share a circle; their captions stack along the outward ray.
    """
    buckets = []
    centres = []
    for xy, label, det in marks:
        xy = np.asarray(xy, float)
        placed = False
        for i, centre in enumerate(centres):
            if float(np.linalg.norm(xy - centre)) <= tol:
                buckets[i].append((label, det, xy))
                placed = True
                break
        if not placed:
            centres.append(xy)
            buckets.append([(label, det, xy)])
    return buckets


def _layout_outward_labels(centre_xy, spots, gap=0.016):
    """Place each caption on the ray from the polyhedron centre through its circle.

    ``spots`` is a list of ``(spot, [labels])``. The anchor sits just outside
    the marker. Two heights at one circle stack along that same ray, so the
    text stays attached to its point.
    """
    centre_xy = np.asarray(centre_xy, float)
    items = []
    for spot, labels in spots:
        away = np.asarray(spot, float) - centre_xy
        nrm = float(np.linalg.norm(away))
        away = away / nrm if nrm > 1e-6 else np.array([0.0, -1.0])
        for i, label in enumerate(labels):
            items.append({
                "pos": np.asarray(spot, float) + away * (gap + 0.020 * i),
                "away": away,
                "text": label,
            })
    return items


def _outward_align(away):
    """Anchor the caption on its inner edge, so the letters grow away from the hull.

    These axes run y down the page, so a positive data-y step is toward the
    bottom of the figure and the text has to hang that way.
    """
    ax_, ay = float(away[0]), float(away[1])
    if abs(ax_) >= abs(ay):
        return ("left" if ax_ > 0 else "right"), "center"
    return "center", ("top" if ay > 0 else "bottom")


def _draw_outward_labels(ax, centre_xy, spots, gap=0.016, fontsize=4.5):
    anchors = []
    for item in _layout_outward_labels(centre_xy, spots, gap=gap):
        pos = item["pos"]
        ha, va = _outward_align(item["away"])
        ax.text(float(pos[0]), float(pos[1]), item["text"], fontsize=fontsize,
                zorder=5, ha=ha, va=va)
        anchors.append((float(pos[0]), float(pos[1])))
    return anchors


def _draw_cubic_polyhedra(ax, sg, frame, perm):
    """Orthogonal polyhedra over the closed cell, with a circle at each corner.

    Centres on the boundary are repeated at the opposite edge, so an F lattice
    projected down c shows the whole half-grid, including x = 1 and y = 1.
    The representative point is the compact seed near the highest site, so
    neighbouring hulls stay apart. Returns the plot-space points the axes
    must keep inside the frame.
    """
    proj_name = next(name for name, (p, *_) in _PROJ.items() if tuple(p) == tuple(perm))
    sets = _polyhedron_sets(sg, _polyhedron_seed(sg), projection=proj_name)
    # Two centres that differ only in the depth (a 3-fold and the same axis
    # half a cell along c) project to one polyhedron. Drawing both stacks
    # the captions. The heights stay, as the corner labels.
    merged = {}
    order = []
    for centre, items in sets:
        c = np.asarray(centre, float)
        key = (round(float(c[perm[1]]) % 1.0, 3), round(float(c[perm[0]]) % 1.0, 3))
        slot = merged.get(key)
        if slot is None:
            merged[key] = [c, list(items)]
            order.append(key)
        else:
            slot[1].extend(items)
    sets = [(merged[k][0], merged[k][1]) for k in order]
    fc = (0.25, 0.45, 0.75, 0.18)
    drawn = []

    for centre, items in sets:
        if not items:
            continue
        for shift in _edge_translates(centre, perm):
            marks = []
            for xyz, label, det in items:
                q = _perm_vec(np.asarray(xyz, float) + shift, perm)
                xy = frame.pt((q[1], q[0]))
                marks.append((xy, label, det))
            _draw_polyhedron_faces(ax, [m[0] for m in marks], fc)
            centre_xy = frame.pt((
                float((centre + shift)[perm[1]]),
                float((centre + shift)[perm[0]]),
            ))
            spots = []
            for members in _corner_buckets(marks):
                xy = members[0][2]
                spot = (float(xy[0]), float(xy[1]))
                uniq = []
                for lab, det, _xy in members:
                    if not any(lab == u[0] for u in uniq):
                        uniq.append((lab, det))
                if len(uniq) == 2:
                    _draw_split_circle(
                        ax, spot,
                        [((0, 0), uniq[0][0], uniq[0][1] < 0),
                         ((0, 1), uniq[1][0], uniq[1][1] < 0)],
                        radius=0.010, fontsize=5, draw_labels=False)
                else:
                    det = -1 if any(d < 0 for _lab, d in uniq) else 1
                    _draw_position_mark(ax, spot, det)
                drawn.append(spot)
                spots.append((spot, [lab for lab, _d in uniq]))
            drawn.extend(_draw_outward_labels(ax, centre_xy, spots))
    return drawn


def _draw_cubic_perspective(ax, sg, point=None):
    """Cabinet view of the same polyhedra, with a circle and height at each corner."""
    if point is None:
        point = _polyhedron_seed(sg)
    sets = _polyhedron_sets(sg, point, projection="c")
    fc = (0.25, 0.45, 0.75, 0.22)

    def cabinet(p):
        p = np.asarray(p, float)
        return np.array([p[0] + 0.35 * p[2], p[1] + 0.22 * p[2]])

    pts = []
    # In-cell centres only. The orthogonal plate repeats the boundary ones at
    # x = 1 and y = 1; repeating them here drops a second copy outside the cube.
    for centre, items in sets:
        if not items:
            continue
        face = []
        labelled = []
        for xyz, label, det in items:
            xy = cabinet(np.asarray(xyz, float))
            face.append(xy)
            labelled.append((xy, label, det))
        _draw_polyhedron_faces(ax, face, fc)
        pts.extend(face)
        centre_xy = cabinet(np.asarray(centre, float))
        spots = []
        for members in _corner_buckets(labelled):
            spot = (float(members[0][2][0]), float(members[0][2][1]))
            uniq = []
            for lab, det, _xy in members:
                if not any(lab == u[0] for u in uniq):
                    uniq.append((lab, det))
            det = -1 if any(d < 0 for _lab, d in uniq) else 1
            _draw_position_mark(ax, spot, det)
            pts.append(spot)
            spots.append((spot, [lab for lab, _d in uniq]))
        pts.extend(_draw_outward_labels(
            ax, centre_xy, spots, fontsize=4))
    corners = {(i, j, k): cabinet([i, j, k])
               for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for i, j, k in corners:
        for step in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (i + step[0], j + step[1], k + step[2])
            if nb in corners:
                a, b = corners[(i, j, k)], corners[nb]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="0.25", lw=0.8,
                        zorder=3)
                pts.extend((a, b))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 0.08
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)
    ax.set_aspect("equal")
    ax.axis("off")


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


def _orient_d_glides(elements, perm):
    """Give successive d planes opposite in-plane arrows.

    The stored glide representative keeps one component's sign and lets the
    alternating component fall along the projection axis, so a setting draws
    every arrow the same way. The first plane of each family takes the sense
    of the symmetry operation, and the next plane points the other way.
    """
    families = {}
    for el in elements:
        if el.get("symbol") != "d" or el.get("axis_exact") is None:
            continue
        axis = list(el["axis_exact"])
        for comp in axis:
            if comp:
                if comp < 0:
                    axis = [-c for c in axis]
                break
        families.setdefault(tuple(axis), []).append(el)

    def _sign_of(comps):
        for i in range(3):
            if i == perm[2]:
                continue
            if comps[i] != 0:
                return 1 if comps[i] > 0 else -1
        return 1

    def _stored_sign(el):
        return _sign_of(el["intrinsic_exact"])

    def _operation_sign(el):
        # The glide in the symmetry operation, not a centring translate that
        # arrived first and flipped both components. (0,1/4,1/4) is that
        # operation; (0,3/4,3/4) is the same glide plus a centring vector.
        raws = list(el.get("_glide_raws") or [])
        if not raws:
            raws = [el["intrinsic_exact"]]
        raw = min(raws, key=lambda g: (
            sum(Fraction(c) % 1 for c in g),
            tuple(Fraction(c) % 1 for c in g)))
        comps = []
        for c in raw:
            c = Fraction(c) % 1
            if c > Fraction(1, 2):
                c -= 1
            comps.append(c)
        return _sign_of(comps)

    for group in families.values():
        group.sort(key=lambda e: sum(
            e["axis_exact"][i] * e["location_exact"][i] for i in range(3)))
        base = _operation_sign(group[0])
        for i, el in enumerate(group):
            want = base if i % 2 == 0 else -base
            g = el["intrinsic_exact"]
            if _stored_sign(el) != want:
                g = tuple(-c for c in g)
            el["_d_arrow"] = g


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


def _distinct_glide_axes(raws, hexagonal=False):
    """True when one plane carries two glide directions and is an e glide.

    Two axial halves along different axes are the orthorhombic e. A c/2
    together with an in-plane diagonal half is the I-tetragonal e. A
    hexagonal g, whose translation is not one of those halves, stays g.
    """
    if hexagonal:
        return False
    axes = set()
    half = Fraction(1, 2)
    has_c = False
    has_diag = False
    for raw in raws:
        if not raw:
            continue
        nonzero = [i for i, comp in enumerate(raw) if comp != 0]
        if len(nonzero) == 1 and abs(raw[nonzero[0]]) == half:
            axes.add(nonzero[0])
            if nonzero[0] == 2:
                has_c = True
        if (len(nonzero) == 2 and 2 not in nonzero
                and all(abs(raw[i]) == half for i in nonzero)):
            has_diag = True
    return len(axes) >= 2 or (has_c and has_diag)


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


def _same_axis_line(a, b, lattice_vecs=None):
    """True when two axes are the same line, even if the points differ.

    The two locations can differ by a lattice vector and still be one line,
    so the comparison is made after subtracting that vector.
    """
    if a.get("axis_exact") is None or b.get("axis_exact") is None:
        return False
    if _axis_sign_key(a["axis_exact"]) != _axis_sign_key(b["axis_exact"]):
        return False
    axis = _axis_sign_key(a["axis_exact"])
    delta = tuple(a["location_exact"][i] - b["location_exact"][i] for i in range(3))
    shifts = lattice_vecs if lattice_vecs is not None else (
        (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
    for L in shifts:
        d = (delta[0] - L[0], delta[1] - L[1], delta[2] - L[2])
        cross = (
            d[1] * axis[2] - d[2] * axis[1],
            d[2] * axis[0] - d[0] * axis[2],
            d[0] * axis[1] - d[1] * axis[0],
        )
        if cross == (0, 0, 0):
            return True
    return False


def _mark_contained_axes(elements, lattice_vecs=None):
    """Mark sub-axes that sit on a higher axis of the same line.

    A 3 on a −3 is tagged even when the −3 is at a different point of the
    line. A 2 or 2₁ inside a 4-fold is tagged the way a 2 or 3 inside a
    6-fold already is. They stay in the list and the plate draws the host.
    """
    axes = [el for el in elements
            if el["type"] in ("rotation", "screw", "rotoinversion")]
    clusters = []
    for el in axes:
        for cluster in clusters:
            if _same_axis_line(el, cluster[0], lattice_vecs):
                cluster.append(el)
                break
        else:
            clusters.append([el])
    for group in clusters:
        host = next((el for el in group
                     if el["type"] in ("rotation", "screw") and el["order"] == 6),
                    None)
        if host is not None:
            for el in group:
                if el is host:
                    continue
                if el["type"] in ("rotation", "screw") and el["order"] in (2, 3):
                    el["contained_in"] = host["symbol"]
            continue
        four = next((el for el in group
                     if el["type"] in ("rotation", "screw") and el["order"] == 4),
                    None)
        if four is not None:
            for el in group:
                if el is four:
                    continue
                if el["type"] in ("rotation", "screw") and el["order"] == 2:
                    el["contained_in"] = four["symbol"]
        if any(el["type"] == "rotoinversion" and el["symbol"] == "-3"
               for el in group):
            for el in group:
                if el["type"] in ("rotation", "screw") and el["order"] == 3:
                    el["contained_in"] = "-3"
        # The 3 on a −6 is the proper part of that rotoinversion. The plate
        # draws the −6, not a second triangle.
        if any(el["type"] == "rotoinversion" and el["order"] == 6
               for el in group):
            for el in group:
                if el.get("contained_in"):
                    continue
                if el["type"] in ("rotation", "screw") and el["order"] == 3:
                    el["contained_in"] = "-6"
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
                        if _distinct_glide_axes(raws, hexagonal=hexagonal):
                            prev["symbol"] = "e"
                            prev["type"] = "glide"
                        elif prev["symbol"] == "g" and el["symbol"] == "c":
                            # Rhombohedral centring can write the same c-glide
                            # as a diagonal g. The ITA letter is c.
                            prev["symbol"] = "c"
                            prev["intrinsic_exact"] = el["intrinsic_exact"]
                            prev["intrinsic"] = el["intrinsic"]
                    continue
                el["_glide_raws"] = [] if raw is None else [raw]
                planes[(axis, loc)] = el
            key = (el["type"], el["symbol"], axis, loc)
            if key in seen:
                if el.get("_coincident_symbol"):
                    twin_key = ("glide", el["_coincident_symbol"], axis, loc)
                    if twin_key not in seen:
                        twin = dict(el)
                        twin["symbol"] = el["_coincident_symbol"]
                        twin["type"] = "glide"
                        twin["contained_in"] = "m"
                        raw_i = el.get("_coincident_intrinsic")
                        if raw_i is not None:
                            twin["intrinsic_exact"] = raw_i
                            twin["intrinsic"] = [float(c) for c in raw_i]
                        seen.add(twin_key)
                        out.append(twin)
                continue
            seen.add(key)
            out.append(el)
            if el.get("_coincident_symbol"):
                twin = dict(el)
                twin["symbol"] = el["_coincident_symbol"]
                twin["type"] = "glide"
                twin["contained_in"] = "m"
                raw_i = el.get("_coincident_intrinsic")
                if raw_i is not None:
                    twin["intrinsic_exact"] = raw_i
                    twin["intrinsic"] = [float(c) for c in raw_i]
                twin.pop("_coincident_symbol", None)
                tkey = (twin["type"], twin["symbol"], axis, loc)
                if tkey not in seen:
                    seen.add(tkey)
                    out.append(twin)
    return _mark_contained_axes(out, lattice.vecs)




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
                             show_centring=False, mark_origin=False):
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
    _orient_d_glides(els, perm)
    omitted = 0

    # Establish the inverted y-axis (a' down the page, ITA convention) BEFORE
    # drawing, so orientation-aware glyphs (the parallel-plane corner bracket)
    # detect the correct screen sense at draw time.
    def _parallel_key(el):
        depth = round(float(np.asarray(el["location"], float)[perm[2]]) % 1.0, 3)
        raws = list(el.get("_glide_raws") or [])
        if not raws and el.get("intrinsic_exact") is not None:
            raws = [el["intrinsic_exact"]]
        gkey = tuple(sorted(
            tuple(round(float(c), 3) for c in raw) for raw in raws))
        return (el["symbol"], depth, gkey)

    parallel_planes = []
    seen_par = set()
    for el in els:
        axis = el.get("axis")
        if el["type"] in ("mirror", "glide") and axis is not None \
                and _dir_class(_perm_vec(axis, perm)) == "c":
            key = _parallel_key(el)
            if key not in seen_par:
                seen_par.add(key)
                parallel_planes.append(el)
    # h and h+1/2 are one plane. Keep the lower height and its arrow.
    # A second symbol, or the same letter with a different glide, stays.
    by_glide = {}
    for el in parallel_planes:
        by_glide.setdefault((_parallel_key(el)[0], _parallel_key(el)[2]), []).append(el)
    collapsed = []
    for group in by_glide.values():
        depths = {}
        for el in group:
            depths.setdefault(_parallel_key(el)[1], el)
        for depth, el in sorted(depths.items()):
            partner = round((depth + 0.5) % 1.0, 3)
            if partner in depths and partner < depth - 1e-9:
                continue
            collapsed.append(el)
    parallel_planes = collapsed
    xl, yl = frame.limits(0.24)
    if parallel_planes:
        # One bracket per plane, stacked above the cell so two glide
        # arrows do not share a corner.
        left = min(c[0] for c in frame.corners)
        top = min(c[1] for c in frame.corners)
        stack = 0.16 + 0.32 * len(parallel_planes)
        xl = (min(xl[0], left - 0.42), xl[1])
        yl = (yl[0], min(yl[1], top - stack))
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

    parallel_drawn = set()   # (symbol, height, glide) already bracketed
    kept_par = {_parallel_key(el) for el in parallel_planes}
    plane_traces = []               # (origin, direction) of planes ⊥ page
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
                if cap:
                    tx, ty = _caption_xy(xy, frame, 0.08, -0.06)
                    ax.text(tx, ty, cap, fontsize=6,
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
                    "el": el, "order": el["order"],
                })
                slot["order"] = max(slot["order"], el["order"])
                if t == "rotation":
                    slot["full"] = True
                else:
                    slot["half"] = True
            elif el["order"] == 2 and dc == "gen":
                inclined2.append(el)
            elif el["order"] == 3 and dc == "gen":
                inclined3.append(el)
            elif el["order"] in (4, 6) and dc == "ab":
                # An axis lying in the page is an arrowhead at the cell edge,
                # the same size as a 2-fold. A square here would be the symbol
                # for an axis perpendicular to the page and would cover the edge.
                slot = inplane_axes.setdefault(_trace_id(el), {
                    "full": False, "half": False, "loc": P(loc), "d": d,
                    "el": el, "order": el["order"],
                })
                slot["order"] = max(slot["order"], el["order"])
                if t == "screw":
                    slot["half"] = True
                else:
                    slot["full"] = True
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
                gdir = None
                if sym == "d":
                    raw = el.get("_d_arrow", el.get("intrinsic"))
                    if raw is not None:
                        gdir = frame.vec(frac_dir(np.asarray(raw, float)))
                plane_traces.append((np.asarray(P(loc), float), np.asarray(d, float)))
                draw_line_family(
                    P(loc), d,
                    lambda p0, p1, _dp, _s=sym, _g=gdir:
                        draw_plane_symbol(
                            ax, p0, p1, _s, glide_dir=_g,
                            erase_under=_on_cell_edge(p0, p1, frame)))
                cap = captions.get(_trace_id(el), "")
                if cap:
                    for xy in edge_copies(P(loc)):
                        tx, ty = _caption_xy(xy, frame, 0.08, -0.06)
                        ax.text(tx, ty, cap, fontsize=6,
                                zorder=8, ha="left", va="top")
            elif dc == "c":
                # One bracket per symbol and glide. The copy half a cell
                # away is the same plane and is not drawn again.
                pkey = _parallel_key(el)
                if pkey not in kept_par:
                    continue
                if pkey not in parallel_drawn:
                    slot = next(i for i, other in enumerate(parallel_planes)
                                if _parallel_key(other) == pkey)
                    if el["symbol"] == "d" and el.get("_d_arrow") is not None:
                        sources = [el["_d_arrow"]]
                    else:
                        raws = el.get("_glide_raws") if el["symbol"] == "e" else None
                        sources = list(raws) if raws else []
                        if not sources and el.get("intrinsic") is not None:
                            sources = [el["intrinsic"]]
                    gdirs = [
                        frame.vec(_cell_edge_sense(
                            frac_dir(np.asarray(raw, float))))
                        for raw in sources
                    ]
                    left = min(c[0] for c in frame.corners)
                    top = min(c[1] for c in frame.corners)
                    size = 0.18
                    corner = (left - 0.06, top - 0.14 - slot * 0.32)
                    draw_parallel_plane_symbol(
                        ax, el["symbol"],
                        corner=corner,
                        size=size, glide_dirs=gdirs or None)
                    parallel_drawn.add(pkey)
                    partner = round((pkey[1] + 0.5) % 1.0, 3)
                    heights = {_parallel_key(other)[1] for other in parallel_planes
                               if other["symbol"] == el["symbol"]}
                    # The copy half a cell away is the same note. Keep the
                    # one below 1/2.
                    if partner in heights and partner < pkey[1]:
                        cap = ""
                    else:
                        cap = frac_label(pkey[1])
                    if cap:
                        ax.text(corner[0] - 0.03, corner[1] + size * 0.45, cap,
                                fontsize=6, zorder=8, ha="right", va="center")
            else:
                omitted += 1
            continue

    drawn_heads = set()

    def _one_head(tip, outward, full, size, order=2):
        """One arrowhead. A second head at the same point, sense and order is dropped.

        A 4-fold head is not dropped because a 2-fold head already sits nearby.
        Returns False when this tip repeats a head already drawn, so the caller
        does not add a second shaft.
        """
        tip = np.asarray(tip, float)
        outward = np.asarray(outward, float)
        outward = outward / (np.linalg.norm(outward) or 1.0)
        key = (round(float(tip[0]), 2), round(float(tip[1]), 2),
               round(float(outward[0]), 1), round(float(outward[1]), 1),
               bool(full), int(order))
        if key in drawn_heads:
            return False
        # Lattice copies and segments that meet on one edge share a tip.
        # Heads a quarter-cell apart are different axes and are kept.
        for prev in drawn_heads:
            if prev[4] != bool(full) or prev[5] != int(order):
                continue
            if abs(prev[0] - key[0]) <= 0.05 and abs(prev[1] - key[1]) <= 0.05:
                if prev[2] == key[2] and prev[3] == key[3]:
                    return False
        drawn_heads.add(key)
        _draw_inplane_arrowhead(ax, tip, outward, full=full, size=size)
        return True

    def _shaft_to(tip, outward, head_size, shaft_len):
        shaft_end = np.asarray(tip, float) - outward * head_size
        shaft_in = shaft_end - outward * shaft_len
        ax.plot([shaft_in[0], shaft_end[0]], [shaft_in[1], shaft_end[1]],
                color="k", lw=1.0, zorder=3)

    def draw_axis_line(p0, p1, full, half, key, stub, order=2):
        """Arrowheads just outside the cell, one pair per in-plane axis.

        The trace itself is not stroked. A solid line through the cell reads
        as a mirror in groups that have no glide on that line. A 2 and a 2₁
        on the same trace sit a head-width apart, each on its own short shaft,
        far enough outside the frame to clear a lens centred on the edge.
        """
        del stub  # every in-plane axis is an edge stub
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        head_size = 0.026
        shaft_len = 0.018
        mid = 0.5 * (p0 + p1)
        ends = []
        for end in (p0, p1):
            outward = end - mid
            nrm = np.linalg.norm(outward) or 1.0
            ends.append((end, outward / nrm))
        # Past the perpendicular glyph (a lens on the edge is about 0.045).
        extra = 0.065
        for end, outward in ends:
            perp = np.array([-outward[1], outward[0]])
            if full and half:
                for sign, is_full in ((1.0, True), (-1.0, False)):
                    tip = end + outward * extra + perp * 0.032 * sign
                    if _one_head(tip, outward, is_full, head_size, order=order):
                        _shaft_to(tip, outward, head_size, shaft_len)
            else:
                tip = end + outward * extra
                if _one_head(tip, outward, bool(full), head_size, order=order):
                    _shaft_to(tip, outward, head_size, shaft_len)
        cap = captions.get(key, "")
        if cap and key not in labelled:
            labelled.add(key)
            end = p0 if p0[1] <= p1[1] else p1
            other = p1 if p0[1] <= p1[1] else p0
            along = other - end
            span = np.linalg.norm(along) or 1.0
            # Beside the trace, clear of a glyph at the centre or the corner.
            pos = end + along * (0.22 if span > 0.4 else 0.5)
            perp = np.array([-along[1], along[0]]) / span * 0.06
            ax.text(pos[0] + perp[0], pos[1] + perp[1], cap,
                    fontsize=7, color="k", zorder=8, ha="left", va="center")

    for key, slot in inplane_axes.items():
        direction = np.asarray(slot["d"], float)
        for fp0, fp1 in _line_lattice_segments(slot["loc"], direction):
            draw_axis_line(
                frame.pt(fp0), frame.pt(fp1),
                slot["full"], slot["half"], key,
                True, order=slot.get("order", 2))

    def _screw_k(el):
        if el["type"] == "screw" and "_" in el["symbol"]:
            return int(el["symbol"].split("_")[1])
        return 0

    def _draw_inclined_family(elements, kind):
        """One glyph where each inclined axis cuts the drawing plane at height 0.

        Several axes can cut the page at the same point. Each keeps its own
        symbol, shifted a short step along its own projected direction so the
        glyphs stay separate. A lone symbol, such as the origin 3-fold, stays
        on the point and is copied to the cell corners.
        """
        sites = {}
        for el in elements:
            k = _screw_k(el)
            roto = el["type"] == "rotoinversion"
            dvec = frac_dir(el["axis"])
            dvec = dvec / (np.linalg.norm(dvec) or 1.0)
            plotted = np.asarray(frame.vec(dvec), float)
            for comp in plotted:
                if abs(comp) > 1e-9:
                    if comp < 0:
                        plotted = -plotted
                    break
            direction = tuple(np.round(plotted, 3))
            for rd, depth in _axis_page_points(el, perm):
                key = (round(rd[0], 3), round(rd[1], 3))
                slot = sites.setdefault(
                    key, {"depths": [], "glyphs": {}, "rd": rd})
                slot["depths"].append(depth)
                # One glyph per symbol. Opposite orientations of one axis,
                # and the several <111> lines through the origin, share it.
                mark = (k, roto)
                slot["glyphs"].setdefault(mark, direction)
        def _near_edge(xy, tol=0.03):
            x, y = float(xy[0]), float(xy[1])
            return min(abs(x), abs(1.0 - x), abs(y), abs(1.0 - y)) < tol

        def _near_c_axis(xy, tol=0.04):
            x, y = float(xy[0]), float(xy[1])
            for ckey in c_sites:
                for other in edge_copies(ckey):
                    if (abs(float(other[0]) - x) <= tol
                            and abs(float(other[1]) - y) <= tol):
                        return True
            return False

        for key, slot in sites.items():
            cap = _height_caption(slot["depths"])
            # Rotoinversions last, so an open −3 is not painted over by a 3₁.
            glyphs = sorted(slot["glyphs"].items(), key=lambda item: (item[0][1], item[0][0]))
            n = len(glyphs)
            for xy in edge_copies(slot["rd"]):
                for i, ((k, roto), direction) in enumerate(glyphs):
                    direction = np.asarray(direction, float)
                    dn = np.linalg.norm(direction) or 1.0
                    direction = direction / dn
                    pos = np.asarray(xy, float)
                    # A 4/m square is drawn later on this same point. Step the
                    # inclined symbol out along its axis so the −3 stays visible.
                    if _near_c_axis(pos):
                        pos = pos + direction * 0.07
                    if n > 1 and kind == "tri":
                        # Side by side, across the axis, so 3₁ and 3₂ do not
                        # sit on top of each other or walk into the next site.
                        perp = np.array([-direction[1], direction[0]])
                        pos = pos + perp * 0.07 * (i - 0.5 * (n - 1))
                    elif n > 1:
                        pos = pos + direction * 0.028 * (i - 0.5 * (n - 1))
                    if kind == "lens" and _near_edge(xy):
                        # The cut is on the cell border. A filled lens there
                        # covers the edge; use the same small head as a 2-fold.
                        _one_head(pos, direction, k == 0, 0.026)
                    elif kind == "lens":
                        ang = math.atan2(float(direction[1]), float(direction[0]))
                        _draw_lens(ax, pos, 0.032, angle=ang, fc="k", ec="k",
                                   lw=0.9, zorder=5)
                        if k:
                            tang = np.array([-math.sin(ang), math.cos(ang)])
                            tip = pos + direction * 0.055
                            hook = tip + tang * 0.032
                            _ink(ax, [pos[0], tip[0], hook[0]],
                                 [pos[1], tip[1], hook[1]],
                                 color="k", lw=1.2, zorder=8)
                    else:
                        draw_axis_symbol(
                            ax, pos, 3, screw_k=k, rotoinv=roto, size=0.018)
                if cap and ("inclined", key) not in labelled:
                    labelled.add(("inclined", key))
                    tx, ty = _caption_xy(xy, frame)
                    ax.text(tx, ty, cap, fontsize=7,
                            zorder=8, ha="left", va="center")

    _draw_inclined_family(inclined2, "lens")
    _draw_inclined_family(inclined3, "tri")

    for key, s in c_sites.items():
        cap = captions.get(("point", tuple(np.round(key, 3))), "")
        for xy in edge_copies(key):
            _draw_combined_axis(ax, xy, s["max_rot"], s["rot_k"], s["roto"],
                                inversion=s.get("inv", False))
            if cap:
                tx, ty = _caption_xy(xy, frame)
                ax.text(tx, ty, cap, fontsize=7,
                        zorder=8, ha="left", va="center")

    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_aspect("equal"); ax.axis("off")
    frame.label_axes(ax, dlab, rlab, off=0.1)
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    _separate_overlapping_labels(ax)
    if mark_origin:
        ax.text(-0.07, -0.05, "0", fontsize=8, ha="right", va="center",
                zorder=9, color="k")
    if show_title:
        extra = f"  (+{omitted} oblique)" if omitted else ""
        num, name = _sg_label(sg)
        pfx = f"#{num}  " if num is not None else ""
        ax.set_title(f"{pfx}{_display_hm(name)}{extra}", fontsize=8)
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
    four_xy = set()
    six_xy = set()

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
                xy = _proj_xy(el)
                if t == "rotation" and el["order"] == 2 and xy is not None:
                    two_xy.add(xy)
                if t == "rotation" and el["order"] == 4 and xy is not None:
                    four_xy.add(xy)
                if t == "rotation" and el["order"] == 6 and xy is not None:
                    six_xy.add(xy)
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
                raws = el.get("_glide_raws") if sym == "e" else None
                sources = list(raws) if raws else []
                if not sources and el.get("intrinsic") is not None:
                    sources = [el["intrinsic"]]
                gdirs = []
                for intr in sources:
                    pv = _perm_vec(np.asarray(intr, float), perm)
                    gdirs.append(frame.vec(_cell_edge_sense(
                        np.array([pv[1], pv[0]]))))
                par_planes.setdefault(sym, gdirs)
    # A trace that carries both a 2 and a 2_1 is drawn with both heads.
    inplane["both"] = inplane["rot"] and inplane["screw"]

    def _axis_lbl(order, k, ro):
        if ro:
            return "\u2212" + str(order)
        if k:
            return str(order) + str(k).translate(_SUBDIGIT)
        return str(order)

    y_top = 9.0
    # Room under the title so "elements present" does not run into the first row.
    y = y_top - 1.45
    ax.set_title("elements present", fontsize=10, pad=12)
    if two_xy and two_xy <= inv_xy and 2 in perp:
        # Every 2-fold shares its site with an inversion, so the plate draws
        # 2/m and the legend does not also list a bare 2.
        perp[2] = {item for item in perp[2] if item != (0, False)}
        if not perp[2]:
            del perp[2]
    if perp:
        ax.text(0.15, y, "Axes ⊥ page:", fontsize=8, style="italic")
        y -= 1.05
        for order in sorted(perp):
            for k, ro in sorted(perp[order]):
                draw_axis_symbol(ax, (0.55, y), order, screw_k=k,
                                 rotoinv=ro, size=0.34)
                ax.text(1.65, y, _axis_lbl(order, k, ro), fontsize=8,
                        va="center")
                y -= 1.15
        y -= 0.2
    if inplane["rot"] or inplane["screw"]:
        ax.text(0.05, y, "In the plane of the page:", fontsize=8, style="italic")
        y -= 1.0
        if inplane["rot"]:
            ax.plot([0.15, 0.95], [y, y], "k-", lw=1.3)
            _draw_inplane_arrowhead(ax, (1.0, y), (1, 0), full=True, size=0.28)
            ax.text(1.3, y, "2  (full head)", fontsize=8, va="center")
            y -= 0.9
        if inplane["screw"]:
            ax.plot([0.15, 0.95], [y, y], "k-", lw=1.3)
            _draw_inplane_arrowhead(ax, (1.0, y), (1, 0), full=False, size=0.28)
            ax.text(1.3, y, "2\u2081  (half head)", fontsize=8, va="center")
            y -= 0.9
        y -= 0.3
    if inplane_n:
        ax.text(0.15, y, "Axes in the page, order > 2:", fontsize=8,
                style="italic")
        y -= 1.0
        for order, k, ro in sorted(inplane_n):
            draw_axis_symbol(ax, (0.55, y), order, screw_k=k, rotoinv=ro,
                             size=0.34)
            ax.text(1.65, y, _axis_lbl(order, k, ro), fontsize=8, va="center")
            y -= 1.05
        y -= 0.2
    if inclined_3:
        ax.text(0.15, y, "Inclined 3-fold:", fontsize=8, style="italic")
        y -= 1.0
        for k, ro in sorted(inclined_3):
            draw_axis_symbol(ax, (0.55, y), 3, screw_k=k, rotoinv=ro, size=0.34)
            ax.text(1.65, y, _axis_lbl(3, k, ro), fontsize=8, va="center")
            y -= 0.95
        y -= 0.2
    if inclined_2["rot"] or inclined_2["screw"]:
        ax.text(0.15, y, "Inclined 2-fold:", fontsize=8, style="italic")
        y -= 0.9
        if inclined_2["rot"]:
            _draw_lens(ax, (0.55, y), 0.12, fc="k", ec="k", lw=0.8, zorder=5)
            ax.text(1.65, y, "2", fontsize=8, va="center")
            y -= 0.9
        if inclined_2["screw"]:
            _draw_lens(ax, (0.55, y), 0.12, angle=0.0, fc="k", ec="k",
                       lw=0.8, zorder=5)
            _ink(ax, [0.55, 0.78, 0.70], [y, y, y + 0.10],
                 color="k", lw=1.0, zorder=6)
            ax.text(1.65, y, "2\u2081", fontsize=8, va="center")
            y -= 0.9
    if planes:
        ax.text(0.05, y, "Planes ⊥ page:", fontsize=8, style="italic")
        y -= 0.9
        for name in sorted(planes):
            x1 = 1.7 if name in ("a", "b", "c", "e", "n", "g", "d") else 1.05
            draw_plane_symbol(ax, (0.15, y), (x1, y), name)
            ax.text(x1 + 0.2, y, name, fontsize=8, va="center")
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
                                       size=0.5, glide_dirs=par_planes[name])
            ax.text(1.3, y, f"{name}  (corner bracket)", fontsize=8,
                    va="center")
            y -= 1.0
        y -= 0.2
    if has_inv:
        ax.text(0.05, y, "Inversion:", fontsize=8, style="italic")
        y -= 0.9
        draw_inversion(ax, (0.55, y), size=0.05)
        ax.text(1.65, y, "\u22121", fontsize=8, va="center")
        y -= 1.0
    if two_xy & inv_xy:
        # A 2-fold and an inversion at one projected site (2/m). The plate
        # draws the lens and the dot together; the legend shows that glyph.
        ax.text(0.2, y, "2/m:", fontsize=8, style="italic")
        y -= 0.95
        _draw_lens(ax, (0.7, y), 0.14, fc="k", ec="k", lw=1.0, zorder=5)
        ax.plot(0.7, y, "o", ms=4, mfc="white", mec="k", mew=0.8, zorder=6)
        ax.text(1.65, y, "2/m", fontsize=8, va="center")
        y -= 1.0
    if four_xy & inv_xy:
        ax.text(0.2, y, "4/m:", fontsize=8, style="italic")
        y -= 0.95
        _draw_combined_axis(ax, (0.7, y), 4, 0, 0, size=0.28, inversion=True)
        ax.text(1.65, y, "4/m", fontsize=8, va="center")
        y -= 1.0
    if six_xy & inv_xy:
        ax.text(0.2, y, "6/m:", fontsize=8, style="italic")
        y -= 0.95
        _draw_combined_axis(ax, (0.7, y), 6, 0, 0, size=0.28, inversion=True)
        ax.text(1.65, y, "6/m", fontsize=8, va="center")
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


def _column_layout(system, m3m, legend):
    """Figure size and column widths. The symmetry-element panel is the widest.

    ITA prints that panel at full page width. The perspective and the general
    positions sit beside it, and the legend is only as wide as its own glyphs
    so it does not leave an empty band.
    """
    if system != "cubic":
        figsize = (10.2, 5.6) if legend else (6.6, 3.8)
        return figsize, None
    # The element panel is the widest. The perspective column is still wide
    # enough that a polyhedron of radius ~0.09 shows its corners.
    ratios = ([1.45] if m3m else []) + [1.35, 2.55]
    if legend:
        # Just wide enough for the legend glyphs. A wider column leaves an
        # empty band, because equal aspect keeps that panel tall and narrow.
        ratios.append(0.58)
    # 2.7 * 4.15 in is the element panel: the largest share of the canvas.
    figsize = (sum(ratios) * 4.15, 9.2)
    return figsize, ratios


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
    projection : {'c', 'a', 'b', 'all'}
        Projection axis. For monoclinic groups the ITA standard plate is the
        unique-axis-b projection (``projection='b'``), where a c-glide plane
        lies parallel to the page and shows its glide-direction arrow.
        ``'all'`` draws the three element projections (or, for a cubic group,
        one element panel) beside the general-position panel.
    """
    import matplotlib.pyplot as plt
    sg = _resolve_sg(sg)
    system = _crystal_system(sg)
    # A rhombohedral-axes setting is drawn in the hexagonal cell. The
    # rhombohedral [111] is the hexagonal c axis, so projection 111 is
    # that same view. The rhombohedral cell is outlined afterwards.
    outline_rhombo = _is_rhombo_axes(sg)
    draw_sg = sg
    if outline_rhombo:
        draw_sg = space_group(_sg_number(sg))
        if projection in ("c", "111"):
            projection = "c"
    elif projection == "111":
        raise ValueError(
            "projection 111 is the body-diagonal view of a rhombohedral-axes setting"
        )
    if projection == "all":
        return _ita_plate_all(sg, figsize=figsize, legend=legend,
                              show_centring=show_centring, system=system)
    num = _sg_number(sg)
    m3m = system == "cubic" and num is not None and 221 <= int(num) <= 230
    if figsize is None:
        figsize, ratios = _column_layout(system, m3m, legend)
    else:
        _ignored, ratios = _column_layout(system, m3m, legend)
    ncol = ((3 if m3m else 2) + (1 if legend else 0)
            if system == "cubic" else (3 if legend else 2))
    fig = plt.figure(figsize=figsize)
    gs_kw = {"wspace": 0.12}
    if ratios:
        gs_kw["width_ratios"] = ratios
    gs = fig.add_gridspec(1, ncol, **gs_kw)
    col = 0
    if m3m:
        axp = fig.add_subplot(gs[col])
        _draw_cubic_perspective(axp, sg)
        axp.set_title("perspective", fontsize=8)
        axp.set_anchor("N")
        col += 1
    axL = fig.add_subplot(gs[col])
    col += 1
    axR = fig.add_subplot(gs[col])
    col += 1
    general_position_diagram(draw_sg, ax=axL, show_title=False,
                             projection=projection)
    symmetry_element_diagram(draw_sg, ax=axR, show_title=False,
                             show_centring=show_centring,
                             projection=projection)
    if outline_rhombo:
        rframe = _Frame(cell_frame(draw_sg, projection))
        _draw_rhombo_outline(axL, rframe)
        _draw_rhombo_outline(axR, rframe)
    axL.set_title("general positions", fontsize=8)
    axR.set_title("symmetry elements", fontsize=8)
    axL.set_anchor("N")
    axR.set_anchor("N")
    if legend:
        lax = fig.add_subplot(gs[col])
        element_legend(sg, ax=lax, projection=projection)
        lax.set_anchor("W")
    _stamp_plate_title(fig, sg, system)
    # tight_layout fights the legend's equal aspect and crops its last row.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04,
                        wspace=0.28)
    return fig


def _stamp_plate_title(fig, sg, system):
    num, name = _sg_label(sg)
    pfx = f"#{num}   " if num is not None else ""
    extra = f"({system}, order {_sg_order(sg)})" if system else f"(order {_sg_order(sg)})"
    fig.suptitle(f"{pfx}{_display_hm(name)}   {extra}", fontsize=9)


def _element_panel_title(sg, projection, system):
    """Title of one symmetry-element panel in the three-projection figure.

    An orthorhombic panel is the ITA symbol of the group after the panel's
    axis permutation, with that change of basis in the subtitle.
    """
    if system == "orthorhombic":
        title, _cob, _ops, _row = orthorhombic_panel(sg, projection)
        return title
    _num, name = _sg_label(sg)
    return _display_hm(name)


def _ita_plate_all(sg, figsize, legend, show_centring, system):
    """Three element projections, or one cubic element panel, plus positions."""
    import matplotlib.pyplot as plt
    default = "b" if system == "monoclinic" else "c"
    num = _sg_number(sg)
    m3m = system == "cubic" and num is not None and 221 <= int(num) <= 230
    if system == "cubic":
        if figsize is None:
            figsize, ratios = _column_layout(system, m3m, legend)
        else:
            _ignored, ratios = _column_layout(system, m3m, legend)
        ncol = (3 if m3m else 2) + (1 if legend else 0)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, ncol, width_ratios=ratios, wspace=0.12)
        col = 0
        if m3m:
            axp = fig.add_subplot(gs[col])
            _draw_cubic_perspective(axp, sg)
            axp.set_title("perspective", fontsize=8)
            axp.set_anchor("N")
            col += 1
        axL = fig.add_subplot(gs[col])
        col += 1
        general_position_diagram(sg, ax=axL, show_title=False, projection="c")
        axL.set_title("general positions", fontsize=8)
        axL.set_anchor("N")
        axR = fig.add_subplot(gs[col])
        col += 1
        symmetry_element_diagram(sg, ax=axR, show_title=False,
                                 show_centring=show_centring, projection="c",
                                 mark_origin=True)
        axR.set_title(_element_panel_title(sg, "c", system), fontsize=8)
        axR.set_anchor("N")
        if legend:
            lax = fig.add_subplot(gs[col])
            element_legend(sg, ax=lax, projection="c")
            lax.set_anchor("W")
    else:
        if figsize is None:
            figsize = (20.0, 5.4) if legend else (16.5, 4.6)
        ncol = 5 if legend else 4
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, ncol, wspace=0.28)
        ax0 = fig.add_subplot(gs[0])
        general_position_diagram(sg, ax=ax0, show_title=False,
                                 projection=default)
        ax0.set_title("general positions", fontsize=8)
        for i, proj in enumerate(("c", "a", "b")):
            ax = fig.add_subplot(gs[i + 1])
            symmetry_element_diagram(sg, ax=ax, show_title=False,
                                     show_centring=show_centring,
                                     projection=proj, mark_origin=True)
            ax.set_title(_element_panel_title(sg, proj, system), fontsize=8)
        if legend:
            element_legend(sg, ax=fig.add_subplot(gs[4]), projection=default)
    _stamp_plate_title(fig, sg, system)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04,
                        wspace=0.30)
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
