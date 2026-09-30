"""Structural checks for the ITA plate renderer (audit R1–R8, L, J5–J9)."""
from matplotlib.patches import Circle, Polygon, RegularPolygon

import numpy as np
import pytest

from agentsg.cell.diagrams import (
    _PLANE_STYLE,
    _PROJ,
    _dir_class,
    _element_copies,
    _height_caption,
    _perm_vec,
    best_general_point,
    cell_frame,
    draw_axis_symbol,
    element_legend,
    general_position_diagram,
    ita_plate,
    symmetry_element_diagram,
)
from agentsg.serve.handlers import ita_plate_json
from agentsg.space_groups import space_group


def _render(num, projection="c"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    symmetry_element_diagram(num, ax=ax, projection=projection)
    return fig, ax


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _long_patterns(ax):
    found = set()
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) < 2:
            continue
        length = ((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) ** 0.5
        if length < 0.4:
            continue
        pattern = getattr(ln, "_unscaled_dash_pattern", (0, None))
        found.add(None if pattern[1] is None else tuple(pattern[1]))
    return found


def _style_pattern(name):
    ls = _PLANE_STYLE[name]["ls"]
    if ls == "-":
        return None
    return tuple(ls[1])


def test_heights_are_printed_for_the_audit_groups():
    cases = [(14, "b"), (19, "c"), (124, "c"), (142, "c"), (176, "c")]
    for num, proj in cases:
        _fig, ax = _render(num, proj)
        blob = " ".join(_texts(ax))
        assert "¼" in blob, (num, blob)
        import matplotlib.pyplot as plt
        plt.close(_fig)


def test_plane_line_styles_match_the_table():
    _fig, ax = _render(49)
    assert _style_pattern("c") in _long_patterns(ax)
    import matplotlib.pyplot as plt
    plt.close(_fig)
    _fig, ax = _render(63)
    found = _long_patterns(ax)
    assert _style_pattern("n") in found
    assert None in found  # solid mirrors
    plt.close(_fig)


def test_inplane_axes_are_edge_stubs_including_the_far_edge():
    import matplotlib.pyplot as plt
    _fig, ax = _render(16)
    shafts = []
    interior = []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) < 2 or ln.get_linewidth() >= 1.5:
            continue
        length = ((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) ** 0.5
        if length < 0.2:
            shafts.append(length)
        else:
            interior.append(length)
    assert shafts
    assert max(shafts) < 0.08
    # An in-plane axis is a head and a short shaft. A line through the cell
    # would read as a mirror.
    assert not interior
    heads = [p.get_xy() for p in ax.patches if isinstance(p, Polygon)]
    xs = [float(xy[:, 0].max()) for xy in heads]
    assert max(xs) > 0.9
    assert min(float(xy[:, 0].min()) for xy in heads) < 0.1
    # Corner heads sit just outside the frame. The head is the small 2-fold
    # size, so the tip clears the edge by about one head-length.
    outside = []
    for xy in heads:
        if xy.shape[0] > 5:
            continue
        if xy[:, 0].min() < 0.0 or xy[:, 1].min() < 0.0:
            outside.append(xy)
    assert outside
    assert min(float(xy[:, 0].min()) for xy in outside) < -0.03
    plt.close(_fig)


def test_monoclinic_glide_arrow_follows_c():
    import matplotlib.pyplot as plt
    _fig, ax = _render(14, "b")
    deltas = []
    for child in ax.get_children():
        if type(child).__name__ == "Annotation" and child.arrow_patch is not None:
            x1, y1 = child.xy
            x0, y0 = child.xyann
            deltas.append((float(x1 - x0), float(y1 - y0)))
    assert deltas
    dx, dy = deltas[0]
    assert abs(dy) > abs(dx) * 3
    plt.close(_fig)


def test_fdd2_arrow_follows_the_operation_glide():
    """x=1/8 glides along +b and y=1/8 glides along +a.

    Those are the translations in -x+1/4,y+1/4,z+1/4 and
    x+1/4,-y+1/4,z+1/4. The neighbour half a quarter away points the other way.
    """
    import matplotlib.pyplot as plt
    fig, ax = _render(43)
    M = np.linalg.inv(np.asarray(cell_frame(43, "c")["matrix"], float))
    found = {}
    for child in ax.get_children():
        if type(child).__name__ != "Annotation" or child.arrow_patch is None:
            continue
        x1, y1 = map(float, child.xy)
        x0, y0 = map(float, child.xyann)
        right, down = M @ np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        dr, dd = M @ np.array([x1 - x0, y1 - y0])
        found[(round(float(right), 1), round(float(down), 1))] = (
            float(dr), float(dd))
    # Plane x=1/8 is the horizontal trace at down=1/8. Arrow along +b (right).
    dr, dd = found[(0.5, 0.1)]
    assert dr > 0 and abs(dr) > abs(dd)
    # Plane y=1/8 is the vertical trace at right=1/8. Arrow along +a (down).
    dr, dd = found[(0.1, 0.5)]
    assert dd > 0 and abs(dd) > abs(dr)
    plt.close(fig)


def test_d_glide_arrows_alternate():
    import matplotlib.pyplot as plt
    _fig, ax = _render(43)
    vertical = []
    for child in ax.get_children():
        if type(child).__name__ != "Annotation" or child.arrow_patch is None:
            continue
        x1, y1 = map(float, child.xy)
        x0, y0 = map(float, child.xyann)
        if abs(x1 - x0) < 1e-6:
            vertical.append(y1 - y0)
    assert any(v > 0 for v in vertical)
    assert any(v < 0 for v in vertical)
    plt.close(_fig)


def test_screw_glyphs_are_distinct_and_hooked():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 4, screw_k=0, size=0.2)
    plain = len(ax.lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 4, screw_k=2, size=0.2)
    flagged = len(ax.lines)
    plt.close(fig)
    assert plain == 0
    assert flagged > plain
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 2, screw_k=1, size=0.2)
    assert any(len(ln.get_xdata()) >= 3 for ln in ax.lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 6, screw_k=3, size=0.2)
    assert len(ax.lines) >= 12
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 6, rotoinv=True, size=0.2)
    assert not any(ln.get_marker() == "o" for ln in ax.lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_ylim(1, 0)
    draw_axis_symbol(ax, (0, 0), 4, rotoinv=True, size=0.2)
    assert any(ln.get_marker() == "o" for ln in ax.lines)
    plt.close(fig)


def test_legend_uses_unicode_subscripts():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    element_legend(173, ax=ax)
    labels = " ".join(_texts(ax))
    assert "2₁" in labels
    assert "6₃" in labels
    assert "2_1" not in labels
    plt.close(fig)


@pytest.mark.parametrize("num", (195, 207, 221, 227))
def test_cubic_plates_draw_inclined_and_fourfold_glyphs(num):
    import matplotlib.pyplot as plt
    _fig, ax = _render(num)
    tris = [p for p in ax.patches
            if isinstance(p, RegularPolygon) and p.numvertices == 3]
    squares = [p for p in ax.patches
               if isinstance(p, RegularPolygon) and p.numvertices == 4]
    assert tris, num
    if num != 195:
        assert squares, num
    plt.close(_fig)


def test_e_glides_and_contained_subaxes_and_compact_json():
    meta = ita_plate_json({"sg": 39}, png_query="sg=39")
    assert "e" in meta["counts"]
    assert meta["sg_hm_2016"] == "A e m 2"
    assert "Wyckoff letters are not assigned" in meta["note"]
    assert "x=1/4" not in meta["note"]
    origin_axes = [
        el for el in ita_plate_json({"sg": 177}, png_query="sg=177")["elements"]
        if all(abs(c) < 1e-9 for c in el["location"])
        and el["axis"] == [0.0, 0.0, 1.0]
    ]
    assert any(el["symbol"] == "6" and "contained_in" not in el
               for el in origin_axes)
    contained = [el for el in origin_axes if el["symbol"] in ("2", "3")]
    assert contained
    assert all(el["contained_in"] == "6" for el in contained)
    big = ita_plate_json({"sg": 225, "compact": True}, png_query="sg=225")
    assert big["n_total"] > len(big["elements"])
    assert big["n_total"] == sum(big["counts"].values())
    assert len(big["elements"]) == len(big["counts"])


def test_rhombohedral_setting_is_accepted_for_r_groups():
    meta = ita_plate_json({"sg": 146, "setting": "R"}, png_query="sg=146")
    assert meta["sg_number"] == 146
    assert meta["n_total"] >= 1
    with pytest.raises(ValueError):
        ita_plate_json({"sg": 16, "setting": "R"}, png_query="sg=16")


@pytest.mark.parametrize("num", range(1, 231))
def test_every_plane_family_is_drawn_in_its_style(num):
    import matplotlib.pyplot as plt
    sg = space_group(num)
    perm = _PROJ["c"][0]
    expected = set()
    for el in _element_copies(sg):
        if el["type"] not in ("mirror", "glide") or el["axis"] is None:
            continue
        if el.get("contained_in"):
            continue
        if _dir_class(_perm_vec(el["axis"], perm)) != "ab":
            continue
        sym = el["symbol"]
        if sym in _PLANE_STYLE:
            expected.add(_style_pattern(sym))
    if not expected:
        return
    _fig, ax = _render(num)
    found = _long_patterns(ax)
    missing = expected - found
    assert not missing, (num, missing)
    covered = _axis_lines_covering_glides(ax)
    plt.close(_fig)
    assert not covered, (num, covered)


def _axis_lines_covering_glides(ax, tol=0.03):
    """Solid in-plane axis strokes that run along a dashed or dotted glide.

    A glide keeps its dash only when the axis on that trace is an edge stub.
    A solid stroke under the dashes reads as a solid line.
    """
    solids = []
    glides = []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) < 2:
            continue
        p0 = np.array([float(x[0]), float(y[0])])
        p1 = np.array([float(x[-1]), float(y[-1])])
        length = float(np.hypot(*(p1 - p0)))
        if length < 0.35:
            continue
        pattern = getattr(ln, "_unscaled_dash_pattern", (0, None))
        if pattern[1] is not None:
            glides.append((p0, p1))
        elif ln.get_linewidth() < 1.5:
            solids.append((p0, p1))
    hits = []
    for s0, s1 in solids:
        su = s1 - s0
        sn = float(np.linalg.norm(su)) or 1.0
        su = su / sn
        for g0, g1 in glides:
            gu = g1 - g0
            gn = float(np.linalg.norm(gu)) or 1.0
            gu = gu / gn
            if abs(su[0] * gu[1] - su[1] * gu[0]) > 0.08:
                continue
            mid = 0.5 * (g0 + g1)
            delta = mid - s0
            dist = abs(float(delta[0] * su[1] - delta[1] * su[0]))
            along = float(np.dot(delta, su))
            if dist < tol and -0.05 < along < sn + 0.05:
                hits.append((np.round(s0, 2), np.round(s1, 2)))
                break
    return hits


def _ita(num):
    return "b" if space_group(num).crystal_system == "monoclinic" else "c"


def _arrow_deltas(ax):
    deltas = []
    for child in ax.get_children():
        if type(child).__name__ == "Annotation" and child.arrow_patch is not None:
            x1, y1 = map(float, child.xy)
            x0, y0 = map(float, child.xyann)
            deltas.append((x1 - x0, y1 - y0))
    return deltas


def _edge(num, right, down):
    matrix = np.asarray(cell_frame(num, "b")["matrix"], float)
    return matrix @ np.array([right, down], float)


def _aligned(delta, edge):
    d = np.asarray(delta, float)
    e = np.asarray(edge, float)
    dn = np.linalg.norm(d) or 1.0
    en = np.linalg.norm(e) or 1.0
    cross = abs(d[0] * e[1] - d[1] * e[0]) / (dn * en)
    dot = float(np.dot(d, e)) / (dn * en)
    return cross < 0.08 and dot > 0.9


def test_group_14_prints_height_and_c_follows_the_edge():
    import matplotlib.pyplot as plt
    _fig, ax = _render(14, "b")
    assert "¼" in " ".join(_texts(ax))
    c_edge = _edge(14, 0.0, 1.0)
    deltas = _arrow_deltas(ax)
    assert deltas
    assert all(_aligned(d, c_edge) for d in deltas)
    hooks = [ln.get_zorder() for ln in ax.lines if len(ln.get_xdata()) >= 3]
    assert hooks and min(hooks) > 6.5
    plt.close(_fig)


def test_group_12_a_arrow_is_horizontal_along_a():
    import matplotlib.pyplot as plt
    _fig, ax = _render(12, "b")
    deltas = _arrow_deltas(ax)
    assert deltas
    a_edge = _edge(12, 1.0, 0.0)
    assert all(_aligned(d, a_edge) for d in deltas)
    plt.close(_fig)


def test_group_15_n_follows_a_plus_c_on_plate_and_legend():
    import matplotlib.pyplot as plt
    _fig, ax = _render(15, "b")
    plate = _arrow_deltas(ax)
    plt.close(_fig)
    fig, ax = plt.subplots()
    element_legend(15, ax=ax, projection="b")
    legend = _arrow_deltas(ax)
    plt.close(fig)
    plus = _edge(15, 1.0, 1.0)
    minus = _edge(15, 1.0, -1.0)
    assert any(_aligned(d, plus) for d in plate)
    assert not any(_aligned(d, minus) for d in plate)
    # The legend axis runs upward, so its data-y is the opposite of the plate.
    legend_down = [(dx, -dy) for dx, dy in legend]
    assert any(_aligned(d, plus) for d in legend_down)
    assert not any(_aligned(d, minus) for d in legend_down)
    # Same screen direction on the plate and in the legend.
    def _screen(deltas, y_down):
        units = []
        for dx, dy in deltas:
            sy = -dy if y_down else dy
            v = np.array([dx, sy], float)
            units.append(v / (np.linalg.norm(v) or 1.0))
        return sorted(units, key=lambda v: (round(v[0], 3), round(v[1], 3)))
    plate_s = _screen(plate, True)
    legend_s = _screen(legend, False)
    for p in plate_s:
        assert any(np.allclose(p, lg, atol=0.08) for lg in legend_s)
    for lg in legend_s:
        assert any(np.allclose(p, lg, atol=0.08) for p in plate_s)


def test_group_11_uses_one_split_circle_per_pair():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    general_position_diagram(11, ax=ax, projection="b")
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    markers = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert circles
    assert not markers
    plt.close(fig)


def test_groups_16_and_18_keep_distinct_off_quarter_seeds():
    p16 = np.asarray(best_general_point(16), float)
    p18 = np.asarray(best_general_point(18), float)
    assert not np.allclose(p16, p18)
    for coord in p16[:2]:
        nearest = min(abs(float(coord) - q) for q in (0.25, 0.5, 0.75))
        assert nearest >= 0.04
    # x = y draws a square. P222 has no diagonal elements, so an offset
    # like P2_12_12 is the readable pattern.
    assert abs(float(p16[0] - p16[1])) >= 0.06


def test_group_17_arrowhead_reaches_the_far_edge():
    import matplotlib.pyplot as plt
    _fig, ax = _render(17)
    heads = [p.get_xy() for p in ax.patches if isinstance(p, Polygon)]
    assert heads
    assert max(float(xy[:, 1].max()) for xy in heads) > 0.95
    plt.close(_fig)


@pytest.mark.parametrize("num", (4, 14, 17))
def test_legend_screw_label_is_unicode(num):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    element_legend(num, ax=ax, projection=_ita(num))
    labels = " ".join(_texts(ax))
    assert "2₁" in labels
    assert "2_1" not in labels
    plt.close(fig)


def test_legend_circle_and_text_stay_inside_the_axes():
    import matplotlib.pyplot as plt
    for num in (1, 10):
        fig = ita_plate(num, legend=True, projection=_ita(num))
        fig.canvas.draw()
        legend = fig.axes[2]
        box = legend.get_window_extent()
        for text in legend.texts:
            tb = text.get_window_extent()
            assert tb.x0 >= box.x0 - 1, (num, text.get_text())
            assert tb.x1 <= box.x1 + 1, (num, text.get_text())
            assert tb.y0 >= box.y0 - 1, (num, text.get_text())
            assert tb.y1 <= box.y1 + 1, (num, text.get_text())
        circles = [p for p in legend.patches if isinstance(p, Circle)]
        assert circles
        for circ in circles:
            cb = circ.get_window_extent()
            assert cb.x0 >= box.x0 - 1 and cb.x1 <= box.x1 + 1
            assert cb.y0 >= box.y0 - 1 and cb.y1 <= box.y1 + 1
        if num == 10:
            points = [t.get_text() for t in legend.texts if t.get_text().startswith("+ /")]
            assert points and "y" in points[0] and "z" not in points[0]
            assert any(t.get_text() == "2/m" for t in legend.texts)
        plt.close(fig)


def test_quarter_pair_is_implicit_like_zero_and_half():
    assert _height_caption([0, 0.5]) == ""
    assert _height_caption([0.25, 0.75]) == "¼"
    assert _height_caption([0.5]) == "½"
    assert _height_caption([0.75]) == "¾"
    assert "¾" not in _height_caption([0, 0.25, 0.5, 0.75])


def _fold_site(right, down):
    def fold(c):
        c = float(c) % 1.0
        return 0.0 if c > 1.0 - 1e-6 else round(c, 3)
    return (fold(right), fold(down))


def _json_inversion_sites(num):
    proj = _ita(num)
    perm = _PROJ[proj][0]
    meta = ita_plate_json({"sg": num}, png_query=f"sg={num}")
    sites = set()
    for el in meta["elements"]:
        if el.get("type") != "inversion":
            continue
        v = _perm_vec(el["location"], perm)
        sites.add(_fold_site(v[1], v[0]))
    return proj, sites


def _drawn_inversion_sites(num, proj):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    symmetry_element_diagram(num, ax=ax, projection=proj, show_title=False)
    matrix = np.asarray(cell_frame(num, proj)["matrix"], float)
    inv = np.linalg.inv(matrix)
    sites = set()
    for patch in ax.patches:
        if isinstance(patch, Circle) and patch.get_facecolor()[0] > 0.9:
            frac = inv @ np.asarray(patch.center, float)
            sites.add(_fold_site(frac[0], frac[1]))
    for ln in ax.lines:
        face = ln.get_markerfacecolor()
        if isinstance(face, str):
            white = face == "white"
        else:
            white = len(face) >= 3 and min(face[:3]) > 0.9
        if ln.get_marker() == "o" and white:
            xy = np.asarray(ln.get_xydata(), float)
            for pt in xy:
                frac = inv @ pt
                sites.add(_fold_site(frac[0], frac[1]))
    texts = _texts(ax)
    plt.close(fig)
    return sites, texts


@pytest.mark.parametrize("num", (10, 11, 12, 14))
def test_inversion_circles_match_projected_json(num):
    proj, expected = _json_inversion_sites(num)
    drawn, texts = _drawn_inversion_sites(num, proj)
    assert drawn == expected
    assert "¼ ½ ¾" not in texts
    assert "¾" not in " ".join(texts)


def test_split_circle_labels_sit_outside_the_disk():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    general_position_diagram(11, ax=ax, projection="b")
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert circles
    for text in ax.texts:
        label = text.get_text()
        if label in ("a", "c") or not label:
            continue
        x, y = text.get_position()
        for circ in circles:
            cx, cy = circ.center
            assert (x - cx) ** 2 + (y - cy) ** 2 > circ.radius ** 2
    plt.close(fig)


def test_legend_two_over_m_is_drawn_on_the_plate():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    element_legend(12, ax=ax, projection="b")
    labels = " ".join(_texts(ax))
    plt.close(fig)
    assert "2/m" in labels
    assert "−1" in labels
    proj, sites = _json_inversion_sites(12)
    drawn, _texts_out = _drawn_inversion_sites(12, proj)
    assert drawn == sites


def test_plate_json_lists_the_glide_vector():
    meta = ita_plate_json({"sg": 14}, png_query="sg=14")
    glides = [el for el in meta["elements"] if el["type"] == "glide"]
    assert glides
    assert all(len(el["glide"]) == 3 for el in glides)
    assert all("axis" in el for el in glides)


def _frac_segments(num, min_length=0.2):
    """Clipped traces in fractional (right, down), long enough to be a line."""
    import matplotlib.pyplot as plt
    fig, ax = _render(num)
    M = np.linalg.inv(np.asarray(cell_frame(num, "c")["matrix"], float))
    segs = []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) < 2:
            continue
        length = ((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) ** 0.5
        if length < min_length:
            continue
        p0 = M @ np.array([x[0], y[0]], float)
        p1 = M @ np.array([x[-1], y[-1]], float)
        segs.append((np.round(p0, 3), np.round(p1, 3), length))
    plt.close(fig)
    return segs


def _seg_key(p0, p1):
    a = tuple(np.round(np.asarray(p0, float) + 0.0, 3))
    b = tuple(np.round(np.asarray(p1, float) + 0.0, 3))
    return tuple(sorted((a, b)))


def _rot90_segment(p0, p1):
    """Image of a fractional segment under the 4-fold (x, y) -> (-y, x)."""
    def turn(p):
        return np.array([p[1], -p[0]], float)
    a, b = turn(p0), turn(p1)
    mid = 0.5 * (a + b)
    shift = np.floor(mid + 1e-9)
    return a - shift, b - shift


def test_diagonal_copies_restore_the_fourfold():
    """P4mm draws both diagonals and both pairs of glide halves."""
    segs = _frac_segments(99)
    full = [s for s in segs if s[2] > 1.2]
    halves = [s for s in segs if 0.6 < s[2] < 0.85]
    assert len(full) >= 2
    assert len(halves) >= 4
    keys = {_seg_key(p0, p1) for p0, p1, _L in segs}
    for p0, p1, _L in segs:
        a, b = _rot90_segment(p0, p1)
        assert _seg_key(a, b) in keys


def test_p4mcc_glides_keep_their_style_and_axes_stay_stubs():
    """c-glides stay dotted and n-glides stay dash-dot; axes do not cover them."""
    import matplotlib.pyplot as plt
    fig, ax = _render(124)
    found = _long_patterns(ax)
    assert _style_pattern("c") in found
    assert _style_pattern("n") in found
    assert not _axis_lines_covering_glides(ax)
    leg, lax = plt.subplots()
    element_legend(124, ax=lax)
    legend = " ".join(_texts(lax))
    plt.close(leg)
    assert "c" in legend.split()
    assert "n" in legend.split()
    M = np.linalg.inv(np.asarray(cell_frame(124, "c")["matrix"], float))
    heads = []
    for p in ax.patches:
        if not isinstance(p, Polygon) or p.get_xy().shape[0] > 5:
            continue
        heads.append(M @ np.mean(p.get_xy(), axis=0))
    # The [1-10] 2-fold meets the cell at the corners. Its heads sit there.
    for corner in ((0.0, 1.0), (1.0, 0.0)):
        assert any(np.linalg.norm(h - corner) < 0.25 for h in heads), corner
    # The height note sits off the 4/m glyph at the centre.
    for text in ax.texts:
        if text.get_text() != "¼":
            continue
        x, y = text.get_position()
        frac = M @ np.array([x, y], float)
        assert np.linalg.norm(frac - np.array([0.5, 0.5])) > 0.15
    plt.close(fig)


def test_p622_crossing_axes_are_edge_heads():
    segs = _frac_segments(177, min_length=0.4)
    diagonal = [
        s for s in segs
        if abs(s[0][0] - s[1][0]) > 0.2 and abs(s[0][1] - s[1][1]) > 0.2
    ]
    assert diagonal == []
    import matplotlib.pyplot as plt
    _fig, ax = _render(177)
    heads = [
        p for p in ax.patches
        if isinstance(p, Polygon) and 2 < p.get_xy().shape[0] <= 5
    ]
    assert len(heads) >= 4
    plt.close(_fig)


def test_p23_threefold_sites_match_the_twofold():
    import matplotlib.pyplot as plt
    fig, ax = _render(195)
    M = np.linalg.inv(np.asarray(cell_frame(195, "c")["matrix"], float))
    centers = []
    for p in ax.patches:
        if isinstance(p, RegularPolygon) and p.numvertices == 3:
            centers.append(M @ np.asarray(p.xy, float))
    snapped = {tuple(np.round(np.asarray(c, float) * 3) / 3 % 1) for c in centers}
    assert any(abs(a) < 0.02 and abs(b) < 0.02 for a, b in snapped)
    assert any(abs(a - 2 / 3) < 0.05 and abs(b - 2 / 3) < 0.05 for a, b in snapped)
    for x, y in snapped:
        image = ((1 - x) % 1, (1 - y) % 1)
        assert any(
            abs(image[0] - a) < 0.05 and abs(image[1] - b) < 0.05
            for a, b in snapped)
    # One 3-fold at each corner, on the corner, not a row of offset copies.
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    for corner in corners:
        near = [c for c in centers if np.linalg.norm(c - corner) < 0.08]
        assert len(near) == 1, (corner, near)
    # Height 0 is left blank. The screw kind is the glyph, not a ⅓ ⅔ pile.
    assert not any("⅓" in t.get_text() for t in ax.texts)
    plt.close(fig)


def _cell_corner_bounds(num, projection):
    M = np.asarray(cell_frame(num, projection)["matrix"], float)
    corners = [M @ np.array([r, d], float)
               for r, d in ((0, 0), (1, 0), (1, 1), (0, 1))]
    return min(c[0] for c in corners), min(c[1] for c in corners)


def test_parallel_brackets_sit_outside_the_upper_left():
    """Groups 14 and 55 put the page-parallel bracket above and left of the origin."""
    import matplotlib.pyplot as plt
    for num, proj in ((14, "b"), (55, "c")):
        fig, ax = _render(num, proj)
        left, top = _cell_corner_bounds(num, proj)
        pts = []
        for ln in ax.lines:
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if len(x) != 2 or ln.get_linewidth() < 1.5:
                continue
            if float(np.max(x)) < left - 0.02 and float(np.min(y)) < top - 0.01:
                pts.extend([(float(x[0]), float(y[0])), (float(x[1]), float(y[1]))])
        plt.close(fig)
        assert pts, num
        corner = min(pts, key=lambda p: p[0] + p[1])
        assert corner[0] < left - 0.05
        assert corner[1] < top - 0.02


def test_e_glide_lists_two_vectors_and_the_2016_title():
    import matplotlib.pyplot as plt
    meta = ita_plate_json({"sg": 64}, png_query="sg=64")
    es = [el for el in meta["elements"] if el["symbol"] == "e"]
    assert es
    assert all(len(el["glide"]) == 2 and len(el["glide"][0]) == 3 for el in es)
    fig = ita_plate(64)
    title = fig._suptitle.get_text()
    assert "Cmce" in title
    assert "Cmca" not in title
    arrows = [
        c for c in fig.axes[1].get_children()
        if type(c).__name__ == "Annotation" and c.arrow_patch is not None
    ]
    assert len(arrows) >= 2
    plt.close(fig)


def test_d_lines_are_dash_dot_and_e_lines_are_dash_dot_dot():
    assert _PLANE_STYLE["d"]["ls"] == (0, (6, 2, 1, 2))
    assert _PLANE_STYLE["e"]["ls"] == (0, (6, 2, 1, 2, 1, 2))
    import matplotlib.pyplot as plt
    fig, ax = _render(43)
    patterns = set()
    for ln in ax.lines:
        pat = getattr(ln, "_unscaled_dash_pattern", None)
        if pat and pat[1] and ln.get_linewidth() > 1.0:
            patterns.add(tuple(pat[1]))
    plt.close(fig)
    assert (6, 2, 1, 2) in patterns


def test_ia3d_origin_threefold_is_inside_bar3():
    origin = [
        el for el in _element_copies(230)
        if el["symbol"] == "3"
        and el.get("location_exact") is not None
        and all(float(c) == 0.0 for c in el["location_exact"])
    ]
    assert origin
    assert all(el.get("contained_in") == "-3" for el in origin)


def test_ia3d_general_positions_are_polyhedra():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    general_position_diagram(230, ax=ax, show_title=False)
    faces = [p for p in ax.patches if isinstance(p, Polygon)]
    circles = [ln for ln in ax.lines if ln.get_marker() == "o"]
    circles += [p for p in ax.patches if isinstance(p, Circle)]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)
    # One face per projected centre, including the copies on x = 1 and y = 1.
    # Centres that differ only along the projection axis share that face.
    assert len(faces) >= 12
    assert circles
    # The cell, including the x = 1 and y = 1 edges, sits inside the axes.
    assert xlim[0] < 0 < 1 < xlim[1]
    assert ylim[0] > 1 and ylim[1] < 0


def test_f_lattice_polyhedra_cover_the_half_grid():
    """Fm-3m down c: a polyhedron at every point of the closed 1/2-grid."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    general_position_diagram(225, ax=ax, show_title=False)
    M = np.linalg.inv(np.asarray(cell_frame(225, "c")["matrix"], float))
    found = set()
    for p in ax.patches:
        if isinstance(p, Polygon) and len(p.get_xy()) > 3:
            xy = np.asarray(p.get_xy(), float)
            centre = M @ xy[:-1].mean(axis=0)
            found.add((round(centre[0] * 2) / 2, round(centre[1] * 2) / 2))
    plt.close(fig)
    expect = {(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)}
    assert expect <= found


def test_cubic_polyhedra_are_compact_and_labelled_outward():
    """A seed near the highest site keeps neighbouring hulls apart."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    general_position_diagram(225, ax=ax, show_title=False)
    M = np.linalg.inv(np.asarray(cell_frame(225, "c")["matrix"], float))
    radii = []
    for p in ax.patches:
        if isinstance(p, Polygon) and len(p.get_xy()) > 3:
            xy = np.asarray(p.get_xy(), float)[:-1]
            centre = xy.mean(axis=0)
            r = float(np.linalg.norm(xy - centre, axis=1).max())
            # The cell frame is the unit square (radius ~0.7). The polyhedra
            # are the compact hulls inside it.
            if r < 0.4:
                radii.append(r)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    assert len(radii) >= 9
    assert max(radii) < 0.16
    assert any(t for t in texts if t and t != "0")


def test_perspective_marks_polyhedron_corners():
    import matplotlib.pyplot as plt
    fig = ita_plate(225, legend=False)
    persp = fig.axes[0]
    circles = [ln for ln in persp.lines if ln.get_marker() == "o"]
    labels = [t.get_text() for t in persp.texts if t.get_text()]
    plt.close(fig)
    assert circles
    assert labels


def test_inclined_threefolds_use_open_and_filled_triangles():
    """3₁ / 3₂ stay filled; −3 is the open triangle, and both are on the plate."""
    from matplotlib.patches import RegularPolygon
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    symmetry_element_diagram(225, ax=ax, show_title=False)
    tris = [p for p in ax.patches if isinstance(p, RegularPolygon) and p.numvertices == 3]
    plt.close(fig)
    assert any(p.get_fill() for p in tris)
    assert any(not p.get_fill() for p in tris)


def test_cubic_element_panel_is_the_wide_one():
    import matplotlib.pyplot as plt
    fig = ita_plate(225, legend=True)
    widths = [ax.get_position().width for ax in fig.axes]
    plt.close(fig)
    # perspective, general positions, symmetry elements, legend
    assert len(widths) == 4
    assert widths[2] > widths[0]
    assert widths[2] > widths[1]
    assert widths[2] > widths[3]


def test_projection_all_uses_the_three_setting_symbols():
    import matplotlib.pyplot as plt
    fig = ita_plate(26, projection="all")
    titles = [ax.get_title() for ax in fig.axes]
    assert "Pmc2\u2081" in titles
    assert "Pcm2\u2081" in titles
    assert "Pm2\u2081b" in titles
    marked = sum(any(t.get_text() == "0" for t in ax.texts) for ax in fig.axes)
    assert marked >= 3
    plt.close(fig)
    cubic = ita_plate(230, projection="all")
    titles = [ax.get_title() for ax in cubic.axes]
    assert titles[0] == "perspective"
    assert titles[1] == "general positions"
    assert len(cubic.axes) == 3
    plt.close(cubic)


def test_projection_all_json_keeps_the_default_elements():
    from agentsg.serve.handlers import plate_png_args
    meta = ita_plate_json(
        {"sg": 26, "projection": "all"}, png_query="sg=26&projection=all")
    assert meta["projection"] == "c"
    assert meta["elements"]
    assert "projection=all" in meta["png_url"]
    _sg, proj, _legend, _cent = plate_png_args({"sg": 26, "projection": "all"})
    assert proj == "all"


def _poly_span(xy):
    xy = np.asarray(xy, float)
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    span = 0.0
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            span = max(span, float(np.linalg.norm(xy[i] - xy[j])))
    return span


def test_fm3m_border_heads_are_small():
    """In-plane heads match a 2-fold, and the cell edge is not a row of squares."""
    import matplotlib.pyplot as plt
    fig, ax = _render(225)
    spans = []
    edge_small_squares = []
    filled_on_edge = []
    for p in ax.patches:
        if isinstance(p, RegularPolygon) and p.numvertices == 4:
            c = np.asarray(p.xy, float)
            on_edge = min(abs(c[0]), abs(1.0 - c[0]), abs(c[1]), abs(1.0 - c[1])) <= 0.02
            if on_edge and float(p.radius) <= 0.032:
                edge_small_squares.append(p)
            if on_edge and p.get_fill():
                filled_on_edge.append(p)
            continue
        if not isinstance(p, Polygon):
            continue
        xy = np.asarray(p.get_xy(), float)
        if len(xy) < 2 or len(xy) - 1 != 3:
            continue
        body = xy[:-1]
        centre = body.mean(axis=0)
        if min(abs(centre[0]), abs(1.0 - centre[0]),
               abs(centre[1]), abs(1.0 - centre[1])) > 0.08:
            continue
        spans.append(_poly_span(body))
    plt.close(fig)
    assert spans
    assert max(spans) < 0.035
    assert edge_small_squares == []
    # One 4/m square at each corner and edge midpoint. A second square there
    # would be an in-plane 4 drawn with the perpendicular-page glyph.
    assert len(filled_on_edge) == 8


def test_inclined_threefold_pairs_are_separated():
    import matplotlib.pyplot as plt
    fig, ax = _render(225)
    tris = [
        p for p in ax.patches
        if isinstance(p, RegularPolygon) and p.numvertices == 3
    ]
    pts = np.array([np.asarray(p.xy, float) for p in tris])
    nearest = []
    for i in range(len(pts)):
        dist = np.linalg.norm(pts - pts[i], axis=1)
        dist[i] = 99.0
        nearest.append(float(dist.min()))
    open_near = [
        p for p in tris
        if not p.get_fill()
        and abs(float(p.xy[0])) < 0.15
        and abs(float(p.xy[1])) < 0.15
    ]
    plt.close(fig)
    assert nearest
    assert min(nearest) >= 0.05
    assert open_near
    assert min(float(np.hypot(p.xy[0], p.xy[1])) for p in open_near) > 0.02


def test_ia3d_captions_sit_beside_their_circles():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    general_position_diagram(230, ax=ax, show_title=False)
    centres = []
    for p in ax.patches:
        if isinstance(p, Circle):
            centres.append(np.asarray(p.center, float))
    for ln in ax.lines:
        if ln.get_marker() != "o":
            continue
        for x, y in zip(ln.get_xdata(), ln.get_ydata()):
            centres.append(np.array([float(x), float(y)]))
    centres = np.asarray(centres, float)
    distances = []
    for t in ax.texts:
        text = t.get_text()
        if not text or text in ("0", ","):
            continue
        pos = t.get_position()
        distances.append(float(np.linalg.norm(
            centres - np.array([pos[0], pos[1]], float), axis=1).min()))
    plt.close(fig)
    assert distances
    assert max(distances) <= 0.04


def test_p4m_split_circles_carry_the_enantiomorph_comma():
    """The page-parallel mirror is the enantiomorph when no 2-fold lies in the page."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    general_position_diagram(83, ax=ax, projection="c")
    commas = [t.get_text() for t in ax.texts if "," in t.get_text()]
    assert len(commas) >= 4
    plt.close(fig)


def test_mm2_page_mirror_carries_the_comma():
    """Both improper images of Pmm2 are enantiomorphs, including z to -z."""
    import matplotlib.pyplot as plt
    from agentsg.setting import SpaceGroupSetting
    fig, ax = plt.subplots()
    general_position_diagram(SpaceGroupSetting.parse("P m m 2 (c,a,b)"),
                             ax=ax, projection="c")
    labels = [t.get_text() for t in ax.texts if t.get_text() not in ("a", "b", "c")]
    assert ",−" in labels
    assert ",+" in labels
    plt.close(fig)


def test_bar6_sites_are_open_hexagons():
    """The 3 on a −6 is not drawn. The site is the open hexagon."""
    import matplotlib.pyplot as plt
    fig, ax = _render(193)
    open6 = [p for p in ax.patches
             if isinstance(p, RegularPolygon) and p.numvertices == 6
             and not p.get_fill()]
    filled3 = [p for p in ax.patches
               if isinstance(p, RegularPolygon) and p.numvertices == 3
               and p.get_fill()]
    assert len(open6) >= 2
    assert filled3 == []
    plt.close(fig)


def test_p4m_squares_carry_the_inversion_circle():
    import matplotlib.pyplot as plt
    fig, ax = _render(83)
    squares = [p for p in ax.patches
               if isinstance(p, RegularPolygon) and p.numvertices == 4
               and p.get_fill()]
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert squares and circles
    for sq in squares:
        assert any(np.hypot(c.center[0] - sq.xy[0], c.center[1] - sq.xy[1]) < 0.02
                   for c in circles)
    plt.close(fig)


def _bracket_rows(ax, num, projection):
    """y of each page-parallel bracket above the cell."""
    _left, top = _cell_corner_bounds(num, projection)
    ys = []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) != 2 or abs(float(y[0]) - float(y[1])) > 1e-9:
            continue
        span = abs(float(x[1]) - float(x[0]))
        if float(y[0]) < top - 0.02 and 0.12 < span < 0.25:
            ys.append(round(float(y[0]), 2))
    return sorted(set(ys))


def test_cc_brackets_collapse_the_half_cell_partner():
    """c at 0 and n at 1/4. The copies at 1/2 and 3/4 are the same planes."""
    import matplotlib.pyplot as plt
    fig, ax = _render(9, "b")
    assert len(_bracket_rows(ax, 9, "b")) == 2
    assert _texts(ax).count("¼") == 1
    plt.close(fig)


def test_page_parallel_partner_is_one_bracket():
    """A plane and the copy half a cell away share one bracket."""
    import matplotlib.pyplot as plt
    fig, ax = _render(83)
    assert len(_bracket_rows(ax, 83, "c")) == 1
    plt.close(fig)
    fig, ax = _render(193)
    rows = _bracket_rows(ax, 193, "c")
    assert len(rows) == 1
    # The mirror is at 1/4, not at 0.
    assert any(t.get_text() == "¼" and abs(t.get_position()[1] - rows[0]) < 0.2
               for t in ax.texts)
    plt.close(fig)


def test_rhombohedral_g_matches_the_dashed_legend():
    import matplotlib.pyplot as plt
    fig, ax = _render(166)
    found = _long_patterns(ax)
    assert _style_pattern("g") in found
    assert _style_pattern("n") not in found
    plt.close(fig)
