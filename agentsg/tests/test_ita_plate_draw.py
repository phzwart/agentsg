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
    # Axes through the cell are the line itself; the edge ones stay stubs.
    assert interior
    assert min(interior) > 0.9
    heads = [p.get_xy() for p in ax.patches if isinstance(p, Polygon)]
    xs = [float(xy[:, 0].max()) for xy in heads]
    assert max(xs) > 0.9
    assert min(float(xy[:, 0].min()) for xy in heads) < 0.1
    # Corner heads sit outside the frame, clear of the 2-fold at the corner.
    outside = []
    for xy in heads:
        if xy.shape[0] > 5:
            continue
        if xy[:, 0].min() < 0.0 or xy[:, 1].min() < 0.0:
            outside.append(xy)
    assert outside
    assert min(float(xy[:, 0].min()) for xy in outside) < -0.04
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
        if _dir_class(_perm_vec(el["axis"], perm)) != "ab":
            continue
        sym = el["symbol"]
        intr = el.get("intrinsic_exact")
        if sym == "g" and intr is not None:
            depth = abs(float(intr[perm[2]])) % 1.0
            if 0.05 < depth < 0.95:
                sym = "n"
        if sym in _PLANE_STYLE:
            expected.add(_style_pattern(sym))
    if not expected:
        return
    _fig, ax = _render(num)
    found = _long_patterns(ax)
    plt.close(_fig)
    missing = expected - found
    assert not missing, (num, missing)


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
    for p, lg in zip(_screen(plate, True), _screen(legend, False)):
        assert np.allclose(p, lg, atol=0.05)


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


def test_p4mcc_diagonal_axis_has_arrowheads():
    import matplotlib.pyplot as plt
    fig, ax = _render(124)
    M = np.linalg.inv(np.asarray(cell_frame(124, "c")["matrix"], float))
    diag = None
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) < 2 or ln.get_linewidth() >= 1.5:
            continue
        p0 = M @ np.array([x[0], y[0]], float)
        p1 = M @ np.array([x[-1], y[-1]], float)
        if abs(p0[0] - p0[1]) < 0.05 and abs(p1[0] + p1[1] - 1) < 0.05:
            diag = (p0, p1)
            break
        if abs(p0[0] + p0[1] - 1) < 0.05 and abs(p1[0] - p1[1]) < 0.05:
            diag = (p0, p1)
            break
    assert diag is not None
    heads = []
    for p in ax.patches:
        if not isinstance(p, Polygon) or p.get_xy().shape[0] > 5:
            continue
        heads.append(M @ np.mean(p.get_xy(), axis=0))
    for end in diag:
        assert any(np.linalg.norm(h - end) < 0.2 for h in heads)
    keys = {_seg_key(p0, p1) for p0, p1, _L in _frac_segments(124)}
    for p0, p1, _L in _frac_segments(124):
        a, b = _rot90_segment(p0, p1)
        assert _seg_key(a, b) in keys
    plt.close(fig)


def test_p622_crossing_axes_are_lines():
    segs = _frac_segments(177, min_length=0.4)
    diagonal = [
        s for s in segs
        if abs(s[0][0] - s[1][0]) > 0.2 and abs(s[0][1] - s[1][1]) > 0.2
    ]
    assert len(diagonal) >= 4


def test_p23_threefold_sites_match_the_twofold():
    import matplotlib.pyplot as plt
    fig, ax = _render(195)
    M = np.linalg.inv(np.asarray(cell_frame(195, "c")["matrix"], float))
    centers = []
    for p in ax.patches:
        if isinstance(p, RegularPolygon) and p.numvertices == 3:
            centers.append(M @ np.asarray(p.xy, float))
    snapped = {tuple(np.round(c * 3) / 3 % 1) for c in centers}
    assert (0.0, 0.0) in snapped
    assert (round(2 / 3, 10), round(2 / 3, 10)) in snapped or any(
        abs(a - 2 / 3) < 0.02 and abs(b - 2 / 3) < 0.02 for a, b in snapped)
    for x, y in snapped:
        image = ((1 - x) % 1, (1 - y) % 1)
        image = (round(image[0] * 3) / 3, round(image[1] * 3) / 3)
        assert image in snapped or any(
            abs(image[0] - a) < 0.02 and abs(image[1] - b) < 0.02
            for a, b in snapped)
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    for corner in corners:
        assert any(np.linalg.norm(c - corner) < 0.08 for c in centers)
    assert any("⅓" in t.get_text() for t in ax.texts)
    plt.close(fig)
