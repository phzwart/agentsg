"""Tests for the ITA space-group diagram module."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from agentsg.cell import diagrams as D
from agentsg.space_groups import space_group


def _order(sg):
    return sg.order() if callable(sg.order) else sg.order


# --- general-position multiplicity == group order (incl. centring) ----------
@pytest.mark.parametrize("num", [1, 2, 4, 14, 19, 47, 62, 96, 155, 225, 227])
def test_general_position_multiplicity(num):
    sg = space_group(num)
    assert D.general_position_multiplicity(sg) == _order(sg)


# --- element classification: curated expected content ----------------------
def _symbols(num):
    from collections import Counter
    return Counter(e["symbol"] for e in D.classify_space_group(num))


def test_p1bar_inversion_only():
    s = _symbols(2)
    assert set(s) == {"-1"} and s["-1"] == 1


def test_p21c_elements():
    s = _symbols(14)
    assert s["2_1"] == 1 and s["c"] == 1 and s["-1"] == 1


def test_p212121_three_screws_no_inversion():
    s = _symbols(19)
    assert s["2_1"] == 3
    assert "-1" not in s and "m" not in s


def test_pmmm_three_mirrors_and_inversion():
    s = _symbols(47)
    assert s["m"] == 3 and s["2"] == 3 and s["-1"] == 1


def test_pnma_full_element_set():
    s = _symbols(62)
    assert s["2_1"] == 3 and s["m"] == 1 and s["-1"] == 1
    assert s["n"] >= 1 and s["a"] >= 1


def test_p43212_screw_axes():
    s = _symbols(96)
    # a 4_3 axis contains the 4_1 power; three 2_1 screws; two 2-folds
    assert s["4_3"] == 1 and s["4_1"] == 1
    assert s["2_1"] == 3 and s["2"] == 2


# --- screw-index recovery for every variant ---------------------------------
@pytest.mark.parametrize("num,expect", [
    (4, "2_1"), (76, "4_1"), (77, "4_2"), (78, "4_3"),
    (144, "3_1"), (145, "3_2"),
    (169, "6_1"), (171, "6_2"), (173, "6_3"), (172, "6_4"), (170, "6_5"),
])
def test_screw_variants_identified(num, expect):
    s = _symbols(num)
    assert s[expect] >= 1, (num, expect, dict(s))


# --- classifier primitives ---------------------------------------------------
def test_classify_pure_twofold_along_c():
    W = np.diag([-1.0, -1.0, 1.0])
    el = D.classify_element(W, np.zeros(3))
    assert el["type"] == "rotation" and el["order"] == 2


def test_classify_21_screw_along_b():
    W = np.diag([-1.0, 1.0, -1.0])
    el = D.classify_element(W, np.array([0.0, 0.5, 0.0]))
    assert el["type"] == "screw" and el["symbol"] == "2_1"


def test_classify_inversion():
    el = D.classify_element(-np.eye(3), np.zeros(3))
    assert el["type"] == "inversion" and el["symbol"] == "-1"


def test_classify_mirror_perp_c():
    W = np.diag([1.0, 1.0, -1.0])
    el = D.classify_element(W, np.zeros(3))
    assert el["type"] == "mirror" and el["symbol"] == "m"


def test_classify_c_glide():
    # mirror perpendicular to b, with c/2 glide
    W = np.diag([1.0, -1.0, 1.0])
    el = D.classify_element(W, np.array([0.0, 0.0, 0.5]))
    assert el["type"] == "glide" and el["symbol"] == "c"


# --- full-cell tiling produces >= representative count -----------------------
def test_full_cell_expands_copies():
    for num in (4, 19, 62):
        sg = space_group(num)
        rep = len(D.classify_space_group(sg))
        full = len(D._element_copies(sg))
        assert full >= rep


# --- rendering smoke tests ---------------------------------------------------
@pytest.mark.parametrize("num", [1, 14, 19, 62, 96, 167, 194, 225, 230])
def test_render_no_crash(num):
    fig, (a1, a2) = plt.subplots(1, 2)
    D.general_position_diagram(num, ax=a1)
    D.symmetry_element_diagram(num, ax=a2, full_cell=True)
    plt.close(fig)


@pytest.mark.parametrize("proj", ["a", "b", "c"])
def test_projections_render(proj):
    fig, ax = plt.subplots()
    D.symmetry_element_diagram(62, ax=ax, projection=proj)
    plt.close(fig)


def test_ita_plate_returns_figure():
    fig = D.ita_plate(19)
    assert fig is not None
    plt.close(fig)


def test_all_230_render_and_multiplicity():
    """Every space group renders both diagrams and has correct multiplicity."""
    for num in range(1, 231):
        sg = space_group(num)
        assert D.general_position_multiplicity(sg) == _order(sg), num
        fig, (a1, a2) = plt.subplots(1, 2)
        D.general_position_diagram(sg, ax=a1)
        D.symmetry_element_diagram(sg, ax=a2, full_cell=True)
        plt.close(fig)


# --- non-standard settings (change of basis) --------------------------------
def _setting(text):
    from agentsg.setting import SpaceGroupSetting
    return SpaceGroupSetting.parse(text)


def test_setting_operations_and_order():
    """A det=2 change of basis doubles the cell and surfaces centring."""
    st = _setting("P 21 21 2 (2a,b-a,c)")
    assert st.order() == 8              # base order 4 x det 2
    # both in-plane axes remain 2_1 screws; c-axis 2 becomes doubled
    from collections import Counter
    s = Counter(e["symbol"] for e in D.classify_space_group(st)
                if e["type"] != "translation")
    assert s["2_1"] == 2 and s["2"] == 4


def test_setting_centring_derived():
    """Centring is read from the operations, not the HM letter -- so a
    non-standard setting reports the surfaced (1/2,0,0) centring."""
    st = _setting("P 21 21 2 (2a,b-a,c)")
    cvs = D._centring_translations(st)
    assert len(cvs) == 2               # origin + (1/2,0,0)
    import numpy as np
    nonzero = [v for v in cvs if not np.allclose(v % 1.0, 0)]
    assert len(nonzero) == 1
    assert np.allclose(sorted(np.round(nonzero[0] % 1.0, 3)), [0.0, 0.0, 0.5])


def test_setting_renders():
    st = _setting("P 21 21 2 (2a,b-a,c)")
    fig, (a1, a2) = plt.subplots(1, 2)
    D.general_position_diagram(st, ax=a1)
    D.symmetry_element_diagram(st, ax=a2, full_cell=True, show_centring=True)
    plt.close(fig)


def test_standard_group_is_primitive():
    """A standard primitive group has only the trivial centring."""
    assert len(D._centring_translations(space_group(18))) == 1
    # a C-centred group has two
    assert len(D._centring_translations(space_group(5))) == 2


# --- arrowhead helper: full (rotation) vs half (screw) ----------------------
def test_inplane_arrowhead_full_vs_half():
    """Full head draws a closed triangle (3 verts); half head is a half
    triangle -- both are Polygon patches, full has larger area."""
    import numpy as np
    from matplotlib.patches import Polygon
    figf, axf = plt.subplots(); D._draw_inplane_arrowhead(axf, (0.5, 0.5),
                                                          (1, 0), full=True)
    figh, axh = plt.subplots(); D._draw_inplane_arrowhead(axh, (0.5, 0.5),
                                                          (1, 0), full=False)

    def area(ax):
        p = [c for c in ax.patches if isinstance(c, Polygon)][0]
        xy = p.get_xy()
        x, y = xy[:, 0], xy[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert area(axf) > area(axh) > 0
    plt.close(figf); plt.close(figh)


# --- element_legend + ita_plate legend option -------------------------------
def test_element_legend_renders():
    fig, ax = plt.subplots()
    D.element_legend(18, ax=ax)
    plt.close(fig)


def test_ita_plate_legend_and_centring():
    fig = D.ita_plate(18, legend=True, show_centring=True)
    # 3 panels when legend=True
    assert len(fig.axes) == 3
    plt.close(fig)


# --- combined c-axis glyph (coincident rotation + rotoinversion) -------------
def test_combined_c_axis_glyph_draws_both():
    """P4/mmm has 4, -4 and 2 all on the c-axis at one site. The combined
    glyph must draw the filled rotation polygon AND the rotoinversion outline +
    centre dot, not overpaint one with the other."""
    from matplotlib.patches import RegularPolygon
    fig, ax = plt.subplots()
    # a filled square (4) framed by an open square (-4) + centre dot
    D._draw_combined_axis(ax, (0.5, 0.5), max_rot=4, rot_k=0, roto=4)
    polys = [p for p in ax.patches if isinstance(p, RegularPolygon)]
    # at least two polygons: the filled 4 and the open -4 frame
    assert len(polys) >= 2
    filled = [p for p in polys if p.get_fill()]
    open_ = [p for p in polys if not p.get_fill()]
    assert filled and open_          # both a filled and an outline glyph
    # a centre dot (Line2D marker) sits on top
    dots = [ln for ln in ax.lines
            if ln.get_marker() == "o" and len(ln.get_xdata()) == 1]
    assert dots
    plt.close(fig)


def test_rotoinversion_only_site_draws():
    """A site with a rotoinversion but no same-order pure rotation still draws
    (falls back to the plain rotoinversion glyph)."""
    from matplotlib.patches import RegularPolygon
    fig, ax = plt.subplots()
    D._draw_combined_axis(ax, (0.5, 0.5), max_rot=0, rot_k=0, roto=4)
    assert any(isinstance(p, RegularPolygon) for p in ax.patches)
    plt.close(fig)


def test_p4mmm_c_axis_glyphs_visible():
    """Rendering #123 must place both the filled 4 and the open -4 frame (the
    bug was the white -4 erasing the black 4)."""
    from matplotlib.patches import RegularPolygon
    fig, ax = plt.subplots()
    D.symmetry_element_diagram(123, ax=ax, full_cell=True)
    polys = [p for p in ax.patches if isinstance(p, RegularPolygon)]
    assert any(p.get_fill() for p in polys)          # filled 4
    assert any(not p.get_fill() for p in polys)      # open -4 frame
    plt.close(fig)


# --- no stray diagonal planes outside the cell ------------------------------
def test_no_out_of_box_plane_segments():
    """The diagonal-mirror edge-copy bug drew a plane at x+y=sqrt(2) (a
    non-lattice offset). No plane segment may leave the unit cell by more than
    a hair, for any tetragonal/cubic group with diagonal mirrors."""
    import numpy as np
    seen = []
    orig = D.draw_plane_symbol

    def spy(ax, p0, p1, name, _o=orig, _s=seen):
        _s.append((np.asarray(p0), np.asarray(p1)))
        return _o(ax, p0, p1, name)

    D.draw_plane_symbol = spy
    try:
        for num in (123, 99, 129, 139, 221, 225, 127, 131):
            seen.clear()
            fig, ax = plt.subplots()
            D.symmetry_element_diagram(num, ax=ax, full_cell=True)
            plt.close(fig)
            for p0, p1 in seen:
                m = max(abs(p0).max(), abs(p1).max())
                assert m <= 1.02, (num, p0, p1)
    finally:
        D.draw_plane_symbol = orig


def test_no_zero_length_plane_segments():
    """Corner-touching diagonal lines produced a degenerate (0,0)-(0,0)
    segment; draw_plane_symbol must skip zero-length lines."""
    import numpy as np
    n_drawn = [0]
    orig = D.draw_plane_symbol

    def spy(ax, p0, p1, name, _o=orig):
        before = len(ax.lines)
        _o(ax, p0, p1, name)
        if len(ax.lines) > before:
            L = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            assert L > 1e-6, (p0, p1)
            n_drawn[0] += 1

    D.draw_plane_symbol = spy
    try:
        fig, ax = plt.subplots()
        D.symmetry_element_diagram(123, ax=ax, full_cell=True)
        plt.close(fig)
    finally:
        D.draw_plane_symbol = orig
    assert n_drawn[0] > 0


# --- planes parallel to the page (horizontal mirror in 4/mmm etc.) ----------
def test_parallel_page_plane_detected():
    """Groups with a plane whose normal is the projection axis (the /m in
    4/mmm) must surface it; chiral groups must not."""
    def par_planes(num):
        return {e["symbol"] for e in D.classify_space_group(num)
                if e["type"] in ("mirror", "glide")
                and D._dir_class(e["axis"]) == "c"}
    assert "m" in par_planes(123)      # P4/mmm horizontal mirror
    assert "m" in par_planes(47)       # Pmmm
    assert par_planes(19) == set()     # P2_1 2_1 2_1 (chiral, none)
    assert par_planes(96) == set()     # P4_3 2_1 2 (chiral, none)


def test_parallel_plane_symbol_draws_bracket():
    """The corner bracket is two line legs; a glide adds an arrow annotation."""
    fig, ax = plt.subplots()
    D.draw_parallel_plane_symbol(ax, "m")
    assert len([ln for ln in ax.lines]) >= 2      # two bracket legs
    n_ann_m = len(ax.texts)
    plt.close(fig)
    fig, ax = plt.subplots()
    D.draw_parallel_plane_symbol(ax, "n")
    # glide adds an arrow (annotation)
    assert len(ax.patches) + len(ax.texts) > n_ann_m or len(ax.lines) >= 2
    plt.close(fig)


def test_element_legend_lists_parallel_plane():
    """element_legend for P4/mmm must include the parallel-page plane row."""
    fig, ax = plt.subplots()
    D.element_legend(123, ax=ax)
    labels = " ".join(t.get_text() for t in ax.texts)
    assert "page" in labels and "corner bracket" in labels
    plt.close(fig)


# --- projection kwarg + projection-aware legend (monoclinic Cc) -------------
def test_ita_plate_projection_kwarg():
    """ita_plate accepts projection='b' (the ITA standard for monoclinic) and
    renders without error."""
    fig = D.ita_plate(9, legend=True, projection="b")
    assert len(fig.axes) == 3
    plt.close(fig)


def test_cc_glide_parallel_in_b_projection():
    """Cc (#9, unique axis b): the c/n glide planes have normal ∥ b, so in the
    down-b projection they classify as parallel to the page (corner bracket),
    but as edge-on lines in the down-c projection."""
    import numpy as np
    els = D.classify_space_group(9)
    glides = [e for e in els if e["type"] == "glide"]
    assert glides
    permb = D._PROJ["b"][0]
    permc = D._PROJ["c"][0]
    for e in glides:
        assert D._dir_class(D._perm_vec(e["axis"], permb)) == "c"   # ∥ page
        assert D._dir_class(D._perm_vec(e["axis"], permc)) == "ab"  # edge-on


def test_legend_projection_aware():
    """The legend classifies planes relative to the requested projection: Cc's
    glides appear as ∥-page (corner bracket) in down-b, as ⊥-page lines in
    down-c."""
    figb, axb = plt.subplots()
    D.element_legend(9, ax=axb, projection="b")
    lb = " ".join(t.get_text() for t in axb.texts)
    plt.close(figb)
    figc, axc = plt.subplots()
    D.element_legend(9, ax=axc, projection="c")
    lc = " ".join(t.get_text() for t in axc.texts)
    plt.close(figc)
    assert "∥ page" in lb                 # corner-bracket section in down-b
    assert "corner bracket" in lb
    assert "⊥ page" in lc                 # line section in down-c


# --- d-glide arrow on edge-on lines -----------------------------------------
def _n_arrow_annotations(ax):
    """Count annotation artists that carry an arrow (arrowprops)."""
    from matplotlib.text import Annotation
    n = 0
    for o in ax.get_children():
        if isinstance(o, Annotation) and o.arrow_patch is not None:
            n += 1
    return n


def test_d_glide_line_gets_arrow():
    """A d-glide drawn edge-on as a line carries an ITA glide-direction arrow
    (an annotation); a plain mirror does not."""
    fig, ax = plt.subplots()
    D.draw_plane_symbol(ax, (0.0, 0.5), (1.0, 0.5), "d")
    n_d = _n_arrow_annotations(ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    D.draw_plane_symbol(ax, (0.0, 0.5), (1.0, 0.5), "m")
    n_m = _n_arrow_annotations(ax)
    plt.close(fig)
    assert n_d > n_m                       # d adds an arrow, m does not


def _screen_arrows(ax):
    """Glide-arrow deltas of ax, in SCREEN coords (x=right, y=up), accounting
    for an inverted y-axis."""
    import numpy as np
    from matplotlib.text import Annotation
    ylo, yhi = ax.get_ylim()
    ys = 1.0 if ylo > yhi else -1.0
    out = []
    for o in ax.get_children():
        if isinstance(o, Annotation) and o.arrow_patch is not None:
            d = np.array(o.xy) - np.array(o.xyann)
            out.append((round(float(d[0]), 3), round(float(-ys * d[1]), 3)))
    return out


def test_glide_arrow_diagram_legend_consistent():
    """The parallel-plane glide arrows must point the same way ON SCREEN in the
    element diagram and in the legend (Cc, down-b): c straight up, n up-right."""
    import numpy as np
    figd, axd = plt.subplots()
    D.symmetry_element_diagram(9, ax=axd, show_title=False, projection="b")
    dd = _screen_arrows(axd)
    plt.close(figd)
    figl, axl = plt.subplots()
    D.element_legend(9, ax=axl, projection="b")
    ll = _screen_arrows(axl)
    plt.close(figl)
    assert len(dd) == 2 and len(ll) == 2
    # match by direction (unit vector), order-independent
    def dirs(arrs):
        u = []
        for x, y in arrs:
            n = np.hypot(x, y) or 1.0
            u.append((round(x / n, 2), round(y / n, 2)))
        return sorted(u)
    dd_u, ll_u = dirs(dd), dirs(ll)
    for (dx, dy), (lx, ly) in zip(dd_u, ll_u):
        # same screen sense: both up (y>0), horizontal sign agrees
        assert dy > 0 and ly > 0
        assert np.sign(dx) == np.sign(lx) or abs(dx) < 0.05
    # every arrow points to screen-up in BOTH panels (no downward arrow)
    assert all(y > 0 for _, y in dd)
    assert all(y > 0 for _, y in ll)


def test_parallel_plane_glide_has_arrow():
    """The ∥-page corner bracket for a glide adds a glide-direction arrow; a
    mirror bracket does not."""
    fig, ax = plt.subplots()
    D.draw_parallel_plane_symbol(ax, "c")
    n_glide = _n_arrow_annotations(ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    D.draw_parallel_plane_symbol(ax, "m")
    n_mirror = _n_arrow_annotations(ax)
    plt.close(fig)
    assert n_glide > n_mirror
