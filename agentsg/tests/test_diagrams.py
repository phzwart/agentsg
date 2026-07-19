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
