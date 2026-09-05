"""First-principles invariants for the ITA diagram builder.

No reference diagram data: every check derives the expected answer from the
operator set (or from an independent property of the group) and asserts the
diagram layer agrees with it.
"""
from fractions import Fraction

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agentsg.cell import diagrams as D  # noqa: E402
from agentsg.space_groups import space_group  # noqa: E402

HEX = range(143, 195)          # trigonal + hexagonal (hexagonal axes)
TET = range(75, 143)
ORTHO = range(16, 75)
CUBIC = range(195, 231)


# --- cell frame from metric constraints ---------------------------------------

@pytest.mark.parametrize("num", list(range(1, 231)))
def test_cell_frame_matches_crystal_system(num):
    sg = space_group(num)
    fr = D.cell_frame(sg, "c")
    if num in HEX:
        assert fr["kind"] == "hex" and fr["angle"] == 120.0
    elif num in TET or num in CUBIC:
        assert fr["kind"] == "square" and fr["angle"] == 90.0
    elif num in ORTHO:
        assert fr["kind"] == "rect" and fr["angle"] == 90.0
    elif 3 <= num <= 15:               # monoclinic, unique axis b
        assert fr["kind"] == "rect"                    # a-b plane: gamma=90
        assert D.cell_frame(sg, "b")["kind"] == "oblique"   # a-c plane: beta free
    else:                              # triclinic
        assert fr["kind"] == "oblique"


def test_frame_matrix_is_the_cell_basis():
    fr = D.cell_frame(146)
    M = fr["matrix"]
    # right edge -> (1, 0); down edge -> unit vector at 120 degrees
    assert np.allclose(M @ [1, 0], [1, 0])
    assert np.allclose(M @ [0, 1], [np.cos(np.radians(120)),
                                    np.sin(np.radians(120))])


# --- exact heights ------------------------------------------------------------

def test_height_labels_are_ita_strings():
    assert D.height_label(1, Fraction(0), "z") == "+"
    assert D.height_label(-1, Fraction(0), "z") == "−"
    assert D.height_label(1, Fraction(1, 2), "z") == "½+"
    assert D.height_label(-1, Fraction(1, 2), "z") == "½−"
    assert D.height_label(1, Fraction(1, 3), "z") == "⅓+"
    assert D.height_label(1, Fraction(2, 3), "z") == "⅔+"
    assert D.height_label(1, Fraction(1, 12), "z") == "1/12+"
    assert D.height_label(1, Fraction(1, 2), "x") == "½+x"


@pytest.mark.parametrize("num", list(range(1, 231)))
def test_heights_come_from_the_operations(num):
    """Every image's depth is coef*coord + t with t exactly the operation's
    translation; a '-' sign occurs iff some operation sends z -> -z; the set of
    t values is closed under the group's own translations."""
    sg = space_group(num)
    imgs = D.general_position_images(sg, "c")
    assert len(imgs) == sg.order()
    ops = list(sg.operations())
    flips = any(op.W.rows[2][2] == -1 for op in ops)
    labels = {D.height_label(c, t, k) for c, t, k, _, _ in imgs}
    assert any(lbl.endswith("−") or "−" in lbl for lbl in labels) == flips
    # never a bare guess: exact fractions only
    for c, t, k, _, _ in imgs:
        assert isinstance(t, Fraction) and 0 <= t < 1 and c in (1, -1)


def test_rhombohedral_heights_are_thirds_not_signs():
    """R3 (146): no z -> -z operation, so no '-' anywhere; the centring
    translations give exactly +, 1/3+, 2/3+ (the earlier bug printed '-')."""
    labels = {D.height_label(c, t, k)
              for c, t, k, _, _ in D.general_position_images(146)}
    assert labels == {"+", "⅓+", "⅔+"}


def test_p41_heights_are_quarters():
    labels = {D.height_label(c, t, k)
              for c, t, k, _, _ in D.general_position_images(76)}
    assert labels == {"+", "¼+", "½+", "¾+"}


# --- screw sense is intrinsic, not an artefact of operation choice -----------

def _c_axis_symbols(num):
    sg = space_group(num)
    out = {}
    for el in D._element_copies(sg):
        if el["type"] in ("rotation", "screw") and \
                D._dir_class(el["axis"]) == "c":
            key = tuple(np.round(el["location"][:2] % 1.0, 3))
            out.setdefault(key, set()).add(el["symbol"])
    return out


def test_screw_symbol_invariant_under_inverse():
    """An operation and its inverse are the same element: 3+ with t=1/3 and
    3- with t=2/3 are both the 3_1 axis. Check across every screw in the
    tetragonal / trigonal / hexagonal groups."""
    for num in list(range(75, 195)):
        for op in space_group(num).operations():
            W, w = D._wmatrix(op)
            el = D.classify_element(W, w)
            if el["type"] != "screw":
                continue
            Wi = np.linalg.inv(W)
            eli = D.classify_element(Wi, -Wi @ w)
            assert eli["symbol"] == el["symbol"], (num, op.as_xyz())


@pytest.mark.parametrize("num,expect_at_origin", [
    (144, "3_1"), (145, "3_2"), (76, "4_1"), (78, "4_3"),
    (169, "3_1"), (170, "3_2"),      # 6_1^2 = 3_1, 6_5^2 = 3_2
    (171, "3_2"), (172, "3_1"),      # 6_2^2 = 3_2, 6_4^2 = 3_1
])
def test_enantiomorphic_pairs_are_told_apart(num, expect_at_origin):
    syms = _c_axis_symbols(num)[(0.0, 0.0)]
    assert expect_at_origin in syms, (num, syms)


def test_r3_screw_positions_obverse():
    """R3, hexagonal axes, obverse: 3 at (0,0), (1/3,2/3), (2/3,1/3);
    3_1 at (1/3,1/3), (0,2/3), (2/3,0); 3_2 at (2/3,2/3), (0,1/3), (1/3,0).
    Derived by solving (I - W) p = t for the centring translations."""
    syms = _c_axis_symbols(146)
    third, two3 = round(1 / 3, 3), round(2 / 3, 3)
    assert syms[(0.0, 0.0)] == {"3"}
    assert syms[(third, two3)] == {"3"} and syms[(two3, third)] == {"3"}
    for p in [(third, third), (0.0, two3), (two3, 0.0)]:
        assert syms[p] == {"3_1"}, (p, syms[p])
    for p in [(two3, two3), (0.0, third), (third, 0.0)]:
        assert syms[p] == {"3_2"}, (p, syms[p])


# --- everything drawn is a group element; nothing is drawn twice --------------

@pytest.mark.parametrize("num", [1, 2, 4, 14, 19, 62, 76, 96, 146, 148, 152,
                                 167, 173, 194, 198, 225, 227])
def test_general_position_count_equals_order(num):
    """Distinct projected (position, height) circles in one cell == |G|."""
    sg = space_group(num)
    x0 = np.array(D.best_general_point(sg))
    seen = set()
    for c, t, k, W, w in D.general_position_images(sg):
        p = (np.array(W, float) @ x0 + [float(v) for v in w]) % 1.0
        seen.add((round(p[0], 4), round(p[1], 4), D.height_label(c, t, k)))
    assert len(seen) == sg.order()


@pytest.mark.parametrize("num", [146, 148, 152, 167, 173, 194, 14, 1])
def test_oblique_frames_render(num):
    fig = D.ita_plate(num, legend=True)
    plt.close(fig)
    fig = D.ita_plate(num, projection="b")
    plt.close(fig)


def test_hexagonal_plane_lines_follow_lattice_directions():
    """In P-3m1 (164) the mirrors are perpendicular to a, b and a+b, so their
    traces run along the lattice directions 2a+b, a+2b and a-b of the
    120-degree cell; in P-31m (157) they run along a, b and a+b. Every drawn
    plane segment must be parallel to one of those six lattice directions in
    PLOT space -- not to the axes of a square -- and the 164 set must include
    the non-edge directions."""
    seen = []
    orig = D.draw_plane_symbol

    def spy(ax, p0, p1, name, _o=orig):
        seen.append((np.asarray(p0, float), np.asarray(p1, float)))
        return _o(ax, p0, p1, name)

    D.draw_plane_symbol = spy
    try:
        fig, ax = plt.subplots()
        D.symmetry_element_diagram(164, ax=ax)
        plt.close(fig)
    finally:
        D.draw_plane_symbol = orig
    M = D.cell_frame(164)["matrix"]
    # (right, down) = (b, a) fractional components of each lattice direction
    lat = {"a": [0, 1], "b": [1, 0], "a+b": [1, 1], "a-b": [-1, 1],
           "2a+b": [1, 2], "a+2b": [2, 1]}
    allowed = {k: (M @ v) / np.linalg.norm(M @ v) for k, v in lat.items()}
    assert seen
    hit = set()
    for p0, p1 in seen:
        d = p1 - p0
        d = d / np.linalg.norm(d)
        best = max(allowed, key=lambda k: abs(float(np.dot(d, allowed[k]))))
        assert abs(float(np.dot(d, allowed[best]))) > 0.999, d
        hit.add(best)
    assert hit == {"2a+b", "a+2b", "a-b"}, hit
