"""Unit-cell metric tests: analytic identities + validation against gemmi."""
import math
import pytest
from agentsg.cell.metric import UnitCell

CELLS = {
    "triclinic":    (6.2, 7.8, 9.1, 78.0, 82.5, 66.3),
    "monoclinic":   (10.0, 12.0, 8.0, 90, 105.0, 90),
    "orthorhombic": (7.0, 9.0, 11.0, 90, 90, 90),
    "tetragonal":   (5.0, 5.0, 13.0, 90, 90, 90),
    "hexagonal":    (6.0, 6.0, 9.0, 90, 90, 120),
    "rhombohedral": (7.0, 7.0, 7.0, 55.0, 55.0, 55.0),
    "cubic":        (5.43, 5.43, 5.43, 90, 90, 90),
}


def test_cubic_volume_and_metric():
    uc = UnitCell(5.0, 5.0, 5.0, 90, 90, 90)
    assert math.isclose(uc.volume(), 125.0, rel_tol=1e-12)
    G = uc.metric_tensor()
    assert math.isclose(G[0][0], 25.0)
    assert math.isclose(G[0][1], 0.0, abs_tol=1e-12)


def test_reciprocal_of_reciprocal_is_original():
    for p in CELLS.values():
        uc = UnitCell(*p)
        rr = uc.reciprocal().reciprocal()
        for x, y in zip((uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma),
                        (rr.a, rr.b, rr.c, rr.alpha, rr.beta, rr.gamma)):
            assert math.isclose(x, y, rel_tol=1e-10)


def test_orthogonalization_preserves_lengths():
    # squared length via metric tensor == squared Cartesian length
    for p in CELLS.values():
        uc = UnitCell(*p)
        frac = [0.1, 0.2, 0.3]
        G = uc.metric_tensor()
        L2_metric = sum(frac[i] * G[i][j] * frac[j] for i in range(3) for j in range(3))
        cart = uc.orthogonalize(frac)
        L2_cart = sum(c * c for c in cart)
        assert math.isclose(L2_metric, L2_cart, rel_tol=1e-10)


def test_fractionalize_inverts_orthogonalize():
    uc = UnitCell(*CELLS["triclinic"])
    frac = [0.37, 0.12, 0.88]
    back = uc.fractionalize(uc.orthogonalize(frac))
    for a, b in zip(frac, back):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def test_d_spacing_cubic_100():
    uc = UnitCell(4.0, 4.0, 4.0, 90, 90, 90)
    assert math.isclose(uc.d_spacing((1, 0, 0)), 4.0, rel_tol=1e-12)
    assert math.isclose(uc.d_spacing((2, 0, 0)), 2.0, rel_tol=1e-12)


def test_two_theta_bragg():
    uc = UnitCell(4.0, 4.0, 4.0, 90, 90, 90)
    # d(100)=4, lambda=1.54 -> 2theta = 2 asin(1.54/8)
    tt = uc.two_theta((1, 0, 0), 1.54)
    assert math.isclose(tt, math.degrees(2 * math.asin(1.54 / 8.0)), rel_tol=1e-10)


# --- oracle validation ---
gemmi = pytest.importorskip("gemmi")


@pytest.mark.parametrize("name,p", list(CELLS.items()))
def test_volume_matches_gemmi(name, p):
    assert math.isclose(UnitCell(*p).volume(), gemmi.UnitCell(*p).volume, rel_tol=1e-12)


@pytest.mark.parametrize("name,p", list(CELLS.items()))
def test_reciprocal_cell_matches_gemmi(name, p):
    r = UnitCell(*p).reciprocal()
    gr = gemmi.UnitCell(*p).reciprocal()
    for x, y in zip((r.a, r.b, r.c, r.alpha, r.beta, r.gamma),
                    (gr.a, gr.b, gr.c, gr.alpha, gr.beta, gr.gamma)):
        assert math.isclose(x, y, rel_tol=1e-10)


@pytest.mark.parametrize("name,p", list(CELLS.items()))
def test_d_spacings_match_gemmi(name, p):
    uc = UnitCell(*p); gc = gemmi.UnitCell(*p)
    for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 3), (0, 0, 2), (3, 2, 1)]:
        assert math.isclose(uc.d_spacing(hkl), gc.calculate_d(hkl), rel_tol=1e-10)
