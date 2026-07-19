"""Tests for conventional-to-primitive cell reduction (centred lattices).

The primitive-cell transform is what makes the root invariant a *lattice*
invariant for centred Bravais types. spglib is used as a TEST-ONLY oracle
(never a runtime dependency): our primitive cell must reduce to the same Niggli
cell spglib's find_primitive produces.
"""
import math
import pytest

from agentsg.cell.primitive import (
    primitive_cell, primitive_transform, lattice_letter,
    CENTRING_MULTIPLICITY, _validate,
)
from agentsg.cell.metric import UnitCell
from agentsg.cell.reduction import niggli_reduce


def _det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def test_validation_against_centring_table():
    # every P must place the hall.LATTICE_CENTERING vectors on integer nodes
    assert _validate() is True


@pytest.mark.parametrize("letter,mult", [
    ("P", 1), ("A", 2), ("B", 2), ("C", 2), ("I", 2), ("F", 4), ("R", 3),
])
def test_determinant_is_inverse_multiplicity(letter, mult):
    P = primitive_transform(letter)
    assert abs(abs(_det3(P)) - 1.0 / mult) < 1e-12
    assert CENTRING_MULTIPLICITY[letter] == mult


def test_h_normalises_to_r():
    assert lattice_letter("H 3") == "R"
    assert lattice_letter("H -3 2") == "R"
    # H and R give identical primitive cells
    conv = (60.0, 60.0, 90.0, 90, 90, 120)
    assert primitive_cell(conv, "H 3") == pytest.approx(primitive_cell(conv, "R 3"))


def test_primitive_leaves_p_unchanged():
    conv = (50.0, 60.0, 70.0, 90, 90, 90)
    prim = primitive_cell(conv, "P 21 21 21")
    assert prim == pytest.approx(conv, abs=1e-9)


@pytest.mark.parametrize("letter,conv", [
    ("C", (80.0, 90.0, 100.0, 90, 90, 90)),
    ("I", (70.0, 70.0, 120.0, 90, 90, 90)),
    ("F", (100.0, 100.0, 100.0, 90, 90, 90)),
    ("R", (60.0, 60.0, 90.0, 90, 90, 120)),
])
def test_primitive_volume_is_conventional_over_multiplicity(letter, conv):
    prim = primitive_cell(conv, letter + " 2")
    vc = UnitCell(*conv).volume()
    vp = UnitCell(*prim).volume()
    assert vp == pytest.approx(vc / CENTRING_MULTIPLICITY[letter], rel=1e-9)


def test_unknown_symbol_raises():
    with pytest.raises(ValueError):
        primitive_cell((10, 10, 10, 90, 90, 90), "Z 1")
    with pytest.raises(ValueError):
        lattice_letter("")


# ---- spglib oracle: primitive Niggli cell must match find_primitive ----
spglib = pytest.importorskip("spglib")
import numpy as np                                      # noqa: E402


def _lattice_rows(cell):
    a, b, c, al, be, ga = cell
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    v0 = [a, 0.0, 0.0]
    v1 = [b * math.cos(ga), b * math.sin(ga), 0.0]
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    v2 = [cx, cy, cz]
    return np.array([v0, v1, v2])


def _params(L):
    A, B, C = (float(np.linalg.norm(L[i])) for i in range(3))
    al = math.degrees(math.acos(np.dot(L[1], L[2]) / (B * C)))
    be = math.degrees(math.acos(np.dot(L[0], L[2]) / (A * C)))
    ga = math.degrees(math.acos(np.dot(L[0], L[1]) / (A * B)))
    return (A, B, C, al, be, ga)


@pytest.mark.parametrize("conv,centring,sym", [
    ((40.0, 50.0, 60.0, 90, 90, 90), [(0, 0, 0), (0.5, 0.5, 0.0)], "C 2 2 2"),
    ((40.0, 50.0, 60.0, 90, 90, 90), [(0, 0, 0), (0.5, 0.5, 0.5)], "I 2 2 2"),
    ((60.0, 60.0, 60.0, 90, 90, 90),
     [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)], "F 2 2 2"),
])
def test_primitive_niggli_matches_spglib(conv, centring, sym):
    ours = niggli_reduce(*primitive_cell(conv, sym))[0]
    lat = _lattice_rows(conv)
    prim = spglib.find_primitive((lat, centring, [1] * len(centring)),
                                 symprec=1e-5)
    sp = _params(spglib.niggli_reduce(prim[0]))
    assert UnitCell(*ours).volume() == pytest.approx(UnitCell(*sp).volume(),
                                                     rel=1e-6)
    # same Niggli cell parameters (sorted lengths + angles)
    assert sorted(ours[:3]) == pytest.approx(sorted(sp[:3]), rel=1e-4)
