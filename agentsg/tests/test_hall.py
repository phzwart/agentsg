"""Unit tests for the Hall-symbol parser (token-level + a few whole groups)."""
from fractions import Fraction as Fr
import pytest
from agentsg.hall import parse_hall, ops_from_hall
from agentsg.group import close_group, point_group


def _orders(symbol):
    return len(ops_from_hall(symbol))


def test_lattice_symbol_centering():
    gens, cent = parse_hall("F 1")
    assert len(cent) == 4
    gens, cent = parse_hall("R 1")
    assert len(cent) == 3
    gens, cent = parse_hall("P 1")
    assert len(cent) == 1


def test_centrosymmetric_prefix_adds_inversion():
    # -P 1 is P-1 (order 2)
    assert _orders("-P 1") == 2
    ops = ops_from_hall("-P 1")
    assert any(op.as_xyz() == "-x,-y,-z" for op in ops)


def test_simple_rotations():
    assert _orders("P 2") == 2      # C2 about z
    assert _orders("P 3") == 3
    assert _orders("P 4") == 4
    assert _orders("P 6") == 6


def test_screw_translation_letters():
    # P 2c is P2_1 (2-fold screw along z via 'c'): 0,0,z+1/2 present
    ops = ops_from_hall("P 2c")
    tris = {op.as_xyz() for op in ops}
    assert "-x,-y,z+1/2" in tris


def test_screw_digit_threefold():
    # P 31 : 3_1 screw, intrinsic translation 1/3 along z
    ops = ops_from_hall("P 31")
    assert len(ops) == 3
    assert any(op.w.v[2] == Fr(1, 3) for op in ops)


def test_default_axis_second_generator():
    # 'P 4 2' must equal 'P 4 2x' (2nd order-2 defaults to x after a 4)
    assert {o.as_xyz() for o in ops_from_hall("P 4 2")} == {o.as_xyz() for o in ops_from_hall("P 4 2x")}


def test_origin_shift_parenthesis():
    # P 61 2 (0 0 5): origin shift present, group order 12
    assert _orders("P 61 2 (0 0 5)") == 12


def test_cubic_body_diagonal_default():
    # 'P 4 2 3' : third generator order-3 defaults to * (body diagonal)
    ops = ops_from_hall("P 4 2 3")
    assert len(point_group(ops)) == 24   # 432 (O)


@pytest.mark.parametrize("hall,order", [
    ("P 1", 1),
    ("-P 1", 2),
    ("P 2yb", 2),         # P2_1 (b-unique screw)
    ("-P 2ybc", 4),       # P2_1/c
    ("-F 4 2 3", 192),    # Fm-3m
    ("F 4d 2 3 -1d", 192) # Fd-3m
])
def test_known_group_orders(hall, order):
    assert _orders(hall) == order
