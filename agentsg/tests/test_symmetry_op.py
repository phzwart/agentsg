import pytest
from fractions import Fraction as Fr
from agentsg.symmetry_op import SymmetryOp
from agentsg.linalg import Vector3


@pytest.mark.parametrize("triplet", [
    "x,y,z", "-x,-y,-z", "-y,x-y,z", "-x,y+1/2,-z",
    "1/2-y,1/2+x,z", "x+1/2,-y+1/2,-z",
])
def test_parse_produces_valid_op(triplet):
    op = SymmetryOp.from_xyz(triplet)
    assert op.W.det() in (1, -1)


def test_canonical_printing_is_order_independent():
    # "1/2-y" and "-y+1/2" are the same operation and must print identically
    a = SymmetryOp.from_xyz("1/2-y,1/2+x,z")
    b = SymmetryOp.from_xyz("-y+1/2,x+1/2,z")
    assert a == b
    assert a.as_xyz() == b.as_xyz()


def test_identity_is_neutral():
    e = SymmetryOp.identity()
    op = SymmetryOp.from_xyz("-y,x-y,z")
    assert op * e == op
    assert e * op == op


def test_inverse_axioms():
    e = SymmetryOp.identity()
    for triplet in ["-x,-y,-z", "-y,x-y,z", "-x,y+1/2,-z", "x,-y,z+1/2"]:
        op = SymmetryOp.from_xyz(triplet)
        assert op * op.inverse() == e
        assert op.inverse() * op == e


def test_21_screw_squared_is_identity_mod_lattice_translation():
    op = SymmetryOp.from_xyz("-x,y+1/2,-z")
    assert (op * op) == SymmetryOp.identity()
