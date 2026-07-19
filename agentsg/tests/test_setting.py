"""Extended setting notation: SG + attached change-of-basis.

Validates against the insulin example from Zwart, Grosse-Kunstleve & Adams,
"Exploring Metric Symmetry" (2006): Hall: I 4 2 3 (y+z,x+z,x+y).
"""
from fractions import Fraction as Fr
import pytest
from agentsg.setting import (parse_setting, parse_cob, format_cob, SpaceGroupSetting)
from agentsg.space_groups import space_group, SPACE_GROUPS
from agentsg.linalg import IDENTITY3, Vector3
from agentsg.symmetry_op import SymmetryOp


def test_parse_cob_columns_are_new_basis_vectors():
    cob = parse_cob("(y+z,x+z,x+y)")
    # column j = new basis vector j in old coords; P[i][j]
    assert cob.P.rows == ((0, 1, 1), (1, 0, 1), (1, 1, 0))
    assert cob.P.det() == 2


def test_abc_and_xyz_spellings_agree():
    assert parse_cob("(a+b,b+c,c+a)").P.rows == parse_cob("(x+y,y+z,z+x)").P.rows


def test_parse_field_forms():
    # 2a, 2*x, fractions, signed
    assert parse_cob("(2a,a+b,c-a)").P.rows == ((2, 1, -1), (0, 1, 0), (0, 0, 1))
    assert parse_cob("(2*x,x+y,-x+z)").P.rows == ((2, 1, -1), (0, 1, 0), (0, 0, 1))
    assert parse_cob("(a/2,b,c)").P.rows == ((Fr(1, 2), 0, 0), (0, 1, 0), (0, 0, 1))


def test_origin_shift_captured():
    _, cob = parse_setting("P 1 (x,y,z+1/2)")
    assert cob.p.v == (0, 0, Fr(1, 2))


def test_parse_setting_strips_hall_prefix():
    base, cob = parse_setting("Hall: I 4 2 3 (y+z,x+z,x+y)")
    assert base == "I 4 2 3"
    assert cob.P.det() == 2


def test_insulin_det2_surfaces_centering():
    """The paper's insulin case: base I432 (order 48) under the det-2 CoB
    gains the I-centring translation and closes to order 96."""
    s = SpaceGroupSetting.parse("Hall: I 4 2 3 (y+z,x+z,x+y)")
    ops = s.operations()
    assert len(ops) == 2 * space_group("I 4 2 3").order()
    half = SymmetryOp(IDENTITY3, Vector3((Fr(1, 2), Fr(1, 2), Fr(1, 2))))
    assert half in ops


@pytest.mark.parametrize("row", SPACE_GROUPS, ids=[str(r[0]) for r in SPACE_GROUPS])
def test_identity_cob_reproduces_base(row):
    n = row[0]
    s = SpaceGroupSetting(space_group(n))
    assert s.operations() == space_group(n).operations()


def test_format_round_trip():
    for txt in ["(y+z,x+z,x+y)", "(2*x,x+y,-x+z)", "(x-y,x+y,z)"]:
        cob = parse_cob(txt)
        # re-parse the formatted string; P must be identical
        assert parse_cob(format_cob(cob)).P.rows == cob.P.rows


def test_inverse_cob_round_trips_op():
    _, cob = parse_setting("Hall: I 4 2 3 (y+z,x+z,x+y)")
    op = sorted(space_group(19).operations(), key=lambda o: (o.W.rows, o.w.v))[1]
    assert cob.inverse().apply_to_op(cob.apply_to_op(op)) == op
