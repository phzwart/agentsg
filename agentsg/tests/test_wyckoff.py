"""Wyckoff / site-symmetry tests: orbit-stabiliser identity, known ITA
multiplicities, and fixed-locus dimensions."""
from fractions import Fraction as Fr
import pytest
from agentsg.space_groups import space_group
from agentsg.wyckoff import (
    multiplicity, site_symmetry_order, site_symmetry_point_group, orbit,
    general_position_multiplicity, fixed_locus,
)
from agentsg.linalg import Vector3, IDENTITY3, Matrix3


ALL = list(range(1, 231))


@pytest.mark.parametrize("n", ALL)
def test_general_position_multiplicity_equals_order(n):
    ops = list(space_group(n).operations())
    x = Vector3((Fr(11, 97), Fr(13, 101), Fr(17, 103)))  # generic
    assert multiplicity(x, ops) == len(ops) == general_position_multiplicity(ops)


@pytest.mark.parametrize("n", ALL)
def test_orbit_stabiliser_identity_holds(n):
    ops = list(space_group(n).operations())
    G = len(ops)
    for coords in [(0, 0, 0), (Fr(1, 2), 0, 0), (Fr(1, 4), Fr(1, 4), Fr(1, 4)),
                   (Fr(1, 3), Fr(2, 3), Fr(1, 4)), (Fr(1, 7), Fr(2, 7), Fr(3, 7))]:
        x = Vector3(coords)
        assert multiplicity(x, ops) * site_symmetry_order(x, ops) == G


def test_fm3m_known_wyckoff_multiplicities():
    ops = list(space_group(225).operations())
    cases = {
        (0, 0, 0): (4, 48),                                   # 4a, m-3m
        (Fr(1, 2), Fr(1, 2), Fr(1, 2)): (4, 48),              # 4b, m-3m
        (Fr(1, 4), Fr(1, 4), Fr(1, 4)): (8, 24),              # 8c, -43m
        (Fr(1, 7), Fr(2, 7), Fr(3, 7)): (192, 1),             # general
    }
    for coords, (mult, sso) in cases.items():
        x = Vector3(coords)
        assert multiplicity(x, ops) == mult
        assert site_symmetry_order(x, ops) == sso


def test_p_bar_1_has_eight_inversion_centres():
    ops = list(space_group(2).operations())
    inv = SymmetryOp_inversion()
    loci = fixed_locus(inv)
    assert len(loci) == 8
    assert all(len(basis) == 0 for _, basis in loci)   # isolated points


def test_p2_two_fold_fixes_lines():
    ops = list(space_group(3).operations())
    rot = [op for op in ops if op.W != IDENTITY3][0]
    loci = fixed_locus(rot)
    assert all(len(basis) == 1 for _, basis in loci)   # fixed lines


def SymmetryOp_inversion():
    from agentsg.symmetry_op import SymmetryOp
    return SymmetryOp(Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), Vector3((0, 0, 0)))
