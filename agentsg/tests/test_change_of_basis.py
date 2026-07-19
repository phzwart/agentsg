from fractions import Fraction as Fr
from agentsg.linalg import Matrix3, Vector3
from agentsg.change_of_basis import ChangeOfBasis
from agentsg.generators import GENERATOR_TABLE
from agentsg.group import close_group

# ITA Table 5.1.3.1, hexagonal -> rhombohedral (obverse setting):
#   a_R =  2/3 a_H + 1/3 b_H + 1/3 c_H
#   b_R = -1/3 a_H + 1/3 b_H + 1/3 c_H
#   c_R = -1/3 a_H - 2/3 b_H + 1/3 c_H
# Columns of P are the new basis vectors expressed in the old ones.
HEX_TO_RHOMB_OBVERSE = ChangeOfBasis(
    Matrix3([
        [Fr(2, 3), Fr(-1, 3), Fr(-1, 3)],
        [Fr(1, 3), Fr(1, 3), Fr(-2, 3)],
        [Fr(1, 3), Fr(1, 3), Fr(1, 3)],
    ]),
    Vector3((0, 0, 0)),
)


def test_determinant_is_one_third():
    # rhombohedral primitive cell is 1/3 the volume of the hexagonal triple cell
    assert HEX_TO_RHOMB_OBVERSE.P.det() == Fr(1, 3)


def test_centering_vectors_collapse_to_lattice_points():
    Pinv = HEX_TO_RHOMB_OBVERSE.P.inverse()
    for v in GENERATOR_TABLE["R3_hex"]["centering"]:
        assert (Pinv @ v).mod1() == Vector3((0, 0, 0))


def test_reindexed_group_loses_centering_and_shrinks_to_point_group():
    spec = GENERATOR_TABLE["R3_hex"]
    hex_ops = close_group(spec["generators"], spec["centering"])
    assert len(hex_ops) == 9  # 3 point ops x 3 centering vectors, hexagonal setting

    rhomb_ops = frozenset(HEX_TO_RHOMB_OBVERSE.apply_to_op(op) for op in hex_ops)
    assert len(rhomb_ops) == 3  # centering has been absorbed into the primitive cell

    # the 3-fold about [111] should come out as the clean cyclic permutation
    expected = {"x,y,z", "y,z,x", "z,x,y"}
    assert {op.as_xyz() for op in rhomb_ops} == expected


def test_change_of_basis_inverse_round_trips_up_to_centering_coset():
    # NOTE: this transform has det(P) = 1/3 (index 3, a coarser->finer cell
    # change). Round-tripping through it is only guaranteed up to a coset of
    # the ORIGINAL (hexagonal) centering, not bit-for-bit identical: the
    # forward transform reduces translations mod 1 in the smaller
    # rhombohedral cell, which erases exactly the distinction between hex
    # operators that differ by a centering vector -- that information is
    # genuinely not recoverable, not a bug. Assert the correct (weaker,
    # group-theoretically honest) invariant instead: same rotation part, and
    # the translation differs by a hex centering vector.
    cob = HEX_TO_RHOMB_OBVERSE
    back = cob.inverse()
    spec = GENERATOR_TABLE["R3_hex"]
    hex_ops = close_group(spec["generators"], spec["centering"])
    centering_set = set(spec["centering"])
    for op in hex_ops:
        rt = back.apply_to_op(cob.apply_to_op(op))
        assert rt.W == op.W
        assert (rt.w - op.w).mod1() in centering_set
