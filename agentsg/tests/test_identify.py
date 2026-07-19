"""Space-group identification (ops -> IT / Hall) and origin-shift recovery."""
from __future__ import annotations
from fractions import Fraction as Fr

import pytest

from agentsg import space_group, ChangeOfBasis, SymmetryOp
from agentsg.identify import identify_space_group, hall_from_ops
from agentsg.linalg import Vector3, IDENTITY3, ZERO3


@pytest.mark.parametrize("n", range(1, 231))
def test_identify_all_230(n):
    sg = space_group(n)
    hit = identify_space_group(sg.operations())
    assert hit is not None
    assert hit.number == n
    assert hit.hall == sg.hall
    assert hit.hermann_mauguin == sg.hermann_mauguin
    assert hit.change_of_basis.P == IDENTITY3
    assert hit.change_of_basis.p == ZERO3


@pytest.mark.parametrize("n", range(1, 231))
def test_hall_from_ops_all_230(n):
    assert hall_from_ops(space_group(n).operations()) == space_group(n).hall


def test_origin_shift_p21c():
    ops = space_group(14).operations()
    q = Vector3((0, 0, Fr(1, 4)))
    conj = frozenset(ChangeOfBasis(IDENTITY3, q).apply_to_op(op) for op in ops)
    assert conj != ops
    hit = identify_space_group(conj)
    assert hit is not None
    assert hit.number == 14
    # Returned CoB must map the conjugated ops back to the standard set.
    recovered = frozenset(
        hit.change_of_basis.apply_to_op(op) for op in conj
    )
    assert recovered == ops


def test_origin_shift_various():
    for n, shift in (
        (2, (Fr(1, 2), 0, 0)),
        (19, (Fr(1, 4), Fr(1, 4), Fr(1, 4))),
        (225, (0, 0, Fr(1, 2))),
    ):
        ops = space_group(n).operations()
        conj = frozenset(
            ChangeOfBasis(IDENTITY3, Vector3(shift)).apply_to_op(op) for op in ops
        )
        hit = identify_space_group(conj)
        assert hit is not None
        assert hit.number == n
        recovered = frozenset(hit.change_of_basis.apply_to_op(op) for op in conj)
        assert recovered == ops


def test_p1_from_identity():
    hit = identify_space_group([SymmetryOp.identity()])
    assert hit is not None
    assert hit.number == 1


def test_hall_from_ops_raises_on_empty():
    with pytest.raises(ValueError):
        hall_from_ops([])


@pytest.mark.parametrize("n,float_axis", [
    (1, None),   # P1: all float
    (3, 1),      # P2
    (4, 1),      # P21
    (75, 2),     # P4
    (143, 2),    # P3
])
def test_floating_origin_reported_and_pinned(n, float_axis):
    ops = space_group(n).operations()
    hit = identify_space_group(ops)
    assert hit is not None
    if n == 1:
        assert len(hit.floating_origin) == 3
    else:
        assert len(hit.floating_origin) == 1
        assert hit.floating_origin[0].v[float_axis] == 1

    # Mixed discrete + floating shift: recovered p has floating component pinned to 0.
    if n == 1:
        return
    shift = [Fr(0), Fr(0), Fr(0)]
    # discrete-ish shift orthogonal to floating axis
    for i in range(3):
        if i != float_axis:
            shift[i] = Fr(1, 4)
            break
    shift[float_axis] = Fr(1, 7)  # pure float — must pin away
    conj = frozenset(
        ChangeOfBasis(IDENTITY3, Vector3(shift)).apply_to_op(op) for op in ops
    )
    hit2 = identify_space_group(conj)
    assert hit2 is not None
    assert hit2.number == n
    assert hit2.change_of_basis.p.v[float_axis] == 0
    recovered = frozenset(
        hit2.change_of_basis.apply_to_op(op) for op in conj
    )
    assert recovered == ops


def test_pure_floating_shift_is_identity_cob():
    """A shift purely along the floating axis does not change the operator set."""
    for n, axis in ((4, 1), (75, 2), (143, 2)):
        ops = space_group(n).operations()
        shift = [Fr(0), Fr(0), Fr(0)]
        shift[axis] = Fr(1, 5)
        conj = frozenset(
            ChangeOfBasis(IDENTITY3, Vector3(shift)).apply_to_op(op) for op in ops
        )
        assert conj == ops
        hit = identify_space_group(conj)
        assert hit is not None
        assert hit.change_of_basis.p == ZERO3
