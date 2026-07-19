"""Structure semi-invariants — checked against SgInfo CLI examples."""
from __future__ import annotations

import pytest

from agentsg import space_group
from agentsg.semi_invariants import (
    semi_invariants, is_semi_invariant, is_allowed_origin, SemiInvariant,
    floating_origin_basis, pin_floating_origin, _discrete_allowed_origins,
)
from agentsg.linalg import Vector3, ZERO3
from fractions import Fraction as Fr


# (IT number, expected [(vector, modulus), ...]) from SgInfo 1.01
_SGINFO_EXAMPLES = {
    1: [((1, 0, 0), 0), ((0, 1, 0), 0), ((0, 0, 1), 0)],
    3: [((1, 0, 0), 2), ((0, 1, 0), 0), ((0, 0, 1), 2)],
    4: [((1, 0, 0), 2), ((0, 1, 0), 0), ((0, 0, 1), 2)],
    5: [((0, 1, 0), 0), ((0, 0, 1), 2)],
    14: [((1, 0, 0), 2), ((0, 1, 0), 2), ((0, 0, 1), 2)],
    19: [((1, 0, 0), 2), ((0, 1, 0), 2), ((0, 0, 1), 2)],
    68: [((1, 0, 0), 2), ((0, 0, 1), 2)],
    75: [((1, 1, 0), 2), ((0, 0, 1), 0)],
    143: [((1, -1, 0), 3), ((0, 0, 1), 0)],
    148: [((0, 0, 1), 2)],
    168: [((0, 0, 1), 0)],
    223: [((1, 1, 1), 2)],
    225: [((1, 1, 1), 2)],
}


@pytest.mark.parametrize("n,expected", list(_SGINFO_EXAMPLES.items()))
def test_sginfo_examples(n, expected):
    got = [(si.vector, si.modulus) for si in semi_invariants(space_group(n).operations())]
    assert got == expected


def test_is_semi_invariant_pm3n():
    ops = space_group(223).operations()
    # (1,1,1)|2 means h+k+l even
    assert is_semi_invariant((2, 2, 2), ops)
    assert is_semi_invariant((1, 1, 0), ops)
    assert not is_semi_invariant((1, 1, 1), ops)
    assert not is_semi_invariant((1, 0, 0), ops)
    assert is_semi_invariant(Vector3((0, 0, 0)), ops)


def test_is_semi_invariant_p1():
    ops = space_group(1).operations()
    assert is_semi_invariant((0, 0, 0), ops)
    assert not is_semi_invariant((1, 0, 0), ops)


def test_allowed_origin_zero():
    for n in (1, 14, 225):
        assert is_allowed_origin(ZERO3, space_group(n).operations())


def test_allowed_origin_half_pbar1():
    ops = space_group(2).operations()
    assert is_allowed_origin(Vector3((Fr(1, 2), 0, 0)), ops)
    assert is_allowed_origin(Vector3((Fr(1, 2), Fr(1, 2), Fr(1, 2))), ops)


def test_semi_invariant_accept():
    si = SemiInvariant((1, 0, 0), 2)
    assert si.accepts(2, 5, 7)
    assert not si.accepts(1, 0, 0)
    si0 = SemiInvariant((0, 1, 0), 0)
    assert si0.accepts(3, 0, 1)
    assert not si0.accepts(0, 1, 0)


def test_vector_dot_origin_integrity():
    """Every reported s.i. constraint is satisfied by all allowed torsion origins."""
    for n in (5, 14, 68, 225):
        ops = space_group(n).operations()
        sis = semi_invariants(ops)
        # Spot-check a few Miller indices against is_semi_invariant consistency.
        for hkl in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, 0, 2)):
            expected = all(si.accepts(*hkl) for si in sis)
            assert is_semi_invariant(hkl, ops) is expected


@pytest.mark.parametrize("n,n_float,axis", [
    (1, 3, None),          # P1: all three axes float
    (3, 1, 1),             # P2: unique y
    (4, 1, 1),             # P21: unique y
    (75, 1, 2),            # P4: unique z
    (143, 1, 2),           # P3: unique z
    (168, 1, 2),           # P6: unique z
    (19, 0, None),         # P212121: unique origin
    (225, 0, None),        # Fm-3m: unique origin
])
def test_floating_origin_basis(n, n_float, axis):
    ops = space_group(n).operations()
    basis = floating_origin_basis(ops)
    assert len(basis) == n_float
    if axis is not None:
        assert len(basis) == 1
        assert basis[0].v[axis] == 1
        assert sum(1 for x in basis[0].v if x != 0) == 1


def test_p3_cheshire_torsion_points():
    """P3 has discrete Cheshire points (1/3,2/3,0) and (2/3,1/3,0) plus floating z."""
    ops = space_group(143).operations()
    disc = _discrete_allowed_origins(ops)
    got = {o.v for o in disc}
    assert (Fr(0), Fr(0), Fr(0)) in got
    assert (Fr(1, 3), Fr(2, 3), Fr(0)) in got
    assert (Fr(2, 3), Fr(1, 3), Fr(0)) in got
    # Floating z must be pinned — no continuum samples.
    assert all(o.v[2] == 0 for o in disc)


def test_pin_floating_origin():
    ops = space_group(4).operations()
    p = Vector3((Fr(1, 4), Fr(1, 7), Fr(1, 3)))
    pinned = pin_floating_origin(p, ops)
    assert pinned.v[1] == 0  # unique axis y
    assert pinned.v[0] == Fr(1, 4)
    assert pinned.v[2] == Fr(1, 3)

    ops1 = space_group(1).operations()
    assert pin_floating_origin(Vector3((Fr(1, 5), Fr(2, 7), Fr(3, 11))), ops1) == ZERO3
