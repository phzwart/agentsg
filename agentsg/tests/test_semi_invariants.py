"""Structure semi-invariants — checked against SgInfo CLI examples."""
from __future__ import annotations

import pytest

from agentsg import space_group
from agentsg.semi_invariants import (
    semi_invariants, is_semi_invariant, is_allowed_origin, SemiInvariant,
    floating_origin_basis, pin_floating_origin, _discrete_allowed_origins,
    origin_lattice, n_alternative_origins, smith_normal_form,
)
from agentsg.group import is_systematically_absent
from itertools import product
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


def _accepted(sis, ops, maxh=4):
    """Non-absent reflections in a box satisfying every congruence (test-only grid)."""
    out = set()
    for h in product(range(-maxh, maxh + 1), repeat=3):
        if is_systematically_absent(Vector3(h), ops):
            continue
        if all(SemiInvariant(v, m).accepts(*h) for v, m in sis):
            out.add(h)
    return out


@pytest.mark.parametrize("n,expected", list(_SGINFO_EXAMPLES.items()))
def test_sginfo_examples(n, expected):
    """Same semi-invariant reflections as SgInfo (representatives may differ)."""
    ops = space_group(n).operations()
    got = [(si.vector, si.modulus) for si in semi_invariants(ops)]
    assert len(got) == len(expected)
    assert sorted(m for _, m in got) == sorted(m for _, m in expected)
    assert _accepted(got, ops) == _accepted(expected, ops)


@pytest.mark.parametrize("n,expected", list(_SGINFO_EXAMPLES.items()))
def test_is_semi_invariant_matches_congruences(n, expected):
    ops = space_group(n).operations()
    acc = _accepted(expected, ops)
    for h in product(range(-3, 4), repeat=3):
        if is_systematically_absent(Vector3(h), ops):
            assert not is_semi_invariant(h, ops)
        else:
            assert is_semi_invariant(h, ops) is (h in acc)


# Number of distinct alternative origins (ITA Table 15.2.1), centred groups included.
_N_ORIGINS = {
    2: 8, 5: 2, 12: 4, 19: 8, 22: 4, 23: 4, 68: 4, 70: 2, 75: 2, 81: 4, 82: 4,
    87: 2, 92: 4, 97: 2, 143: 3, 146: 1, 147: 2, 148: 2, 149: 6, 150: 2, 155: 2,
    160: 1, 166: 2, 168: 1, 174: 6, 195: 2, 196: 4, 197: 1, 213: 2, 216: 4,
    217: 1, 225: 2, 227: 2, 229: 1, 230: 1,
}


@pytest.mark.parametrize("n,count", list(_N_ORIGINS.items()))
def test_alternative_origin_count(n, count):
    ops = space_group(n).operations()
    assert n_alternative_origins(ops) == count
    assert len(_discrete_allowed_origins(ops)) == count


def test_invariant_factors():
    assert origin_lattice(space_group(2).operations()).invariant_factors == (2, 2, 2)
    assert origin_lattice(space_group(216).operations()).invariant_factors == (4,)
    assert origin_lattice(space_group(81).operations()).invariant_factors == (2, 2)
    assert origin_lattice(space_group(229).operations()).invariant_factors == ()


@pytest.mark.parametrize("n", [2, 5, 12, 68, 143, 146, 148, 216, 225, 229])
def test_discrete_origins_are_allowed_and_closed(n):
    ops = space_group(n).operations()
    lat = origin_lattice(ops)
    disc = _discrete_allowed_origins(ops)
    keys = {o.v for o in disc}
    for o in disc:
        assert is_allowed_origin(o, ops)
        assert lat.reduce(o) == o
    for a in disc:
        for b in disc:
            assert lat.reduce(a + b).v in keys
    # centring translations are never reported as alternative origins
    for c in lat.centering:
        assert lat.reduce(c) == ZERO3


def test_smith_normal_form_transforms():
    A = [[-1, -1, 0], [1, -1, 0], [0, 0, 0], [-2, 0, 0], [0, -2, 0], [0, 0, 0]]
    U, s, V = smith_normal_form(A)
    assert s == [1, 2, 0]
    n, m = len(A), len(A[0])
    UA = [[sum(U[i][k] * A[k][j] for k in range(n)) for j in range(m)] for i in range(n)]
    UAV = [[sum(UA[i][k] * V[k][j] for k in range(m)) for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            assert UAV[i][j] == (s[i] if i == j and i < len(s) else 0)


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
    """Congruence list and is_semi_invariant agree on non-absent reflections."""
    for n in (5, 14, 68, 225):
        ops = space_group(n).operations()
        sis = semi_invariants(ops)
        for hkl in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, 0, 2)):
            if is_systematically_absent(Vector3(hkl), ops):
                continue
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
