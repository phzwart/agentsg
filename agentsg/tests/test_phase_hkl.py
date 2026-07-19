"""Phase restrictions and equivalent reflections vs gemmi."""
from __future__ import annotations
from fractions import Fraction
from math import pi

import pytest

gemmi = pytest.importorskip("gemmi")

from agentsg import space_group
from agentsg.linalg import Vector3
from agentsg.group import (
    phase_restriction, is_reflection_centric, is_systematically_absent,
    phase_shift, is_centrosymmetric,
)
from agentsg.reflections import (
    equivalent_reflections, are_equivalent_reflections,
    epsilon_factor, reflection_multiplicity, laue_multiplicity,
)


_GROUPS = (1, 2, 4, 14, 15, 19, 75, 148, 195, 225)
_HKL = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    (2, 0, 0), (0, 0, 2), (1, 2, 3), (3, 2, 1),
]


@pytest.mark.parametrize("n", _GROUPS)
def test_absence_and_centric_vs_gemmi(n):
    ops = list(space_group(n).operations())
    gops = gemmi.SpaceGroup(n).operations()
    for hkl in _HKL:
        v = Vector3(hkl)
        pr = phase_restriction(v, ops)
        assert pr.absent == gops.is_systematically_absent(hkl)
        assert pr.absent == is_systematically_absent(v, ops)
        assert pr.centric == gops.is_reflection_centric(hkl)
        assert pr.centric == is_reflection_centric(v, ops)


@pytest.mark.parametrize("n", _GROUPS)
def test_phase_shift_vs_gemmi(n):
    from agentsg import SymmetryOp
    from agentsg.linalg import frac_mod1

    for gop in gemmi.SpaceGroup(n).operations():
        aop = SymmetryOp.from_xyz(gop.triplet())
        for hkl in ((1, 0, 0), (1, 2, 3), (0, 0, 2)):
            turns = phase_shift(Vector3(hkl), aop)
            g_turns = frac_mod1(
                Fraction(gop.phase_shift(hkl) / (2 * pi)).limit_denominator(48)
            )
            # gemmi's phase_shift uses the opposite sign convention (−h·w).
            assert turns == g_turns or turns == frac_mod1(-g_turns)


def test_phase_restriction_values():
    # P-1: all reflections centric with phase 0
    ops = list(space_group(2).operations())
    pr = phase_restriction(Vector3((1, 2, 3)), ops)
    assert not pr.absent and pr.centric and pr.phase == Fraction(0)

    # P1: unrestricted
    ops1 = list(space_group(1).operations())
    pr1 = phase_restriction(Vector3((1, 2, 3)), ops1)
    assert not pr1.absent and not pr1.centric and pr1.phase is None

    # P21: 010 absent
    ops4 = list(space_group(4).operations())
    pr4 = phase_restriction(Vector3((0, 1, 0)), ops4)
    assert pr4.absent


@pytest.mark.parametrize("n", _GROUPS)
def test_equivalent_orbit_closed(n):
    ops = list(space_group(n).operations())
    for hkl in ((1, 0, 0), (1, 1, 1), (1, 2, 3)):
        if is_systematically_absent(Vector3(hkl), ops):
            continue
        v = Vector3(hkl)
        eq = equivalent_reflections(v, ops)
        assert eq.multiplicity == reflection_multiplicity(v, ops)
        assert eq.epsilon == epsilon_factor(v, ops)
        assert eq.laue_multiplicity == 2 * eq.N
        assert len(ops) == eq.multiplicity * eq.epsilon
        # Every listed mate is equivalent to the seed.
        for mate in eq.hkls:
            assert are_equivalent_reflections(
                Vector3(hkl), Vector3(mate), ops
            )
        # Friedel mate also equivalent.
        assert are_equivalent_reflections(
            Vector3(hkl), Vector3((-hkl[0], -hkl[1], -hkl[2])), ops
        )


@pytest.mark.parametrize("n", _GROUPS)
def test_epsilon_and_multiplicity_vs_gemmi(n):
    ops = list(space_group(n).operations())
    gops = gemmi.SpaceGroup(n).operations()
    for hkl in _HKL:
        v = Vector3(hkl)
        eps = epsilon_factor(v, ops)
        assert eps == gops.epsilon_factor(hkl)
        assert reflection_multiplicity(v, ops) == len(ops) // eps
        assert laue_multiplicity(v, ops) >= reflection_multiplicity(v, ops)


def test_equivalent_fm3m_multiplicity():
    ops = list(space_group(225).operations())
    eq = equivalent_reflections(Vector3((1, 2, 3)), ops)
    # General reflection in m-3m: geometric multiplicity 48; 24 Friedel pairs.
    assert eq.N == 24
    assert eq.multiplicity == 48
    assert eq.laue_multiplicity == 48
    assert eq.epsilon == 4  # 192 / 48


def test_p1_and_p212121_multiplicities():
    ops1 = list(space_group(1).operations())
    assert reflection_multiplicity(Vector3((1, 2, 3)), ops1) == 1
    assert laue_multiplicity(Vector3((1, 2, 3)), ops1) == 2

    ops19 = list(space_group(19).operations())
    # General reflection in 222: multiplicity 4; Laue mmm expands to 8.
    assert reflection_multiplicity(Vector3((1, 2, 3)), ops19) == 4
    assert laue_multiplicity(Vector3((1, 2, 3)), ops19) == 8
    # Axial centric reflection: −h already in the orbit.
    assert reflection_multiplicity(Vector3((1, 0, 0)), ops19) == 2
    assert laue_multiplicity(Vector3((1, 0, 0)), ops19) == 2


def test_centrosymmetric_flag():
    assert not is_centrosymmetric(space_group(1).operations())
    assert is_centrosymmetric(space_group(2).operations())
    assert is_centrosymmetric(space_group(14).operations())
    assert not is_centrosymmetric(space_group(19).operations())
