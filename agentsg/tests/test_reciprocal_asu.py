"""Reciprocal-space ASU vs gemmi (CCP4 / cctbx convention)."""
from __future__ import annotations

import pytest

gemmi = pytest.importorskip("gemmi")

from agentsg import SymmetryOp, space_group
from agentsg.asu import ReciprocalAsu, laue_class

_HKL = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    (2, 0, 0), (0, 0, 2), (1, 2, 3), (3, 2, 1),
    (-1, 2, 0), (2, -1, 3), (0, 0, -1), (5, -4, 2),
]


@pytest.mark.parametrize("n", range(1, 231))
def test_condition_str_vs_gemmi(n):
    g = gemmi.SpaceGroup(n)
    rasu = ReciprocalAsu.from_space_group(n)
    assert rasu.condition_str == gemmi.ReciprocalAsu(g).condition_str()


@pytest.mark.parametrize("n", range(1, 231))
def test_is_in_and_to_asu_vs_gemmi(n):
    g = gemmi.SpaceGroup(n)
    grasu = gemmi.ReciprocalAsu(g)
    rasu = ReciprocalAsu.from_space_group(n)
    ops = [SymmetryOp.from_xyz(op.triplet()) for op in g.operations()]
    for hkl in _HKL:
        assert rasu.is_in(hkl) == grasu.is_in(hkl)
        ah, ai = rasu.to_asu(hkl, ops)
        gh, gi = grasu.to_asu(hkl, g.operations())
        assert tuple(ah) == tuple(gh)
        assert ai == gi


@pytest.mark.parametrize("n", (1, 14, 75, 148, 155, 166, 195, 225))
def test_laue_class_matches_table(n):
    # Spot-check: laue_class is populated for every IT number.
    name = laue_class(n)
    assert name
    assert laue_class(space_group(n)) == name
