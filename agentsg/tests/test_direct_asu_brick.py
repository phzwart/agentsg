"""Conventional real-space ASU bricks vs gemmi ``find_asu_brick``."""
from __future__ import annotations
from fractions import Fraction

import pytest

gemmi = pytest.importorskip("gemmi")

from agentsg.asu import DirectAsuBrick


@pytest.mark.parametrize("n", range(1, 231))
def test_brick_string_vs_gemmi(n):
    g = gemmi.SpaceGroup(n)
    brick = DirectAsuBrick.from_space_group(n)
    assert str(brick) == gemmi.find_asu_brick(g).str()


@pytest.mark.parametrize("n", range(1, 231))
def test_brick_bounds_consistent_with_string(n):
    brick = DirectAsuBrick.from_space_group(n)
    # Interior sample of the brick should be inside; a point just outside
    # the open upper bound (when hi < 1) should be outside after mod1.
    hx, hy, hz = brick.x.hi, brick.y.hi, brick.z.hi
    mid = (
        hx / 2 if hx > 0 else Fraction(0),
        hy / 2 if hy > 0 else Fraction(0),
        hz / 2 if hz > 0 else Fraction(0),
    )
    assert brick.contains(mid)

    # Point with a coordinate past a closed upper bound (when hi < 1).
    if brick.x.closed_hi and hx < 1:
        assert not brick.contains((hx + Fraction(1, 1000), mid[1], mid[2]))
    if not brick.x.closed_hi and hx < 1:
        assert not brick.contains((hx, mid[1], mid[2]))
        assert brick.contains((hx - Fraction(1, 1000), mid[1], mid[2]))
