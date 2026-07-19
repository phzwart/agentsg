"""G6/S6 embedding + boundary-aware distance tests.

The lattice manifold: cells embed as points in G6/S6, distances are robust to
cell choice and continuous across the Niggli reduction-flip boundary, and
symmetry is a continuous distance-to-subspace rather than a binary test.
"""
import math
import pytest
from agentsg.cell.g6 import (
    g6, s6, g6_distance, s6_distance, distance_to_symmetry, g6_from_metric,
    _transform_metric,
)
from agentsg.cell.metric import UnitCell
from agentsg.space_groups import space_group
from agentsg.group import point_group


def _cell_of(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)), ang(G[0][1] / (a * b)))


def test_g6_identity_zero_distance():
    A = (40, 50, 60, 88, 92, 103)
    assert g6_distance(A, A) < 1e-9


def test_g6_setting_invariance():
    """Same lattice in a sheared (unimodular) basis -> distance ~ 0."""
    A = (40, 50, 60, 88, 92, 103)
    G = UnitCell(*A).metric_tensor()
    A2 = _cell_of(_transform_metric(G, ((1, 1, 0), (0, 1, 0), (0, 0, 1))))
    assert g6_distance(A, A2) < 1e-6
    assert s6_distance(A, A2) < 1e-6


def test_boundary_aware_removes_reduction_flip():
    """Across the a=b boundary the raw distance JUMPS; boundary-aware stays smooth."""
    ref = (40.0, 40.0, 60.0, 90, 91, 90)
    hi = (40.0, 40.001, 60.0, 90, 91, 90)
    lo = (40.0, 39.999, 60.0, 90, 91, 90)
    raw_hi = g6_distance(ref, hi, boundary_aware=False)
    raw_lo = g6_distance(ref, lo, boundary_aware=False)
    ba_hi = g6_distance(ref, hi)
    ba_lo = g6_distance(ref, lo)
    # raw distance is discontinuous across the boundary (differs by > 100)
    assert abs(raw_hi - raw_lo) > 50
    # boundary-aware distance is continuous (nearly equal on both sides)
    assert abs(ba_hi - ba_lo) < 1e-2
    assert ba_hi < 1.0 and ba_lo < 1.0


def test_distance_to_symmetry_zero_when_symmetric():
    cubic = point_group(space_group(225).operations())
    assert distance_to_symmetry((50, 50, 50, 90, 90, 90), cubic) < 1e-9


def test_distance_to_symmetry_smooth_and_monotone():
    """Distance to cubic subspace grows monotonically as one axis is stretched."""
    cubic = point_group(space_group(225).operations())
    ds = [distance_to_symmetry((50, 50, 50 * t, 90, 90, 90), cubic)
          for t in (1.0, 1.01, 1.03, 1.08)]
    assert ds[0] < 1e-9
    assert ds[0] < ds[1] < ds[2] < ds[3]


def test_tetragonal_stays_satisfied_along_its_axis():
    """Stretching c keeps the c-axis tetragonal symmetry exact."""
    tetr = point_group(space_group(123).operations())   # 4-fold ‖ c
    for t in (1.0, 1.05, 1.15):
        assert distance_to_symmetry((50, 50, 50 * t, 90, 90, 90), tetr) < 1e-9


def test_g6_and_s6_vectors_length6():
    assert len(g6((40, 50, 60, 88, 92, 103))) == 6
    assert len(s6((40, 50, 60, 88, 92, 103))) == 6
