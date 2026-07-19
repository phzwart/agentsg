"""Tests for the root-distance volume/shape decomposition and similarity key.

The root invariant scales linearly with the length scale factor s=(V'/V)**(1/3),
so a pure isotropic volume change contributes an exactly predictable amount to
the root distance; the volume-normalised invariant RI/V**(1/3) is a similarity
(shape-only) key, matching the manuscript's similarity relation.
"""
import math
import pytest

from agentsg.cell.rootform import (
    root_invariant, root_distance, _cell_volume,
    similarity_invariant, similarity_distance, root_volume_decomposition,
)
from agentsg.cell.metric import UnitCell

CELLS = [
    (50.0, 50.0, 50.0, 90, 90, 90),       # cubic
    (78.0, 78.0, 37.0, 90, 90, 90),       # tetragonal
    (45.0, 55.0, 67.0, 82.0, 96.0, 103.0),  # triclinic
]


@pytest.mark.parametrize("cell", CELLS)
def test_cell_volume_matches_unitcell(cell):
    assert _cell_volume(cell) == pytest.approx(UnitCell(*cell).volume(), rel=1e-9)


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("s", [0.5, 0.9, 1.02, 1.3, 2.0])
def test_exact_scaling_law(cell, s):
    """d(cell, s*cell) == |s-1| * ||RI(cell)||, exactly."""
    scaled = (cell[0] * s, cell[1] * s, cell[2] * s, cell[3], cell[4], cell[5])
    d = root_distance(cell, scaled)
    norm = math.sqrt(sum(x * x for x in root_invariant(cell)))
    assert d == pytest.approx(abs(s - 1.0) * norm, rel=1e-9)


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("s", [0.5, 0.9, 1.3, 2.0])
def test_similarity_distance_is_scale_invariant(cell, s):
    scaled = (cell[0] * s, cell[1] * s, cell[2] * s, cell[3], cell[4], cell[5])
    assert similarity_distance(cell, scaled) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("cell", CELLS)
def test_similarity_invariant_scales_out_volume(cell):
    s = 1.7
    scaled = (cell[0] * s, cell[1] * s, cell[2] * s, cell[3], cell[4], cell[5])
    assert similarity_invariant(cell) == pytest.approx(
        similarity_invariant(scaled), rel=1e-9)


@pytest.mark.parametrize("cell", CELLS)
def test_decomposition_pure_scaling(cell):
    s = 1.2
    scaled = (cell[0] * s, cell[1] * s, cell[2] * s, cell[3], cell[4], cell[5])
    dec = root_volume_decomposition(cell, scaled)
    assert dec["shape_residual"] == pytest.approx(0.0, abs=1e-6)
    assert dec["volume_component"] == pytest.approx(dec["total"], rel=1e-9)
    assert dec["scale_factor"] == pytest.approx(s, rel=1e-9)
    assert dec["volume_ratio"] == pytest.approx(s ** 3, rel=1e-9)


def test_decomposition_total_equals_root_distance():
    a = (45.0, 55.0, 67.0, 82.0, 96.0, 103.0)
    b = (46.0, 56.0, 66.0, 83.0, 95.0, 102.0)
    dec = root_volume_decomposition(a, b)
    assert dec["total"] == pytest.approx(root_distance(a, b), rel=1e-12)


def test_decomposition_triangle_inequality():
    a = (45.0, 55.0, 67.0, 82.0, 96.0, 103.0)
    b = (49.0, 52.0, 70.0, 85.0, 92.0, 100.0)
    dec = root_volume_decomposition(a, b)
    assert dec["total"] <= dec["volume_component"] + dec["shape_residual"] + 1e-9


def test_cutoff_analytic_single_and_all_edges():
    from agentsg.cell.rootform import root_cutoff_for_edge_tolerance as cut
    assert cut(10) == pytest.approx(10.0)
    assert cut(10, n_edges=3) == pytest.approx(math.sqrt(3) * 10.0)
    assert cut(2.5) == pytest.approx(2.5)


def test_cutoff_orthogonal_exact():
    from agentsg.cell.rootform import root_cutoff_for_edge_tolerance as cut
    # orthogonal cell: worst case is all three edges -> sqrt(3)*delta, exactly
    assert cut(10, cell=(60.0, 70.0, 80.0, 90, 90, 90)) == pytest.approx(
        math.sqrt(3) * 10.0, rel=1e-9)


def test_cutoff_is_conservative_bound():
    """Every cell within max_edge_change per edge is within the calibrated cutoff."""
    import random
    tri = (45.0, 55.0, 67.0, 82.0, 96.0, 103.0)
    from agentsg.cell.rootform import root_cutoff_for_edge_tolerance as cut
    c = cut(8.0, cell=tri)
    random.seed(1)
    for _ in range(300):
        p = tuple([tri[i] + random.uniform(-8.0, 8.0) for i in range(3)]
                  + list(tri[3:]))
        assert root_distance(tri, p) <= c + 1e-9


def test_shape_residual_matches_scaled_similarity():
    """shape_residual == root_distance(scaled_A, B) by construction."""
    a = (45.0, 55.0, 67.0, 82.0, 96.0, 103.0)
    b = (49.0, 52.0, 70.0, 85.0, 92.0, 100.0)
    dec = root_volume_decomposition(a, b)
    s = dec["scale_factor"]
    scaled_a = (a[0] * s, a[1] * s, a[2] * s, a[3], a[4], a[5])
    assert dec["shape_residual"] == pytest.approx(root_distance(scaled_a, b),
                                                  rel=1e-9)


def test_volume_ratio_distance_round_trip():
    from agentsg.cell.rootform import (root_distance_to_volume_ratio,
        volume_ratio_to_root_distance)
    cell = (78.0, 78.0, 37.0, 90, 90, 90)
    for vr in [1.05, 1.20, 1.50, 2.0]:
        d = volume_ratio_to_root_distance(vr, cell)
        assert root_distance_to_volume_ratio(d, cell) == pytest.approx(vr, rel=1e-9)


def test_volume_ratio_to_distance_matches_actual_scaling():
    from agentsg.cell.rootform import volume_ratio_to_root_distance
    cell = (78.0, 78.0, 37.0, 90, 90, 90)
    vr = 1.05
    s = vr ** (1.0 / 3.0)
    scaled = (cell[0] * s, cell[1] * s, cell[2] * s, 90, 90, 90)
    assert volume_ratio_to_root_distance(vr, cell) == pytest.approx(
        root_distance(cell, scaled), rel=1e-9)


def test_symmetry_cutoff_volume_mode():
    from agentsg.cell.rootform import symmetry_cutoff, volume_ratio_to_root_distance
    cell = (78.0, 78.0, 37.0, 90, 90, 90)
    assert symmetry_cutoff(cell, volume_tol=0.05) == pytest.approx(
        volume_ratio_to_root_distance(1.05, cell), rel=1e-9)


def test_symmetry_cutoff_noise_mode_scales_with_rho():
    from agentsg.cell.rootform import symmetry_cutoff, root_invariant
    cell = (78.0, 78.0, 37.0, 90, 90, 90)
    nrho = math.sqrt(sum(x * x for x in root_invariant(cell)))
    assert symmetry_cutoff(cell, noise_frac=0.01) == pytest.approx(
        11.0 * 0.01 * nrho, rel=1e-9)


def test_symmetry_cutoff_requires_exactly_one_mode():
    from agentsg.cell.rootform import symmetry_cutoff
    cell = (78.0, 78.0, 37.0, 90, 90, 90)
    with pytest.raises(ValueError):
        symmetry_cutoff(cell)
    with pytest.raises(ValueError):
        symmetry_cutoff(cell, volume_tol=0.05, noise_frac=0.01)
