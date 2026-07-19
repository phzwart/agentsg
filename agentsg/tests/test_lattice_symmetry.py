"""Lattice-symmetry (Le Page / Lebedev) tests.

Reproduces the Lebedev et al. (2006) counts (480 matrices / 81 two-folds) and
the insulin example from Zwart, Grosse-Kunstleve & Adams (2006): the cell
68.4 68.4 68.3 109.5 109.4 109.5 has cubic (order 48) metric symmetry.
"""
import pytest
from agentsg.lattice_symmetry import (
    lattice_symmetry, LEBEDEV_MATRICES, TWO_FOLD_MATRICES, le_page_delta,
    kurlin_distance_to_two_fold, evaluate_two_folds, _two_fold_axis_direct,
)
from agentsg.cell.reduction import niggli_reduce


def test_lebedev_counts():
    assert len(LEBEDEV_MATRICES) == 480
    assert len(TWO_FOLD_MATRICES) == 81


def test_two_folds_square_to_identity():
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for M in TWO_FOLD_MATRICES:
        MM = [[sum(M[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        assert MM == I and M != I


def test_insulin_is_cubic():
    red, _ = niggli_reduce(68.4, 68.4, 68.3, 109.5, 109.4, 109.5)
    ls = lattice_symmetry(red, max_delta=3.0)
    assert ls.order == 48
    assert ls.crystal_system == "cubic"


@pytest.mark.parametrize("cell,order,system", [
    ((5, 5, 5, 90, 90, 90), 48, "cubic"),
    ((5, 5, 8, 90, 90, 90), 16, "tetragonal"),
    ((5, 5, 8, 90, 90, 120), 24, "hexagonal"),
    ((5, 6, 7, 90, 90, 90), 8, "orthorhombic"),
    ((5, 6, 7, 90, 95, 90), 4, "monoclinic"),
    ((5, 6, 7, 80, 85, 95), 2, "triclinic"),
    ((5, 5, 5, 70, 70, 70), 12, "trigonal"),
])
def test_ideal_cells_recover_holohedry(cell, order, system):
    red, _ = niggli_reduce(*cell)
    ls = lattice_symmetry(red, max_delta=1.0)
    assert ls.order == order
    assert ls.crystal_system == system


def test_tolerance_controls_pseudosymmetry():
    # a cubic cell perturbed by 1 degree: triclinic at tight tol, cubic at loose
    cell = (5.0, 5.0, 5.0, 91.0, 90.0, 90.0)
    red, _ = niggli_reduce(*cell)
    assert lattice_symmetry(red, max_delta=0.1).order < 48
    assert lattice_symmetry(red, max_delta=2.0).order == 48


def test_le_page_delta_zero_for_true_symmetry():
    # perfect cubic: the two-fold along [100] has delta 0
    cubic = (5.0, 5.0, 5.0, 90, 90, 90)
    zeros = [d for (M, u, h, d, k) in lattice_symmetry(cubic, max_delta=0.5).deltas]
    assert all(abs(d) < 1e-6 for d in zeros)


def test_kurlin_distance_zero_for_exact_two_fold():
    cubic = (5.0, 5.0, 5.0, 90, 90, 90)
    ls = lattice_symmetry(cubic, max_delta=0.5)
    assert ls.two_fold_scores
    assert all(s.kurlin_distance < 1e-9 for s in ls.two_fold_scores)
    # 5-tuple deltas carry Kurlin as the last field
    assert all(abs(k) < 1e-9 for (*_, k) in ls.deltas)


def test_kurlin_grows_under_metric_distortion():
    perfect = (5.0, 5.0, 5.0, 90, 90, 90)
    distorted = (5.0, 5.0, 5.0, 91.0, 90.0, 90.0)
    # Pick a cubic two-fold that survives a loose Le Page cut on the distorted cell
    scores = evaluate_two_folds(distorted, sort_by="le_page")
    # Best two-folds still near-exact on angles should have small but nonzero Kurlin
    # relative to the perfect cell's zero.
    assert scores[0].kurlin_distance >= 0.0
    assert kurlin_distance_to_two_fold(perfect, scores[0].matrix) < 1e-9
    # Distortion increases Kurlin for at least some accepted cubic two-folds
    ls = lattice_symmetry(distorted, max_delta=2.0)
    assert any(s.kurlin_distance > 1e-6 for s in ls.two_fold_scores)


def test_evaluate_two_folds_lists_all_81():
    scores = evaluate_two_folds((5, 6, 7, 90, 95, 90), sort_by="kurlin")
    assert len(scores) == 81
    assert scores[0].kurlin_distance <= scores[-1].kurlin_distance


# --- oracle: spglib on a single-atom lattice ---
spglib = pytest.importorskip("spglib")


@pytest.mark.parametrize("seed", range(20))
def test_holohedry_order_matches_spglib(seed):
    import random
    import numpy as np
    from agentsg.cell.metric import UnitCell
    random.seed(seed)
    cell = (random.uniform(4, 10), random.uniform(4, 10), random.uniform(4, 10),
            random.uniform(80, 100), random.uniform(80, 100), random.uniform(80, 100))
    red, _ = niggli_reduce(*cell)
    mine = lattice_symmetry(red, max_delta=0.05).order
    lat = np.array(UnitCell(*red).orthogonalization_matrix()).T
    sym = spglib.get_symmetry((lat, [[0.0, 0, 0]], [1]), symprec=1e-3)
    assert mine == len(sym["rotations"])
