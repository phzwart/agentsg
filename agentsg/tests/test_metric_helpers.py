"""Shared cell ↔ metric helpers: round-trip and cross-module consistency."""
import math
import pytest
from agentsg.cell.metric import (
    UnitCell, metric_tensor, params_from_metric, cell_from_metric,
)
from agentsg.lattice_symmetry import (
    _metric_tensor as ls_metric_tensor,
    _params_from_metric as ls_params_from_metric,
    _cell_params as ls_cell_params,
)
from agentsg.cell.g6 import _cell_from_metric as g6_cell_from_metric
from agentsg.cell.primitive import (
    _metric as prim_metric,
    _cell_from_metric as prim_cell_from_metric,
)

CELLS = [
    (6.2, 7.8, 9.1, 78.0, 82.5, 66.3),
    (10.0, 12.0, 8.0, 90.0, 105.0, 90.0),
    (7.0, 9.0, 11.0, 90.0, 90.0, 90.0),
    (5.0, 5.0, 13.0, 90.0, 90.0, 90.0),
    (6.0, 6.0, 9.0, 90.0, 90.0, 120.0),
    (7.0, 7.0, 7.0, 55.0, 55.0, 55.0),
    (5.43, 5.43, 5.43, 90.0, 90.0, 90.0),
]


def _close_params(p, q, *, rel=1e-12, abs_=1e-12):
    for x, y in zip(p, q):
        assert math.isclose(x, y, rel_tol=rel, abs_tol=abs_)


def _close_G(A, B, *, rel=1e-12, abs_=1e-12):
    for i in range(3):
        for j in range(3):
            assert math.isclose(A[i][j], B[i][j], rel_tol=rel, abs_tol=abs_)


@pytest.mark.parametrize("cell", CELLS)
def test_round_trip_params_metric(cell):
    G = metric_tensor(cell)
    back = params_from_metric(G)
    _close_params(cell, back)
    assert cell_from_metric is params_from_metric
    _close_params(cell, cell_from_metric(G))


@pytest.mark.parametrize("cell", CELLS)
def test_unitcell_matches_module_helper(cell):
    uc = UnitCell(*cell)
    _close_G(uc.metric_tensor(), metric_tensor(cell))


@pytest.mark.parametrize("cell", CELLS)
def test_call_sites_agree_on_G_and_params(cell):
    G = metric_tensor(cell)
    _close_G(G, ls_metric_tensor(cell))
    _close_G(G, prim_metric(cell))
    _close_G(G, UnitCell(*cell).metric_tensor())

    params = params_from_metric(G)
    _close_params(params, ls_params_from_metric(G))
    _close_params(params, ls_cell_params(G))
    _close_params(params, g6_cell_from_metric(G))
    _close_params(params, prim_cell_from_metric(G))


@pytest.mark.parametrize("cell", CELLS)
def test_compare_reindex_sublattice_use_shared_helper(cell):
    from agentsg.cell.compare import _params_from_G
    from agentsg.cell.sublattice import apply_to_cell
    from agentsg.cell.reindex import _transform_metric

    G = metric_tensor(cell)
    _close_params(params_from_metric(G), _params_from_G(G))
    # apply_to_cell with identity must recover the same params
    _close_params(cell, apply_to_cell(cell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    # reindex path: transform with I then params_from_metric
    _close_params(
        cell,
        params_from_metric(_transform_metric(G, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))),
    )


@pytest.mark.parametrize("cell", CELLS)
def test_reciprocal_uses_shared_clamping(cell):
    uc = UnitCell(*cell)
    rec = uc.reciprocal()
    # Round-trip via G* should match shared helper
    Gs = uc.reciprocal_metric_tensor()
    p = params_from_metric(Gs)
    _close_params((rec.a, rec.b, rec.c, rec.alpha, rec.beta, rec.gamma), p)


def test_non_positive_edge_raises():
    with pytest.raises(ValueError, match="non-positive"):
        params_from_metric([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_cosine_clamp_near_orthogonal_noise():
    # Off-diagonal slightly past |ab| so raw cos would be > 1 without clamp.
    a = b = c = 10.0
    G = [
        [a * a, a * b * 1.0000000001, 0.0],
        [a * b * 1.0000000001, b * b, 0.0],
        [0.0, 0.0, c * c],
    ]
    p = params_from_metric(G)
    assert math.isclose(p[5], 0.0, abs_tol=1e-6)
