"""Tests for the exact<->numeric bridge: W^T G W = G, symmetrisation, and
derived crystal-system free parameters."""
import pytest
from agentsg.space_groups import space_group, SPACE_GROUPS
from agentsg.group import point_group
from agentsg.cell.constraints import (
    metric_is_invariant, symmetrize_metric, free_metric_parameters,
)
from agentsg.cell.metric import UnitCell

# representative space group per crystal system -> expected free metric params
_SYS_FREE = {
    "triclinic": 6, "monoclinic": 4, "orthorhombic": 3,
    "tetragonal": 2, "trigonal": 2, "hexagonal": 2, "cubic": 1,
}


@pytest.mark.parametrize("row", SPACE_GROUPS, ids=[str(r[0]) for r in SPACE_GROUPS])
def test_free_parameters_match_crystal_system(row):
    n, hm, hall, cs = row
    pg = point_group(space_group(n).operations())
    assert free_metric_parameters(pg) == _SYS_FREE[cs]


def test_cubic_metric_is_invariant_under_its_point_group():
    pg = point_group(space_group(225).operations())     # m-3m
    G = UnitCell(5.0, 5.0, 5.0, 90, 90, 90).metric_tensor()
    assert metric_is_invariant(G, pg)
    # a non-cubic metric is NOT invariant under m-3m
    Gbad = UnitCell(5.0, 6.0, 5.0, 90, 90, 90).metric_tensor()
    assert not metric_is_invariant(Gbad, pg)


def test_symmetrize_projects_onto_invariant_subspace():
    pg = point_group(space_group(225).operations())     # cubic
    # a noisy near-cubic cell
    G = UnitCell(5.01, 4.99, 5.02, 90.1, 89.9, 90.0).metric_tensor()
    Gs = symmetrize_metric(G, pg)
    assert metric_is_invariant(Gs, pg)
    # cubic invariant metric is a scalar multiple of identity
    assert abs(Gs[0][0] - Gs[1][1]) < 1e-9
    assert abs(Gs[1][1] - Gs[2][2]) < 1e-9
    assert abs(Gs[0][1]) < 1e-9 and abs(Gs[0][2]) < 1e-9 and abs(Gs[1][2]) < 1e-9


def test_symmetrize_is_idempotent():
    pg = point_group(space_group(75).operations())      # tetragonal 4
    G = UnitCell(5.0, 5.2, 8.0, 90, 90, 90).metric_tensor()
    G1 = symmetrize_metric(G, pg)
    G2 = symmetrize_metric(G1, pg)
    for i in range(3):
        for j in range(3):
            assert abs(G1[i][j] - G2[i][j]) < 1e-12
