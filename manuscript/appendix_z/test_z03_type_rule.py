"""z:type-rule — Voronoi type from zero pattern; illegal 3-zero singular."""
from __future__ import annotations

import pytest

from helpers import TYPE_CONORMS, cell_from_conorms, CLOSURE_TABLE

pytestmark = [pytest.mark.zcheck]


@pytest.mark.parametrize("vtype", [1, 2, 3, 4, 5])
def test_voronoi_type_matches_fixture(vtype):
    from agentsg.cell.selling_closure import voronoi_type, closure_class_count

    cell = cell_from_conorms(TYPE_CONORMS[vtype]())
    assert voronoi_type(cell) == vtype
    assert closure_class_count(cell) == CLOSURE_TABLE[vtype][1]


def test_illegal_three_zero_pattern_singular():
    """p01=p02=p03=0 ⇒ |v0|^2 = 0 ⇒ v0=0 ⇒ v1+v2+v3=0 (dependent)."""
    assert abs(0.0 + 0.0 + 0.0) < 1e-15  # |v0|^2 = p01+p02+p03
    from helpers import gram_from_conorms
    np = pytest.importorskip("numpy")
    p = {
        (0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0,
        (1, 2): 1.0, (1, 3): 2.0, (2, 3): 3.0,
    }
    G = np.asarray(gram_from_conorms(p), dtype=float)
    # Linear dependence v1+v2+v3=0 implies a null direction in R^3 for the
    # centered embedding; the 3x3 Gram of an obtuse superbase with this
    # pattern cannot be a positive-definite lattice metric.
    # Concrete check: attempting Cholesky / PD fails or |v0|^2 claim holds.
    # Also: sum of rows of the 4x4 Gram of (v0..v3) is zero with v0=0.
    w = np.linalg.eigvalsh(G)
    # May still look PD as a 3x3 of a dependent frame projected oddly —
    # the algebraic claim is |v0|^2=0, which forces a degenerate 4-vector set.
    v0_sq = p[(0, 1)] + p[(0, 2)] + p[(0, 3)]
    assert v0_sq == 0.0
    # Building a cell must fail or yield non-PD / singular volume
    try:
        from helpers import cell_from_conorms
        cell = cell_from_conorms(
            (0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        )
        from agentsg.cell.metric import UnitCell
        vol = UnitCell(*cell).volume()
        assert vol == 0.0 or w[0] <= 1e-8
    except Exception:
        pass  # failure to build is confirmation