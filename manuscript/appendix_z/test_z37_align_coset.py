"""z:align-coset — alignment ambiguity is one coset of the eps-automorphisms.

Aut_eps(B) = unimodular Q with ||Q^T G_B Q - G_B|| <= eps. The alignments of
two closures within eps of optimal form a single coset P0 * Aut_{2 eps}(B)
(triangle inequality both ways), so the lineup is unique exactly when
2 eps lies below B's pseudo-symmetry margin s(B) -- the residual of the
nearby symmetric lattice's extra operators, which scales linearly with the
distance to the stratum. Monoclinic beta = 90 + delta against itself:
coset 4 (exact 2/m) below the margin, 8 (pseudo-mmm) above it, margin
proportional to delta, and the returned band is group-closed.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zcheck]


def _np():
    import numpy
    return numpy


def _self_coset(delta, boundary_rel, verify_rel):
    import numpy as np
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.metric import UnitCell

    B = (50.0, 70.0, 60.0, 90.0, 90.0 + delta, 90.0)
    GB = np.array(UnitCell(*B).metric_tensor())
    ops = reindexing_via_canonical(B, B, boundary_rel=boundary_rel,
                                   verify_rel=verify_rel)
    Pm = [np.array(P, float) for P in ops]
    res = [float(np.linalg.norm(P.T @ GB @ P - GB)) for P in Pm]
    return GB, Pm, res


def test_coset_counts_margin_scaling_and_group_closure():
    np = _np()
    for delta, lo, hi in ((2.0, 250.0, 350.0), (0.2, 25.0, 35.0)):
        # tight band: the exact automorphism coset of 2/m
        _, Pm, res = _self_coset(delta, 0.0, 1e-6)
        assert len(Pm) == 4
        assert max(res) < 1e-6
        # loose band past the margin: pseudo-mmm inflation to 8
        GB, Pm, res = _self_coset(delta, 0.05, 2e-2)
        assert len(Pm) == 8
        # the margin s(B) is the extra operators' residual, linear in delta
        assert lo < max(res) < hi
        # group closure: every P_i^{-1} P_j is an eps-automorphism within
        # twice the accepted band (coset structure, not a loose bag)
        worst = 0.0
        for Pi in Pm:
            Ii = np.linalg.inv(Pi)
            for Pj in Pm:
                Q = Ii @ Pj
                worst = max(worst, float(
                    np.linalg.norm(Q.T @ GB @ Q - GB)))
        assert worst <= 2.0 * max(res) + 1e-6
    # uniqueness criterion in one line: same loose band, both deltas -> the
    # inflation is decided by the margin, not by the band alone
    _, Pm_tight, _ = _self_coset(2.0, 0.05, 1e-3)
    assert len(Pm_tight) == 4  # band below the delta=2 margin: still unique
