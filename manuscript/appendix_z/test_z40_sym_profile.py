"""z:sym-profile — the ideal self-spectrum is a symmetry-distance profile.

Every rung of the self-spectrum sits at twice the minimum-norm
symmetrisation cost onto the stratum where that operator becomes exact
(exactly so for involutions: Q^T G Q reflects G through the operator's
fixed subspace, so the nearest fixed metric is the average and the
residual is twice the distance). Wall strata contribute the
reduction-setting ("Selling jump") operators, tie strata the
pseudo-symmetry ones -- one ladder covers both. Worked cells: the
(30, 40, 50, 90, 92, 90) parent and its two-wall nudge.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zcheck]

PARENT = (30.0, 40.0, 50.0, 90.0, 92.0, 90.0)
DAUGHTER = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)


def _np():
    import numpy
    return numpy


def _metric(cell):
    import numpy as np
    from agentsg.cell.metric import UnitCell
    return np.array(UnitCell(*cell).metric_tensor())


def _self_ops(cell, boundary_rel, verify_rel):
    import numpy as np
    from agentsg.cell.canonical import reindexing_via_canonical
    G = _metric(cell)
    ops = reindexing_via_canonical(cell, cell, boundary_rel=boundary_rel,
                                   verify_rel=verify_rel)
    res = sorted(float(np.linalg.norm(np.array(P, float).T @ G
                                      @ np.array(P, float) - G))
                 for P in ops)
    return res


def test_rungs_are_twice_the_strata_costs():
    """daughter self-spectrum: rung 1 = 2 x (cost to the monoclinic wall
    stratum), rung 2 = 2 x (cost to mmm), with the strata costs computed
    independently by the constrained minimum-norm solve (zero the named
    off-diagonal entries)."""
    import math
    G = _metric(DAUGHTER)
    g12, g13, g23 = float(G[0, 1]), float(G[0, 2]), float(G[1, 2])
    cost_mono = math.sqrt(2.0 * (g12 ** 2 + g23 ** 2))   # p12 = p23 = 0
    cost_mmm = math.sqrt(2.0 * (g12 ** 2 + g13 ** 2 + g23 ** 2))
    assert 13.0 < cost_mono < 14.5
    assert 76.0 < cost_mmm < 78.5

    res0 = _self_ops(DAUGHTER, 0.0, 1e-6)
    assert len(res0) == 2 and max(res0) < 1e-6          # triclinic: 1, -1

    res1 = _self_ops(DAUGHTER, 0.05, 1.2e-2)
    assert len(res1) == 4
    rung1 = [r for r in res1 if r > 1.0]
    assert len(rung1) == 2
    for r in rung1:                                      # Selling-jump ops
        assert abs(r - 2.0 * cost_mono) < 0.02 * 2.0 * cost_mono

    res2 = _self_ops(DAUGHTER, 0.05, 8e-2)
    assert len(res2) == 8
    rung2 = [r for r in res2 if r > 2.0 * cost_mono + 5.0]
    assert len(rung2) == 4
    for r in rung2:                                      # pseudo-mmm ops
        assert abs(r - 2.0 * cost_mmm) < 0.02 * 2.0 * cost_mmm


def test_involution_identity_residual_is_twice_distance():
    """for every accepted non-trivial operator (all involutions up to
    sign), the residual equals exactly twice the Frobenius distance to
    that operator's own fixed metric (the average of G and its image)."""
    import numpy as np
    from agentsg.cell.canonical import reindexing_via_canonical
    G = _metric(DAUGHTER)
    ops = reindexing_via_canonical(DAUGHTER, DAUGHTER, boundary_rel=0.05,
                                   verify_rel=8e-2)
    checked = 0
    for P in ops:
        Q = np.array(P, float)
        img = Q.T @ G @ Q
        r = float(np.linalg.norm(img - G))
        if r < 1e-9:
            continue
        fixed = 0.5 * (G + img)                      # reflection average
        assert abs(r - 2.0 * float(np.linalg.norm(G - fixed))) < 1e-9
        checked += 1
    assert checked >= 6


def test_constrained_symmetrisation_of_the_parent():
    """minimum-norm solve of the single constraint p13 = 0 on the parent:
    lands exactly on (30, 40, 50, 90, 90, 90), costs sqrt(2) x 52.349 =
    74.03 = s(B)/2, leaves all eight mmm operators exact, and the closure
    grows from 6 to 16 members."""
    import numpy as np
    from agentsg.cell.canonical import (_closure_for_match,
                                        reindexing_via_canonical)
    from agentsg.cell.metric import params_from_metric
    G = _metric(PARENT)
    Gs = G.copy()
    Gs[0, 2] = Gs[2, 0] = 0.0                        # the one constraint
    cost = float(np.linalg.norm(G - Gs))
    assert abs(cost - np.sqrt(2.0) * 52.349) < 0.05

    # s(B)/2 from the parent's own pseudo band (z37 protocol)
    ops = reindexing_via_canonical(PARENT, PARENT, boundary_rel=0.05,
                                   verify_rel=8e-2)
    res = [float(np.linalg.norm(np.array(P, float).T @ G
                                @ np.array(P, float) - G)) for P in ops]
    sB = max(res)
    assert abs(cost - sB / 2.0) < 0.1

    cell_star = tuple(params_from_metric(Gs))
    assert all(abs(x - y) < 1e-6 for x, y in
               zip(cell_star, (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)))
    for P in ops:                                    # exact afterwards
        Q = np.array(P, float)
        assert float(np.linalg.norm(Q.T @ Gs @ Q - Gs)) < 1e-9

    def n_members(cell):
        ms = _closure_for_match(cell, boundary_rel=0.0)
        return len({_vs(M) for M in ms})

    def _vs(S):
        S = [tuple(int(round(x)) for x in v) for v in S]
        Sn = [tuple(-x for x in v) for v in S]
        return min(tuple(sorted(S)), tuple(sorted(Sn)))

    assert n_members(PARENT) == 6
    assert n_members(cell_star) == 16
