"""z:lepage — Table tab:flip: Le Page gate drops exchange; closure recovers it."""
from __future__ import annotations

import math

import pytest

from helpers import (
    metric_tensor, transform_metric, cell_from_metric, key_l2,
)

pytestmark = [pytest.mark.zcheck]


def _flip_delta_deg(a, c, beta_deg):
    """Angular residual of the a↔c exchange as a would-be 2-fold (degrees).

    Matches the manuscript Table tab:flip slope (~1.04° per Å of |a-c| for the
    published monoclinic family). Uses the metric mismatch of P0 as an
    automorphism, converted to an angle via atan(frobenius/scale).
    """
    s1 = (a, 60.0, c, 90.0, beta_deg, 90.0)
    P0 = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    G = metric_tensor(s1)
    Gp = transform_metric(G, P0)
    fro = math.sqrt(sum((Gp[i][j] - G[i][j]) ** 2
                        for i in range(3) for j in range(3)))
    scale = (abs(G[0][0]) + abs(G[1][1]) + abs(G[2][2])) / 3.0
    return math.degrees(math.atan(fro / max(scale, 1e-12)))


def test_table_flip_lepage_vs_closure():
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.rootform import sorted_root_key

    P0 = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    a, b, beta = 50.0, 60.0, 100.0
    rows = []
    for d in (0.0, 1.0, 2.0, 2.85, 4.0, 5.0, 10.0):
        s1 = (a, b, a + d, 90.0, beta, 90.0)
        G = metric_tensor(s1)
        s2 = cell_from_metric(transform_metric(G, P0))
        delta = _flip_delta_deg(a, a + d, beta)
        gate = delta <= 3.0
        ops = reindexing_via_canonical(s1, s2, boundary_rel=0, verify_rel=1e-9)
        assert ops, f"|a-c|={d}: no operators"
        kd = key_l2(sorted_root_key(s1), sorted_root_key(s2))
        assert kd < 1e-10
        rows.append((d, delta, gate, len(ops), kd))

    # delta grows with |a-c|; gate drops at large distortion
    assert rows[0][1] < 0.1
    assert rows[-1][1] > 3.0
    assert rows[-1][2] is False  # gate drops
    # find crossover: some small d keeps, some large d drops
    assert any(r[2] for r in rows[:4])
    assert any(not r[2] for r in rows[4:])
    print("z:lepage rows (d, delta, gate, n_ops, key_dist):")
    for r in rows:
        print(" ", r)
