"""z:pseudo — pseudo-symmetry as cosets between closures."""
from __future__ import annotations

import pytest

from helpers import (
    metric_tensor, transform_metric, cell_from_metric, det3, key_l2,
)

pytestmark = [pytest.mark.zcheck]


def _self_ops(cell, verify_rel=1e-9):
    from agentsg.cell.canonical import reindexing_via_canonical
    return reindexing_via_canonical(
        cell, cell, boundary_rel=0, verify_rel=verify_rel,
    )


def _residual(P, G):
    """||P^T G P - G||_max."""
    Gp = transform_metric(G, P)
    return max(abs(Gp[i][j] - G[i][j]) for i in range(3) for j in range(3))


def test_monoclinic_family_pseudo_orthorhombic():
    """Near a=c: H order 4; impose equality → H* order 8; exchange in H*\\H."""
    from agentsg.cell.canonical import reindexing_via_canonical

    P_ex = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    a, b, beta = 120.0, 189.1, 91.2
    ratios = []
    for d in (0.5, 1.0, 2.0, 4.0, 10.0):
        R = (a, b, a + d, 90.0, beta, 90.0)
        H = _self_ops(R)
        assert len(H) == 4, f"d={d}: |H|={len(H)}"
        # Impose a=c stratum
        Rstar = (a, b, a, 90.0, beta, 90.0)
        Hstar = _self_ops(Rstar)
        assert len(Hstar) == 8, f"|H*|={len(Hstar)}"
        Hset = set(H)
        Hstar_set = set(Hstar)
        assert Hset <= Hstar_set
        assert len(Hstar_set) // len(Hset) == 2
        assert P_ex in Hstar_set
        assert P_ex not in Hset
        # Residual of exchange on the distorted metric is linear in d
        G = metric_tensor(R)
        ratios.append(_residual(P_ex, G) / d)

    # residual/d constant to ~5%
    mean = sum(ratios) / len(ratios)
    for r in ratios:
        assert abs(r - mean) / mean < 0.05, f"ratios={ratios}"


def test_near_tetragonal_cuboid_pseudo():
    """Near-tetragonal cuboid: H=8 (mmm); H*=16 (4/mmm); 4-fold in H*\\H."""
    _4z = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    for e in (0.01, 0.1, 0.5):
        R = (3.0, 3.0 + e, 5.0, 90.0, 90.0, 90.0)
        H = _self_ops(R)
        assert len(H) == 8, f"e={e}: |H|={len(H)}"
        Rstar = (3.0, 3.0, 5.0, 90.0, 90.0, 90.0)
        Hstar = _self_ops(Rstar)
        assert len(Hstar) == 16, f"|H*|={len(Hstar)}"
        assert set(H) <= set(Hstar)
        assert len(Hstar) // len(H) == 2
        # A 4-fold (or its coset mate) is in H* but not H
        four = [P for P in Hstar if P not in set(H)]
        assert four
        G = metric_tensor(R)
        # residual of a non-H operator scales with e
        res = min(_residual(P, G) for P in four)
        assert res > 0
        # order-of-magnitude ~ e (metric units Ang^2)
        assert res < 50 * e + 1.0


def test_ops_integer_unimodular_and_close_closure():
    from agentsg.cell.selling_closure import selling_superbase_closure

    Rstar = (3.0, 3.0, 5.0, 90.0, 90.0, 90.0)
    Hstar = _self_ops(Rstar)
    cl = selling_superbase_closure(Rstar)
    for P in Hstar:
        assert abs(det3(P)) == 1
        for row in P:
            for x in row:
                assert abs(x - int(x)) < 1e-9
    assert len(cl) == 32  # V5 cuboid
