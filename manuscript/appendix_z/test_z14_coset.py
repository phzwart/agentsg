"""z:coset — full-closure matching recovers automorphism coset."""
from __future__ import annotations

import pytest

from helpers import (
    ORTHO, P63, CUBOID, even_cuboid_cell, metric_tensor, transform_metric,
    brute_automorphisms, det3,
)

pytestmark = [pytest.mark.zcheck]


def _check_ops(ops, GA, GB):
    assert ops
    for P in ops:
        assert abs(det3(P)) == 1
        for row in P:
            for x in row:
                assert isinstance(x, int) or abs(x - int(x)) < 1e-9
        Gp = transform_metric(GA, P)
        resid = max(abs(Gp[a][b] - GB[a][b]) for a in range(3) for b in range(3))
        assert resid < 1e-6


def test_orthorhombic_self_coset_order_8():
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.selling_closure import selling_closure_representatives
    from agentsg.cell.canonical import _PERMS, _inv3_unimod, _matmul_int, _transform_metric_int
    from agentsg.cell.selling_closure import selling_superbase_closure

    ops = reindexing_via_canonical(ORTHO, ORTHO, boundary_rel=0, verify_rel=1e-9)
    assert len(ops) == 8
    G = metric_tensor(ORTHO)
    _check_ops(ops, G, G)
    # independent brute
    aut = brute_automorphisms(G, entry_max=2)
    assert len(aut) == len(ops)

    # representatives-only → 2
    reps = selling_closure_representatives(ORTHO)
    # match reps only (manual, like selling_closure tests)
    found = set()
    for CA in reps:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in reps:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_unimod(W)
                    if Winv is None:
                        continue
                    P = _matmul_int(U, Winv)
                    if abs(det3(P)) != 1:
                        continue
                    Gp = _transform_metric_int(G, P)
                    if max(abs(Gp[a][b] - G[a][b]) for a in range(3) for b in range(3)) <= 1e-8:
                        found.add(tuple(tuple(int(x) for x in row) for row in P))
    assert len(found) == 2


def test_hexagonal_self_coset_order_24():
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.selling_closure import selling_closure_representatives
    from agentsg.cell.canonical import _PERMS, _inv3_unimod, _matmul_int, _transform_metric_int

    ops = reindexing_via_canonical(P63, P63, boundary_rel=0, verify_rel=1e-9)
    assert len(ops) == 24
    G = metric_tensor(P63)
    _check_ops(ops, G, G)

    reps = selling_closure_representatives(P63)
    found = set()
    for CA in reps:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in reps:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_unimod(W)
                    if Winv is None:
                        continue
                    P = _matmul_int(U, Winv)
                    if abs(det3(P)) != 1:
                        continue
                    Gp = _transform_metric_int(G, P)
                    if max(abs(Gp[a][b] - G[a][b]) for a in range(3) for b in range(3)) <= 1e-8:
                        found.add(tuple(tuple(int(x) for x in row) for row in P))
    assert len(found) == 4


def test_odd_vs_even_cuboid_coset_order_8():
    from agentsg.cell.canonical import reindexing_via_canonical, canonical_superbase
    from agentsg.cell.canonical import _PERMS, _inv3_unimod, _matmul_int, _transform_metric_int

    even, _ = even_cuboid_cell(CUBOID)
    ops = reindexing_via_canonical(CUBOID, even, boundary_rel=0, verify_rel=1e-9)
    assert len(ops) == 8
    GA, GB = metric_tensor(CUBOID), metric_tensor(even)
    _check_ops(ops, GA, GB)

    # 24 relabellings of a single superbase → 0
    C0, _ = canonical_superbase(CUBOID)
    found = set()
    for perm in _PERMS:
        for s in (1, -1):
            # only match C0 against itself style on B's reduced — use even's C0
            CB, _ = canonical_superbase(even)
            U = [[C0[1][r], C0[2][r], C0[3][r]] for r in range(3)]
            W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                 for r in range(3)]
            Winv = _inv3_unimod(W)
            if Winv is None:
                continue
            P = _matmul_int(U, Winv)
            if abs(det3(P)) != 1:
                continue
            Gp = _transform_metric_int(GA, P)
            if max(abs(Gp[a][b] - GB[a][b]) for a in range(3) for b in range(3)) <= 1e-8:
                found.add(tuple(tuple(int(x) for x in row) for row in P))
    assert len(found) == 0
