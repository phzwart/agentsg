"""z:reindex-proc — end-to-end reindex to reference recovers Aut·M (or M·Aut)."""
from __future__ import annotations

import random

import pytest

from helpers import (
    SEED_REINDEX_PROC, TRICLINIC, MONOCLINIC_P, ORTHO, P63, TETRAGONAL,
    sample_unimodular, metric_tensor, transform_metric, cell_from_metric,
    reindex_to_reference, matmul_int, det3, inv3,
)

pytestmark = [pytest.mark.zcheck]


REFERENCES = [
    ("triclinic", TRICLINIC, None),
    ("monoclinic_P", MONOCLINIC_P, None),
    ("monoclinic_C", (50.0, 60.0, 70.0, 90.0, 110.0, 90.0), "C2"),
    ("orthorhombic_P", ORTHO, None),
    ("orthorhombic_I", (40.0, 50.0, 60.0, 90.0, 90.0, 90.0), "I222"),
    ("tetragonal", TETRAGONAL, None),
    ("hexagonal", P63, None),
    ("cubic_F", (50.0, 50.0, 50.0, 90.0, 90.0, 90.0), "F23"),
]


def _int_inv(M):
    Mi = inv3([list(r) for r in M])
    return tuple(tuple(int(round(x)) for x in row) for row in Mi)


def _matches_coset(ops, M, aut):
    got = {tuple(tuple(int(x) for x in row) for row in P) for P in ops}
    candidates = [M, _int_inv(M)]
    for Mc in candidates:
        right = {matmul_int(Mc, a) for a in aut}
        left = {matmul_int(a, Mc) for a in aut}
        if got == right or got == left:
            return True
    return False


@pytest.mark.parametrize("label,cell,symbol", REFERENCES)
def test_reindex_recovers_unimodular_resetting(label, cell, symbol):
    from agentsg.cell.primitive import primitive_cell
    from agentsg.cell.canonical import reindexing_via_canonical

    rng = random.Random(SEED_REINDEX_PROC + hash(label) % 1000)
    R_prim = primitive_cell(cell, symbol) if symbol else cell
    G = metric_tensor(R_prim)
    aut = reindexing_via_canonical(R_prim, R_prim, boundary_rel=0, verify_rel=1e-9)
    assert len(aut) >= 1

    for M in sample_unimodular(rng, entry_max=2, n=10):
        N_prim = cell_from_metric(transform_metric(G, M))
        ops, _ref = reindex_to_reference(R_prim, N_prim, verify_rel=1e-9)
        assert ops, f"{label}: empty ops for M={M}"
        for P in ops:
            assert abs(det3(P)) == 1
        assert len(ops) == len(aut)
        assert _matches_coset(ops, M, aut), (
            f"{label}: ops not a left/right coset of Aut for M={M}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("label,cell,symbol", REFERENCES)
def test_reindex_recovers_unimodular_resetting_50(label, cell, symbol):
    """Full manuscript: 50 unimodular matrices per reference."""
    from agentsg.cell.primitive import primitive_cell
    from agentsg.cell.canonical import reindexing_via_canonical

    rng = random.Random(SEED_REINDEX_PROC + hash(label) % 1000 + 99)
    R_prim = primitive_cell(cell, symbol) if symbol else cell
    G = metric_tensor(R_prim)
    aut = reindexing_via_canonical(R_prim, R_prim, boundary_rel=0, verify_rel=1e-9)
    for M in sample_unimodular(rng, entry_max=2, n=50):
        N_prim = cell_from_metric(transform_metric(G, M))
        ops, _ = reindex_to_reference(R_prim, N_prim, verify_rel=1e-9)
        assert ops and len(ops) == len(aut)
        assert _matches_coset(ops, M, aut)


def test_centred_conventional_operator_denominators():
    from agentsg.cell.primitive import primitive_cell, primitive_transform, CENTRING_MULTIPLICITY
    from agentsg.cell.canonical import reindexing_via_canonical
    from helpers import matmul

    rng = random.Random(SEED_REINDEX_PROC)
    cell = (50.0, 60.0, 70.0, 90.0, 110.0, 90.0)
    symbol = "C2"
    C = primitive_transform(symbol)
    R_prim = primitive_cell(cell, symbol)
    G = metric_tensor(R_prim)
    M = sample_unimodular(rng, entry_max=2, n=1)[0]
    N_prim = cell_from_metric(transform_metric(G, M))
    ops = reindexing_via_canonical(R_prim, N_prim, boundary_rel=0, verify_rel=1e-9)
    assert ops
    P = [list(row) for row in ops[0]]
    C_inv = inv3(C)
    P_conv = matmul(C, matmul(P, C_inv))
    mult = CENTRING_MULTIPLICITY["C"]
    for row in P_conv:
        for x in row:
            assert abs(x * mult - round(x * mult)) < 1e-8


def test_reindex_proc_lepage_crosscheck():
    from agentsg.cell.canonical import reindexing_via_canonical

    P0 = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    a, b, beta = 50.0, 60.0, 100.0
    for d in (0.0, 1.0, 2.0, 2.85, 4.0, 5.0, 10.0):
        s1 = (a, b, a + d, 90.0, beta, 90.0)
        G = metric_tensor(s1)
        s2 = cell_from_metric(transform_metric(G, P0))
        ops = reindexing_via_canonical(s1, s2, boundary_rel=0, verify_rel=1e-9)
        assert ops
        got = {tuple(tuple(int(x) for x in row) for row in P) for P in ops}
        assert P0 in got or any(
            max(abs(transform_metric(G, P)[i][j] - metric_tensor(s2)[i][j])
                for i in range(3) for j in range(3)) < 1e-6
            for P in ops
        )


def test_space_group_branch_count_equals_ops_over_laue():
    """With space group given: |Aut|/|L| equals the number of Laue cosets."""
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.space_groups import space_group
    from agentsg.cell.ambiguity import _laue_matrices

    R = P63  # hexagonal: Aut typically 24, Laue 6/m order 12 → 2 branches
    H = {
        tuple(tuple(int(x) for x in row) for row in P)
        for P in reindexing_via_canonical(R, R, boundary_rel=0, verify_rel=1e-9)
    }
    L = {
        tuple(tuple(int(x) for x in row) for row in W.rows)
        for W in _laue_matrices(space_group("P63"))
    }
    L_in_H = L & H
    assert L_in_H, "Laue should embed in Aut(R)"
    assert len(H) % len(L_in_H) == 0
    n_branches = len(H) // len(L_in_H)
    assert n_branches >= 1

    # One representative per coset; prefer identity then minimal order
    def _ord(M):
        P = M
        for k in range(1, 7):
            if P == ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                return k
            P = matmul_int(P, M)
        return 99

    remaining = set(H)
    reps = []
    while remaining:
        cand = min(
            remaining,
            key=lambda M: (
                0 if M == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) else 1,
                _ord(M),
                M,
            ),
        )
        reps.append(cand)
        remaining -= {matmul_int(cand, ell) for ell in L_in_H}
    assert len(reps) == n_branches
    assert reps[0] == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    # Representatives are of minimal order within their coset
    for r in reps:
        coset = {matmul_int(r, ell) for ell in L_in_H}
        assert _ord(r) == min(_ord(x) for x in coset)
