"""z:brute-complete — when a {-1,0,1} unimodular search is complete.

Also pins the Niggli CoB identity N^T G N == G(red) (main_v8 claim iv).
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from helpers import (
    SEED_REINDEX_PROC, sample_unimodular, metric_tensor, transform_metric,
    cell_from_metric, det3, matmul_int, inv3, TYPE_CONORMS, cell_from_conorms,
    perturb_angles, key_l2,
)

pytestmark = [pytest.mark.zcheck]

# Fixed seed for this check
SEED_BRUTE = 29


def _max_abs_entry(M):
    return max(abs(int(x)) for row in M for x in row)


def _unimodular_pm1():
    from agentsg.cell.g6 import _unimodular_pm1
    return _unimodular_pm1()


def test_i_most_random_resettings_need_entry_mag_2():
    """(i) 200 random M with |entries|<=2: most have max|entry|>1."""
    rng = random.Random(SEED_BRUTE)
    Ms = sample_unimodular(rng, entry_max=2, n=200)
    n_large = sum(1 for M in Ms if _max_abs_entry(M) > 1)
    # manuscript observed 192/200; pin ±10%
    assert 170 <= n_large <= 200, f"got {n_large}/200 with max|entry|>1"


def test_iv_niggli_cob_identity_and_convergence():
    """(iv) N^T G N == G(red), |det|=1; converge on noisy cells."""
    from agentsg.cell.reduction import niggli_reduce
    from agentsg.cell.reduction import _transform_metric, _gram_from_params

    rng = random.Random(SEED_BRUTE + 1)
    base = (40.0, 50.0, 60.0, 80.0, 95.0, 110.0)
    n_ok = 0
    n_fail_conv = 0
    for _ in range(500):  # fast subset; full 10^4 is slow-marked below
        cell = (
            base[0] * (1 + rng.gauss(0, 0.02)),
            base[1] * (1 + rng.gauss(0, 0.02)),
            base[2] * (1 + rng.gauss(0, 0.02)),
            base[3] + rng.gauss(0, 0.5),
            base[4] + rng.gauss(0, 0.5),
            base[5] + rng.gauss(0, 0.5),
        )
        try:
            red, N = niggli_reduce(*cell, max_iter=2000)
        except RuntimeError:
            n_fail_conv += 1
            continue
        G = metric_tensor(cell)
        Gp = _transform_metric(G, N)
        Gr = _gram_from_params(*red)
        err = max(abs(Gp[i][j] - Gr[i][j]) for i in range(3) for j in range(3))
        scale = max(abs(x) for row in Gr for x in row) or 1.0
        assert err <= 1e-9 * max(scale, 1.0) or err < 1e-6
        assert abs(det3(N)) == 1
        n_ok += 1
    assert n_ok >= 480, f"too many non-convergences: fail={n_fail_conv}"
    # After the CoB fix, noisy convergence should be rare
    assert n_fail_conv / max(n_ok + n_fail_conv, 1) < 0.05


@pytest.mark.slow
def test_iv_niggli_cob_10k_noisy():
    """Full manuscript 10^4 noisy-cell CoB + convergence check."""
    from agentsg.cell.reduction import niggli_reduce, _transform_metric, _gram_from_params

    rng = random.Random(SEED_BRUTE + 2)
    base = (40.0, 50.0, 60.0, 80.0, 95.0, 110.0)
    n_fail = 0
    for _ in range(10_000):
        cell = (
            base[0] * (1 + rng.gauss(0, 0.02)),
            base[1] * (1 + rng.gauss(0, 0.02)),
            base[2] * (1 + rng.gauss(0, 0.02)),
            base[3] + rng.gauss(0, 0.5),
            base[4] + rng.gauss(0, 0.5),
            base[5] + rng.gauss(0, 0.5),
        )
        try:
            red, N = niggli_reduce(*cell, max_iter=2000)
        except RuntimeError:
            n_fail += 1
            continue
        G = metric_tensor(cell)
        err = max(
            abs(_transform_metric(G, N)[i][j] - _gram_from_params(*red)[i][j])
            for i in range(3) for j in range(3)
        )
        assert err < 1e-6
        assert abs(det3(N)) == 1
    assert n_fail / 10000 < 0.02


def _max_entry_closure_match(cell):
    """Max |P_ij| over typed-closure matching (canonical U W^{-1} path)."""
    from agentsg.cell.selling_closure import selling_superbase_closure
    from agentsg.cell.canonical import _inv3_unimod, _matmul_int, _PERMS

    cl = selling_superbase_closure(cell)
    worst = 0
    worst_P = None
    for CA in cl:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in cl:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [
                        [s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                        for r in range(3)
                    ]
                    Winv = _inv3_unimod(W)
                    if Winv is None:
                        continue
                    P = _matmul_int(U, Winv)
                    if abs(det3(P)) != 1:
                        continue
                    m = _max_abs_entry(P)
                    if m > worst:
                        worst = m
                        worst_P = P
    return worst, worst_P


def _max_entry_s4_orbit(cell):
    """Max |P_ij| within the S4×{±I} orbit of one reduced superbase."""
    from agentsg.cell.canonical import canonical_superbase, _inv3_unimod, _matmul_int, _PERMS

    C0, _ = canonical_superbase(cell)
    orb = []
    for perm in _PERMS:
        for s in (1, -1):
            orb.append([[s * C0[perm[i]][j] for j in range(3)] for i in range(4)])
    worst = 0
    for CA in orb:
        for drop_a in range(4):
            idx_a = [i for i in range(4) if i != drop_a]
            A = [[CA[idx_a[j]][r] for j in range(3)] for r in range(3)]
            if abs(det3(A)) != 1:
                continue
            for CB in orb:
                for drop_b in range(4):
                    idx_b = [i for i in range(4) if i != drop_b]
                    B = [[CB[idx_b[j]][r] for j in range(3)] for r in range(3)]
                    Binv = _inv3_unimod(B)
                    if Binv is None:
                        continue
                    P = _matmul_int(A, Binv)
                    if abs(det3(P)) != 1:
                        continue
                    worst = max(worst, _max_abs_entry(P))
    return worst


def test_iii_v3_needs_entry_mag_2_others_do_not():
    """(iii) V3 closure matching needs |entry|=2; single-class orbit stays at 1."""
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.g6 import _unimodular_pm1

    # Single-class S4 orbit: |entry|<=1 for every Voronoi type
    for vt in (1, 2, 3, 4, 5):
        w = _max_entry_s4_orbit(cell_from_conorms(TYPE_CONORMS[vt]()))
        assert w <= 1, f"V{vt} S4-orbit: max|entry|={w}"

    # Typed V3 closure: an entry of magnitude 2 occurs (manuscript example class)
    c3 = cell_from_conorms(TYPE_CONORMS[3]())  # conorms 0,2,3,4,5,0
    w3, P3 = _max_entry_closure_match(c3)
    assert w3 >= 2, f"V3 expected max|entry|>=2, got {w3}"
    assert P3 is not None

    G = metric_tensor(c3)
    cB = cell_from_metric(transform_metric(G, P3))
    ops = reindexing_via_canonical(c3, cB, boundary_rel=0, verify_rel=1e-9)
    assert ops, "closure route should recover the V3 relating operator"
    assert any(_max_abs_entry(P) >= 2 for P in ops)

    pm1 = set(_unimodular_pm1())
    GB = metric_tensor(cB)
    found_pm1 = False
    for M in pm1:
        Gp = transform_metric(G, M)
        if max(abs(Gp[i][j] - GB[i][j]) for i in range(3) for j in range(3)) < 1e-6:
            found_pm1 = True
            break
    assert not found_pm1, "{-1,0,1} search should miss this V3 pair"


def test_ii_niggli_cells_relating_op_in_pm1():
    """(ii) Between Niggli cells of one lattice, relating op has |entries|<=1."""
    from agentsg.cell.reduction import niggli_reduce
    from agentsg.cell.primitive import primitive_cell

    rng = random.Random(SEED_BRUTE + 3)
    refs = [
        ("oP", (3.0, 4.0, 5.0, 90.0, 90.0, 90.0), None),
        ("cP", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), None),
        ("hP", (3.0, 3.0, 5.0, 90.0, 90.0, 120.0), None),
        ("mP", (120.0, 189.1, 120.3, 90.0, 91.2, 90.0), None),
        ("tric", (40.0, 50.0, 60.0, 80.0, 95.0, 110.0), None),
        ("oI", (3.0, 4.0, 5.0, 90.0, 90.0, 90.0), "I222"),
        ("cF", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), "F23"),
    ]
    worst = 0
    for label, cell, sym in refs:
        R = primitive_cell(cell, sym) if sym else cell
        G0 = metric_tensor(R)
        for _ in range(30):  # 30 x 7 = 210; manuscript 150 per type is slow
            M1 = sample_unimodular(rng, 2, 1)[0]
            M2 = sample_unimodular(rng, 2, 1)[0]
            # noise before resetting
            noisy = perturb_angles(
                (
                    R[0] * (1 + rng.gauss(0, 0.02 / R[0])),
                    R[1] * (1 + rng.gauss(0, 0.02 / R[1])),
                    R[2] * (1 + rng.gauss(0, 0.02 / R[2])),
                    R[3], R[4], R[5],
                ),
                0.05, rng,
            )
            Gn = metric_tensor(noisy)
            c1 = cell_from_metric(transform_metric(Gn, M1))
            c2 = cell_from_metric(transform_metric(Gn, M2))
            r1, N1 = niggli_reduce(*c1)
            r2, N2 = niggli_reduce(*c2)
            # P relating r1 -> r2 in reduced bases: approx via reindexing
            from agentsg.cell.canonical import reindexing_via_canonical
            ops = reindexing_via_canonical(r1, r2, boundary_rel=0, verify_rel=1e-3)
            if not ops:
                continue
            for P in ops:
                worst = max(worst, _max_abs_entry(P))
    assert worst <= 1, f"Niggli-cell relating op had max|entry|={worst}"
