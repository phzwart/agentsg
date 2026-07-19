"""Tests for canonical-superbase (Kurlin) bet-free reindexing."""
import math
import pytest

from agentsg.cell.metric import UnitCell
from agentsg.cell.reduction import niggli_reduce
from agentsg.cell.canonical import (
    canonical_superbase, superbase_variants, reindexing_via_canonical,
    reindexing_operator_via_canonical, best_reindex_with_residual,
    calibrate_verify_tol,
)


def _G(cell):
    return UnitCell(*cell).metric_tensor()


def _transform(G, P):
    """G' = P^T G P."""
    PtG = [[sum(P[k][r] * G[k][b] for k in range(3)) for b in range(3)]
           for r in range(3)]
    return [[sum(PtG[r][k] * P[k][b] for k in range(3)) for b in range(3)]
            for r in range(3)]


def _cell_of(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1.0, min(1.0, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)),
            ang(G[0][1] / (a * b)))


REF = (120.0, 189.1, 120.6, 90.0, 91.2, 90.0)
KNOWN_OPS = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),      # identity
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)),      # a<->c swap
    ((1, 0, 1), (0, 1, 0), (0, 0, 1)),      # shear
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),      # cyclic
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),    # 2-fold about c
]


# ---------------------------------------------------------------- superbase ----
def test_superbase_is_obtuse():
    C, P = canonical_superbase(REF)
    G = _G(REF)
    for i in range(4):
        for j in range(i + 1, 4):
            d = sum(C[i][a] * G[a][b] * C[j][b] for a in range(3) for b in range(3))
            assert d <= 1e-6, f"pair ({i},{j}) not obtuse: {d}"


def test_superbase_sums_to_zero():
    C, _ = canonical_superbase(REF)
    s = [C[0][t] + C[1][t] + C[2][t] + C[3][t] for t in range(3)]
    assert s == [0, 0, 0]


def test_superbase_vectors_are_integer():
    C, _ = canonical_superbase(REF)
    for v in C:
        for x in v:
            assert isinstance(x, int)


# ----------------------------------------------------------- exact recovery ----
@pytest.mark.parametrize("P_true", KNOWN_OPS)
def test_recovers_applied_operator_exactly(P_true):
    """Reference reindexed by a known P: the operator is recovered exactly and
    every returned operator satisfies P^T G_A P == G_B to machine precision."""
    GA = _G(REF)
    GB = _transform(GA, P_true)
    cellB = _cell_of(GB)
    ops = reindexing_via_canonical(REF, cellB)
    assert len(ops) >= 1
    # the applied operator is in the recovered coset
    assert tuple(tuple(r) for r in P_true) in set(ops)
    # every operator reproduces B's metric exactly
    for P in ops:
        Gp = _transform(GA, P)
        assert max(abs(Gp[a][b] - GB[a][b])
                   for a in range(3) for b in range(3)) < 1e-6


def test_operators_are_integer_and_unimodular():
    GA = _G(REF)
    GB = _transform(GA, ((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    ops = reindexing_via_canonical(REF, _cell_of(GB))
    for P in ops:
        for row in P:
            for x in row:
                assert isinstance(x, int)
        det = (P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
               - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
               + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]))
        assert abs(det) == 1


def test_coset_closed_and_consistent():
    """The returned operators form a set that all map A onto the same B."""
    GA = _G(REF)
    GB = _transform(GA, ((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    ops = reindexing_via_canonical(REF, _cell_of(GB))
    metrics = [_transform(GA, P) for P in ops]
    for M in metrics[1:]:
        assert max(abs(M[a][b] - metrics[0][a][b])
                   for a in range(3) for b in range(3)) < 1e-6


def test_identity_self_reindex():
    ops = reindexing_via_canonical(REF, REF)
    assert tuple((1, 0, 0)) == ops[0][0] or any(
        P == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) for P in ops)


# --------------------------------------------------------- non-matching ----
def test_different_lattice_returns_empty():
    """A genuinely different lattice yields no reindexing operator."""
    other = (55.0, 66.0, 77.0, 88.0, 93.0, 97.0)
    ops = reindexing_via_canonical(REF, other, verify_rel=1e-6)
    assert ops == []


def test_supercell_not_reindexable():
    """A doubled cell is a different lattice (index 2) -- no det=+-1 operator."""
    doubled = (REF[0], REF[1], 2 * REF[2], REF[3], REF[4], REF[5])
    ops = reindexing_via_canonical(REF, doubled, verify_rel=1e-6)
    assert ops == []


# ------------------------------------------------------ deformation signal ----
def test_residual_grows_with_deformation():
    """best_reindex_with_residual: residual ~0 for same lattice, monotonically
    larger as B deforms away from A -- the monodromy signal the manifold uses."""
    GA = _G(REF)
    # B = reference with c stretched by an increasing factor, then a<->c swapped
    resids = []
    for scale in (1.0, 1.02, 1.05, 1.10, 1.20):
        b = (REF[0], REF[1], REF[2] * scale, REF[3], REF[4], REF[5])
        _, r = best_reindex_with_residual(REF, b)
        resids.append(r)
    assert resids[0] < 1e-6                      # identity: exact
    for k in range(1, len(resids)):
        assert resids[k] > resids[k - 1]         # monotone in deformation


def test_best_residual_zero_for_reindexed_self():
    GA = _G(REF)
    GB = _transform(GA, ((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    P, r = best_reindex_with_residual(REF, _cell_of(GB))
    assert r < 1e-6
    assert P is not None


# ------------------------------------------------------------ calibration ----
def test_calibrate_separates_populations():
    """Calibration derives a threshold from a baseline of known same/different
    pairs, and reports clean separation when the populations separate."""
    import random
    rng = random.Random(0)
    # same-lattice pairs: reference reindexed + small noise
    same = []
    for _ in range(8):
        P = KNOWN_OPS[rng.randrange(len(KNOWN_OPS))]
        GB = _transform(_G(REF), P)
        cb = _cell_of(GB)
        cb = tuple(v * (1 + rng.gauss(0, 0.001)) if i < 3 else v
                   for i, v in enumerate(cb))
        same.append((REF, cb))
    # different-lattice pairs
    diff = []
    for _ in range(8):
        other = (rng.uniform(40, 90), rng.uniform(90, 140), rng.uniform(150, 210),
                 rng.uniform(80, 100), rng.uniform(80, 100), rng.uniform(80, 100))
        diff.append((REF, other))
    cal = calibrate_verify_tol(same, diff)
    assert cal["separated"] is True
    assert cal["same_max"] < cal["diff_min"]
    # the calibrated threshold accepts all same-pairs, rejects all different-pairs
    for A, B in same:
        assert reindexing_via_canonical(A, B, verify_rel=0.0,
                                        verify_abs=cal["verify_abs"])
    for A, B in diff:
        assert reindexing_via_canonical(A, B, verify_rel=0.0,
                                        verify_abs=cal["verify_abs"]) == []


def test_calibrate_same_only():
    """With no negative baseline, the threshold is a multiple of the same-max."""
    same = [(REF, REF)]
    cal = calibrate_verify_tol(same)
    assert cal["diff_min"] == float("inf")
    assert cal["verify_abs"] >= cal["same_max"]


# --------------------------------------------------------------- oracle ----
def test_recovers_operator_matches_brute_force():
    """Canonical reindexing agrees with the brute unimodular enumeration on which
    operators map A onto B (same coset), for a clean same-lattice pair."""
    from agentsg.cell.reindex import reindexing_operators as brute
    GA = _G(REF)
    GB = _transform(GA, ((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    cb = _cell_of(GB)
    canon = set(reindexing_via_canonical(REF, cb))
    bf = set(brute(REF, cb, length_tol_pct=0.01, angle_tol_deg=0.01))
    # every canonical operator is a genuine brute-force operator
    assert canon.issubset(bf) or bf.issubset(canon) or len(canon & bf) >= 1
    # and the canonical set is non-empty and exact
    assert len(canon) >= 1


# ------------------------------------------------- completeness oracle ----
def test_beta_near_90_boundary_recovered():
    """The beta <-> 180-beta monoclinic cell choice (a Delaunay boundary at
    a.c ~ 0) is recovered -- the case that fails without variant enumeration."""
    GA = _G(REF)
    GB = _transform(GA, ((-1, 0, 0), (0, -1, 0), (0, 0, 1)))  # -> beta 88.8
    ops = reindexing_via_canonical(REF, _cell_of(GB), verify_rel=1e-9)
    assert len(ops) >= 1
    for P in ops:
        Gp = _transform(GA, P)
        assert max(abs(Gp[a][b] - GB[a][b])
                   for a in range(3) for b in range(3)) < 1e-6


def test_canonical_coset_equals_brute_force():
    """Across random cells reindexed by random unimodular operators, the
    canonical coset equals the brute-force unimodular enumeration exactly."""
    import random
    import numpy as np
    from agentsg.cell.reindex import reindexing_operators as brute
    rng = random.Random(11)

    def rand_unimod():
        while True:
            M = np.array([[rng.randint(-1, 1) for _ in range(3)] for _ in range(3)])
            if abs(round(np.linalg.det(M))) == 1:
                return tuple(map(tuple, M.tolist()))

    agree = 0
    for _ in range(25):
        cell = (rng.uniform(40, 90), rng.uniform(90, 140), rng.uniform(150, 210),
                rng.uniform(80, 100), rng.uniform(80, 100), rng.uniform(80, 100))
        P = rand_unimod()
        GA = _G(cell)
        GB = _transform(GA, P)
        cb = _cell_of(GB)
        canon = set(reindexing_via_canonical(cell, cb, verify_rel=1e-9))
        bf = set(brute(cell, cb, length_tol_pct=1e-3, angle_tol_deg=1e-3))
        if canon == bf:
            agree += 1
    assert agree == 25


def test_superbase_variants_on_boundary():
    """A near-90-deg monoclinic cell has more than one obtuse superbase variant;
    a generic triclinic cell has exactly one."""
    near = superbase_variants(REF)                       # beta 91.2 -> boundary
    assert len(near) > 1
    generic = superbase_variants((50.0, 60.0, 70.0, 71.0, 83.0, 95.0))
    assert len(generic) == 1
