"""z:transport — alignment is whole-set: COB transport and the three-level
candidate spectrum.

An operator accepted by the closure match never pairs one member with one
other in isolation: it induces a bijection of the member sets and conjugates
the internal change-of-basis web, W'_{s(i)s(j)} = P W_ij P^{-1}. The full
member-pair residual spectrum is three-level -- the distance, the
pseudo-symmetry margin s(B), the bulk -- with the pseudo level proposed only
by pairings into the widened extras. Worked monoclinic pair: parent
(30, 40, 50, 90, 92, 90) on two Selling walls, daughter nudged off both by
eps = 14.4 A^2; at tau = 3 eps levels 14.4 / >= 1378 A^2; past the
pseudo-wall (tau >= 45.0) levels 14.4 / 150.7 / >= 992 A^2.
"""
from __future__ import annotations

from itertools import combinations, permutations

import pytest

pytestmark = [pytest.mark.zcheck]

PARENT = (30.0, 40.0, 50.0, 90.0, 92.0, 90.0)
DAUGHTER = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)

# parent closure, slide order: superbases as rows (b0, b1, b2, b3)
PARENT_MEMBERS = {
    1: [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    2: [[-1, 1, -1], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
    3: [[-1, -1, 0], [1, 0, 1], [0, 1, 0], [0, 0, -1]],
    4: [[-1, 1, 0], [1, 0, 1], [0, -1, 0], [0, 0, -1]],
    5: [[-1, 0, 0], [1, 0, 1], [0, 1, 0], [0, -1, -1]],
    6: [[-1, 0, 0], [1, 0, 1], [0, -1, 0], [0, 1, -1]],
}


def _np():
    import numpy
    return numpy


def _metric(cell):
    import numpy as np
    from agentsg.cell.metric import UnitCell
    return np.array(UnitCell(*cell).metric_tensor())


def _vecset(S):
    S = [tuple(int(round(x)) for x in v) for v in S]
    Sn = [tuple(-x for x in v) for v in S]
    return min(tuple(sorted(S)), tuple(sorted(Sn)))


def _widened_members(cell, tau):
    """the widened closure by its definition: every superbase whose six
    conorms are >= -tau, from all superbases with entries in [-2, 2] (input
    coordinates; for this monoclinic pair every member of the relevant
    stratum closures has entries in {-1, 0, 1})."""
    import numpy as np
    G = _metric(cell)
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    SB = np.concatenate([-(B.sum(1))[:, None, :], B], 1)
    P = -np.einsum("nia,ab,njb->nij", SB, G, SB)
    q = np.stack([P[:, i, j] for i, j in
                  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]], 1)
    out = {}
    for S in SB[q.min(1) >= -tau]:
        out.setdefault(_vecset(S), S)
    return out


def _pair_min_residual(Si, Sj, GP, GD):
    """One operator per pair, minimised over the S4 x +- labelling of B."""
    import numpy as np
    Si = np.array(Si, float)
    CiInvT = np.linalg.inv(Si[1:4].T)
    best, bestP = np.inf, None
    for pm in permutations(range(4)):
        for sg in (1.0, -1.0):
            Cj = (sg * np.array(Sj, float)[list(pm)])[1:4]
            Pm = Cj.T @ CiInvT
            Pr = np.round(Pm)
            if not np.allclose(Pm, Pr, atol=1e-9):
                continue
            r = float(np.linalg.norm(Pr.T @ GD @ Pr - GP))
            if r < best:
                best, bestP = r, Pr.astype(int)
    return best, bestP


def test_three_level_candidate_spectrum():
    """At tau = 3 eps the widened closure of the daughter is exactly the six
    parent members, and the 6 x 6 pairings give two levels: twelve at eps
    (the 2/m coset) and nothing below 1300. The pseudo-mmm level s(B) is
    proposed only once the band passes 45.0 A^2 (= 49.54 - 4.54, the depth
    of the first frame across the pseudo-wall); at tau = 60 the 14 members
    give three levels: twelve at eps, ten at s(B) (only via the new
    members), the rest >= 992."""
    import numpy as np
    GP, GD = _metric(PARENT), _metric(DAUGHTER)
    eps = float(np.linalg.norm(GD - GP))
    assert 14.0 < eps < 15.0
    parent_sets = {_vecset(S) for S in PARENT_MEMBERS.values()}

    def spectrum(tau):
        wide = _widened_members(DAUGHTER, tau)
        lv1, lv2, bulk = 0, 0, float("inf")
        for Si in PARENT_MEMBERS.values():
            for key, Sj in wide.items():
                r, _ = _pair_min_residual(Si, Sj, GP, GD)
                if r < 30.0:
                    lv1 += 1
                    assert abs(r - eps) < 1.0 and key in parent_sets
                elif r < 300.0:
                    lv2 += 1
                    assert 140.0 < r < 160.0 and key not in parent_sets
                else:
                    bulk = min(bulk, r)
        return wide, lv1, lv2, bulk

    wide, lv1, lv2, bulk = spectrum(3.0 * eps)
    assert set(wide) == parent_sets               # overlap, and nothing else
    assert (lv1, lv2) == (12, 0) and bulk > 1300.0
    assert len(_widened_members(DAUGHTER, 44.9)) == 6
    assert len(_widened_members(DAUGHTER, 45.1)) == 7
    wide, lv1, lv2, bulk = spectrum(60.0)
    assert len(wide) == 14 and parent_sets <= set(wide)
    assert (lv1, lv2) == (12, 10) and bulk > 900.0


def test_cob_transport_under_accepted_operator():
    """an accepted operator aligns the whole set: the twofold along b
    permutes the six members as (1 2)(3 4)(5 6), and the internal COB web
    is conjugated, W'_{s(i)s(j)} = P W_ij P^{-1}, for every member pair."""
    import numpy as np
    GP = _metric(PARENT)
    T = np.diag([-1.0, 1.0, -1.0])                   # twofold along b
    assert float(np.linalg.norm(T.T @ GP @ T - GP)) < 1e-9

    C = {k: np.array(S, float)[1:4] for k, S in PARENT_MEMBERS.items()}
    sets = {k: _vecset(S) for k, S in PARENT_MEMBERS.items()}

    sigma = {}
    for k, S in PARENT_MEMBERS.items():
        img = [(T @ np.array(v, float)) for v in S]
        tgt = [m for m, s in sets.items() if s == _vecset(img)]
        assert len(tgt) == 1
        sigma[k] = tgt[0]
    assert sigma == {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}

    def W(i, j):
        # change of basis member i -> member j, expressed in member i's frame
        return np.round(np.linalg.inv(C[i]) @ C[j]).astype(int)

    for i in PARENT_MEMBERS:
        for j in PARENT_MEMBERS:
            lhs = W(sigma[i], sigma[j]).astype(float)
            rhs = T @ W(i, j) @ np.linalg.inv(T)
            assert np.allclose(lhs, rhs, atol=1e-9)

    # the concrete instance quoted on the slide and in the SI
    assert np.array_equal(W(1, 3), W(2, 4))


def test_worked_example_closure_table():
    """the six parent members carry one conorm multiset (two zeros at the
    two right angles) and therefore one sorted linear key."""
    import numpy as np
    from itertools import combinations
    from agentsg.cell.rootform import sorted_linear_key
    G = _metric(PARENT)
    want = sorted([847.65, 1600.0, 2447.65, 0.0, 52.35, 0.0])
    for S in PARENT_MEMBERS.values():
        M = np.array(S, float)
        p = sorted(float(-M[i] @ G @ M[j]) for i, j in combinations(range(4), 2))
        assert all(abs(a - b) < 0.01 for a, b in zip(p, want))
    k = sorted_linear_key(PARENT)
    assert all(abs(a - b) < 0.006 for a, b in
               zip(k, (0.0, 0.0, 0.744, 12.051, 22.747, 34.798)))
