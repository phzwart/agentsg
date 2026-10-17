"""z:selling-groupoid — the Selling reduction as a groupoid.

(i) For each of the 24 Delone sorts (Selling patterns of Andrews, Bernstein &
Sauter, 2023), a generic lattice's closure groupoid: number of members,
loop group Aut (equal to the holohedry order), the part of Aut that fixes
the reduced superbase (relabellings, visible in one reduced cell), and the
number of members Aut reaches from it. (ii) Reindexing reads an arrow: a new
cell whose only reduced frame is a hidden member of the reference's widened
closure is matched from that single frame; the full near-coset, including the
true operator, is proposed only by the cache built as Theorem 2 prescribes.
"""
from __future__ import annotations

import itertools

import pytest

pytestmark = [pytest.mark.zcheck]

from test_z44_widened_complete import (  # noqa: E402
    LATTICES, S0, _cn, _generate, _gram, _key, _lam1,
)

# name, Bravais type, Selling pattern s1..s6 (s1=b.c, s2=a.c, s3=a.b, s4=a.d, s5=b.d, s6=c.d)
SORTS = [("C1", "cI", "rrrrrr"), ("C3", "cF", "rr0rr0"), ("C5", "cP", "000rrr"),
         ("T1", "tI", "rrsrrs"), ("T2", "tI", "rr0rrs"), ("T5", "tP", "000rrs"),
         ("R1", "hR", "rrrsss"), ("R3", "hR", "rr0sr0"), ("O1A", "oF", "rrsrrt"),
         ("O1B", "oI", "rstrst"), ("O2", "oI", "rs0srt"), ("O3", "oI", "rs0rs0"),
         ("O4", "oS", "00rsst"), ("O5", "oP", "000rst"), ("M1A", "mC", "rrsttu"),
         ("M1B", "mC", "rstrsu"), ("M2A", "mC", "rs0stu"), ("M2B", "mC", "rs0rst"),
         ("M3", "mC", "rs0ts0"), ("M4", "mP", "00rstu"), ("H4", "hP", "00rrrs"),
         ("A1", "aP", "rstuvw"), ("A2", "aP", "rs0tuv"), ("A3", "aP", "rs0tu0")]
HOLOHEDRY = {"cI": 48, "cF": 48, "cP": 48, "tI": 16, "tP": 16, "hR": 12, "hP": 24,
             "oF": 8, "oI": 8, "oS": 8, "oP": 8, "mC": 4, "mP": 4, "aP": 2}
# sort: (zeros, members, |Aut|, fixing the reduced superbase, members reached by Aut)
EXPECTED = {"C1": (0, 1, 48, 48, 1), "C3": (2, 3, 48, 16, 3), "C5": (3, 16, 48, 12, 4),
            "T1": (0, 1, 16, 16, 1), "T2": (1, 2, 16, 8, 2), "T5": (3, 16, 16, 4, 4),
            "R1": (0, 1, 12, 12, 1), "R3": (2, 3, 12, 4, 3), "O1A": (0, 1, 8, 8, 1),
            "O1B": (0, 1, 8, 8, 1), "O2": (1, 2, 8, 4, 2), "O3": (2, 3, 8, 8, 1),
            "O4": (2, 6, 8, 4, 2), "O5": (3, 16, 8, 2, 4), "M1A": (0, 1, 4, 4, 1),
            "M1B": (0, 1, 4, 4, 1), "M2A": (1, 2, 4, 2, 2), "M2B": (1, 2, 4, 4, 1),
            "M3": (2, 3, 4, 4, 1), "M4": (2, 6, 4, 2, 2), "H4": (2, 6, 24, 4, 6),
            "A1": (0, 1, 2, 2, 1), "A2": (1, 2, 2, 2, 1), "A3": (2, 3, 2, 2, 1)}


def sort_groupoid(pattern, rnd):
    """closure members (unordered, up to sign), Aut, the part of Aut fixing
    the reduced superbase S0, and the members Aut reaches from S0."""
    import numpy as np
    letters = sorted(set(pattern) - {"0"})
    val = dict(zip(letters, rnd.uniform(300, 2500, len(letters))))
    s = [0.0 if ch == "0" else val[ch] for ch in pattern]
    p = np.array([s[3], s[4], s[5], s[2], s[1], s[0]])     # (p01 p02 p03 p12 p13 p23)
    G = _gram(p)
    _, members = _generate(p, 0.0, 1e18)
    k0 = _key(np.array(S0, float))
    aut, fix, reached = set(), set(), set()
    for k, S in members.items():
        for pm in itertools.permutations(range(4)):
            for sg in (1, -1):
                W = (sg * S[list(pm)])[1:4].T
                if np.allclose(W.T @ G @ W, G, atol=1e-7 * np.abs(G).max()):
                    key = tuple(np.round(W).astype(int).ravel())
                    aut.add(key)
                    reached.add(k)
                    if k == k0:
                        fix.add(key)
    return int((p < 1e-12).sum()), len(members), len(aut), len(fix), len(reached)


def test_closure_groupoid_of_every_delone_sort():
    import numpy as np
    rnd = np.random.default_rng(24)
    assert len(SORTS) == 24
    for name, bravais, pattern in SORTS:
        got = sort_groupoid(pattern, rnd)
        assert got == EXPECTED[name], (name, got)
        assert got[2] == HOLOHEDRY[bravais], name            # loops = point group
        assert got[2] % got[3] == 0                         # Aut = orbit x stabiliser


def test_reindexing_coset_needs_the_theorem_cache():
    import numpy as np
    from agentsg.cell.canonical import _closure_for_match, canonical_superbase
    from agentsg.cell.metric import params_from_metric
    base = np.array(LATTICES["oP three walls"], float)
    tau = 0.1 * _lam1(base) / 32
    pR = base.copy()
    pR[3], pR[4], pR[5] = 1.25 * tau, 0.5 * tau, 0.0
    GR = _gram(pR)

    def band(frames):
        return [S for S in frames.values() if _cn(S, GR).min() >= -tau - 1e-9]

    cache_thm = band(_generate(pR, 4 * tau, tau)[1])
    cache_tau = band(_generate(pR, tau, tau)[1])
    known = {_key(S) for S in cache_tau}
    hidden = [S for S in cache_thm if _key(S) not in known]
    assert len(cache_thm) == 7 and len(cache_tau) == 6 and len(hidden) == 1
    H = hidden[0]
    q = _cn(H, GR)
    Binv = np.linalg.inv(H[1:4].T)
    GN = Binv.T @ _gram(np.where(q <= 0, 0.25, q)) @ Binv   # H is N's only reduced frame
    assert len(_closure_for_match(tuple(params_from_metric(GN)), boundary_rel=0.0)) == 1
    SN = np.array(canonical_superbase(tuple(params_from_metric(GN)))[0], float)

    def operators(cache):
        BinvT = np.linalg.inv(SN[1:4].T)
        out = {}
        for S in cache:
            for pm in itertools.permutations(range(4)):
                for sg in (1.0, -1.0):
                    P = np.round((sg * S[list(pm)])[1:4].T @ BinvT)
                    Pi = np.linalg.inv(P)
                    out[tuple(P.astype(int).ravel())] = float(np.linalg.norm(Pi.T @ GN @ Pi - GR))
        return out

    thm, snap = operators(cache_thm), operators(cache_tau)
    near_thm = sorted(round(r, 2) for r in thm.values() if r < 40)
    near_tau = sorted(round(r, 2) for r in snap.values() if r < 40)
    assert near_thm == [7.83, 7.83, 8.01, 8.01, 8.84, 8.84]
    assert near_tau == [7.83, 7.83, 8.01, 8.01]
    identity = tuple(np.eye(3, dtype=int).ravel())
    assert identity in thm and identity not in snap           # the true operator
    assert abs(thm[identity] - float(np.linalg.norm(GN - GR))) < 1e-9


@pytest.fixture(scope="module")
def superbases():
    import numpy as np
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    return np.concatenate([-(B.sum(1))[:, None, :], B], 1)


def test_hidden_members_strip_corners_and_threshold(superbases):
    """members missed by snapping only at tau: they occur at three-wall
    (triangle) corners exactly when one wall z <= tau < y <= z + tau, never at
    the two-wall corners of mP and hP; snapping at 2 tau already finds them;
    the oblique wall carries no symmetry (Aut = +-I)."""
    import numpy as np
    from test_z44_widened_complete import _brute
    grid = np.arange(0.0, 2.51, 0.25)
    for name in ("oP three walls", "cP"):
        base = np.array(LATTICES[name], float)
        tau = 0.1 * _lam1(base) / 32
        for y in grid:
            for z in grid:
                p = base.copy()
                p[3], p[4], p[5] = y * tau, z * tau, 0.0
                truth, _ = _brute(superbases, _gram(p), tau)
                at_tau, _ = _generate(p, tau, tau)
                hidden = bool(truth - at_tau)
                lo, hi = min(y, z), max(y, z)
                assert hidden == (0 < lo <= 1 and 1 < hi <= lo + 1), (name, y, z)
                assert _generate(p, 2 * tau, tau)[0] == truth
                assert _generate(p, 4 * tau, tau)[0] == truth
    for name in ("mP two walls", "hP"):
        base = np.array(LATTICES[name], float)
        tau = 0.1 * _lam1(base) / 32
        w = np.where(base < 1e-12)[0]
        for y in grid:
            for z in grid:
                p = base.copy()
                p[w[0]], p[w[1]] = y * tau, z * tau
                truth, _ = _brute(superbases, _gram(p), tau)
                assert _generate(p, tau, tau)[0] == truth, (name, y, z)
    # a lattice on the oblique wall p12 = p13 (with p23 = 0) has only +-I
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    U = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    G = _gram(np.array([900, 1600, 2500, 3.0, 3.0, 0.0]))
    R = np.einsum("nai,ab,nbj->nij", U, G, U)
    assert int((np.abs(R - G).max((1, 2)) < 1e-9 * np.abs(G).max()).sum()) == 2


def test_partial_snaps_are_nested_in_the_full_snap():
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    for c in [(30.02, 39.97, 50.03, 90.25, 92.05, 89.80), (30.01, 29.98, 30.02, 90.12, 89.93, 90.08)]:
        G = np.array(UnitCell(*c).metric_tensor())
        p0 = _cn(np.array(canonical_superbase(c)[0], float), G)
        small = [n for n in range(6) if p0[n] <= 4 * 43.2]

        def closure(Z):
            q = p0.copy()
            q[list(Z)] = 0
            return set(_generate(q, 0.0, 1e18)[1])
        full = closure(small)
        union = set()
        for r_ in range(len(small) + 1):
            for Z in itertools.combinations(small, r_):
                C = closure(Z)
                assert C <= full
                union |= C
        assert union == full and len(full) == 16
