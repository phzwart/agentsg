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


def test_primitive_monoclinic_worked_example(superbases):
    """a = 30, b = 40, c = 50 A, beta = 100 deg: six members on a ring of
    flips; the symmetries 1 and -1 are loops at the reduced cell, the 2-fold
    and the mirror are arrows from it to the opposite member of the ring; no
    frame outside the exact closure lies within a band below 260.5 A^2."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    from test_z44_widened_complete import SL, _brute, _flip
    cell = (30, 40, 50, 90, 100, 90)
    G = np.array(UnitCell(*cell).metric_tensor())
    F0 = np.array(canonical_superbase(cell)[0], float)
    assert np.allclose(F0[1:4], np.eye(3))                  # reduced basis = a, b, c
    p0 = _cn(np.array(S0, float), G)
    assert np.allclose(sorted(p0), [0, 0, 260.5, 639.5, 1600.0, 2239.5], atol=0.1)
    _, mem = _generate(p0, 0.0, 1e18)
    expected = {
        "M0": [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        "M1": [(0, -1, -1), (-1, 0, 0), (0, 1, 0), (1, 0, 1)],
        "M2": [(-1, 0, -1), (1, 1, 0), (0, -1, 0), (0, 0, 1)],
        "M3": [(1, 0, 1), (0, 1, -1), (0, -1, 0), (-1, 0, 0)],
        "M4": [(1, -1, 1), (0, 0, -1), (0, 1, 0), (-1, 0, 0)],
        "M5": [(1, -1, 0), (0, 0, 1), (0, 1, 0), (-1, 0, -1)],
    }
    name = {_key(np.array(v, float)): n for n, v in expected.items()}
    assert set(mem) == set(name)
    edges = set()
    for k, S in mem.items():
        q = _cn(S, G)
        for n, (i, j) in enumerate(SL):
            if abs(q[n]) < 1e-9:
                edges.add(frozenset((name[k], name[_key(_flip(S, i, j))])))
    ring = [("M0", "M1"), ("M1", "M5"), ("M5", "M4"), ("M4", "M3"), ("M3", "M2"), ("M2", "M0")]
    assert edges == {frozenset(e) for e in ring}
    targets = {}
    for k, S in mem.items():
        for pm in itertools.permutations(range(4)):
            for sg in (1, -1):
                W = (sg * S[list(pm)])[1:4].T
                if np.allclose(W.T @ G @ W, G, atol=1e-7 * np.abs(G).max()):
                    targets[tuple(np.round(W).astype(int).ravel())] = name[k]
    assert targets == {(1, 0, 0, 0, 1, 0, 0, 0, 1): "M0", (-1, 0, 0, 0, -1, 0, 0, 0, -1): "M0",
                       (-1, 0, 0, 0, 1, 0, 0, 0, -1): "M4", (1, 0, 0, 0, -1, 0, 0, 0, 1): "M4"}
    # M4 in its listed order (-c, b, -a) has a different metric tensor (conorms
    # 639.5 and 2239.5 exchanged); in the twofold's order (-a, b, -c) the same one
    M4 = np.array(expected["M4"], float)
    assert not np.allclose(M4[1:4] @ G @ M4[1:4].T, G)
    assert np.allclose(_cn(M4, G), [2239.5, 1600.0, 639.5, 0.0, 260.5, 0.0], atol=0.1)
    twofold_order = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)
    assert np.allclose(twofold_order @ G @ twofold_order.T, G)
    for tau in (10.0, 100.0, 260.0):
        truth, _ = _brute(superbases, G, tau)
        assert truth == set(mem)                            # no near-wall frame yet
    truth, _ = _brute(superbases, G, 261.0)
    assert len(truth) > 6


def _orbits_and_shapes(p, G=None):
    """Aut-orbits of the closure members, and distinct cell shapes (conorm
    vectors up to relabelling)."""
    import numpy as np
    from test_z44_widened_complete import SL
    G = _gram(p) if G is None else G
    _, mem = _generate(p, 0.0, 1e18)
    keys = list(mem)
    idx = {k: i for i, k in enumerate(keys)}
    aut = []
    for S in mem.values():
        for pm in itertools.permutations(range(4)):
            for sg in (1, -1):
                W = (sg * S[list(pm)])[1:4].T
                if np.allclose(W.T @ G @ W, G, atol=1e-7 * np.abs(G).max()):
                    aut.append(W)
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for k, S in mem.items():
        for W in aut:
            kk = _key(np.round(S @ W.T))        # rows are coordinate vectors; W acts on columns
            assert kk in idx                   # a symmetry maps members to members
            parent[find(idx[kk])] = find(idx[k])
    orbit_of = {k: find(idx[k]) for k in keys}
    I = {e: n for n, e in enumerate(SL)}
    perms = [[I[tuple(sorted((g[a], g[b])))] for a, b in SL] for g in itertools.permutations(range(4))]
    shape_of = {k: min(tuple(np.round(_cn(S, G), 6)[q]) for q in perms) for k, S in mem.items()}
    return mem, orbit_of, shape_of, len(aut)


def test_symmetry_orbits_are_the_distinct_cell_shapes():
    """in every sort, two members are related by a symmetry iff they have the
    same conorm vector up to relabelling; the number of Aut-orbits equals the
    number of distinct reduced-cell shapes. mP: 6 members, 3 shapes,
    {M0,M4}, {M1,M3}, {M2,M5}; 288 labelled steps out of M0, 4 symmetries;
    the M1 shape is the cell 30, 40, 53.7 A, beta = 113.4 deg."""
    import numpy as np
    from agentsg.cell.metric import UnitCell
    rnd = np.random.default_rng(3)
    expected = {"C5": 2, "T5": 3, "O3": 2, "O4": 2, "O5": 4, "M2B": 2, "M3": 2,
                "M4": 3, "A2": 2, "A3": 3}
    for name, bt, pat in SORTS:
        letters = sorted(set(pat) - {"0"})
        val = dict(zip(letters, rnd.uniform(300, 2500, len(letters))))
        s = [0.0 if ch == "0" else val[ch] for ch in pat]
        p = np.array([s[3], s[4], s[5], s[2], s[1], s[0]])
        mem, orbit_of, shape_of, naut = _orbits_and_shapes(p)
        same_orbit = {(a, b) for a in mem for b in mem if orbit_of[a] == orbit_of[b]}
        same_shape = {(a, b) for a in mem for b in mem if shape_of[a] == shape_of[b]}
        assert same_orbit == same_shape, name
        assert len(set(orbit_of.values())) == expected.get(name, 1), name
    cell = (30, 40, 50, 90, 100, 90)
    G = np.array(UnitCell(*cell).metric_tensor())
    p0 = _cn(np.array(S0, float), G)
    mem, orbit_of, shape_of, naut = _orbits_and_shapes(p0, G)
    M = {n: _key(np.array(v, float)) for n, v in {
        "M0": [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        "M1": [(0, -1, -1), (-1, 0, 0), (0, 1, 0), (1, 0, 1)],
        "M2": [(-1, 0, -1), (1, 1, 0), (0, -1, 0), (0, 0, 1)],
        "M3": [(1, 0, 1), (0, 1, -1), (0, -1, 0), (-1, 0, 0)],
        "M4": [(1, -1, 1), (0, 0, -1), (0, 1, 0), (-1, 0, 0)],
        "M5": [(1, -1, 0), (0, 0, 1), (0, 1, 0), (-1, 0, -1)]}.items()}
    pairs = [("M0", "M4"), ("M1", "M3"), ("M2", "M5")]
    for a, b in pairs:
        assert orbit_of[M[a]] == orbit_of[M[b]]
    assert len(set(orbit_of.values())) == 3
    assert 48 * len(mem) == 288 and naut == 4
    S1 = mem[M["M1"]]
    Gm = S1[1:4] @ G @ S1[1:4].T
    a, b, c = np.sqrt(np.diag(Gm))
    beta = np.degrees(np.arccos(Gm[0, 2] / (a * c)))
    assert np.allclose([a, b, c, beta], [30.0, 40.0, 53.7, 113.4], atol=0.05)


def test_twofolds_and_inversion_generate_every_point_group():
    """Lebedev et al. (2006): the holohedry is generated by its twofold axes
    with the inversion. Checked on the symmetry steps of every sort; and the
    number of twofolds that relabel the reduced superbase itself (0 of 3 for
    oP, 0 of 1 for mP, 1 of 5 for tP, 3 of 9 for cP, 1 of 7 for hP)."""
    import numpy as np
    rnd = np.random.default_rng(3)
    visible = {"O5": (3, 0), "M4": (1, 0), "T5": (5, 1), "C5": (9, 3), "H4": (7, 1), "C1": (9, 9)}
    for name, bt, pat in SORTS:
        letters = sorted(set(pat) - {"0"})
        val = dict(zip(letters, rnd.uniform(300, 2500, len(letters))))
        s = [0.0 if ch == "0" else val[ch] for ch in pat]
        p = np.array([s[3], s[4], s[5], s[2], s[1], s[0]])
        G = _gram(p)
        _, mem = _generate(p, 0.0, 1e18)
        k0 = _key(np.array(S0, float))
        aut = []
        for k, S in mem.items():
            for pm in itertools.permutations(range(4)):
                for sg in (1, -1):
                    W = np.round((sg * S[list(pm)])[1:4].T).astype(int)
                    if np.allclose(W.T @ G @ W, G, atol=1e-7 * np.abs(G).max()):
                        aut.append((W, k == k0))
        two = [(W, v) for W, v in aut if round(np.linalg.det(W)) == 1 and np.trace(W) == -1]
        gens = [W for W, _ in two] + [-np.eye(3, dtype=int)]
        E = np.eye(3, dtype=int)
        seen, front = {E.tobytes()}, [E]
        while front:
            A = front.pop()
            for g in gens:
                B = A @ g
                if B.tobytes() not in seen:
                    seen.add(B.tobytes())
                    front.append(B)
        assert len(seen) == len(aut), name
        if name in visible:
            assert (len(two), sum(v for _, v in two)) == visible[name], name
