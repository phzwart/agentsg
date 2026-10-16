"""z:s6dist — how an S6 distance relates to closure matching.

(i) One boundary transformation at each zero face, with the 24 relabellings,
reaches every relabelling orbit of the closure (V2..V5): an S6 distance that
applies them is complete in reach. (ii) A one-sided minimisation -- one cell
fixed, the other searched over its *exact* closure -- fails when the fixed
cell sits on walls the other has just left (worked pair: 1126.7 against a
true 13.4); two-sided matching, or one-sided against the *widened* closure,
recovers it. (iii) Optional, when NCDIST_DIR points at a checkout of
github.com/yayahjb/ncdist: CS6Dist is two-sided and near-face and agrees
with the closure on the worked pair; its on-face mirrors are orthogonal and
its unreduction maps are not (largest singular value phi^2 = 2.618), which
bounds by how much its path distance can exceed a chord.
"""
from __future__ import annotations

import itertools
import os

import pytest

pytestmark = [pytest.mark.zcheck]

SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PARENT = (30.0, 40.0, 50.0, 90.0, 92.0, 90.0)
DAUGHTER = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)


def _perms():
    out = []
    idx = {s: n for n, s in enumerate(SL)}
    for g in itertools.permutations(range(4)):
        out.append([idx[tuple(sorted((g[a], g[b])))] for a, b in SL])
    return out


def _gram_from_p(p):
    import numpy as np
    return np.array([[p[0] + p[3] + p[4], -p[3], -p[4]],
                     [-p[3], p[1] + p[3] + p[5], -p[5]],
                     [-p[4], -p[5], p[2] + p[4] + p[5]]])


def _cn(S, G):
    import numpy as np
    S = np.asarray(S, float)
    return np.array([-S[i] @ G @ S[j] for i, j in SL])


def _move(S, i, j):
    k, l = [m for m in range(4) if m not in (i, j)]
    U = S.copy()
    U[i], U[j], U[k], U[l] = -S[i], S[j], S[i] + S[k], S[i] + S[l]
    return U


def test_one_boundary_move_per_zero_face_reaches_every_orbit():
    import numpy as np
    P = _perms()

    def orbit(c):
        return min(tuple(round(c[q], 6) for q in pm) for pm in P)

    cases = {(0, 3, 5, 7, 11, 13): 2, (0, 3, 5, 7, 11, 0): 3,
             (0, 0, 5, 7, 11, 13): 3, (4, 6, 9, 0, 0, 0): 4}
    for p0, n_orbits in cases.items():
        G = _gram_from_p(np.array(p0, float))
        S0 = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)

        def zeros(S):
            return [(i, j) for (i, j) in SL if abs(-S[i] @ G @ S[j]) < 1e-9]

        seen = {tuple(np.round(_cn(S0, G), 6))}
        front = [S0]
        while front:
            nxt = []
            for S in front:
                for (i, j) in zeros(S):
                    U = _move(S, i, j)
                    c = _cn(U, G)
                    if c.min() < -1e-9:
                        continue
                    key = tuple(np.round(c, 6))
                    if key not in seen:
                        seen.add(key)
                        nxt.append(U)
            front = nxt
        full = {orbit(np.array(c)) for c in seen}
        one_move = {orbit(_cn(S0, G))} | {orbit(_cn(_move(S0, i, j), G))
                                          for (i, j) in zeros(S0)}
        assert len(full) == n_orbits
        assert one_move == full


def test_one_sided_exact_search_fails_two_sided_or_widened_recovers():
    import numpy as np
    from agentsg.cell.canonical import _closure_for_match, superbase_variants
    from agentsg.cell.metric import UnitCell
    P = _perms()

    def od(x, y):
        return min(np.linalg.norm(x - y[pm]) for pm in P)

    GP = np.array(UnitCell(*PARENT).metric_tensor())
    GD = np.array(UnitCell(*DAUGHTER).metric_tensor())
    A = [_cn(M, GP) for M in _closure_for_match(PARENT, boundary_rel=0.0)]
    Bx = [_cn(M, GD) for M in _closure_for_match(DAUGHTER, boundary_rel=0.0)]
    eta = float(np.linalg.norm(GD - GP))
    Bw = [_cn(M, GD) for M in superbase_variants(
        DAUGHTER, boundary_rel=3 * eta / 2447.65, max_variants=512)]
    a0 = _cn([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], GP)
    one_sided = min(od(a0, b) for b in Bx)
    two_sided = min(od(a, b) for a in A for b in Bx)
    widened = min(od(a0, b) for b in Bw)
    assert 1100 < one_sided < 1150          # the trap
    assert 13.0 < two_sided < 14.0
    assert 14.0 < widened < 15.0
    assert one_sided > 50 * two_sided


@pytest.fixture(scope="module")
def cs6():
    d = os.environ.get("NCDIST_DIR")
    if not d or not os.path.exists(os.path.join(d, "CS6Dist.c")):
        pytest.skip("NCDIST_DIR not set to a checkout of yayahjb/ncdist")
    import ctypes
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "lib.c")
    with open(src, "w") as f:
        f.write('#include "CS6Dist.c"\n'
                'double cs6_g6(double* a, double* b)'
                '{ return CS6Dist_in_G6(a, b); }\n')
    so = os.path.join(tmp, "libcs6.so")
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-I", d, src, "-lm",
                    "-o", so], check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    lib.cs6_g6.restype = ctypes.c_double
    lib.cs6_g6.argtypes = [ctypes.POINTER(ctypes.c_double)] * 2
    return d, lib


def _g6(cell):
    import ctypes
    import numpy as np
    from agentsg.cell.metric import UnitCell
    G = np.array(UnitCell(*cell).metric_tensor())
    return (ctypes.c_double * 6)(G[0, 0], G[1, 1], G[2, 2],
                                 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])


def test_cs6dist_agrees_with_closure_on_worked_pair(cs6):
    _, lib = cs6
    d = lib.cs6_g6(_g6(PARENT), _g6(DAUGHTER))
    assert 13.0 < d < 14.0                   # = two-sided closure, not 1126.7
    relabelled = (40.0, 30.0, 50.0, 92.0, 90.0, 90.0)
    assert lib.cs6_g6(_g6(PARENT), _g6(relabelled)) < 1e-9


def test_cs6dist_boundary_maps(cs6):
    import re
    import numpy as np
    d, _ = cs6
    t = open(os.path.join(d, "CS6Dist.h")).read()

    def grab(name, n):
        m = re.search(name + r"\[\d+\]\[36\]\s*=\s*\{(.*?)\};", t, re.S)
        body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
        nums = [int(x) for x in re.findall(r"-?\d+", body)]
        return [np.array(nums[36 * k:36 * (k + 1)], float).reshape(6, 6)
                for k in range(n)]

    for M in grab("S6MS", 6):                # on-face mirrors: permutations
        assert np.allclose(M @ M.T, np.eye(6))
    smax = max(np.linalg.svd(M, compute_uv=False).max()
               for M in grab("S6MSX", 12))
    assert abs(smax - (3 + 5 ** 0.5) / 2) < 1e-6   # phi^2


def _bracket_trials(lib, fams, noise, n, seed):
    """yield (CS6Dist, E, K) for noisy reindexed copies: E = straight-segment
    upper bound from the exact closures (a legal path in the reduced cone),
    K = sorted-key distance of reduced presentations."""
    import numpy as np
    from agentsg.cell.canonical import _closure_for_match
    from agentsg.cell.metric import UnitCell, params_from_metric
    P = _perms()
    rnd = np.random.default_rng(seed)

    def pts(G):
        cell = tuple(params_from_metric(G))
        A = np.array([_cn(M, G) for M in _closure_for_match(cell, boundary_rel=0.0)])
        return np.unique(np.round(A, 9), axis=0)

    def g6(G):
        import ctypes
        return (ctypes.c_double * 6)(G[0, 0], G[1, 1], G[2, 2],
                                     2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])
    for cell in fams:
        GA = np.array(UnitCell(*cell).metric_tensor())
        for _ in range(n):
            while True:
                M = rnd.integers(-1, 2, size=(3, 3)).astype(float)
                if abs(round(np.linalg.det(M))) == 1:
                    break
            E_ = rnd.normal(size=(3, 3)); E_ = (E_ + E_.T) / 2
            GB = M.T @ (GA + noise * np.linalg.norm(GA) * E_ / np.linalg.norm(E_)) @ M
            A, B = pts(GA), pts(GB)
            Bp = np.concatenate([B[:, pm] for pm in P])
            E = float(np.sqrt(((A[:, None] - Bp[None]) ** 2).sum(-1).min()))
            K = float(np.linalg.norm(np.sort(A[0]) - np.sort(B[0])))
            yield lib.cs6_g6(g6(GA), g6(GB)), E, K


FAMS = [(30, 30, 30, 90, 90, 90), (30, 40, 50, 90, 90, 90),
        (30, 30, 50, 90, 90, 90), (30, 30, 50, 90, 90, 120),
        (30, 40, 50, 90, 92, 90), (30, 40, 50, 80, 85, 95)]


def test_cs6dist_never_exceeds_the_closure_straight_path(cs6):
    """in its own metric CS6Dist never misses: it stays below the straight
    segment between exact closure points; it can fall below the reduced
    key bound because its outer pass also measures to one-step unreduced
    presentations (a partial widening)."""
    _, lib = cs6
    below_key = 0
    for C, E, K in _bracket_trials(lib, FAMS, 1e-3, 10, seed=11):
        assert C <= 1.001 * E
        below_key += C < K * (1 - 1e-9)
    assert below_key >= 1


def test_cs6dist_is_a_partial_widening(cs6):
    """against the full widened chord minimum CS6Dist is larger near
    several walls, by a bounded factor (no jumps): one gated unreduction
    per side cannot compose multi-wall moves."""
    import numpy as np
    from agentsg.cell.canonical import superbase_variants
    from agentsg.cell.metric import UnitCell, params_from_metric
    _, lib = cs6
    P = _perms()
    rnd = np.random.default_rng(5)

    def frames(G, br):
        cell = tuple(params_from_metric(G))
        F = np.array([_cn(M, G) for M in superbase_variants(
            cell, boundary_rel=br, max_variants=512)])
        return np.unique(np.round(F, 9), axis=0)

    ratios = []
    for cell in [(30, 40, 50, 90, 90, 90), (30, 30, 50, 90, 90, 90)]:
        GA = np.array(UnitCell(*cell).metric_tensor())
        for _ in range(8):
            E_ = rnd.normal(size=(3, 3)); E_ = (E_ + E_.T) / 2
            GB = GA + 1e-3 * np.linalg.norm(GA) * E_ / np.linalg.norm(E_)
            FA, FB = frames(GA, 0.02), frames(GB, 0.02)
            FBp = np.concatenate([FB[:, pm] for pm in P])
            clos = float(np.sqrt(((FA[:, None] - FBp[None]) ** 2).sum(-1).min()))
            import ctypes
            g = lambda G: (ctypes.c_double * 6)(G[0, 0], G[1, 1], G[2, 2],
                                                2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])
            ratios.append(lib.cs6_g6(g(GA), g(GB)) / clos)
    ratios = np.array(ratios)
    assert np.median(ratios) >= 1.0
    assert ratios.max() < 6.0
