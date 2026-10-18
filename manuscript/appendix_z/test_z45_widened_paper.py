"""z:widened-paper — checks behind the standalone widened-closure paper.

(i) Reduction with the boundary moved to -tau stops at a member, but which
member depends on the start and the flip order; restarts do not reliably
cover the set. (ii) Walks on the measured metric are incomplete for any
band; the snap-walk-filter construction is exact. (iii) One walk on the
snapped lattice gives every frame's depth, so the widened closure at every
band is a re-filter (worked monoclinic pair).
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zcheck]

from test_z44_widened_complete import (  # noqa: E402
    LATTICES, S0, SL, _brute, _cn, _flip, _generate, _gram, _key, _lam1, superbases,  # noqa: F401
)

DAUGHTER = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)


def _reduce_tau(S, G, tau, most_negative):
    for _ in range(10000):
        p = _cn(S, G)
        bad = [n for n in range(6) if p[n] < -tau]
        if not bad:
            return S
        n = min(bad, key=lambda m: p[m]) if most_negative else bad[0]
        S = _flip(S, *SL[n])
    raise RuntimeError("no termination")


def test_tau_reduction_endpoints_are_members_but_not_unique(superbases):
    import numpy as np
    from agentsg.cell.metric import UnitCell
    rnd = np.random.default_rng(8)
    cells = {"worked daughter": DAUGHTER,
             "noisy oP": (30.01, 39.98, 50.02, 90.15, 89.9, 90.1),
             "noisy cP": (30.01, 29.98, 30.02, 90.12, 89.93, 90.08)}
    seen_incomplete = False
    for name, c in cells.items():
        G = np.array(UnitCell(*c).metric_tensor())
        for tau in (10.0, 43.2):
            truth, _ = _brute(superbases, G, tau)
            ends = set()
            for trial in range(400):
                while True:
                    U = rnd.integers(-3, 4, (3, 3)).astype(float)
                    if abs(round(np.linalg.det(U))) == 1:
                        break
                S = np.concatenate([-(U.sum(1))[None, :], U.T], 0)
                ends.add(_key(_reduce_tau(S, G, tau, trial % 2 == 1)))
            assert ends <= truth, name          # every stopping frame is a member
            assert len(ends) > 1, name          # but the stopping frame is not unique
            seen_incomplete |= ends != truth
    assert seen_incomplete                      # restarts are not an enumeration


def _walk(p0, tau, band, stay_inside):
    import numpy as np
    G = _gram(p0)
    s0 = np.array(S0, float)
    seen, front = {_key(s0): s0}, [s0]
    while front:
        S = front.pop()
        p = _cn(S, G)
        for n, (i, j) in enumerate(SL):
            if abs(p[n]) <= band:
                U = _flip(S, i, j)
                k = _key(U)
                if k in seen or (stay_inside and _cn(U, G).min() < -tau - 1e-9):
                    continue
                seen[k] = U
                front.append(U)
    return {k for k, S in seen.items() if _cn(S, G).min() >= -tau - 1e-9}


def test_walks_on_the_measured_metric_are_incomplete(superbases):
    import numpy as np
    rnd = np.random.default_rng(5)
    cases = []
    for name in ("oP three walls", "cP", "plate", "mP two walls", "hP"):
        base = np.array(LATTICES[name], float)
        lam = _lam1(base)
        for frac in (0.1, 0.5):
            tau = frac * lam / 32
            w = np.where(base < 1e-12)[0]
            for vals in ([1.2, 0.5], [1.6, 0.9], [0.3, 0.8], [2.5, 1.8]):
                p = base.copy()
                p[w[:2]] = np.array(vals) * tau
                if len(w) > 2:
                    p[w[2]] = rnd.uniform(0, 3) * tau
                cases.append((p, tau))
            for _ in range(4):
                p = base.copy()
                p[w] = rnd.uniform(0, 4, len(w)) * tau
                cases.append((p, tau))
    wrong = {"tau inside": 0, "tau filter": 0, "2tau inside": 0, "2tau filter": 0, "theorem": 0}
    for p, tau in cases:
        truth, _ = _brute(superbases, _gram(p), tau)
        wrong["tau inside"] += _walk(p, tau, tau, True) != truth
        wrong["tau filter"] += _walk(p, tau, tau, False) != truth
        wrong["2tau inside"] += _walk(p, tau, 2 * tau, True) != truth
        wrong["2tau filter"] += _walk(p, tau, 2 * tau, False) != truth
        wrong["theorem"] += _generate(p, 4 * tau, tau)[0] != truth
    assert len(cases) == 80
    assert wrong == {"tau inside": 13, "tau filter": 13, "2tau inside": 13,
                     "2tau filter": 2, "theorem": 0}


def test_depth_ladder_of_the_worked_pair(superbases):
    """snap at 4 x 43.2: an orthorhombic stratum, 16 frames; their depths in
    the measured metric give the widened closure at every band by filtering."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    G = np.array(UnitCell(*DAUGHTER).metric_tensor())
    p0 = _cn(np.array(canonical_superbase(DAUGHTER)[0], float), G)
    _, frames = _generate(p0, 4 * 43.2, 43.2)
    Gf = _gram(p0)
    depth = sorted(max(0.0, -_cn(S, Gf).min()) for S in frames.values())
    assert len(depth) == 16
    assert np.allclose(depth, [0, 4.19, 4.54, 4.54, 8.73, 8.73, 45.0, 49.54, 49.54,
                               53.73, 53.73, 53.73, 53.73, 57.91, 62.45, 62.45], atol=0.02)
    for tau, n in ((43.2, 6), (44.9, 6), (45.1, 7), (50.0, 9), (60.0, 14)):
        truth, _ = _brute(superbases, Gf, tau)
        assert sum(d <= tau for d in depth) == len(truth) == n


def test_worked_filter_table():
    """Table 'filter' of the paper: the sixteen members of the idealised
    closure of the measured worked cell, their conorms under the measured
    metric (as sorted six-tuples) and their entry bands; idealised cell edges."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    G = np.array(UnitCell(*DAUGHTER).metric_tensor())
    F0 = np.array(canonical_superbase(DAUGHTER)[0], float)
    Gf = F0[1:4] @ G @ F0[1:4].T
    p0 = _cn(np.array(S0, float), Gf)
    assert np.allclose(p0, [49.54, 1588.88, 2444.74, 4.19, 847.47, 4.54], atol=0.01)
    ps = np.where(p0 <= 4 * 43.2, 0.0, p0)
    Gs = _gram(ps)
    assert np.allclose(np.sqrt(np.diag(Gs)), [29.1, 39.9, 57.4], atol=0.05)
    # the idealised reduced cell is monoclinic-looking (beta 120.5); the orthogonal cell is (a, b, a+c)
    assert abs(np.degrees(np.arccos(Gs[0, 2] / np.sqrt(Gs[0, 0] * Gs[2, 2]))) - 120.5) < 0.1
    ac = np.array([1.0, 0.0, 1.0])
    assert abs(np.sqrt(ac @ Gs @ ac) - 49.4) < 0.05 and abs(Gs[0] @ ac) < 1e-6 and abs(Gs[1] @ ac) < 1e-6
    _, frames = _generate(p0, 4 * 43.2, 43.2)
    rows = sorted((round(-min(_cn(S, Gf)), 1), tuple(np.round(sorted(_cn(S, Gf)), 1)))
                  for S in frames.values())
    enters = [r[0] for r in rows]
    assert enters == [-4.2, 4.2, 4.5, 4.5, 8.7, 8.7, 45.0, 49.5, 49.5, 53.7, 53.7, 53.7, 53.7, 57.9, 62.5, 62.5]
    table = {
        (4.2, 847.5, 1588.9, 2444.7, 4.5, 49.5),
        (-4.2, 8.7, 53.7, 851.7, 1593.1, 2440.6),
        (-4.5, 8.7, 45.0, 852.0, 1593.4, 2449.3),
        (-45.0, -49.5, 53.7, 897.0, 1638.4, 2494.3),
        (-62.5, -57.9, 53.7, 905.4, 1664.2, 2511.7),
    }
    got = {r[1] for r in rows}
    for t_ in table:
        assert tuple(sorted(t_)) in got, t_
    assert sum(e <= 43.2 for e in enters) == 6 and sum(e <= 60 for e in enters) == 14
