"""z:single-frame — a new cell needs no closure of its own.

Against a reference whose widened closure is cached, one Selling-reduced
frame of the new cell suffices to find the setting: by the overlap
proposition, that frame (a member of the new cell's exact closure) has a
counterpart in the reference's widened closure at the same integer
coordinates. The coset is then copied from the reference: the accepted
alignments are Aut(R) P0, so the reference's cached automorphisms (exact
and pseudo, with their prices) give the whole coset without a closure for
the new cell. The cache is deduplicated by labelling at no loss.
"""
from __future__ import annotations

import itertools

import pytest

pytestmark = [pytest.mark.zcheck]

REFS = {
    "worked mP, two walls": (30.0, 40.0, 50.0, 90.0, 92.0, 90.0),
    "oP, three walls": (30.0, 40.0, 50.0, 90.0, 90.0, 90.0),
    "P6_3 needle": (41.8, 41.8, 233.0, 90.0, 90.0, 120.0),
    "cP": (30.0, 30.0, 30.0, 90.0, 90.0, 90.0),
    "aP generic": (30.0, 40.0, 50.0, 80.0, 85.0, 95.0),
}
PERMS = list(itertools.permutations(range(4)))


def _G(cell):
    import numpy as np
    from agentsg.cell.metric import UnitCell
    return np.array(UnitCell(*cell).metric_tensor())


def _vecset(S):
    S = [tuple(int(round(x)) for x in v) for v in S]
    Sn = [tuple(-x for x in v) for v in S]
    return min(tuple(sorted(S)), tuple(sorted(Sn)))


def _cache(cell, tau):
    """reference cache: widened closure, one frame per labelling class."""
    import numpy as np
    from agentsg.cell.canonical import superbase_variants
    G = _G(cell)
    W = superbase_variants(cell, boundary_rel=tau / np.max(np.abs(G)),
                           max_variants=512)
    reps = {}
    for S in W:
        reps.setdefault(_vecset(S), np.array(S, float))
    return list(reps.values()), len(W)


def _single_frame(GN):
    import numpy as np
    from agentsg.cell.canonical import _closure_for_match
    from agentsg.cell.metric import params_from_metric
    cell = tuple(params_from_metric(GN))
    return np.array(_closure_for_match(cell, boundary_rel=0.0)[0], float)


def _best(cache, SN, GR, GN):
    """match one reduced frame of N against the cached reference; operators
    satisfy G_N = P^T G_R P."""
    import numpy as np
    BinvT = np.linalg.inv(SN[1:4].T)
    best = (np.inf, None)
    for S in cache:
        for pm in PERMS:
            for s in (1.0, -1.0):
                P = (s * S[list(pm)])[1:4].T @ BinvT
                Pr = np.round(P)
                if np.allclose(P, Pr, atol=1e-6):
                    r = float(np.linalg.norm(Pr.T @ GR @ Pr - GN))
                    if r < best[0]:
                        best = (r, Pr)
    return best


def _noisy_reindexed(GR, eta, rnd, entries=2):
    import numpy as np
    E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2; E *= eta / np.linalg.norm(E)
    while True:
        M = rnd.integers(-entries, entries + 1, size=(3, 3)).astype(float)
        if abs(round(np.linalg.det(M))) == 1:
            break
    return M.T @ (GR + E) @ M, M, E


def test_cache_dedup_by_labelling():
    import numpy as np
    for name, cell in REFS.items():
        tau = 3e-3 * np.linalg.norm(_G(cell))
        reps, labelled = _cache(cell, tau)
        assert labelled % len(reps) == 0 or labelled >= len(reps)
        if name in ("oP, three walls", "cP"):
            assert labelled == 384 and len(reps) == 16


def test_one_reduced_frame_finds_the_setting():
    """for every reference and every noisy, reindexed new cell, the best
    candidate from the single reduced frame is at least as good as the true
    operator: no closure is enumerated for the new cell."""
    import numpy as np
    rnd = np.random.default_rng(7)
    for name, cell in REFS.items():
        GR = _G(cell)
        eta = 1e-3 * np.linalg.norm(GR)
        cache, _ = _cache(cell, 3 * eta)
        for _ in range(12):
            GN, M, E = _noisy_reindexed(GR, eta, rnd)
            true_res = float(np.linalg.norm(M.T @ E @ M))
            r, P0 = _best(cache, _single_frame(GN), GR, GN)
            assert r <= true_res + 1e-6, name


def test_coset_copied_from_reference_equals_full_matching():
    """pseudo-symmetric reference (beta = 90.2, pseudo-mmm): the coset
    Aut_band(R) P0, copied from the reference's cached self-match, equals
    the set returned by full two-sided matching, every time."""
    import numpy as np
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.metric import params_from_metric
    R = (50.0, 70.0, 60.0, 90.0, 90.2, 90.0)
    GR = _G(R)
    H = [np.array(P, float) for P in reindexing_via_canonical(
        R, R, boundary_rel=0.05, verify_rel=2e-2)]
    res = sorted(float(np.linalg.norm(h.T @ GR @ h - GR)) for h in H)
    assert len(H) == 8 and res[3] < 1e-6 and 25 < res[4] < 35   # 2/m + pseudo-mmm
    cache, _ = _cache(R, 0.02 * np.max(np.abs(GR)))
    rnd = np.random.default_rng(3)

    def key(P):
        return tuple(np.round(P).astype(int).ravel())

    for _ in range(12):
        GN, M, _ = _noisy_reindexed(GR, 0.5, rnd, entries=1)
        _, P0 = _best(cache, _single_frame(GN), GR, GN)
        copied = {key(h @ P0) for h in H}
        full = {key(np.array(P, float)) for P in reindexing_via_canonical(
            R, tuple(params_from_metric(GN)), boundary_rel=0.05, verify_rel=2e-2)}
        assert copied == full
