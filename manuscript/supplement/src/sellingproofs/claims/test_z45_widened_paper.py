"""z45 — walks of flips on the measured metric against brute force (SI Table S3).

Walks on the measured metric are incomplete for any band; the construction of SI Section S4
(idealise, enumerate the exact closure, filter) is exact on the same cells.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import pytest

from .test_z44_widened_complete import (  # noqa: E402
    LATTICES, S0, SL, _brute, _cn, _flip, _generate, _gram, _key, _lam1, superbases,  # noqa: F401
)

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


@claim("z45", "SI Table S3", kind="sampled")
def test_walks_on_the_measured_metric_are_incomplete(superbases):
    r"""Walks of flips on the measured metric are wrong on 13, 13, 13, 2 and 0 of 80 cells (SI Table S3)."""
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

