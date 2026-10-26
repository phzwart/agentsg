"""z:band-beyond — the proved band lambda_1 > 32 tau (about 1.8 degrees of
angular error for an isotropic cell) against the bands practitioners use.

On lattices of five zero patterns with their walls placed at random up to
4 tau and the large conorms perturbed, idealise-enumerate-filter is compared
with brute force at bands lambda_1/32 (1.8 deg) down to lambda_1/6 (9.5 deg).
It is exact in every case in which the idealised lattice exists; the only
failures of the construction at wide bands are singular idealised lattices,
which step 2 reports. Pseudo-symmetry at Le Page's 3 degrees is not a band
question at all: at the proved band 4 tau = lambda_1/8, a wall 7 degrees
from a right angle is already idealised and its symmetries priced.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402
from .test_z44_widened_complete import superbases  # noqa: E402, F401


@claim("z55", "SI §S4", kind="sampled")
def test_exact_far_beyond_the_proved_band(superbases):
    r"""The construction is exact at every band from lambda1/32 to lambda1/6 wherever the idealised lattice exists."""
    rnd = np.random.default_rng(9)
    out = {}
    for div in (32, 24, 19, 16, 12, 10, 8, 6):
        ok = n = flagged = 0
        for name in ("oP three walls", "cP", "mP two walls", "hP", "plate"):
            base = np.array(t.LATTICES[name], float)
            lam = t._lam1(base)
            tau = lam / div
            w = np.where(base < 1e-12)[0]
            for _ in range(12):
                p = base.copy()
                p[w] = rnd.uniform(0, 4, len(w)) * tau
                big = p > 1e-9 * lam
                p[big] *= 1 + 0.02 * rnd.normal(size=big.sum())
                if np.linalg.eigvalsh(t._gram(p)).min() <= 0:
                    continue
                truth, _ = t._brute(superbases, t._gram(p), tau)
                ps = np.where(p <= 4 * tau, 0.0, p)
                if np.linalg.eigvalsh(t._gram(ps)).min() <= 1e-9 * p.max():
                    flagged += 1
                    continue
                ok += t._generate(p, 4 * tau, tau)[0] == truth
                n += 1
        out[div] = (ok, n, flagged)
    assert all(ok == n for ok, n, _ in out.values()), out
    assert out[32] == (60, 60, 0) and out[19] == (60, 60, 0) and out[10] == (60, 60, 0)
    assert out[8] == (54, 54, 6) and out[6] == (48, 48, 12)
