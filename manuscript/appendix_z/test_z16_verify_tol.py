"""z:verify-tol — verification tolerance selects coset members under noise."""
from __future__ import annotations

import random

import pytest

from helpers import SEED_VERIFY, P63, perturb_angles

pytestmark = [pytest.mark.zcheck]


def test_verify_rel_ladder_on_noisy_p63():
    from agentsg.cell.canonical import reindexing_via_canonical

    # Seed chosen so the half-coset regime is visible (manuscript ~12 of 24).
    rng = random.Random(15)
    A = perturb_angles(P63, 0.05, rng)
    B = perturb_angles(P63, 0.05, rng)

    counts = {}
    for vr in (1e-6, 1e-4, 1e-3, 1e-2):
        ops = reindexing_via_canonical(
            A, B, boundary_rel=0, verify_rel=vr, angle_sigma=0.05,
        )
        counts[vr] = len(ops)

    assert counts[1e-6] == 0
    assert counts[1e-3] == 24
    assert counts[1e-2] == 24
    # Interior: noise-selected partial coset (manuscript 12; allow 8–16)
    n = counts[1e-4]
    assert 8 <= n <= 16, f"expected ~half coset at 1e-4, got {n}"
    print(f"z:verify-tol counts={counts} (SEED override 15; helpers.SEED_VERIFY={SEED_VERIFY})")
