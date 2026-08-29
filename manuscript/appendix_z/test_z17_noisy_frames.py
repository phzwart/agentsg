"""z:noisy-frames — noisy P6_3 classified V4; full coset with angle_sigma."""
from __future__ import annotations

import random

import pytest

from helpers import SEED_NOISY_FRAMES, P63, perturb_angles

pytestmark = [pytest.mark.zcheck]


@pytest.mark.parametrize("sigma", [0.01, 0.05])
def test_noisy_p63_v4_and_full_coset(sigma):
    from agentsg.cell.selling_closure import voronoi_type
    from agentsg.cell.canonical import reindexing_via_canonical

    rng = random.Random(SEED_NOISY_FRAMES + int(sigma * 1000))
    A = perturb_angles(P63, sigma, rng)
    B = perturb_angles(P63, sigma, rng)

    assert voronoi_type(A, angle_sigma=sigma) == 4
    assert voronoi_type(B, angle_sigma=sigma) == 4
    ops = reindexing_via_canonical(
        A, B, boundary_rel=0, verify_rel=1e-3, angle_sigma=sigma,
    )
    assert len(ops) == 24
