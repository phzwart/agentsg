"""z:noise — angular noise linear in conorms; √ amplification on zeros."""
from __future__ import annotations

import statistics
import random

import pytest

from helpers import (
    SEED_NOISE, P63, TRICLINIC_NOISE, TABLE_C1, assert_within_pct,
    perturb_angles, key_l2,
)

pytestmark = [pytest.mark.zcheck]


def test_p63_sqrt_noise_medians():
    from agentsg.cell.rootform import sorted_root_key, conorms

    rng = random.Random(SEED_NOISE)
    r0 = sorted_root_key(P63)
    p0 = conorms(P63)

    for sigma, expected in TABLE_C1["sqrt"].items():
        shifts = []
        for _ in range(300):
            noisy = perturb_angles(P63, sigma, rng)
            shifts.append(key_l2(r0, sorted_root_key(noisy)))
        med = statistics.median(shifts)
        assert_within_pct(med, expected, pct=10, label=f"√ σ={sigma}")

    # conorm sd ~ |vi||vj| sigma_rad at a near-zero slot — spot check
    import math
    from agentsg.cell.rootform import delaunay_superbase, _dot
    S = delaunay_superbase(P63)
    lengths = [math.sqrt(max(_dot(S[i], S[i]), 0.0)) for i in range(4)]
    sigma = 0.05
    deltas = []
    for _ in range(300):
        noisy = perturb_angles(P63, sigma, rng)
        pn = conorms(noisy)
        # pick a near-zero conorm pair if any
        for ij in p0:
            if p0[ij] < 1e-6:
                deltas.append(pn[ij] - p0[ij])
                break
    if deltas:
        sd = statistics.pstdev(deltas)
        # order-of-magnitude: σ_p ≈ |vi||vj| σ_θ
        expect_scale = lengths[0] * lengths[1] * math.radians(sigma)
        assert sd < 20 * expect_scale  # loose sanity


def test_triclinic_less_amplified():
    from agentsg.cell.rootform import sorted_root_key

    rng = random.Random(SEED_NOISE + 1)
    r0 = sorted_root_key(TRICLINIC_NOISE)
    shifts = [
        key_l2(r0, sorted_root_key(perturb_angles(TRICLINIC_NOISE, 0.1, rng)))
        for _ in range(300)
    ]
    med = statistics.median(shifts)
    # manuscript ~0.3 Å at 0.1 deg
    assert_within_pct(med, 0.3, pct=50, label="triclinic √ @0.1°")  # wider: order-of-mag
    assert med < 1.0  # clearly << P63 ~4.6
