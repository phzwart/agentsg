"""z:reindex-bench — 6960 brute vs cached PREPARE_REFERENCE / REINDEX_FRAME."""
from __future__ import annotations

import statistics
import time
import random

import pytest

from helpers import (
    SEED_BENCH, prepare_reference, reindex_frame, metric_tensor,
    transform_metric, cell_from_metric, perturb_angles,
)

pytestmark = [pytest.mark.zcheck, pytest.mark.slow]


def test_reindex_bench_brute_vs_cached():
    from agentsg.cell.g6 import _unimodular_pm1
    from agentsg.cell.reindex import reindexing_operators

    rng = random.Random(SEED_BENCH)
    ref = (120.7, 189.1, 129.4, 90.0, 91.2, 90.0)
    reindex_ops = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    ]
    Gref = metric_tensor(ref)

    frames = []
    for _ in range(100):
        M = rng.choice(reindex_ops)
        cell = cell_from_metric(transform_metric(Gref, M))
        frames.append(perturb_angles(cell, 0.08, rng))
    for _ in range(100):
        a, b, c, al, be, ga = perturb_angles(ref, 0.03, rng)
        frames.append((a + rng.gauss(0, 0.02), b, c, al, be, ga))

    n_unimod = len(list(_unimodular_pm1()))
    assert n_unimod == 6960

    brute_times = []
    for frame in frames:
        t0 = time.perf_counter()
        reindexing_operators(ref, frame, length_tol_pct=2.0, angle_tol_deg=2.0)
        brute_times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    cache = prepare_reference(ref, sigma_theta=0.1, radius=5.0)
    setup_ms = (time.perf_counter() - t0) * 1000
    assert cache.closure_call_count == 1

    cached_times = []
    for frame in frames:
        t0 = time.perf_counter()
        reindex_frame(cache, frame, sigma_theta=0.1, verify_rel=1e-3, screen=True)
        cached_times.append((time.perf_counter() - t0) * 1000)

    # Reference closure must not be re-enumerated
    assert cache.closure_call_count == 1

    brute_p50 = statistics.median(brute_times)
    cached_p50 = statistics.median(cached_times)
    ratio = brute_p50 / max(cached_p50, 1e-9)
    amortised = (sum(brute_times)) / max(setup_ms + sum(cached_times), 1e-9)
    print(
        f"z:reindex-bench n_unimod={n_unimod} brute_p50={brute_p50:.3f}ms "
        f"cached_p50={cached_p50:.3f}ms ratio={ratio:.1f}x "
        f"amortised={amortised:.1f}x setup={setup_ms:.2f}ms "
        f"ref_closure_calls={cache.closure_call_count}"
    )
    # Structural asserts only — do not pin wall-clock (hardware varies).
    # The manuscript's 0.07 ms figure uses a tighter Ambiguity coset cache;
    # this check verifies PREPARE_REFERENCE enumerates the closure once.
    assert cache.closure_call_count == 1
    assert n_unimod == 6960
    assert len(frames) == 200
