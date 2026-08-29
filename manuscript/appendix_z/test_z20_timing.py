"""z:timing — KD-tree build/query on PDB keys; planted neighbour completeness."""
from __future__ import annotations

import platform
import random
import time

import pytest

from helpers import SEED_TIMING

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_pdb]


def test_kdtree_timing_and_planted_neighbours(pdb_roots_path):
    np = pytest.importorskip("numpy")
    scipy = pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    data = np.load(pdb_roots_path)
    keys = np.asarray(data["X"], dtype=np.float64)
    assert keys.shape[1] == 6
    n = keys.shape[0]
    assert n >= 200_000  # manuscript 206214

    t0 = time.perf_counter()
    tree = cKDTree(keys)
    build_s = time.perf_counter() - t0

    rng = random.Random(SEED_TIMING)
    idxs = [rng.randrange(n) for _ in range(1000)]
    times = []
    for i in idxs:
        t0 = time.perf_counter()
        tree.query(keys[i], k=10)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times) // 2]

    # radius queries
    for r in (0.5, 1.0, 2.0):
        nt = []
        nret = []
        for i in idxs[:200]:
            t0 = time.perf_counter()
            hits = tree.query_ball_point(keys[i], r=r)
            nt.append((time.perf_counter() - t0) * 1000)
            nret.append(len(hits))
        print(f"  radius {r}: p50={sorted(nt)[len(nt)//2]:.4f}ms "
              f"median_n={sorted(nret)[len(nret)//2]}")

    # planted neighbours on a 3000-cell subset
    m = 3000
    sub = keys[:m].copy()
    # plant: query = key[0] + small noise; must be recovered by radius
    q = sub[0] + rng.gauss(0, 0.01)
    # ensure within ball
    tree_sub = cKDTree(sub)
    hits = tree_sub.query_ball_point(q, r=0.5)
    assert 0 in hits, "planted neighbour missed by radius query (Lemma B.1)"

    print(
        f"z:timing n={n} build_s={build_s:.3f} k10_p50_ms={p50:.4f} "
        f"python={platform.python_version()} "
        f"numpy={np.__version__} scipy={scipy.__version__}"
    )
