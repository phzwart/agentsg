"""z:effrank — local entropy effective rank of key neighbourhoods."""
from __future__ import annotations

import math
import random

import pytest

from helpers import SEED_EMBED, assert_within_pct

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_pdb]


def test_local_effective_rank(pdb_roots_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    data = np.load(pdb_roots_path)
    X = np.asarray(data["X"], dtype=np.float64)
    vol = np.asarray(data["volume"], dtype=np.float64)
    scale = np.cbrt(np.maximum(vol, 1e-12))
    Xs = X / scale[:, None]

    rng = random.Random(SEED_EMBED)
    idx = np.array(rng.sample(range(len(Xs)), min(30000, len(Xs))))
    Xs_s = Xs[idx]
    tree = cKDTree(Xs_s)
    k = 40
    M = 500
    ranks = []
    centres = rng.sample(range(len(Xs_s)), M)
    for i in centres:
        _, nn = tree.query(Xs_s[i], k=k)
        block = Xs_s[nn]
        block = block - block.mean(axis=0)
        s = np.linalg.svd(block, compute_uv=False)
        s = np.maximum(s, 0.0)
        if s.sum() <= 0:
            continue
        p = s / s.sum()
        p = p[p > 0]
        r_eff = float(math.exp(-np.sum(p * np.log(p))))
        ranks.append(r_eff)

    assert ranks
    rmin, rmax, rmed = min(ranks), max(ranks), float(np.median(ranks))
    print(f"z:effrank k={k} M={len(ranks)} min={rmin:.2f} max={rmax:.2f} "
          f"median={rmed:.2f}")
    assert 1.0 <= rmin
    assert_within_pct(rmed, 4.12, pct=10, label="effrank median")
    assert_within_pct(rmax, 5.1, pct=15, label="effrank max")
