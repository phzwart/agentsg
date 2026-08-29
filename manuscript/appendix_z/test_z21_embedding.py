"""z:embedding — crystal-system homogeneity of key-space neighbourhoods."""
from __future__ import annotations

import random

import pytest

from helpers import SEED_EMBED, assert_within_pct

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_pdb]


def test_embedding_homogeneity(pdb_roots_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree

    data = np.load(pdb_roots_path)
    X = np.asarray(data["X"], dtype=np.float64)
    vol = np.asarray(data["volume"], dtype=np.float64)
    sg = np.asarray(data["sg_number"], dtype=np.int32)

    # shape key
    scale = np.cbrt(np.maximum(vol, 1e-12))
    Xs = X / scale[:, None]

    # crystal system from sg number (rough IT buckets)
    def system(n):
        if n <= 2:
            return "triclinic"
        if n <= 15:
            return "monoclinic"
        if n <= 74:
            return "orthorhombic"
        if n <= 142:
            return "tetragonal"
        if n <= 167:
            return "trigonal"
        if n <= 194:
            return "hexagonal"
        return "cubic"

    labels = np.array([system(int(s)) for s in sg])
    rng = random.Random(SEED_EMBED)
    # subsample for speed
    idx = np.array(rng.sample(range(len(Xs)), min(20000, len(Xs))))
    Xs_s = Xs[idx]
    lab_s = labels[idx]
    tree = cKDTree(Xs_s)
    k = 15
    fracs = []
    for i in range(0, len(Xs_s), 20):
        _, nn = tree.query(Xs_s[i], k=k + 1)
        nn = nn[1:]  # drop self
        same = sum(1 for j in nn if lab_s[j] == lab_s[i])
        fracs.append(same / k)

    med = float(np.median(fracs))
    # shuffled null
    lab_shuff = lab_s.copy()
    rng.shuffle(lab_shuff)
    null = []
    for i in range(0, len(Xs_s), 20):
        _, nn = tree.query(Xs_s[i], k=k + 1)
        nn = nn[1:]
        same = sum(1 for j in nn if lab_shuff[j] == lab_s[i])
        null.append(same / k)
    null_med = float(np.median(null))
    assert med > null_med + 0.05, (
        f"homogeneity {med:.3f} not above null {null_med:.3f}"
    )
    print(f"z:embedding homogeneity median={med:.3f} null={null_med:.3f}")
