"""z:euclid — Schoenberg: key metric Euclidean; S4-orbit metric not."""
from __future__ import annotations

import math
import random

import pytest

from helpers import SEED_EUCLID, s4_slot_permutations, apply_slot_perm, key_l2

pytestmark = [pytest.mark.zcheck]


def _schoenberg_eigmin(D2):
    """K = -1/2 J D2 J; return smallest eigenvalue."""
    np = pytest.importorskip("numpy")
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * J @ D2 @ J
    w = np.linalg.eigvalsh(K)
    return float(w[0]), float(w[-1]), int(np.sum(w > 1e-8))


def test_g4_has_no_single_transposition():
    G4 = s4_slot_permutations()
    for g in G4:
        # as a permutation of 0..5: g[i] = source index for position i
        # a single transposition swaps exactly two positions
        moved = [i for i in range(6) if g[i] != i]
        if len(moved) == 2 and g[moved[0]] == moved[1] and g[moved[1]] == moved[0]:
            # check it's only those two — already true
            pytest.fail(f"G4 contains a transposition: {g}")


def test_schoenberg_key_psd_orbit_not():
    np = pytest.importorskip("numpy")
    rng = random.Random(SEED_EUCLID)
    G4 = s4_slot_permutations()
    n = 150
    X = [tuple(0.1 + rng.random() * 5 for _ in range(6)) for _ in range(n)]

    D2_key = np.zeros((n, n))
    D2_orb = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dk = key_l2(sorted(X[i]), sorted(X[j])) ** 2
            do = min(key_l2(X[i], apply_slot_perm(X[j], g)) for g in G4) ** 2
            D2_key[i, j] = D2_key[j, i] = dk
            D2_orb[i, j] = D2_orb[j, i] = do

    emin_k, emax_k, rank_k = _schoenberg_eigmin(D2_key)
    emin_o, emax_o, _ = _schoenberg_eigmin(D2_orb)

    assert emin_k >= -1e-8, f"key eigmin={emin_k}"
    assert emin_o < -1.0, f"orbit eigmin={emin_o} (expected clearly negative)"
    print(f"z:euclid key eigmin={emin_k:.3e} orbit eigmin={emin_o:.3f} "
          f"eigmax_orb={emax_o:.1f}")
