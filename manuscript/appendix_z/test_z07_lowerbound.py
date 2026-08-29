"""z:lowerbound — rearrangement inequality / sorted-key lower bound."""
from __future__ import annotations

import itertools
import random

import pytest

from helpers import (
    SEED_LOWERBOUND, s4_slot_permutations, apply_slot_perm, key_l2,
)

pytestmark = [pytest.mark.zcheck]


def test_rearrangement_lower_bound_random():
    """1000 pairs (fast); identity is exact, not ±10%."""
    rng = random.Random(SEED_LOWERBOUND)
    G4 = s4_slot_permutations()
    assert len(G4) == 24

    for _ in range(1000):
        x = tuple(rng.random() * 5 for _ in range(6))
        y = tuple(rng.random() * 5 for _ in range(6))
        sorted_d = key_l2(sorted(x), sorted(y))
        m6 = min(key_l2(x, yp) for yp in itertools.permutations(y))
        m4 = min(key_l2(x, apply_slot_perm(y, g)) for g in G4)
        assert abs(sorted_d - m6) < 1e-12
        assert m6 <= m4 + 1e-12


@pytest.mark.slow
def test_rearrangement_lower_bound_10k():
    """Full manuscript 10^4-pair check."""
    rng = random.Random(SEED_LOWERBOUND)
    G4 = s4_slot_permutations()
    for _ in range(10_000):
        x = tuple(rng.random() * 5 for _ in range(6))
        y = tuple(rng.random() * 5 for _ in range(6))
        sorted_d = key_l2(sorted(x), sorted(y))
        m6 = min(key_l2(x, yp) for yp in itertools.permutations(y))
        assert abs(sorted_d - m6) < 1e-12
        m4 = min(key_l2(x, apply_slot_perm(y, g)) for g in G4)
        assert m6 <= m4 + 1e-12
