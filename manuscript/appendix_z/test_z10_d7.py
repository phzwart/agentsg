"""z:d7 — sorted √D7 separates generic V1 fibre; Kurlin Ex. 6.5."""
from __future__ import annotations

import itertools
import math
import random

import pytest

from helpers import (
    SEED_D7, _PAIRS, s4_slot_permutations, apply_slot_perm,
    conorm_dict_from_six, gram_from_conorms,
)
from agentsg.cell.rootform import vonorms_from_conorms

pytestmark = [pytest.mark.zcheck]


def _orbit_representatives(six_values):
    G4 = s4_slot_permutations()
    seen = set()
    reps = []
    for perm in itertools.permutations(six_values):
        if perm in seen:
            continue
        reps.append(perm)
        for g in G4:
            seen.add(apply_slot_perm(perm, g))
    assert len(reps) == 30
    return reps


def _key13_from_roots(root_assignment):
    p = {pair: root_assignment[k] ** 2 for k, pair in enumerate(_PAIRS)}
    r6 = tuple(sorted(root_assignment))
    v7 = tuple(sorted(math.sqrt(max(v, 0.0)) for v in vonorms_from_conorms(p)))
    return r6 + v7


def test_v1_fibre_sorted_d7_separates_300():
    rng = random.Random(SEED_D7)
    for _ in range(300):
        roots = []
        while len(roots) < 6:
            x = rng.uniform(0.5, 8.0)
            if all(abs(x - y) > 1e-3 for y in roots):
                roots.append(x)
        roots = tuple(roots)
        keys = [_key13_from_roots(a) for a in _orbit_representatives(roots)]
        assert len(set(keys)) == 30


def test_kurlin_example_6_5():
    """Coforms share vonorm multiset but not conorm multiset."""
    a = (5.0, 3.0, 4.0, 1.0, 1.0, 4.0)
    b = (6.0, 3.0, 3.0, 2.0, 1.0, 3.0)
    pa = conorm_dict_from_six(a)
    pb = conorm_dict_from_six(b)
    va = sorted(vonorms_from_conorms(pa))
    vb = sorted(vonorms_from_conorms(pb))
    assert all(abs(x - y) < 1e-9 for x, y in zip(va, vb))
    assert sorted(a) != sorted(b)
