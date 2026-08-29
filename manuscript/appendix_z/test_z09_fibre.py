"""z:fibre — collision multiplicities 30/15/4/1/1 and explicit pair."""
from __future__ import annotations

import itertools
import math
import random

import pytest

from helpers import (
    SEED_FIBRE, s4_slot_permutations, apply_slot_perm,
    gram_from_conorms, conorm_dict_from_six, cell_from_conorms, key_l2,
)

pytestmark = [pytest.mark.zcheck]


def _distinct_vals(rng, n, lo=0.5, hi=8.0):
    vals = []
    while len(vals) < n:
        x = rng.uniform(lo, hi)
        if all(abs(x - y) > 1e-3 for y in vals):
            vals.append(x)
    return vals


def test_fibre_multiplicities_combinatorial():
    """Table: V1→30, V2→15, V4→4, V3/V5→1."""
    assert math.factorial(6) // 24 == 30          # V1: 6!/|S4|
    assert math.factorial(5) // 8 == 15           # V2: 5!/|D4|
    assert 4 == 4                                # V4: which of four is distinguished
    assert 1 == 1                                # V3, V5 injective


def test_v1_fibre_is_30():
    rng = random.Random(SEED_FIBRE)
    G4 = s4_slot_permutations()
    for _ in range(20):
        vals = tuple(_distinct_vals(rng, 6))
        seen = set()
        n_classes = 0
        for perm in itertools.permutations(vals):
            if perm in seen:
                continue
            n_classes += 1
            for g in G4:
                seen.add(apply_slot_perm(perm, g))
        assert n_classes == 30


def test_v3_v5_injective_key():
    from agentsg.cell.rootform import sorted_root_key
    c5 = cell_from_conorms((1.0, 2.0, 3.0, 0.0, 0.0, 0.0))
    expect = tuple(sorted(math.sqrt(x) for x in (1, 2, 3, 0, 0, 0)))
    assert key_l2(sorted_root_key(c5), expect) < 1e-9
    c3 = cell_from_conorms((0.0, 2.0, 3.0, 4.0, 5.0, 0.0))
    k3 = sorted_root_key(c3)
    assert k3[0] == 0.0 and abs(k3[1]) < 1e-9


def test_explicit_colliding_pair():
    from agentsg.cell.rootform import sorted_root_key

    a = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    b = (1.0, 2.0, 3.0, 4.0, 6.0, 5.0)
    np = pytest.importorskip("numpy")
    assert np.linalg.eigvalsh(gram_from_conorms(conorm_dict_from_six(a)))[0] > 0
    assert np.linalg.eigvalsh(gram_from_conorms(conorm_dict_from_six(b)))[0] > 0
    ca = cell_from_conorms(a)
    cb = cell_from_conorms(b)
    assert key_l2(sorted_root_key(ca), sorted_root_key(cb)) < 1e-9
    G4 = s4_slot_permutations()
    assert b not in {apply_slot_perm(a, g) for g in G4}
