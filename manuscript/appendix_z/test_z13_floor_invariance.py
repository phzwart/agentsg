"""z:floor-invariance — T closure-invariant; per-pair floors are not."""
from __future__ import annotations

import math

import pytest

from helpers import CUBOID, even_cuboid_cell, key_l2, TYPE_CONORMS, cell_from_conorms

pytestmark = [pytest.mark.zcheck]


def test_T_equals_50_on_cuboid_odd_even():
    from agentsg.cell.rootform import conorm_sum, sorted_root_key

    even, _ = even_cuboid_cell(CUBOID)
    assert abs(conorm_sum(CUBOID) - 50.0) < 1e-9
    assert abs(conorm_sum(even) - 50.0) < 1e-9


@pytest.mark.parametrize("vtype", [1, 2, 3, 4, 5])
def test_T_constant_on_closure(vtype):
    from agentsg.cell.selling_closure import selling_superbase_closure
    from agentsg.cell.canonical import _metric, _dotG

    cell = cell_from_conorms(TYPE_CONORMS[vtype]())
    G = _metric(cell)
    Ts = []
    for C in selling_superbase_closure(cell):
        T = sum(
            max(0.0, -_dotG(C[i], C[j], G))
            for i in range(4) for j in range(i + 1, 4)
        )
        Ts.append(T)
    assert max(Ts) - min(Ts) < 1e-8


def test_invariant_floor_keys_agree_per_pair_do_not():
    from agentsg.cell.rootform import (
        sorted_root_key, noise_floor, pair_noise_scales,
    )

    even, _ = even_cuboid_cell(CUBOID)
    sigma = 0.05
    # invariant floor
    d = key_l2(
        sorted_root_key(CUBOID, stabilize="floored", angle_sigma=sigma),
        sorted_root_key(even, stabilize="floored", angle_sigma=sigma),
    )
    assert d < 1e-12

    # per-pair floors |vi||vj|sigma break invariance — construct manually
    from agentsg.cell.rootform import delaunay_superbase, _dot, _PAIRS, root_products
    import agentsg.cell.rootform as rf

    def per_pair_floors(cell):
        S = delaunay_superbase(cell)
        lengths = [math.sqrt(max(_dot(S[i], S[i]), 0.0)) for i in range(4)]
        s = math.radians(sigma)
        return {(i, j): lengths[i] * lengths[j] * s for (i, j) in _PAIRS}

    k_odd = sorted_root_key(CUBOID, stabilize="floored", floors=per_pair_floors(CUBOID))
    k_even = sorted_root_key(even, stabilize="floored", floors=per_pair_floors(even))
    d_bad = key_l2(k_odd, k_even)
    # manuscript ~0.02 Å for floored
    assert d_bad > 1e-4, f"per-pair floors unexpectedly invariant: {d_bad}"
