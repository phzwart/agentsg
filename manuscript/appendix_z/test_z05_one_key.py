"""z:one-key — one sorted key per lattice on the closure."""
from __future__ import annotations

import pytest

from helpers import (
    TYPE_CONORMS, CUBOID, cell_from_conorms, even_cuboid_cell, key_l2,
)

pytestmark = [pytest.mark.zcheck]


@pytest.mark.parametrize("vtype", [1, 2, 3, 4, 5])
def test_one_sorted_key_on_closure(vtype):
    from agentsg.cell.selling_closure import selling_superbase_closure
    from agentsg.cell.canonical import _metric, _dotG
    from agentsg.cell.rootform import sorted_root_key
    import math

    cell = cell_from_conorms(TYPE_CONORMS[vtype]())
    G = _metric(cell)
    keys = set()
    for C in selling_superbase_closure(cell):
        vals = tuple(sorted(
            round(max(0.0, -_dotG(C[i], C[j], G)), 10)
            for i in range(4) for j in range(i + 1, 4)
        ))
        keys.add(vals)
    assert len(keys) == 1
    # sqrt key agrees with sorted_root_key
    k_pkg = sorted_root_key(cell)
    k_clo = tuple(math.sqrt(max(v, 0.0)) for v in next(iter(keys)))
    assert key_l2(k_pkg, k_clo) < 1e-6


def test_odd_vs_even_cuboid_same_key():
    from agentsg.cell.rootform import sorted_root_key

    even, _ = even_cuboid_cell(CUBOID)
    assert key_l2(sorted_root_key(CUBOID), sorted_root_key(even)) < 1e-6
