"""z:closure-count — closure sizes / class counts by Voronoi type."""
from __future__ import annotations

import itertools

import pytest

from helpers import (
    P63, TYPE_CONORMS, CLOSURE_TABLE, cell_from_conorms, metric_tensor,
)

pytestmark = [pytest.mark.zcheck]


def _dotG(u, v, G):
    return sum(u[a] * G[a][b] * v[b] for a in range(3) for b in range(3))


def _ukey_mod_pm(C):
    pos = frozenset(tuple(int(x) for x in v) for v in C)
    neg = frozenset(tuple(-int(x) for x in v) for v in C)
    return frozenset([pos, neg])


def brute_obtuse_superbases(G, R):
    vals = range(-R, R + 1)
    found = {}
    for c1 in itertools.product(vals, repeat=3):
        for c2 in itertools.product(vals, repeat=3):
            for c3 in itertools.product(vals, repeat=3):
                det = (c1[0] * (c2[1] * c3[2] - c2[2] * c3[1])
                       - c1[1] * (c2[0] * c3[2] - c2[2] * c3[0])
                       + c1[2] * (c2[0] * c3[1] - c2[1] * c3[0]))
                if abs(det) != 1:
                    continue
                c0 = tuple(-c1[t] - c2[t] - c3[t] for t in range(3))
                if any(abs(c0[t]) > R for t in range(3)):
                    continue
                C = [list(c0), list(c1), list(c2), list(c3)]
                if any(_dotG(C[i], C[j], G) > 1e-8
                       for i in range(4) for j in range(i + 1, 4)):
                    continue
                k = frozenset(tuple(v) for v in C)
                found[k] = C
    return list(found.values())


@pytest.mark.parametrize("vtype", [1, 2, 3, 4, 5])
def test_closure_table_and_implementation(vtype):
    from agentsg.cell.selling_closure import (
        selling_superbase_closure, closure_class_count, voronoi_type,
    )

    cell = cell_from_conorms(TYPE_CONORMS[vtype]())
    G = metric_tensor(cell)
    n_exp, n_cls, _ = CLOSURE_TABLE[vtype]

    assert voronoi_type(cell) == vtype
    impl = selling_superbase_closure(cell)
    if vtype == 1:
        assert len(impl) in (1, 2)
        assert closure_class_count(cell) == 1
    else:
        assert len(impl) == n_exp
        assert closure_class_count(cell) == n_cls

    # Brute R=2 matches implementation mod ±I (R=3 agreement is a slow check)
    b2 = brute_obtuse_superbases(G, 2)
    k2 = {_ukey_mod_pm(C) for C in b2}
    k_impl = {_ukey_mod_pm(C) for C in impl}
    assert k_impl == k2, f"V{vtype}: impl {k_impl} vs bruteR2 {k2}"


@pytest.mark.slow
def test_brute_R2_R3_agree_v1():
    """Manuscript: R=2 and R=3 brute sets must agree (checked on V1)."""
    cell = cell_from_conorms(TYPE_CONORMS[1]())
    G = metric_tensor(cell)
    k2 = {_ukey_mod_pm(C) for C in brute_obtuse_superbases(G, 2)}
    k3 = {_ukey_mod_pm(C) for C in brute_obtuse_superbases(G, 3)}
    assert k2 == k3


def test_hexagonal_one_class_twelve_superbases():
    from agentsg.cell.selling_closure import (
        selling_superbase_closure, closure_class_count, voronoi_type,
    )
    assert voronoi_type(P63) == 4
    assert len(selling_superbase_closure(P63)) == 12
    assert closure_class_count(P63) == 1
