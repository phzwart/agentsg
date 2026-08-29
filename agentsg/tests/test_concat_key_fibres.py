"""Generic V1 fibre: sorted 6-key collisions vs sorted √D7 separation.

On a generic Voronoi-type V1 lattice the sorted six-root key forgets tetrahedral
edge pairing and merges 30 lattices per fibre (6!/|S4| = 720/24). ABS D7 /
Kurlin vonorms supply a complementary seven-tuple; empirically the sorted
square-root vonorm key separates all 30 members.
"""
from __future__ import annotations

import itertools
import random
from math import sqrt

from agentsg.cell.rootform import (
    _PAIRS,
    conorms,
    sorted_root_key,
    sorted_vonorm_key,
    vonorms,
    vonorms_from_conorms,
)


def _s4_perms():
    return list(itertools.permutations(range(4)))


def _apply_s4_to_assignment(values, sigma):
    """Relabel edge assignment ``values`` (tuple over ``_PAIRS``) by vertex perm."""
    inv = {sigma[i]: i for i in range(4)}
    # values[k] sits on _PAIRS[k]; after sigma, that edge becomes (sigma[i], sigma[j])
    # equivalently: new value on pair (a,b) was formerly on (inv[a], inv[b]).
    out = []
    for a, b in _PAIRS:
        ia, ib = inv[a], inv[b]
        if ia > ib:
            ia, ib = ib, ia
        out.append(values[_PAIRS.index((ia, ib))])
    return tuple(out)


def _orbit_representatives(six_values):
    """Return the 30 S4-orbit representatives of assignments of six values to edges."""
    assert len(six_values) == 6
    assert len(set(six_values)) == 6
    seen = set()
    reps = []
    for perm in itertools.permutations(six_values):
        if perm in seen:
            continue
        reps.append(perm)
        for sigma in _s4_perms():
            seen.add(_apply_s4_to_assignment(perm, sigma))
    assert len(reps) == 30, f"expected 30 fibre members, got {len(reps)}"
    return reps


def _sorted_sqrt_vonorms_from_roots(root_assignment):
    p = {pair: root_assignment[k] ** 2 for k, pair in enumerate(_PAIRS)}
    return tuple(sorted(sqrt(max(v, 0.0)) for v in vonorms_from_conorms(p)))


def test_vonorms_consistent_with_superbase_lengths():
    """Kurlin (2.6a): vertex vonorms match |v_i|^2 on a real obtuse superbase."""
    from agentsg.cell.rootform import delaunay_superbase, _dot

    cell = (50.0, 60.0, 70.0, 80.0, 85.0, 95.0)
    S = delaunay_superbase(cell)
    p = conorms(cell)
    v = vonorms_from_conorms(p)
    for i in range(4):
        assert abs(v[i] - _dot(S[i], S[i])) < 1e-8
    # three opposite-edge pair vonorms
    opp = ((0, 1), (0, 2), (0, 3))
    for t, (i, j) in enumerate(opp):
        sij = [S[i][k] + S[j][k] for k in range(3)]
        assert abs(v[4 + t] - _dot(sij, sij)) < 1e-8
    assert len(sorted_vonorm_key(cell)) == 7
    assert len(sorted_root_key(cell) + sorted_vonorm_key(cell)) == 13


def test_v1_fibre_sorted_d7_separates_300():
    """In 300 random generic V1 fibres, all 30 members have distinct sorted √D7."""
    rng = random.Random(20260828)
    n_fibres = 300
    for _ in range(n_fibres):
        # distinct positive root products
        roots = []
        while len(roots) < 6:
            x = rng.uniform(0.5, 8.0)
            if all(abs(x - y) > 1e-3 for y in roots):
                roots.append(x)
        roots = tuple(roots)
        reps = _orbit_representatives(roots)
        sorted6 = tuple(sorted(roots))
        d7_keys = []
        for assignment in reps:
            assert tuple(sorted(assignment)) == sorted6
            d7_keys.append(_sorted_sqrt_vonorms_from_roots(assignment))
        assert len(set(d7_keys)) == 30, (
            f"sorted √D7 failed to separate a fibre; "
            f"unique={len(set(d7_keys))} roots={roots}"
        )


def test_sorted_concat_key_length():
    cell = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
    from agentsg.cell.rootform import sorted_concat_key
    assert len(sorted_concat_key(cell)) == 13
    assert sorted_concat_key(cell)[:6] == sorted_root_key(cell)
    assert sorted_concat_key(cell)[6:] == sorted_vonorm_key(cell)
    assert vonorms(cell) == vonorms_from_conorms(conorms(cell))
