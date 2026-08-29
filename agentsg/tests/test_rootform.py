"""Sorted root-product search key tests (main_v5 architecture).

The sorted six-tuple is a continuous Euclidean retrieval key: invariant under
basis change, continuous across reduction flips, lower-bounding relabelling-orbit
distance. It is *not* Kurlin Def. 5.1 and is many-to-one except on V3/V5.
Identity is certified by the Selling-superbase closure, not by key equality.
"""
import math
import random
from itertools import permutations

import pytest
from agentsg.cell.rootform import (
    delaunay_superbase, conorms, root_invariant, root_distance, _dot,
    sorted_root_key, sorted_root_distance, sorted_key_lower_bound,
    root_products, _PAIRS,
)
from agentsg.cell.metric import UnitCell
from agentsg.cell.reduction import niggli_reduce
from agentsg.cell.g6 import _transform_metric


def _valid_cell(rng):
    while True:
        c = (rng.uniform(20, 90), rng.uniform(20, 90), rng.uniform(20, 90),
             rng.uniform(65, 115), rng.uniform(65, 115), rng.uniform(65, 115))
        try:
            if UnitCell(*c).volume() > 1e3:
                return c
        except Exception:
            pass


def _cell_of(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)), ang(G[0][1] / (a * b)))


def test_obtuse_superbase_all_conorms_nonnegative():
    rng = random.Random(0)
    for _ in range(100):
        p = conorms(_valid_cell(rng))
        assert all(v > -1e-6 for v in p.values())


def test_superbase_sums_to_zero():
    rng = random.Random(1)
    S = delaunay_superbase(_valid_cell(rng))
    tot = [S[0][k] + S[1][k] + S[2][k] + S[3][k] for k in range(3)]
    assert all(abs(t) < 1e-9 for t in tot)


def test_sorted_root_key_length6_and_sorted():
    """Search key is the ascending six-tuple of root products."""
    c = (40, 50, 60, 88, 92, 103)
    ri = sorted_root_key(c)
    assert len(ri) == 6
    assert list(ri) == sorted(ri)
    assert ri == root_invariant(c)                     # back-compat alias
    assert all(x >= -1e-12 for x in ri)


def test_invariance_under_unimodular_setting():
    """Same lattice in different bases -> identical sorted key."""
    rng = random.Random(2)
    Ms = [((1, 1, 0), (0, 1, 0), (0, 0, 1)),
          ((1, 0, 0), (1, 1, 0), (0, 0, 1)),
          ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
          ((2, 1, 0), (1, 1, 0), (0, 0, 1))]
    for _ in range(30):
        c = _valid_cell(rng)
        G = UnitCell(*c).metric_tensor()
        for M in Ms:
            assert sorted_root_distance(c, _cell_of(_transform_metric(G, M))) < 1e-6


def test_continuity_across_reduction_flip_boundary():
    """No orbit search, yet continuous across a=b (where raw G6 jumps)."""
    ref = (40.0, 40.0, 60.0, 90, 91, 90)
    d_hi = sorted_root_distance(ref, (40.0, 40.001, 60.0, 90, 91, 90))
    d_lo = sorted_root_distance(ref, (40.0, 39.999, 60.0, 90, 91, 90))
    assert abs(d_hi - d_lo) < 1e-3
    assert d_hi < 0.02 and d_lo < 0.02


def test_niggli_invariance():
    rng = random.Random(4)
    for _ in range(20):
        c = _valid_cell(rng)
        r, _ = niggli_reduce(*c)
        assert sorted_root_distance(c, r) < 1e-6


def test_empirical_few_volume_collisions_among_random_cells():
    """Empirical sanity: equal keys among random cells imply equal volumes.

    Not a completeness proof — the sorted key is known many-to-one for V1/V2/V4.
    """
    rng = random.Random(5)
    cells = [_valid_cell(rng) for _ in range(300)]
    ris = [sorted_root_key(c) for c in cells]
    for i in range(len(ris)):
        for j in range(i + 1, len(ris)):
            same = all(abs(ris[i][k] - ris[j][k]) < 1e-6 for k in range(6))
            if same:
                assert abs(UnitCell(*cells[i]).volume() - UnitCell(*cells[j]).volume()) < 1e-3


def test_rearrangement_lower_bound_random_vectors():
    """||sort(x)-sort(y)|| = min_{S6} ||x-σy|| ≤ min_G ||x-σy|| for G ⊆ S6."""
    rng = random.Random(11)
    for _ in range(20):
        x = [rng.random() for _ in range(6)]
        y = [rng.random() for _ in range(6)]
        sorted_d, s6_d = sorted_key_lower_bound(x, y, G=None)
        assert abs(sorted_d - s6_d) < 1e-12
        # subset of permutations (physically allowed G)
        G = list(permutations(range(6)))[::120]  # sparse subset of S6
        sd, orbit_d = sorted_key_lower_bound(x, y, G=G)
        assert sd <= orbit_d + 1e-12


def test_rearrangement_lower_bound_on_root_products():
    """Sorted-key distance lower-bounds S4-induced permutations of products."""
    rng = random.Random(12)
    cA, cB = _valid_cell(rng), _valid_cell(rng)
    x = [root_products(cA)[ij] for ij in _PAIRS]
    y = [root_products(cB)[ij] for ij in _PAIRS]
    # All of S6: equality
    sd, s6 = sorted_key_lower_bound(x, y)
    assert abs(sd - s6) < 1e-12
    assert abs(sd - sorted_root_distance(cA, cB)) < 1e-12


def test_v5_sorted_key_injective_on_distinct_edge_lengths():
    """Two orthorhombic (V5) lattices with different edge multisets differ in key."""
    a = (50.0, 60.0, 70.0, 90, 90, 90)
    b = (50.0, 60.0, 80.0, 90, 90, 90)
    assert sorted_root_distance(a, b) > 1.0


def test_v4_pairing_collision_same_sorted_multiset():
    """V4 hexagonal: two distinct opposite-edge pairings can share a multiset.

    Kurlin keeps a distinguished singleton vs ordered triple; sorting forgets
    that. Construct two 6-tuples that arise as root-product lists with the same
    multiset but different distinguished slots — the sorted keys collide while
    the unpaired layouts differ.
    """
    # Synthetic root-product layouts (not cells): triple (1,2,3) + singleton 4
    # vs triple (1,2,4) + singleton 3 — same multiset {1,2,3,4,0,0} after padding.
    x = (0.0, 0.0, 1.0, 2.0, 3.0, 4.0)   # already a possible sorted key
    # Two different unordered lists with the same multiset:
    layout_a = [0.0, 4.0, 1.0, 2.0, 3.0, 0.0]
    layout_b = [0.0, 3.0, 1.0, 2.0, 4.0, 0.0]
    assert sorted(layout_a) == sorted(layout_b) == list(x)
    assert layout_a != layout_b
    sd, s6 = sorted_key_lower_bound(layout_a, layout_b)
    assert sd < 1e-15
    # Some permutation (swap the "singleton" slots) makes layouts match:
    assert s6 < 1e-15


def test_cubic_signature():
    """Cubic P: three equal root products = edge length, three zero."""
    ri = sorted_root_key((50, 50, 50, 90, 90, 90))
    nz = sorted(x for x in ri if x > 1e-6)
    assert len(nz) == 3
    assert all(abs(x - 50.0) < 1e-6 for x in nz)


def test_metric_axioms():
    """sorted_root_distance is a pseudometric: identity, symmetry, triangle."""
    rng = random.Random(6)
    a, b, c = (_valid_cell(rng) for _ in range(3))
    assert sorted_root_distance(a, a) < 1e-9
    assert abs(sorted_root_distance(a, b) - sorted_root_distance(b, a)) < 1e-9
    assert (sorted_root_distance(a, c)
            <= sorted_root_distance(a, b) + sorted_root_distance(b, c) + 1e-9)


def test_invariance_high_symmetry_cells():
    """Orthorhombic/tetragonal/cubic/monoclinic must be setting-invariant."""
    Ms = [((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
          ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
          ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
          ((1, 0, 1), (0, 1, 0), (0, 0, 1)),
          ((2, 1, 0), (1, 1, 0), (0, 0, 1)),
          ((0, 0, 1), (1, 0, 0), (0, 1, 0))]
    cells = {
        "orthorhombic": (50, 60, 70, 90, 90, 90),
        "tetragonal": (50, 50, 70, 90, 90, 90),
        "cubic": (50, 50, 50, 90, 90, 90),
        "hexagonal": (50, 50, 70, 90, 90, 120),
        "monoclinic": (50, 60, 70, 90, 100, 90),
    }
    for name, c in cells.items():
        G = UnitCell(*c).metric_tensor()
        for M in Ms:
            c2 = _cell_of(_transform_metric(G, M))
            assert sorted_root_distance(c, c2) < 1e-4, f"{name} not invariant under {M}"


def test_orthorhombic_axis_swap_is_isometry():
    """A pure axis relabelling (det=+1) must give distance exactly 0."""
    base = (50.0, 60.0, 70.0, 90, 90, 90)
    G = UnitCell(*base).metric_tensor()
    swapped = _cell_of(_transform_metric(G, ((0, -1, 0), (-1, 0, 0), (0, 0, -1))))
    assert sorted_root_distance(base, swapped) < 1e-9


def test_spglib_oracle_same_and_different_lattices():
    """Independent oracle: sorted_root_distance ~ 0 iff spglib same Niggli cell.

    Exercises same-lattice collapse under setting change. Not a theoretical
    injectivity proof for the sorted key.
    """
    np = pytest.importorskip("numpy")
    spglib = pytest.importorskip("spglib")
    import random as _random
    from agentsg.cell.g6 import _transform_metric, _unimodular_pm1

    def cell_of(G):
        a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
        ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
        return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)),
                ang(G[0][1] / (a * b)))

    def spglib_niggli_full(cell):
        a, b, c, al, be, ga = cell
        al, be, ga = map(math.radians, (al, be, ga))
        va = [a, 0, 0]; vb = [b * math.cos(ga), b * math.sin(ga), 0]
        cx = c * math.cos(be)
        cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
        vc = [cx, cy, math.sqrt(max(0, c * c - cx * cx - cy * cy))]
        red = spglib.niggli_reduce(np.array([va, vb, vc]))
        if red is None:
            return None
        g = red @ red.T
        ls = [math.sqrt(g[i][i]) for i in range(3)]
        ang = lambda i, j: math.degrees(math.acos(max(-1, min(1, g[i][j] / (ls[i] * ls[j])))))
        return (ls[0], ls[1], ls[2], ang(1, 2), ang(0, 2), ang(0, 1))

    rng = _random.Random(7)
    Ms = _unimodular_pm1()
    small = [m for m in Ms if abs(sum(m[i][i] for i in range(3))) < 6]
    bases = [(40, 50, 60, 85, 95, 100), (50, 50, 70, 90, 90, 90),
             (50, 50, 50, 90, 90, 90), (40, 50, 60, 90, 90, 90),
             (50, 50, 70, 90, 90, 120), (45, 55, 65, 90, 105, 90)]
    cells = []
    for bcell in bases:
        G = UnitCell(*bcell).metric_tensor()
        for M in rng.sample(small, 5):
            cells.append(cell_of(_transform_metric(G, M)))

    n_same = 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            rd = sorted_root_distance(cells[i], cells[j])
            ni, nj = spglib_niggli_full(cells[i]), spglib_niggli_full(cells[j])
            if ni is None or nj is None:
                continue
            same_spglib = (max(abs(ni[k] - nj[k]) for k in range(3)) < 1e-2 and
                           max(abs(ni[3 + k] - nj[3 + k]) for k in range(3)) < 1e-1)
            same_root = rd < 1e-2
            assert same_root == same_spglib, (cells[i], cells[j], rd)
            if same_spglib:
                n_same += 1
    assert n_same >= 20
