"""Root invariant (Kurlin 2022) tests: invariance, continuity, completeness.

The root invariant is a single ordered vector per lattice (obtuse superbase ->
conorms -> sorted root products, canonicalised over 24 index permutations). It
needs NO orbit minimisation and is continuous across the reduction-flip boundary.
"""
import math
import random
import pytest
from agentsg.cell.rootform import (
    delaunay_superbase, conorms, root_invariant, root_distance, _dot,
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


def test_root_invariant_length6_and_sorted():
    """The canonical tuple is the SORTED six-tuple of root products (Kurlin Def
    5.1 realised uniformly across Voronoi types) -- deterministic and ascending."""
    c = (40, 50, 60, 88, 92, 103)
    ri = root_invariant(c)
    assert len(ri) == 6
    assert list(ri) == sorted(ri)                      # canonical form is ascending
    assert ri == root_invariant(c)                     # deterministic
    assert all(x >= -1e-12 for x in ri)                # root products are >= 0


def test_invariance_under_unimodular_setting():
    """Same lattice in different bases -> identical root invariant."""
    rng = random.Random(2)
    Ms = [((1, 1, 0), (0, 1, 0), (0, 0, 1)),
          ((1, 0, 0), (1, 1, 0), (0, 0, 1)),
          ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
          ((2, 1, 0), (1, 1, 0), (0, 0, 1))]
    for _ in range(30):
        c = _valid_cell(rng)
        G = UnitCell(*c).metric_tensor()
        for M in Ms:
            assert root_distance(c, _cell_of(_transform_metric(G, M))) < 1e-6


def test_continuity_across_reduction_flip_boundary():
    """No orbit search, yet continuous across a=b (where raw G6 jumps)."""
    ref = (40.0, 40.0, 60.0, 90, 91, 90)
    d_hi = root_distance(ref, (40.0, 40.001, 60.0, 90, 91, 90))
    d_lo = root_distance(ref, (40.0, 39.999, 60.0, 90, 91, 90))
    assert abs(d_hi - d_lo) < 1e-3
    assert d_hi < 0.02 and d_lo < 0.02


def test_niggli_invariance():
    rng = random.Random(4)
    for _ in range(20):
        c = _valid_cell(rng)
        r, _ = niggli_reduce(*c)
        assert root_distance(c, r) < 1e-6


def test_completeness_no_collisions_distinct_lattices():
    rng = random.Random(5)
    cells = [_valid_cell(rng) for _ in range(300)]
    ris = [root_invariant(c) for c in cells]
    for i in range(len(ris)):
        for j in range(i + 1, len(ris)):
            same = all(abs(ris[i][k] - ris[j][k]) < 1e-6 for k in range(6))
            if same:
                assert abs(UnitCell(*cells[i]).volume() - UnitCell(*cells[j]).volume()) < 1e-3


def test_cubic_signature():
    """Cubic P: three equal root products = edge length, three zero."""
    ri = root_invariant((50, 50, 50, 90, 90, 90))
    nz = sorted(x for x in ri if x > 1e-6)
    assert len(nz) == 3
    assert all(abs(x - 50.0) < 1e-6 for x in nz)


def test_metric_axioms():
    """root_distance is a metric: identity, symmetry, triangle inequality."""
    rng = random.Random(6)
    a, b, c = (_valid_cell(rng) for _ in range(3))
    assert root_distance(a, a) < 1e-9
    assert abs(root_distance(a, b) - root_distance(b, a)) < 1e-9
    assert root_distance(a, c) <= root_distance(a, b) + root_distance(b, c) + 1e-9


def test_invariance_high_symmetry_cells():
    """Regression: orthorhombic/tetragonal/cubic/monoclinic (zero-conorm Voronoi
    types) must be setting-invariant. A flat lex-min over index permutations is
    NOT -- the canonical form must be the sorted multiset (Kurlin Def 5.1)."""
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
            assert root_distance(c, c2) < 1e-4, f"{name} not invariant under {M}"


def test_orthorhombic_axis_swap_is_isometry():
    """A pure axis relabelling (det=+1) must give distance exactly 0."""
    base = (50.0, 60.0, 70.0, 90, 90, 90)
    G = UnitCell(*base).metric_tensor()
    swapped = _cell_of(_transform_metric(G, ((0, -1, 0), (-1, 0, 0), (0, 0, -1))))
    assert root_distance(base, swapped) < 1e-9


def test_spglib_oracle_same_and_different_lattices():
    """Independent oracle: root_distance ~ 0 iff spglib reports the same Niggli
    cell (edges AND angles). Crucially this exercises the completeness-critical
    case -- same lattice in different bases must collapse -- not only random
    (always-distinct) lattices."""
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
            rd = root_distance(cells[i], cells[j])
            ni, nj = spglib_niggli_full(cells[i]), spglib_niggli_full(cells[j])
            if ni is None or nj is None:
                continue
            same_spglib = (max(abs(ni[k] - nj[k]) for k in range(3)) < 1e-2 and
                           max(abs(ni[3 + k] - nj[3 + k]) for k in range(3)) < 1e-1)
            same_root = rd < 1e-2
            assert same_root == same_spglib, (cells[i], cells[j], rd)
            if same_spglib:
                n_same += 1
    # guard: the test must actually contain same-lattice pairs, else it's vacuous
    assert n_same >= 20
