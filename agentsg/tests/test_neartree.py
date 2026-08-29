"""NearTree exactness + lattice_index (root-invariant) search tests."""
import math
import random
import pytest

pytest.importorskip("scipy")

from agentsg.cell.neartree import NearTree, build_neartree, lattice_index
from agentsg.cell.rootform import root_distance
from agentsg.cell.metric import UnitCell


def _euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def test_neartree_nearest_matches_bruteforce():
    rng = random.Random(0)
    pts = [tuple(rng.uniform(0, 100) for _ in range(6)) for _ in range(500)]
    tree = build_neartree([(p, i) for i, p in enumerate(pts)], _euclid)
    for _ in range(30):
        q = tuple(rng.uniform(0, 100) for _ in range(6))
        pid, d = tree.nearest(q)
        bf = min(range(len(pts)), key=lambda i: _euclid(q, pts[i]))
        assert abs(_euclid(q, pts[pid]) - _euclid(q, pts[bf])) < 1e-9


def test_neartree_knn_matches_bruteforce():
    rng = random.Random(1)
    pts = [tuple(rng.uniform(0, 100) for _ in range(6)) for _ in range(400)]
    tree = build_neartree([(p, i) for i, p in enumerate(pts)], _euclid)
    for _ in range(20):
        q = tuple(rng.uniform(0, 100) for _ in range(6))
        got = set(p for p, _ in tree.k_nearest(q, 5))
        bf = set(sorted(range(len(pts)), key=lambda i: _euclid(q, pts[i]))[:5])
        assert got == bf


def test_neartree_within_matches_bruteforce():
    rng = random.Random(2)
    pts = [tuple(rng.uniform(0, 100) for _ in range(6)) for _ in range(300)]
    tree = build_neartree([(p, i) for i, p in enumerate(pts)], _euclid)
    q = tuple(rng.uniform(0, 100) for _ in range(6))
    R = sorted(_euclid(q, p) for p in pts)[15]
    got = set(p for p, _ in tree.within(q, R))
    bf = set(i for i, p in enumerate(pts) if _euclid(q, p) <= R)
    assert got == bf


def _valid_cell(rng):
    while True:
        c = (rng.uniform(20, 120), rng.uniform(20, 120), rng.uniform(20, 120),
             rng.uniform(70, 110), rng.uniform(70, 110), rng.uniform(70, 110))
        try:
            if UnitCell(*c).volume() > 1e3:
                return c
        except Exception:
            pass


def test_lattice_index_exact_vs_bruteforce():
    rng = random.Random(3)
    db = [_valid_cell(rng) for _ in range(400)]
    idx = lattice_index([(c, i) for i, c in enumerate(db)])
    for _ in range(20):
        q = _valid_cell(rng)
        pid, d = idx.nearest_cell(q)
        bf = min(range(len(db)), key=lambda i: root_distance(q, db[i]))
        assert abs(root_distance(q, db[pid]) - root_distance(q, db[bf])) < 1e-9


def test_lattice_index_reduction_flip_robust():
    """A query on either side of the a=b boundary finds the same reference."""
    ref = (40.0, 40.0, 60.0, 90, 91, 90)
    rng = random.Random(4)
    idx = lattice_index([(ref, "REF")] + [(_valid_cell(rng), i) for i in range(200)])
    hi, dhi = idx.nearest_cell((40.0, 40.02, 60.0, 90, 91, 90))
    lo, dlo = idx.nearest_cell((40.0, 39.98, 60.0, 90, 91, 90))
    assert hi == "REF" and lo == "REF"
    assert abs(dhi - dlo) < 1e-6
