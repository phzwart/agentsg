"""z:ca-benchmark — carbonic anhydrase II: one protein, one deformed family.

Data: agentsg/tests/data/ca_cells.csv — 1065 carbonic anhydrase PDB entries
(user-supplied list), cells and space groups from the agentsg PDB snapshot
(doi 10.5281/zenodo.22986222), harvested by a single sorted-linear-key
search around 1CA2 at cutoff 2.5 A (every entry on the list fell inside
that one ball; retrieval recall 1.0 from one query).

The family is one crystal form, continuously deformed: 1062 P2_1 cells and
3 P1 deposits of the same lattice shape, one key-connected component at
radius 0.5 A, nearest-neighbour key distances with median 0.05 A. The
medoid's anisotropy budget is positive (D_max ~ 1155 A^2), so the radius-1
ball is certified across the family; the interior-margin certificate holds
on every sampled pair, making the closure minimum exactly the ideal there.
Strains run from ~1% (median) to 10.3% (6BBS, the farthest member,
certified at 639 A^2).
"""
from __future__ import annotations

import csv
import math
import os
import random

import pytest

pytestmark = [pytest.mark.zcheck]

_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "agentsg",
                    "tests", "data", "ca_cells.csv")


def _np():
    import numpy
    return numpy


def _load():
    cells, sgs = {}, {}
    with open(os.path.normpath(_CSV)) as fh:
        for r in csv.DictReader(fh):
            cells[r["pdb"]] = tuple(
                float(r[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma"))
            sgs[r["pdb"]] = r["sg_hm"]
    return cells, sgs


def _keys(cells):
    from agentsg.cell.rootform import sorted_linear_key
    np = _np()
    pdbs = list(cells)
    K = np.array([sorted_linear_key(cells[p]) for p in pdbs])
    return pdbs, K


def test_one_family_one_ball():
    """1065 entries, 1062 P2_1 + 3 P1; all within 2.5 A of 1CA2's key; a
    single key-connected component (>=1060 members) at radius 0.5 A."""
    import collections

    from scipy.spatial import cKDTree
    np = _np()
    cells, sgs = _load()
    assert len(cells) == 1065
    counts = collections.Counter(sgs.values())
    assert counts == {"P 1 21 1": 1062, "P 1": 3}

    pdbs, K = _keys(cells)
    k_1ca2 = K[pdbs.index("1CA2")]
    dists = np.linalg.norm(K - k_1ca2, axis=1)
    assert dists.max() <= 2.5 + 1e-6      # one retrieval ball catches all

    tree = cKDTree(K)
    dnn, _ = tree.query(K, k=2)
    assert float(np.median(dnn[:, 1])) < 0.1

    parent = list(range(len(pdbs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in tree.query_pairs(0.5):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    comp = collections.Counter(find(i) for i in range(len(pdbs)))
    assert max(comp.values()) >= 1060     # one family
    # the P1 deposits live inside it, not apart from it
    main = max(comp, key=comp.get)
    for p in (q for q in pdbs if sgs[q] == "P 1"):
        assert find(pdbs.index(p)) == main


def test_medoid_budget_and_certified_pairs():
    """Medoid 4L5V; positive anisotropy budget; on seeded random pairs the
    radius-1 ideal is certified and the interior-margin certificate makes
    the closure minimum exactly the ideal."""
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.deformation import (_ball_min, _metric, _radius,
                                          _unimodular_ball)
    from agentsg.cell.reduction import niggli_reduce
    np = _np()
    cells, _ = _load()
    pdbs, K = _keys(cells)

    D2 = np.linalg.norm(K[:, None, :] - K[None, :, :], axis=2)
    med = pdbs[int(D2.sum(axis=1).argmin())]
    assert med == "4L5V"

    red, _ = niggli_reduce(*cells[med])
    lam = np.linalg.eigvalsh(_metric(red))
    dmax = 4 * lam[0] - lam[-1]
    assert 1000 < dmax < 1300             # certified regime, unlike 7/17 Scotty

    ball1 = _unimodular_ball(1)
    rng = random.Random(7)
    n = cert = marg = 0
    ds = []
    keys = {p: K[i] for i, p in enumerate(pdbs)}
    while n < 25:
        i, j = rng.randrange(len(pdbs)), rng.randrange(len(pdbs))
        if i == j:
            continue
        GA, GB = _metric(cells[pdbs[i]]), _metric(cells[pdbs[j]])
        v1, _ = _ball_min(GA, GB, ball1)
        v2, _ = _ball_min(GB, GA, ball1)
        d = max(v1, v2)
        n += 1
        ds.append(d)
        cert += (_radius(GA, GB, v1) < 2.0 and _radius(GB, GA, v2) < 2.0)
        _, D0, _ = closure_distance(cells[pdbs[i]], cells[pdbs[j]])
        mink = min(min(x for x in keys[pdbs[i]] if x > 1e-9),
                   min(x for x in keys[pdbs[j]] if x > 1e-9))
        marg += (D0 < mink)
    assert cert == n                      # the whole family is radius-1 certified
    assert marg == n                      # closure minimum == ideal, provably
    assert 40 < float(np.median(ds)) < 130


def test_strain_extreme_is_certified():
    """The farthest member (6BBS) is a 10% swollen copy of the same form,
    with a certified ideal distance near 639 A^2."""
    from agentsg.cell.deformation import deformation_distance
    cells, _ = _load()
    r = deformation_distance(cells["4L5V"], cells["6BBS"],
                             schedule=(0.0,), rmax=2)
    assert r["certified"]
    assert 600 < r["d"] < 680
    assert 0.09 < r["strain_norm"] < 0.12
