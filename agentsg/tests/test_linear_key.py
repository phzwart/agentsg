"""Properties of the sorted linear search key.

The square-root key stays available through ``stabilize="sqrt"``. These checks
are the Lipschitz key: one key per lattice, ångström scaling, the cube
identity, the rearrangement lower bound, and the Scotty pairs.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from agentsg.cell.canonical import _transform_metric_int, closure_distance
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.rootform import (
    sorted_linear_distance,
    sorted_linear_key,
    sorted_root_distance,
)
from agentsg.cell.selling_closure import selling_superbase_closure
from agentsg.cell.selling_cob import _primitive_or_self

_DATA = Path(__file__).parent / "data" / "scotty_cases.csv"

# Query, hit, expected sorted-linear distance (Å).
_SCOTTY = (
    ("4amt", "5mlg", 0.60),
    ("4amt", "5mkt", 5.04),
    ("2d4k", "1vdp", 2.75),
    ("1fe5", "1u4j", 2.72),
    ("1fe5", "1g2x", 2.87),
    ("7mg1", "1hqw", 1.42),
    ("7mg1", "5wey", 3.20),
    ("4nl6", "1jr7", 0.94),
    ("4nl6", "2r6s", 0.89),
    ("4nl6", "8cdf", 1.56),
    ("4nl7", "2yht", 5.43),
)
_FORMER_FAILURES = ("1u4j", "1g2x", "2yht")

# 4nl7 primitive (C2, a = b), the Selling-boundary cell used for the cusp.
_HFQ_PRIMITIVE = (
    61.951, 61.951, 57.10, 94.925, 94.925, 60.373,
)


def _cell_of_member(cell, superbase):
    """Cell of superbase vectors 1, 2, 3 in the input basis."""
    G = UnitCell(*cell).metric_tensor()
    P = [[superbase[1 + j][i] for j in range(3)] for i in range(3)]
    return params_from_metric(_transform_metric_int(G, P))


def _same_key(a, b, rel=1e-9):
    scale = max(1.0, max(abs(x) for x in a), max(abs(x) for x in b))
    return all(abs(x - y) <= rel * scale for x, y in zip(a, b))


def test_closure_invariance():
    """Every Selling-closure member of one lattice shares the linear key."""
    cells = (
        (40.0, 50.0, 60.0, 85.0, 95.0, 100.0),
        _HFQ_PRIMITIVE,
        (4.913, 4.913, 5.405, 90.0, 90.0, 120.0),
        (50.0, 60.0, 70.0, 90.0, 90.0, 90.0),
    )
    for cell in cells:
        key = sorted_linear_key(cell)
        members = selling_superbase_closure(cell)
        assert members
        for member in members:
            assert _same_key(key, sorted_linear_key(_cell_of_member(cell, member)))


def test_length_scaling():
    cell = (40.0, 50.0, 60.0, 85.0, 95.0, 100.0)
    lam = 2.3
    scaled = (lam * cell[0], lam * cell[1], lam * cell[2], *cell[3:])
    assert _same_key(
        sorted_linear_key(scaled),
        tuple(lam * x for x in sorted_linear_key(cell)),
    )


def test_cube_identity():
    a = 17.0
    key = sorted_linear_key((a, a, a, 90.0, 90.0, 90.0))
    slot = a / math.sqrt(3.0)
    assert _same_key(key, (0.0, 0.0, 0.0, slot, slot, slot))
    delta = 0.4
    dist = sorted_linear_distance(
        (a, a, a, 90.0, 90.0, 90.0),
        (a + delta, a + delta, a + delta, 90.0, 90.0, 90.0),
    )
    assert abs(dist - delta) <= 1e-9


def test_rearrangement_lower_bound():
    rng = random.Random(0)

    def cell():
        return (
            rng.uniform(15.0, 80.0), rng.uniform(15.0, 80.0), rng.uniform(15.0, 80.0),
            rng.uniform(70.0, 110.0), rng.uniform(70.0, 110.0), rng.uniform(70.0, 110.0),
        )

    for _ in range(50):
        a, b = cell(), cell()
        bound = sorted_linear_distance(a, b)
        aligned = closure_distance(a, b, boundary_rel=0)[1]
        assert bound <= aligned + 1e-9


def test_lipschitz_on_the_boundary_cell():
    """The square-root slope steepens at a vanishing conorm; the linear slope does not.

    G_13 of the 4nl7 primitive is a boundary direction. It is shifted by
    ``eps * 0.3 * max|G|`` so that eps = 0.1 stays inside the local regime.
    """
    cell = _primitive_or_self(
        (107.100, 62.300, 57.100, 90.0, 95.70, 90.0), "C 1 2 1",
    )
    G = UnitCell(*cell).metric_tensor()
    scale = max(abs(G[i][j]) for i in range(3) for j in range(3))

    def slope(eps, distance):
        Gp = [row[:] for row in G]
        Gp[0][2] += eps * 0.3 * scale
        Gp[2][0] += eps * 0.3 * scale
        return distance(cell, params_from_metric(Gp)) / eps

    lin = [slope(eps, sorted_linear_distance) for eps in (1e-3, 1e-1)]
    sqr = [
        slope(eps, lambda a, b: sorted_root_distance(a, b, stabilize="sqrt"))
        for eps in (1e-3, 1e-1)
    ]
    assert sqr[0] / sqr[1] > 5.0
    assert lin[0] / lin[1] < 1.5


def _load_scotty():
    rows = {}
    with _DATA.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["pdb"]] = row
    return rows


def _primitive(row):
    cell = tuple(float(row[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma"))
    # 1fe5 is R32 already on rhombohedral axes.
    if row["pdb"] == "1fe5":
        return cell
    return _primitive_or_self(cell, row["sg_hm"])


def test_scotty_pairs():
    rows = _load_scotty()
    for query, hit, expected in _SCOTTY:
        dist = sorted_linear_distance(_primitive(rows[query]), _primitive(rows[hit]))
        assert abs(dist - expected) <= 0.02 * expected
        if hit in _FORMER_FAILURES:
            assert dist < 6.0
