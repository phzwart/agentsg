"""z:conograph — the seven sample cells distributed with Oishi-Tomiyasu's
error-stable Bravais-lattice determination (BLDConograph, github.com/rtomiyasu,
commit 93b58ed), compared with the priced operators of the closure.

For each measured primitive cell: idealise at the largest band the theorem
allows (tau = lambda_1/32), enumerate the closure of the idealised lattice,
and price every operator out of the reduced cell (all members, all
labellings) by ||W^T G W - G||_F in the measured metric. The price at which
the count of operators reaches the order of each holohedry is the band at
which that pseudo-symmetry is offered. BLDConograph's minimal distance to a
cell of the sample's Bravais type (its output.py) is listed for comparison;
the two are different measures, so only the order of the ladder and the
rough size are compared. This is the absolute conorm band of Section S4 of the
supporting information; Table S5 of the paper gives the strain distances of the sort
spectrum for the same cells, checked in z60.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402

PERM4 = list(itertools.permutations(range(4)))

# name, primitive cell as distributed (input.py), Bravais type in the sample
# folder name, holohedry order, BLDConograph minimal distance for that type
SAMPLES = [
    ("HERMES_Sr327_250K", (3.8417, 3.8524, 10.389, 100.41, 100.53, 90.046), "tI", 16, 0.5672),
    ("1339prbasrcuo_bs1", (3.8792, 3.8830, 11.654, 90.018, 90.039, 90.057), "tP", 16, 0.05641),
    ("sample3", (16.332, 16.348, 16.357, 109.41, 109.52, 109.34), "cI", 48, 2.0738),
    ("CeO2_matsushita", (3.7250, 3.7269, 3.7286, 90.100, 119.96, 119.91), "cF", 48, 0.084427),
    ("M402-pf-119673", (10.654, 22.895, 23.003, 60.600, 89.377, 89.119), "hP", 24, 20.549),
    ("Acrinol-B-D8CuKa1", (9.1568, 9.1712, 20.946, 81.634, 81.751, 80.086), "mC", 4, 0.71525),
    ("Alumina676", (4.7428, 4.7802, 5.1093, 116.89, 90.170, 119.69), "hR", 12, 1.9315),
]
ZEROS = {"tI": 1, "tP": 3, "cI": 0, "cF": 2, "hP": 2, "mC": 0, "hR": 2}


def price_ladder(cell):
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    G = np.array(UnitCell(*cell).metric_tensor())
    F0 = np.array(canonical_superbase(tuple(cell))[0], float)
    Gf = F0[1:4] @ G @ F0[1:4].T
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, Gf)
    lam1 = min(float(s @ Gf @ s) for s in s0)
    tau = lam1 / 32
    ps = np.where(p0 <= 4 * tau, 0.0, p0)
    _, mem = t._generate(p0, 4 * tau, 1e18)
    prices = []
    for S in mem.values():
        for pm in PERM4:
            for sg in (1, -1):
                W = (sg * S[list(pm)])[1:4].T
                prices.append(np.linalg.norm(W.T @ Gf @ W - Gf))
    prices = np.sort(prices)
    ladder = {n: float(prices[n - 1]) for n in (2, 4, 8, 12, 16, 24, 48) if n <= len(prices)}
    return int((ps == 0).sum()), len(mem), tau, ladder


@claim("z51", "SI Table S5", kind="exhaustive")
def test_conograph_samples_reach_their_type_within_the_hypothesis():
    r"""The seven BLDConograph sample cells reach the program's Bravais type within the hypothesis of the band theorem."""
    rows = []
    for name, cell, bt, order, dist in SAMPLES:
        zeros, members, tau, ladder = price_ladder(cell)
        # the band the theorem allows already idealises the type's walls
        assert zeros >= ZEROS[bt], (name, zeros)
        # the holohedry is offered: its order is reached in the price ladder
        assert order in ladder, (name, ladder)
        price = ladder[order]
        rows.append((name, bt, members, price, dist))
        # same scale as BLDConograph's minimal distance (within a factor 2.5)
        assert 0.4 < price / dist < 2.5, (name, price, dist)
        # the next holohedry up, where one exists, is priced clearly higher
        higher = [ladder[n] for n in ladder if n > order]
        if higher and bt not in ("tI", "tP", "hR"):
            assert min(higher) > 2 * price, (name, ladder)
    # the worked numbers of the paper's table
    expect = {"HERMES_Sr327_250K": (2, 0.77), "1339prbasrcuo_bs1": (16, 0.10), "sample3": (1, 2.98),
              "CeO2_matsushita": (3, 0.11), "M402-pf-119673": (6, 22.5), "Acrinol-B-D8CuKa1": (1, 0.96),
              "Alumina676": (3, 1.69)}
    for name, bt, members, price, dist in rows:
        m, p = expect[name]
        assert members == m and abs(price - p) < 0.05 * max(p, 0.2), (name, members, price)
