"""z:guarantee — retrieval, coverage, coset and soundness under one noise scale."""
from __future__ import annotations

import math
import random

import pytest

pytestmark = [pytest.mark.zcheck]


def _gram_from_p(p):
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _cell_from_p(p):
    from agentsg.cell.metric import params_from_metric
    return params_from_metric(_gram_from_p(p))


def _type_cells():
    """One test cell per Voronoi type, in angstrom-ish sizes."""
    scale = 25.0
    return {
        "V1": _cell_from_p([scale * x for x in (1, 2, 3, 4, 5, 6)]),
        "V2": _cell_from_p([scale * x for x in (1, 2, 3, 4, 5, 0)]),
        "V3": _cell_from_p([scale * x for x in (0, 2, 3, 4, 5, 0)]),
        "V4": _cell_from_p([scale * x for x in (1, 2, 3, 4, 0, 0)]),
        "V5": (3.0, 4.0, 5.0, 90.0, 90.0, 90.0),
    }


def _noisy(cell, sigma_deg, rng):
    a, b, c, al, be, ga = cell
    return (a, b, c,
            al + rng.gauss(0.0, sigma_deg),
            be + rng.gauss(0.0, sigma_deg),
            ga + rng.gauss(0.0, sigma_deg))


def test_full_coset_recovered_at_noise_scale():
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.rootform import sorted_linear_distance

    sigma = 0.02
    rng = random.Random(7)
    for name, cell in _type_cells().items():
        expected = len(reindexing_via_canonical(cell, cell, boundary_rel=0.0))
        assert expected >= 2, name
        A = _noisy(cell, sigma, rng)
        B = _noisy(cell, sigma, rng)
        # retrieval: the noisy pair sits inside a noise-scale radius
        assert sorted_linear_distance(A, B) < 0.5, name
        ops = reindexing_via_canonical(A, B, angle_sigma=sigma,
                                       verify_rel=1e-3)
        assert len(ops) == expected, (name, len(ops), expected)


def test_failure_onset_below_the_multiple():
    from agentsg.cell.canonical import reindexing_via_canonical

    H = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
    sigma = 0.05
    rng = random.Random(2)
    A = _noisy(H, sigma, rng)
    B = _noisy(H, sigma, rng)
    counts = [len(reindexing_via_canonical(A, B, angle_sigma=sigma,
                                           verify_rel=rel))
              for rel in (1e-6, 1e-4, 1e-3)]
    assert counts == sorted(counts)      # tightening never adds operators
    assert counts[0] < 24                # below the multiple: coset lost
    assert counts[-1] == 24              # at the multiple: full coset
