"""z:isometry — on one-chamber families the key is exact, not a bound.

If every member's conorms, aligned to a reference, deviate by less than
half the reference's margin and half its smallest conorm gap, then every
member reduces into the same chamber with the same conorm ordering: the
sorting slack vanishes, key distances equal exact aligned distances for
every pair, and the ideal follows through one fixed linear map -- no
pairwise search. Violating the gap condition leaves a bounded underestimate
(the sorting swap slack), never an overestimate.
"""
from __future__ import annotations

import math
import random

import pytest

pytestmark = [pytest.mark.zcheck]


def _np():
    import numpy
    return numpy


def _gram_from_p(p):
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _cell_from_p(p):
    from agentsg.cell.metric import params_from_metric
    return params_from_metric(_gram_from_p(p))


def test_one_chamber_family_key_is_exact():
    """margin and gap both > 2 delta: key distance == aligned distance ==
    the exact conorm distance, pair by pair; closure match is the identity
    pairing and the margin certificate holds."""
    import numpy as np
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.rootform import sorted_linear_key

    p_ref = (10.0, 14.0, 18.0, 22.0, 26.0, 30.0)   # margin 10, gap 4
    delta = 1.5                                     # < min(margin, gap) / 2
    rng = random.Random(5)
    fam = []
    for _ in range(25):
        p = [x + rng.uniform(-delta, delta) for x in p_ref]
        fam.append((tuple(p), _cell_from_p(p)))

    for i in range(0, len(fam), 5):
        for j in range(i + 1, len(fam), 3):
            (pi, ci), (pj, cj) = fam[i], fam[j]
            Ti, Tj = sum(pi), sum(pj)
            aligned = math.sqrt(sum(
                (a / math.sqrt(Ti) - b / math.sqrt(Tj)) ** 2
                for a, b in zip(pi, pj)))
            ki = np.array(sorted_linear_key(ci))
            kj = np.array(sorted_linear_key(cj))
            keyd = float(np.linalg.norm(ki - kj))
            assert abs(keyd - aligned) < 1e-9 * max(1.0, aligned)
            _, D0, _ = closure_distance(ci, cj)
            # margin certificate: the closure value is below every positive
            # normalised conorm, so it is the exact frame infimum
            margin = min(min(x for x in ki if x > 1e-9),
                         min(x for x in kj if x > 1e-9))
            assert D0 < margin


def test_gap_violation_gives_bounded_underestimate_only():
    """two conorms closer than 2 delta: sorting may swap them; the key then
    underestimates the aligned distance by a bounded slack and never
    overestimates it."""
    import numpy as np
    from agentsg.cell.rootform import sorted_linear_key

    p_ref = (10.0, 14.0, 18.0, 21.0, 21.5, 30.0)   # one gap of 0.5
    delta = 1.5
    rng = random.Random(8)
    worst_slack = 0.0
    n_swapped = 0
    for _ in range(60):
        pa = [x + rng.uniform(-delta, delta) for x in p_ref]
        pb = [x + rng.uniform(-delta, delta) for x in p_ref]
        Ta, Tb = sum(pa), sum(pb)
        aligned = math.sqrt(sum(
            (a / math.sqrt(Ta) - b / math.sqrt(Tb)) ** 2
            for a, b in zip(pa, pb)))
        ka = np.array(sorted_linear_key(_cell_from_p(pa)))
        kb = np.array(sorted_linear_key(_cell_from_p(pb)))
        keyd = float(np.linalg.norm(ka - kb))
        assert keyd <= aligned + 1e-9          # never an overestimate
        slack = aligned - keyd
        worst_slack = max(worst_slack, slack)
        if slack > 1e-9:
            n_swapped += 1
    assert n_swapped >= 1                      # the tie genuinely bites
    # slack bounded by the swap scale: ~ the deviation on the tied pair
    assert worst_slack < 4.0 * delta / math.sqrt(sum(p_ref))
