"""z:skeleton — Bravais skeleton of the normalised key space: points, nexus, seams."""
from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.zcheck]


def _nkey(cell, sg=None):
    from agentsg.cell.rootform import sorted_linear_key
    from agentsg.cell.selling_cob import _primitive_or_self

    prim = _primitive_or_self(cell, sg) if sg else cell
    k = sorted_linear_key(prim)
    a, b, c, al, be, ga = prim
    r = math.radians
    arg = (1.0 - math.cos(r(al)) ** 2 - math.cos(r(be)) ** 2
           - math.cos(r(ga)) ** 2
           + 2.0 * math.cos(r(al)) * math.cos(r(be)) * math.cos(r(ga)))
    V = a * b * c * math.sqrt(max(arg, 1e-300))
    s = V ** (1.0 / 3.0)
    return tuple(x / s for x in k)


def _d(x, y):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y)))


_CP = _nkey((4, 4, 4, 90, 90, 90))
_CI = _nkey((4, 4, 4, 90, 90, 90), "I 2 3")
_CF = _nkey((4, 4, 4, 90, 90, 90), "F 2 3")
_ALPHA_BCC = math.degrees(math.acos(-1.0 / 3.0))  # 109.471...


def _hR(alpha):
    return _nkey((4, 4, 4, alpha, alpha, alpha))


def _tI(ratio):
    return _nkey((4, 4, 4 * ratio, 90, 90, 90), "I 4")


def _tP(ratio):
    return _nkey((4, 4, 4 * ratio, 90, 90, 90))


def _hP(ratio):
    return _nkey((4, 4, 4 * ratio, 90, 90, 120))


def _alpha_for_hex_ratio(R):
    """hR alpha whose hexagonal-setting c/a equals R."""
    return 2.0 * math.degrees(math.asin(3.0 / (2.0 * math.sqrt(R * R + 3.0))))


def test_cubic_fixed_points():
    s3 = 1.0 / math.sqrt(3.0)
    assert _d(_CP, (0, 0, 0, s3, s3, s3)) < 1e-3
    assert max(abs(x - 0.2572) for x in _CI) < 1e-3
    assert _d(_CF, (0, 0, 0.3969, 0.3969, 0.3969, 0.3969)) < 1e-3
    # scale invariance of the normalised key
    assert _d(_nkey((97, 97, 97, 90, 90, 90)), _CP) < 1e-9


def test_quintuple_nexus_at_cP():
    # tP terminus, linear approach
    d1 = _d(_tP(1.1), _CP)
    d2 = _d(_tP(1.01), _CP)
    assert d2 < 0.012 and 7.0 < d1 / d2 < 13.0
    # oP corner
    assert _d(_nkey((4, 4.004, 4.008, 90, 90, 90)), _CP) < 0.01
    # oC coincides with tP exactly, and its tP edge ends on cP
    assert _d(_nkey((4, 4, 7, 90, 90, 90), "C 2 2 2"),
              _nkey((4 / math.sqrt(2), 4 / math.sqrt(2), 7, 90, 90, 90))) < 1e-9
    assert _d(_nkey((4, 4, 4 / math.sqrt(2), 90, 90, 90), "C 2 2 2"), _CP) < 1e-8
    # rhombohedral alpha -> 90, linear approach
    e1 = _d(_hR(91.0), _CP)
    e2 = _d(_hR(90.1), _CP)
    assert e2 < 0.005 and 7.0 < e1 / e2 < 13.0


def test_double_nexus_hR_tI():
    assert _d(_hR(_ALPHA_BCC), _CI) < 1e-8
    assert _d(_tI(1.0), _CI) < 1e-8
    assert _d(_hR(60.0), _CF) < 1e-8
    assert _d(_tI(math.sqrt(2.0)), _CF) < 1e-8
    # no other contact: coarse scan of both curves
    alphas = [40.0 + 2.5 * i for i in range(31)]
    ratios = [0.4 + 0.1 * i for i in range(25)]
    KH = [(a, _hR(a)) for a in alphas]
    KT = [(r, _tI(r)) for r in ratios]
    for a, kh in KH:
        for r, kt in KT:
            if _d(kh, kt) < 0.05:
                near_cI = abs(a - _ALPHA_BCC) < 5.0 and abs(r - 1.0) < 0.2
                near_cF = abs(a - 60.0) < 5.0 and abs(r - math.sqrt(2.0)) < 0.2
                assert near_cI or near_cF, (a, r)


def test_centred_orthorhombic_feeds():
    # oI(a=b) = tI exactly; oI cubic = cI
    assert _d(_nkey((4, 4, 7, 90, 90, 90), "I 2 2 2"), _tI(7.0 / 4.0)) < 1e-9
    assert _d(_nkey((4, 4, 4, 90, 90, 90), "I 2 2 2"), _CI) < 1e-9
    # oF(a=b) = tI at sqrt2 * c/a; oF cubic = cF   (the F222 side)
    assert _d(_nkey((4, 4, 7, 90, 90, 90), "F 2 2 2"),
              _tI(7.0 * math.sqrt(2.0) / 4.0)) < 1e-8
    assert _d(_nkey((4, 4, 4, 90, 90, 90), "F 2 2 2"), _CF) < 1e-9


def test_seam_hP_hR_three_to_one():
    d = {R: _d(_hP(R), _hR(_alpha_for_hex_ratio(3.0 * R)))
         for R in (1.0, 4.0, 16.0, 32.0)}
    assert d[1.0] > d[4.0] > d[16.0] > d[32.0]
    assert d[4.0] < 0.02 and d[32.0] < 0.002


def test_seam_tP_tI_two_to_one():
    d = {R: _d(_tP(R), _tI(2.0 * R)) for R in (1.0, 4.0, 16.0, 64.0)}
    assert d[1.0] > d[4.0] > d[16.0] > d[64.0]
    assert d[16.0] < 0.04 and d[64.0] < 0.006
    # flat mirror branch via the 45-degree relation
    assert _d(_tP(0.1), _tI(0.1 / math.sqrt(2.0))) < 0.02


def test_tetragonal_curve_separation_near_cubic():
    ratios = [0.05 * 1.2 ** i for i in range(40)]  # 0.05 .. ~37
    d_cP = min(_d(_CP, _tI(r)) for r in ratios)
    d_cI = min(_d(_CI, _tP(r)) for r in ratios)
    assert 0.45 < d_cP < 0.56
    assert 0.65 < d_cI < 0.78
