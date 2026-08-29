"""z:trajectory — raw reduced representative jumps; sorted key continuous."""
from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.zcheck]


def test_monoclinic_trajectory_s6_jump_key_continuous():
    from agentsg.cell.g6 import g6_distance, g6
    from agentsg.cell.reduction import niggli_reduce
    from agentsg.cell.rootform import sorted_root_key
    from helpers import key_l2

    N = 401
    cs = [150.0 - i * (50.0 / (N - 1)) for i in range(N)]
    cells = [(120.0, 189.1, c, 90.0, 91.2, 90.0) for c in cs]

    d_raw = []
    for k in range(N - 1):
        na, _ = niggli_reduce(*cells[k])
        nb, _ = niggli_reduce(*cells[k + 1])
        a, b = g6(na), g6(nb)
        d_raw.append(math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(6))))
    ib = max(range(len(d_raw)), key=lambda i: d_raw[i])
    assert d_raw[ib] > 30.0, f"max raw step {d_raw[ib]}"

    # Equal-perturbation flip across a=b: raw G6 discontinuous (manuscript style)
    ref = (40.0, 40.0, 60.0, 90.0, 91.0, 90.0)
    hi = (40.0, 40.001, 60.0, 90.0, 91.0, 90.0)
    lo = (40.0, 39.999, 60.0, 90.0, 91.0, 90.0)
    raw_hi = g6_distance(ref, hi, boundary_aware=False)
    raw_lo = g6_distance(ref, lo, boundary_aware=False)
    assert abs(raw_hi - raw_lo) > 50.0

    dkey = [
        key_l2(sorted_root_key(cells[k]), sorted_root_key(cells[k + 1]))
        for k in range(N - 1)
    ]
    step = abs(cs[0] - cs[1])
    far = [dkey[i] for i in range(len(dkey))
           if abs(cs[i] - 120.0) > 2.0 and abs(cs[i + 1] - 120.0) > 2.0]
    assert max(far) <= 5.0 * step + 1e-6
    assert key_l2(sorted_root_key(hi), sorted_root_key(lo)) < 1.0

    print(f"z:trajectory max niggli-G6 step={d_raw[ib]:.3f} at c~{cs[ib]:.2f}; "
          f"a=b flip |raw_hi-raw_lo|={abs(raw_hi-raw_lo):.3f}; "
          f"max dkey far={max(far):.4g}")
