"""z:xfel — CXIDB 83 stream statistics and orientation diagnostic."""
from __future__ import annotations

import math
import statistics

import pytest

from helpers import assert_within_pct

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_xfel]


def test_xfel_stream_stats(xfel_stream_path):
    np = pytest.importorskip("numpy")
    from agentsg.cell.crystfel_stream import parse_stream, stream_summary
    from agentsg.cell.rootform import sorted_conorm_key

    summary = stream_summary(str(xfel_stream_path))
    assert summary["n_frames"] == 14445
    assert summary["n_indexed"] == 12474

    cells = [d["cell"] for d in parse_stream(str(xfel_stream_path))]
    c_vals = [c[2] for c in cells]
    c_med = statistics.median(c_vals)
    c_mad = statistics.median(abs(x - c_med) for x in c_vals)
    assert_within_pct(c_med, 233.3, pct=10, label="c median")
    assert_within_pct(c_mad, 0.13, pct=10, label="c MAD")
    p999 = sorted(c_vals)[int(0.999 * (len(c_vals) - 1))]
    assert p999 > 240.0  # heavy tail toward ~245

    # linear / sorted-conorm keys + PCA loadings (smoke)
    keys = np.asarray([sorted_conorm_key(c) for c in cells[:5000]], dtype=np.float64)
    keys = keys - keys.mean(axis=0)
    _, _, vt = np.linalg.svd(keys, full_matrices=False)
    print(f"z:xfel PCA loadings row0={vt[0]}")

    # orientation diagnostic
    with_cstar = list(parse_stream(str(xfel_stream_path), with_orientation=True))
    parallel, oblique = [], []
    beam = np.array([0.0, 0.0, 1.0])
    for rec in with_cstar:
        c = rec["cell"][2]
        cstar = np.array(rec["cstar"], dtype=float)
        nrm = np.linalg.norm(cstar)
        if nrm < 1e-12:
            continue
        cstar /= nrm
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(beam, cstar))))))
        if ang < 10.0:
            parallel.append(c)
        elif ang > 80.0:
            oblique.append(c)

    def mad(vals):
        med = statistics.median(vals)
        return statistics.median(abs(v - med) for v in vals)

    if len(parallel) >= 20 and len(oblique) >= 20:
        sc_par = mad(parallel)
        sc_obl = mad(oblique)
        ratio = sc_par / max(sc_obl, 1e-9)
        print(f"z:xfel scatter parallel={sc_par:.3f} oblique={sc_obl:.3f} "
              f"ratio={ratio:.2f} n_par={len(parallel)} n_obl={len(oblique)}")
        # manuscript ~1.53 vs ~0.5 → ratio ~3; prior CSV showed different binning —
        # pin ratio ~3 ± large margin, or at least parallel > oblique
        assert sc_par > sc_obl
        assert_within_pct(ratio, 3.0, pct=50, label="orientation scatter ratio")
    else:
        pytest.skip("insufficient orientation-binned crystals")
