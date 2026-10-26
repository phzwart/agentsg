"""Share of PDB pairs of one crystal form (cut 20 %) whose price, with the identity as the
operator, lies within a given band: the measurements a band covers.

    PYTHONPATH=agentsg/src python manuscript/benchmark/pdb_band_coverage.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdb_band_controls as bc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = bc.pairs()
fit = np.array([p["fit"] for p in P]); raw = np.array([p["raw"] for p in P])
f = fit[raw < 0.2]
lines = [f"{len(f)} pairs of one crystal form (cut 20 %)"]
for b in (0.074, 0.099, 0.148):
    lines.append(f"band {100 * b:.1f} %: {100 * (f <= b).mean():.2f} % of pairs within; "
                 f"{100 * (f > b).mean():.2f} % beyond (median price of those {100 * np.median(f[f > b]):.1f} %)")
lines.append("pooled 97.5th / 99th percentile: " + " / ".join(f"{100 * x:.2f} %" for x in np.percentile(f, [97.5, 99])))
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "results_pdb_band_coverage.txt"), "w").write(txt + "\n")
