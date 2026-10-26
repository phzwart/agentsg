"""The lookup of the 6096 matrices of U between the two Selling-reduced cells (widened
closure with certify=False, Remark A8) on the benchmark and the stress family, scored as in
compare_methods.py, with the number of references whose adjacency radius r_1 is below the
strain e^band - 1 that certifies the lookup.

    PYTHONPATH=agentsg/src python manuscript/benchmark/lookup_benchmark.py
"""
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm                                       # noqa: E402 (imports cctbx first)
import numpy as np                                                 # noqa: E402
import window_failures as wf                                       # noqa: E402

from agentsg.cell import strain_closure as sc                      # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    need = math.expm1(cm.BAND)
    for label, cases in (("benchmark", rows), ("stress", cm.stress_cases(rows))):
        agg = collections.defaultdict(collections.Counter)
        for c in cases:
            fam = c.get("family", "?")
            M = np.array(c["true_operator"]); M = -M if round(np.linalg.det(M)) == -1 else M
            Ps, ps = wf.lookup(c["reference_cell"], c["measured_cell"], cm.BAND)
            ok = cm.score(Ps, ps, M)[0]
            r1 = sc.adjacency_radius(sc.reduce_cell(c["reference_cell"]).G)
            for k in (fam, "all"):
                agg[k]["n"] += 1; agg[k]["found"] += ok; agg[k]["uncertified"] += r1 <= need
                agg[k]["missed, certified"] += (not ok) and r1 > need and not fam.startswith("t=1")
        print(f"{label}: lookup of U, band {cm.BAND}")
        for k, a in sorted(agg.items()):
            print(f"  {k:16s} found {a['found']}/{a['n']}  references with r_1 <= {100 * need:.1f} %: {a['uncertified']}  "
                  f"missed with a certified reference (true price within the band): {a['missed, certified']}")


if __name__ == "__main__":
    main()
