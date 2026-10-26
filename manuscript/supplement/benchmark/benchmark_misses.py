"""The benchmark cases missed by the {-1,0,1} window between Selling-reduced cells and by the
lattice-symmetry pipeline (compare_methods.py): family, reference cell, the reference's
Niggli type, and the largest entry of the true operator between the two reduced cells.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/benchmark_misses.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm                                       # noqa: E402 (imports cctbx first)
import numpy as np                                                 # noqa: E402
import window_failures as wf                                       # noqa: E402

from agentsg.cell.strain_closure import reduce_cell                # noqa: E402


def B(c):
    return np.array([[float(x) for x in row] for row in reduce_cell(c).basis.rows])


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    for name in ("small1", "pipeline"):
        for r in rows:
            M = np.array(r["true_operator"]); M = -M if round(np.linalg.det(M)) == -1 else M
            Ps, ps = cm.METHODS[name](r["reference_cell"], r["measured_cell"], cm.BAND)
            if cm.score(Ps, ps, M)[0]:
                continue
            W = np.linalg.inv(B(r["reference_cell"])) @ M @ B(r["measured_cell"])
            print(f"{name}: {r.get('family')} {r.get('sort', '')} reference {[round(x, 2) for x in r['reference_cell']]} "
                  f"Niggli type {wf.niggli_type(r['reference_cell'])}; largest entry between reduced cells {int(np.abs(np.round(W)).max())}")
