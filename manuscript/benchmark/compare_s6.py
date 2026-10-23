"""Matching against a library of references: which reference does a measured cell
belong to? On benchmark.jsonl, with the distinct reference cells as the library.

  CS6Dist, NCDist  (Andrews & Bernstein; github.com/yayahjb/ncdist, built as shared
                   libraries libcs6dist.so and libncdist.so in NCDIST_DIR): the nearest
                   reference by the distance between the two cells' G6 vectors (NCDist on
                   Niggli-reduced cells, cctbx). These are metrics on lattice space and
                   give no operator.
  closure          the smallest price of any candidate operator between the measured
                   cell and a reference, over the references (restricted to the 10
                   nearest by CS6Dist, as a prefilter), which also returns the operator.

The library is built from the benchmark's references, many of which are near-duplicates
by construction (lattices a small distance from one specialisation), so the nearest
reference is often not unique at the error. Also reported: how often the true reference
is among the 10 nearest by CS6Dist, and among the references the closure accepts at the
band (true operator found).

    NCDIST_DIR=/path/to/ncdist PYTHONPATH=agentsg/src python manuscript/benchmark/compare_s6.py
"""
from __future__ import annotations

import ctypes
import json
import os

from cctbx import crystal                                           # noqa: E402 (before numpy users)
import numpy as np                                                 # noqa: E402

from agentsg.cell.metric import metric_tensor
from agentsg.cell.strain_closure import band_for, reindex, widened_closure

HERE = os.path.dirname(os.path.abspath(__file__))


def g6(cell):
    G = np.array(metric_tensor(tuple(cell)), float)
    return [G[0, 0], G[1, 1], G[2, 2], 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1]]


def g6_niggli(cell):
    cs = crystal.symmetry(unit_cell=tuple(cell), space_group_symbol="P1").niggli_cell()
    return g6(cs.unit_cell().parameters())


def load(lib, fn):
    f = getattr(ctypes.CDLL(os.path.join(os.environ["NCDIST_DIR"], lib)), fn)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double * 6, ctypes.c_double * 6]
    return lambda a, b: f((ctypes.c_double * 6)(*a), (ctypes.c_double * 6)(*b))


def main(band=0.074):
    cs6 = load("libcs6dist.so", "CS6Dist_in_G6")
    ncd = load("libncdist.so", "NCDist")
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    refs, key = [], {}
    for r in rows:
        k = tuple(round(x, 6) for x in r["reference_cell"])
        if k not in key:
            key[k] = len(refs)
            refs.append(r["reference_cell"])
    G6r = [g6(c) for c in refs]
    N6r = [g6_niggli(c) for c in refs]
    widened = {}
    hit = {"CS6Dist": 0, "NCDist": 0, "closure": 0, "true in 10 nearest by CS6Dist": 0,
           "true reference accepted by the closure": 0}
    for r in rows:
        true = key[tuple(round(x, 6) for x in r["reference_cell"])]
        gN = g6(r["measured_cell"])
        dc = np.array([cs6(gN, g) for g in G6r])
        nN = g6_niggli(r["measured_cell"])
        dn = np.array([ncd(nN, g) for g in N6r])
        hit["CS6Dist"] += int(np.argmin(dc) == true)
        hit["NCDist"] += int(np.argmin(dn) == true)
        best, bi = 9.0, -1
        M = np.array(r["true_operator"], int)
        M = -M if round(np.linalg.det(M)) == -1 else M
        near = list(np.argsort(dc)[:10])
        hit["true in 10 nearest by CS6Dist"] += int(true in near)
        for i in near:
            if i not in widened:
                widened[i] = widened_closure(refs[i], eps=band_for(band))
            c, _ = reindex(widened[i], r["measured_cell"], accept=band)
            if c and c[0].price < best:
                best, bi = c[0].price, i
            if i == true and any([[int(x) for x in row] for row in k.operator.rows] == M.tolist() for k in c):
                hit["true reference accepted by the closure"] += 1
        hit["closure"] += int(bi == true)
    print(f"library of {len(refs)} distinct references, {len(rows)} measured cells")
    print("  (first three: the nearest reference is the true one)")
    for k, v in hit.items():
        print(f"  {k:40s} {v}/{len(rows)}")


if __name__ == "__main__":
    main()
