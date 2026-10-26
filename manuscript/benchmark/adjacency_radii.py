"""The first-layer adjacency radius r_1 (Remark A8) of the benchmark references and of a
random sample of PDB cells: below it, the lookup of the 6096 matrices of U between two
Selling-reduced cells is a complete search. Reports quantiles, the share with r_1 above
the strain e^0.074 - 1 = 7.7 % that certifies the default price band of 7.4 %, and the
time per cell. Also times reindexing per benchmark case.

    PYTHONPATH=agentsg/src python manuscript/benchmark/adjacency_radii.py [n_pdb]
"""
import json
import math
import os
import sys
import time

import duckdb
import numpy as np

from agentsg.cell import strain_closure as sc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
NEED = math.expm1(sc.DEFAULT_ACCEPT)


def radii(cells, label):
    r, t, fail = [], [], 0
    for cell, sg in cells:
        t0 = time.perf_counter()
        try:
            r.append(sc.adjacency_radius(sc.reduce_cell(cell, sg).G))
        except ValueError:
            fail += 1
            continue
        t.append(time.perf_counter() - t0)
    r = np.array(r)
    q = np.percentile(100 * r, [1, 5, 10, 50])
    print(f"{label}: {len(r)} cells ({fail} refused); r_1 % quantiles 1/5/10/50: "
          + "/".join(f"{x:.1f}" for x in q) + f"; at the 25 % cap: {100 * (r >= 0.25).mean():.1f} %; "
          f"above {100 * NEED:.1f} %: {100 * (r > NEED).mean():.1f} %; median time {1000 * np.median(t):.0f} ms")
    return r


def main(n_pdb=2000):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    refs = {tuple(round(x, 6) for x in r["reference_cell"]): r["reference_cell"] for r in rows}
    radii([(c, None) for c in refs.values()], "benchmark references")
    con = duckdb.connect(os.path.join(ROOT, "data", "pdb_cells.duckdb"), read_only=True)
    pdb = con.execute(f"select a, b, c, alpha, beta, gamma, sg_hm from cells using sample {n_pdb} rows (reservoir, 7)").fetchall()
    radii([(tuple(p[:6]), p[6] or None) for p in pdb], f"PDB sample")
    for pid in ("3HR2", "3HQV", "6BZP", "1CGM"):                    # the thin tail of the archive
        p = con.execute("select a, b, c, alpha, beta, gamma, sg_hm from cells where pdb_id = ?", [pid]).fetchone()
        try:
            r1 = sc.adjacency_radius(sc.reduce_cell(tuple(p[:6]), p[6] or None).G)
            print(f"{pid} {tuple(round(x, 2) for x in p[:6])} {p[6]}: r_1 {100 * r1:.2f} %")
        except ValueError as e:
            print(f"{pid} {tuple(round(x, 2) for x in p[:6])} {p[6]}: refused ({e})")
    for cell in ((300.0, 300.0, 3.0, 90, 90, 90), (5.0, 5.0, 200.0, 90, 90, 90), (150.0, 120.0, 4.8, 90, 90.4, 90)):
        t0 = time.perf_counter(); W = sc.widened_closure(cell); t1 = time.perf_counter()
        try:
            r1 = sc.adjacency_radius(sc.reduce_cell(cell).G); t2 = time.perf_counter()
            print(f"thin cell {cell}: widening {t1 - t0:.1f} s (layer {W.layer}), r_1 {100 * r1:.1f} % in {t2 - t1:.1f} s")
        except ValueError:
            print(f"thin cell {cell}: widening {t1 - t0:.1f} s (layer {W.layer}); r_1 refused (ball above the cap at 25 %)")
    sys.path.insert(0, HERE)
    import pdb_band_controls as bc
    t0 = time.perf_counter()
    for r in rows[:50]:
        bc.best_price(r["reference_cell"], r["measured_cell"])
    print(f"vectorised lookup of U for one new cell, reductions included: {(time.perf_counter() - t0) / 50:.3f} s on average")
    t = []
    for r in rows[::10]:
        t0 = time.perf_counter()
        ref = sc.widened_closure(r["reference_cell"])
        t1 = time.perf_counter()
        sc.reindex(ref, r["measured_cell"])
        t.append((t1 - t0, time.perf_counter() - t1))
    t = np.array(t)
    print(f"benchmark, every 10th case ({len(t)}): widening the reference median {np.median(t[:, 0]):.2f} s "
          f"(90th {np.percentile(t[:, 0], 90):.2f}); reindexing one new cell median {np.median(t[:, 1]):.2f} s "
          f"(90th {np.percentile(t[:, 1], 90):.2f})")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
