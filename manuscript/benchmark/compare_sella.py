"""Compare the sort spectrum (strain norm, frames of the universal set) with SELLA
(Andrews, Bernstein & Sauter, 2023; S6 distances in A^2 over the boundary images) on the
same cells. SELLA is run from the LatticeRepLib checkout given by SELLA_BIN (the
CmdSella executable). For each cell: both rankings of the 24 sorts, Spearman's rank
correlation, each method's nearest sort of every Bravais type, and the sorts that each
method places within its own small band. The two scales are related first: over all
(cell, sort) pairs with a strain distance between 0.01 % and 25 %, the median ratio r of
SELLA's distance divided by the cell's mean squared edge to the strain distance; SELLA's
band is then 0.074 x r x the mean squared edge.

    SELLA_BIN=/path/to/CmdSella PYTHONPATH=agentsg/src python manuscript/benchmark/compare_sella.py
"""
from __future__ import annotations

import os
import re
import subprocess

import numpy as np

from agentsg.cell.strain_closure import BRAVAIS, adjacency_radius, reduce_cell, sort_spectrum

CELLS = {
    "worked cell": ("P", (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)),
    "lysozyme in P1": ("P", (79.06, 79.13, 37.93, 90.03, 89.96, 90.05)),
    "oP 40 x 40.3 x 80": ("P", (40.0, 40.3, 80.0, 90, 90, 90)),
    "all angles 92": ("P", (30.0, 40.0, 50.0, 92, 92, 92)),
    "needle 5.58 x 5.21 x 132": ("P", (5.583, 5.21, 131.965, 90.0, 90.0, 91.118)),
    "plate 100 x 100.1 x 5": ("P", (100.0, 100.1, 5.0, 90.03, 89.98, 90.05)),
    "plate 150 x 120 x 4.8": ("P", (150.0, 120.0, 4.8, 90.0, 90.4, 90.0)),
    "HERMES_Sr327_250K": ("P", (3.8417, 3.8524, 10.389, 100.41, 100.53, 90.046)),
    "1339prbasrcuo_bs1": ("P", (3.8792, 3.8830, 11.654, 90.018, 90.039, 90.057)),
    "sample3": ("P", (16.332, 16.348, 16.357, 109.41, 109.52, 109.34)),
    "CeO2_matsushita": ("P", (3.7250, 3.7269, 3.7286, 90.100, 119.96, 119.91)),
    "M402-pf-119673": ("P", (10.654, 22.895, 23.003, 60.600, 89.377, 89.119)),
    "Acrinol-B-D8CuKa1": ("P", (9.1568, 9.1712, 20.946, 81.634, 81.751, 80.086)),
    "Alumina676": ("P", (4.7428, 4.7802, 5.1093, 116.89, 90.170, 119.69)),
}


def sella(letter, cell, binary):
    txt = f"{letter} " + " ".join(f"{x}" for x in cell) + "\nend\n"
    out = subprocess.run([binary], input=txt, capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(binary)).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r";\s*best fit in SellaFit\S*\s+S6\s+((?:-?[\d.]+\s+){6})([\d.eE+-]+)\s+(\w+)", line)
        if m:
            d[m.group(3)] = float(m.group(2))
    return d


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    binary = os.environ["SELLA_BIN"]
    runs = {}
    ratios = []
    for name, (letter, cell) in CELLS.items():
        sp = sort_spectrum(cell)
        se = sella(letter, cell, binary)
        red = reduce_cell(cell)
        scale = float(np.mean(np.diag(red.G)))
        runs[name] = (sp, se, red, scale)
        ratios += [se[s] / scale / sp[s] for s in sp if s in se and 1e-4 < sp[s] < 0.25]
    r = float(np.median(ratios))
    q = np.percentile(ratios, [10, 90])
    print(f"SELLA distance / mean squared edge, against the strain distance: median {r:.3f} (10-90 % {q[0]:.3f}-{q[1]:.3f})")
    print(f"{'cell':26s} {'rho':>5s} {'r_1':>6s}  nearest types, strain (%) | SELLA (A^2)")
    same = 0
    for name, (sp, se, red, scale) in runs.items():
        sorts = [s for s in sp if s in se]
        rho = spearman([sp[s] for s in sorts], [se[s] for s in sorts])
        d1 = adjacency_radius(red.G)
        by_s, by_e = {}, {}
        for s in sorts:
            b = BRAVAIS[s]
            by_s[b] = min(by_s.get(b, 9), sp[s]); by_e[b] = min(by_e.get(b, 1e9), se[s])
        in_s = sorted(b for b in by_s if b != "aP" and by_s[b] <= 0.074)
        in_e = sorted(b for b in by_e if b != "aP" and by_e[b] <= 0.074 * r * scale)
        mark = lambda b: "*" if by_s[b] > d1 else ""
        lad_s = ", ".join(f"{b} {100 * by_s[b]:.2f}{mark(b)}" for b in sorted((b for b in by_s if b != "aP"), key=by_s.get)[:4])
        lad_e = ", ".join(f"{b} {by_e[b]:.3g}" for b in sorted((b for b in by_e if b != "aP"), key=by_e.get)[:4])
        print(f"{name:26s} {rho:5.2f} {d1:6.3f}  {lad_s} | {lad_e}")
        diff = set(in_s) ^ set(in_e)
        same += not diff
        print(f"{'':26s} types within band: strain {in_s}  SELLA {in_e}" + (f"   differ: {sorted(diff)}" if diff else "   same"))
    print(f"same types within band in {same} of {len(runs)} cells")
    print("* above the cell's adjacency radius r_1: an upper bound over the frames used (true distance >= r_1)")


if __name__ == "__main__":
    main()
