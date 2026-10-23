"""Non-isomorphism in the PDB snapshot: the strain between cells of one crystal form.

Proxy for "one crystal form of one protein": entries with the same space group whose PDB
ids are adjacent in sorted order and share the first three characters (deposition series
of one group usually get consecutive ids). Both cells are in the standard setting of the
same space group, so the identity is the operator (an exact lattice symmetry of either cell
leaves the price unchanged). The strain is reported with and without the common scale fit.
Writes the quantiles; pairs above 20 % are counted as different crystal forms.

    PYTHONPATH=agentsg/src python manuscript/benchmark/pdb_nonisomorphism.py
"""
from __future__ import annotations

import os

import duckdb
import numpy as np

from agentsg.cell.metric import metric_tensor
from agentsg.cell.strain_closure import _strain

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "pdb_cells.duckdb")


def main():
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells order by pdb_id").fetchall()
    fit, raw, edge = [], [], []
    for r1, r2 in zip(rows, rows[1:]):
        if r1[0][:3] != r2[0][:3] or r1[7] != r2[7]:
            continue
        G1 = np.array(metric_tensor(tuple(r1[1:7])), float)
        G2 = np.array(metric_tensor(tuple(r2[1:7])), float)
        fit.append(_strain(G2, G1)[0]); raw.append(_strain(G2, G1, fit_scale=False)[0])
        edge.append(max(abs(x / y - 1) for x, y in zip(r1[1:4], r2[1:4])))
    fit, raw, edge = map(np.array, (fit, raw, edge))
    same = raw < 0.2
    print(f"{len(raw)} id-adjacent pairs with the same space group; {same.sum()} below 20 % strain (one crystal form)")
    print(f"{'quantile':>9s} {'strain, no scale fit':>21s} {'after scale fit':>16s} {'largest edge change':>20s}")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f"{q:8d}% {100 * np.percentile(raw[same], q):20.2f}% {100 * np.percentile(fit[same], q):15.2f}% "
              f"{100 * np.percentile(edge[same], q):19.2f}%")
    for t in (0.005, 0.01, 0.02, 0.04, 0.074):
        print(f"  pairs within {100 * t:4.1f} %: {100 * (raw[same] <= t).mean():5.1f} % (no fit), "
              f"{100 * (fit[same] <= t).mean():5.1f} % (fit)")
    print(f"  exactly identical cells: {(raw[same] < 1e-9).sum()}")


if __name__ == "__main__":
    main()
