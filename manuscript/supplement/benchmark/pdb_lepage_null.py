"""A null for pseudo-symmetry in the PDB. For the most common space groups of each
low-symmetry type, real cells are compared with cells whose free parameters are shuffled
independently across entries of the same space group (marginal distributions kept, any
relation between parameters destroyed), and with a shuffle within 20 quantiles of cell
volume (which keeps the edges of one size class together). Reported for each: the share
with a higher lattice at Le Page's 3 degrees, and the shares whose cheapest priced
pseudo-symmetric alternative (pdb_lepage.one, the reported lattice's own holohedry
excluded) is within 1 % and within 4.2 %, with a 95 % binomial interval, over two seeds.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/pdb_lepage_null.py [n]
"""
from cctbx import crystal                                          # noqa: F401,E402 (before numpy)
import os                                                          # noqa: E402
import sys                                                         # noqa: E402

import duckdb                                                      # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdb_lepage as pl                                            # noqa: E402

DATA = os.environ.get("SELLINGPROOFS_DATA", os.path.join(HERE, "..", "..", "..", "data"))
DB = os.path.join(DATA, "pdb_cells.duckdb")
GROUPS = (("P 1", "a,b,c,alpha,beta,gamma"), ("P 1 21 1", "a,b,c,beta"), ("C 1 2 1", "a,b,c,beta"),
          ("P 21 21 21", "a,b,c"), ("P 21 21 2", "a,b,c"))


def cell_of(x, cols):
    d = dict(zip(cols.split(","), x))
    return (d["a"], d["b"], d["c"], d.get("alpha", 90.0), d.get("beta", 90.0), d.get("gamma", 90.0))


def volume(c):
    a, b, cc, al, be, ga = c
    ca, cb, cg = (np.cos(np.radians(x)) for x in (al, be, ga))
    return a * b * cc * np.sqrt(max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0))


def rates(X, cols, sg):
    from concurrent.futures import ProcessPoolExecutor
    hi = p1 = p4 = n = 0
    with ProcessPoolExecutor(max(1, os.cpu_count() or 1)) as ex:
        res = list(ex.map(pl.one, [("x",) + cell_of(x, cols) + (sg,) for x in X], chunksize=20))
    for r in res:
        if "error" in r:
            continue
        n += 1
        if r.get("extra") is not None:
            hi += 1
            v = [p for ps in r["extra"].values() for p in ps]
            if v:
                p1 += min(v) <= 0.01; p4 += min(v) <= 0.042
    return n, hi, p1, p4


def ci(k, n):
    p = k / n
    return 100 * p, 196 * np.sqrt(p * (1 - p) / n)


def main(n=1500):
    con = duckdb.connect(DB, read_only=True)
    print(f"share of entries (%, with 95 % interval) with a higher lattice at 3 deg / cheapest alternative <= 1 % / <= 4.2 %; {n} cells per arm")
    for sg, cols in GROUPS:
        rows = np.array(con.execute(f"select {cols} from cells where sg_hm='{sg}'").fetchall(), float)
        vol = np.array([volume(cell_of(x, cols)) for x in rows])
        bins = np.digitize(vol, np.quantile(vol, np.linspace(0, 1, 21)[1:-1]))
        for seed in (1, 2):
            rng = np.random.default_rng(seed)
            real = rows[rng.choice(len(rows), min(n, len(rows)), replace=False)]
            shuf = np.column_stack([rows[rng.choice(len(rows), len(real))][:, k] for k in range(rows.shape[1])])
            strat = []
            for x in real:                                          # same volume bin, parameters from different entries
                b = bins[np.argmin(np.abs(rows - x).sum(1))]
                pool = np.where(bins == b)[0]
                strat.append([rows[rng.choice(pool)][k] for k in range(rows.shape[1])])
            out = []
            for lab, X in (("real", real), ("shuffled", shuf), ("shuffled within volume", np.array(strat))):
                m, hi, p1, p4 = rates(X, cols, sg)
                out.append(f"{lab} " + " / ".join("%.1f±%.1f" % ci(k, m) for k in (hi, p1, p4)))
            print(f"  {sg:11s} seed {seed}: " + ";  ".join(out), flush=True)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
