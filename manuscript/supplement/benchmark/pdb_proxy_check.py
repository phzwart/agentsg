"""How often does the crystal-form proxy admit unrelated crystals?  Random pairs of PDB
entries with the same space group whose identifiers do not share their first three
characters (so not one deposition series), drawn with the space-group mix of the
id-adjacent pairs, are priced as in pdb_band_controls.py; reported is the share that would
pass the 20 % cut between crystal forms (and the 5 and 10 % cuts).

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/pdb_proxy_check.py
"""
import collections
import os

import duckdb
import numpy as np

from agentsg.cell.metric import metric_tensor
from agentsg.cell.strain_closure import _price

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SELLINGPROOFS_DATA", os.path.join(HERE, "..", "..", "..", "data"))
OUTDIR = os.environ.get("SELLINGPROOFS_OUT", HERE)
con = duckdb.connect(os.path.join(DATA, "pdb_cells.duckdb"), read_only=True)
rows = con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells order by pdb_id").fetchall()
rng = np.random.default_rng(0)
bysg = collections.defaultdict(list)
for i, r in enumerate(rows):
    bysg[r[7]].append(i)
adj = [r1[7] for r1, r2 in zip(rows, rows[1:]) if r1[0][:3] == r2[0][:3] and r1[7] == r2[7]]
raw = []
for k in rng.choice(len(adj), 20000):
    L = bysg[adj[k]]
    i, j = rng.choice(L, 2, replace=False)
    if rows[i][0][:3] == rows[j][0][:3]:
        continue
    G1 = np.array(metric_tensor(tuple(rows[i][1:7])), float)
    G2 = np.array(metric_tensor(tuple(rows[j][1:7])), float)
    raw.append(_price(G2, G1, fit_scale=False)[0])
raw = np.array(raw)
txt = (f"{len(raw)} random same-space-group pairs from different deposition prefixes: "
       + ", ".join(f"{100 * (raw < c).mean():.2f} % below a {100 * c:.0f} % cut" for c in (0.05, 0.1, 0.2)))
print(txt)
open(os.path.join(OUTDIR, "results_pdb_proxy_check.txt"), "w").write(txt + "\n")
