"""A null for the Le Page pseudo-symmetry rate in the PDB. For the most common space groups
of each low-symmetry type, the share of entries with a higher lattice symmetry at 3 deg is
compared with the share for cells whose free parameters are shuffled independently across
entries of the same space group (the marginal distributions kept, any coincidence between
parameters destroyed). The excess of the real over the shuffled rate is the part of the
pseudo-symmetry that is not chance.

    PYTHONPATH=agentsg/src python manuscript/benchmark/pdb_lepage_null.py [n]
"""
from cctbx import crystal                                          # noqa: E402 (before numpy)
from cctbx.sgtbx import bravais_types, lattice_symmetry            # noqa: E402
import os                                                          # noqa: E402
import sys                                                         # noqa: E402

import duckdb                                                      # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "data", "pdb_cells.duckdb")
ORDER = {"aP": 2, "mP": 4, "mC": 4, "oP": 8, "oS": 8, "oI": 8, "oF": 8, "tP": 16, "tI": 16, "hR": 12,
         "hP": 24, "cP": 48, "cI": 48, "cF": 48}
GROUPS = (("P 1", "aP", "a,b,c,alpha,beta,gamma"), ("P 1 21 1", "mP", "a,b,c,beta"),
          ("C 1 2 1", "mC", "a,b,c,beta"), ("P 21 21 21", "oP", "a,b,c"), ("P 21 21 2", "oP", "a,b,c"))


def higher(cell, sg, rep):
    cs = crystal.symmetry(unit_cell=cell, space_group_symbol=sg)
    p1 = crystal.symmetry(unit_cell=cs.primitive_setting().niggli_cell().unit_cell(), space_group_symbol="P1")
    g = lattice_symmetry.metric_subgroups(input_symmetry=p1, max_delta=3.0).result_groups[0]
    s = str(bravais_types.bravais_lattice(group=g["best_subsym"].space_group()))
    s = "oS" if s[:2] in ("oC", "oA", "oB") else ("mC" if s in ("mI", "mA") else s)
    return ORDER.get(s, 0) > ORDER[rep]


def cell_of(x, cols):
    d = dict(zip(cols.split(","), x))
    return (d["a"], d["b"], d["c"], d.get("alpha", 90.0), d.get("beta", 90.0), d.get("gamma", 90.0))


def main(n=2000):
    rng = np.random.default_rng(1)
    con = duckdb.connect(DB, read_only=True)
    print(f"Le Page 3 deg: share of entries with a higher lattice symmetry, real cells against shuffled ({n} each)")
    for sg, rep, cols in GROUPS:
        rows = np.array(con.execute(f"select {cols} from cells where sg_hm='{sg}'").fetchall(), float)
        real = rows[rng.choice(len(rows), min(n, len(rows)), replace=False)]
        shuf = np.column_stack([rows[rng.choice(len(rows), len(real))][:, k] for k in range(rows.shape[1])])
        out = []
        for X in (real, shuf):
            hit = tot = 0
            for x in X:
                try:
                    hit += higher(cell_of(x, cols), sg, rep); tot += 1
                except Exception:                                   # noqa: BLE001 (impossible shuffled cells)
                    continue
            out.append((hit, tot))
        (h1, t1), (h2, t2) = out
        print(f"  {sg:11s} ({rep}, {len(rows)} entries): real {100 * h1 / t1:5.1f} %  shuffled {100 * h2 / t2:5.1f} %  "
              f"excess {100 * (h1 / t1 - h2 / t2):+5.1f} points")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
