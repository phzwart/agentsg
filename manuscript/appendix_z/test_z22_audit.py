"""z:audit — collision / edge audit of a key-space graph subsample."""
from __future__ import annotations

import random

import pytest

from helpers import SEED_EMBED

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_pdb]


def test_edge_audit_subsample(pdb_db_path):
    """Classify a subsample of k-NN edges via reindexing_via_canonical."""
    duckdb = pytest.importorskip("duckdb")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.primitive import primitive_cell, lattice_letter
    from agentsg.cell.rootform import sorted_root_key

    con = duckdb.connect(str(pdb_db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT a,b,c,alpha,beta,gamma,sg_hm FROM cells USING SAMPLE 800"
        ).fetchall()
    except Exception:
        rows = con.execute(
            "SELECT a,b,c,alpha,beta,gamma,sg_hm FROM cells LIMIT 800"
        ).fetchall()
    con.close()

    cells = []
    keys = []
    for a, b, c, al, be, ga, sg in rows:
        cell = (float(a), float(b), float(c), float(al), float(be), float(ga))
        try:
            if sg and lattice_letter(sg) != "P":
                cell = primitive_cell(cell, sg)
            keys.append(sorted_root_key(cell))
            cells.append(cell)
        except Exception:
            continue

    X = np.asarray(keys, dtype=np.float64)
    tree = cKDTree(X)
    rng = random.Random(SEED_EMBED)
    same = collision = neighbour = 0
    n_edges = 0
    for i in rng.sample(range(len(cells)), min(100, len(cells))):
        _, nn = tree.query(X[i], k=6)
        for j in nn[1:]:
            j = int(j)
            n_edges += 1
            dkey = float(np.linalg.norm(X[i] - X[j]))
            ops = reindexing_via_canonical(
                cells[i], cells[j], boundary_rel=0, verify_rel=1e-3,
            )
            if ops:
                same += 1
            elif dkey < 0.05:
                collision += 1
            else:
                neighbour += 1

    print(
        f"z:audit edges={n_edges} same={same} collision={collision} "
        f"neighbour={neighbour}"
    )
    assert n_edges > 0
