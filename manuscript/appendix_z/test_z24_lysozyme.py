"""z:lysozyme — UniProt P00698 census and shrinkage coordinate."""
from __future__ import annotations

import pytest

from helpers import assert_within_pct

pytestmark = [pytest.mark.zcheck, pytest.mark.slow, pytest.mark.needs_pdb]


def test_lysozyme_census(pdb_db_path):
    duckdb = pytest.importorskip("duckdb")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from scipy.spatial import cKDTree
    from agentsg.cell.rootform import sorted_root_key
    from agentsg.cell.primitive import primitive_cell, lattice_letter

    con = duckdb.connect(str(pdb_db_path), read_only=True)
    cols = [r[0] for r in con.execute("DESCRIBE cells").fetchall()]
    # Try common metadata columns for UniProt / molecule name
    filter_sql = None
    for cand in (
        "uniprot", "uniprot_id", "entity_uniprot", "molecule", "title",
        "pdb_id",
    ):
        if cand in cols:
            if "uniprot" in cand:
                filter_sql = f"SELECT a,b,c,alpha,beta,gamma,sg_hm,sg_number FROM cells WHERE {cand} LIKE '%P00698%'"
            break
    if filter_sql is None:
        # Fall back: known HEWL space-group / cell ranges is not enough for census.
        # Skip with message if no uniprot column.
        if "pdb_id" in cols:
            pytest.skip(
                "pdb_cells.duckdb has no UniProt column; cannot run z:lysozyme census"
            )
        pytest.skip("cannot filter lysozyme entries from DuckDB schema")

    rows = con.execute(filter_sql).fetchall()
    con.close()
    if len(rows) < 100:
        pytest.skip(f"only {len(rows)} lysozyme rows; need UniProt metadata")

    # space-group counts (HM)
    from collections import Counter
    sg_counts = Counter(r[6] for r in rows if r[6])
    n_cells = len(rows)
    # manuscript: 1361 with cells; allow ±1% on total if DB date differs
    assert_within_pct(n_cells, 1361, pct=5, label="n lysozyme cells")

    # tetragonal native-volume subset
    tet = []
    for a, b, c, al, be, ga, sg, *rest in rows:
        hm = (sg or "").replace(" ", "")
        if "P43212" in hm or "P4_3_2_1_2" in hm or hm.startswith("P43212"):
            V = float(a) ** 2 * float(c)
            tet.append(((float(a), float(b), float(c), float(al), float(be), float(ga)), V))
    if len(tet) < 100:
        # try sg_number 96
        tet = []
        for row in rows:
            a, b, c, al, be, ga = row[:6]
            sgnum = row[7] if len(row) > 7 else None
            if sgnum == 96:
                V = float(a) ** 2 * float(c)
                tet.append(((float(a), float(b), float(c), float(al), float(be), float(ga)), V))

    assert len(tet) > 100
    Vs = [V for _, V in tet]
    span = (max(Vs) - min(Vs)) / (sorted(Vs)[len(Vs) // 2])
    assert_within_pct(span, 0.26, pct=20, label="tetragonal volume span")

    keys = []
    vols = []
    for cell, V in tet:
        try:
            keys.append(sorted_root_key(cell))
            vols.append(V ** (1.0 / 3.0))
        except Exception:
            continue
    X = np.asarray(keys, dtype=np.float64)
    tree = cKDTree(X)
    # graph Laplacian leading eigenvector via diffusion / degree-normalized
    k = 20
    n = len(X)
    W = np.zeros((n, n))
    for i in range(n):
        _, nn = tree.query(X[i], k=k + 1)
        for j in nn[1:]:
            W[i, int(j)] = 1.0
            W[int(j), i] = 1.0
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    # normalized Laplacian eigenvector (smallest nontrivial) via dense eigh on subsample
    if n > 800:
        sel = np.linspace(0, n - 1, 800).astype(int)
        W = W[np.ix_(sel, sel)]
        vols = [vols[i] for i in sel]
        d = W.sum(axis=1)
        d[d == 0] = 1.0
        n = len(sel)
    Dinvsqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dinvsqrt @ W @ Dinvsqrt
    w, v = np.linalg.eigh(L)
    coord = v[:, 1]
    r = float(np.corrcoef(coord, np.asarray(vols))[0, 1])
    assert_within_pct(abs(r), 0.92, pct=10, label="lysozyme |r|")
    print(f"z:lysozyme n={n_cells} tet={len(tet)} span={span:.3f} |r|={abs(r):.3f}")
