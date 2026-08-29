"""z:pdb-types — monoclinic P → V4, orthorhombic P → V5; PDB census."""
from __future__ import annotations

import pytest

from helpers import MONOCLINIC_P, ORTHO

pytestmark = [pytest.mark.zcheck]


def test_monoclinic_orthorhombic_types():
    from agentsg.cell.selling_closure import voronoi_type
    assert voronoi_type(MONOCLINIC_P) == 4
    assert voronoi_type(ORTHO) == 5


@pytest.mark.needs_pdb
@pytest.mark.slow
def test_pdb_type_census(pdb_db_path):
    """Tabulate Voronoi type over PDB; report fraction type != 1."""
    duckdb = pytest.importorskip("duckdb")
    from agentsg.cell.selling_closure import voronoi_type
    from agentsg.cell.primitive import primitive_cell, lattice_letter

    con = duckdb.connect(str(pdb_db_path), read_only=True)
    # Sample for speed; full census is optional
    try:
        rows = con.execute(
            "SELECT a,b,c,alpha,beta,gamma,sg_hm FROM cells USING SAMPLE 5000"
        ).fetchall()
    except Exception:
        rows = con.execute(
            "SELECT a,b,c,alpha,beta,gamma,sg_hm FROM cells LIMIT 5000"
        ).fetchall()
    con.close()

    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "fail": 0}
    for a, b, c, al, be, ga, sg in rows:
        cell = (float(a), float(b), float(c), float(al), float(be), float(ga))
        try:
            letter = lattice_letter(sg or "P1")
            if letter != "P":
                cell = primitive_cell(cell, sg)
            t = voronoi_type(cell)
            counts[t] = counts.get(t, 0) + 1
        except Exception:
            counts["fail"] += 1

    n = sum(v for k, v in counts.items() if k != "fail")
    assert n > 100
    frac_non_v1 = 1.0 - counts[1] / n
    # manuscript: V2--V5 are a large fraction of the PDB
    assert frac_non_v1 > 0.3, f"unexpectedly few non-V1: {counts}"
    print(f"z:pdb-types census sample n={n} counts={counts} "
          f"frac_type!=1={frac_non_v1:.3f}")
