#!/usr/bin/env python3
"""Neighbors of one archive cell on the volume-normalised linear key.

The metric net is the Euclidean ball of radius 0.02 on

    sort(p_ij / sqrt(sum p)) / V_primitive**(1/3)

Both reported distances are computed on primitive cells scaled to unit volume,
so they measure shape. Pairs inside the ball are ranked by the sorted linear
distance of those cells. The Selling closure then replaces that length for the
nearest few: the aligned distance of the exact closure match, which stays put
or rises.

Usage::

    PYTHONPATH="agentsg/src:.venv/lib/python3.12/site-packages" \\
    ../leanmap/.venv/bin/python linear_embed/query.py 1ENX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embed import DEFAULT_CACHE, DEFAULT_PDB, load_features  # noqa: E402

RADIUS = 0.02
CLOSURE_K = 8


def _unit_volume(cell):
    """Isotropic scaling to primitive volume 1. Angles are unchanged."""
    from agentsg.cell.rootform import _cell_volume

    scale = _cell_volume(cell) ** (1.0 / 3.0)
    if scale <= 0.0:
        return cell
    a, b, c, al, be, ga = cell
    return (a / scale, b / scale, c / scale, al, be, ga)


_REF_COB: dict[str, tuple] = {}


def warm_reference_settings() -> None:
    """Build the space-group identification cache before the first click."""
    from agentsg.hall import ops_from_hall
    from agentsg.identify import identify_space_group

    identify_space_group(ops_from_hall("P 1"))


def reference_parameters(conv, sg_hm: str) -> dict:
    """Volume-1 conventional cell in the Hall reference setting.

    The change of basis takes the deposited setting onto the reference
    setting of the same space-group number. Edges are then divided by the
    cube root of that cell's volume. Angles are left as the reference cell
    gives them.
    """
    from agentsg.hall import ops_from_hall
    from agentsg.identify import identify_space_group
    from agentsg.ita_settings import lookup_setting
    from agentsg.cell.sublattice import apply_to_cell

    deposited = str(sg_hm).strip()
    row = lookup_setting(deposited)
    number = None
    if row is None:
        cell = _unit_volume(tuple(float(x) for x in conv))
        reference = deposited
        matched = False
    else:
        hall = row[2]
        cached = _REF_COB.get(hall)
        if cached is None:
            hit = identify_space_group(ops_from_hall(hall))
            basis = [[float(x) for x in r] for r in hit.change_of_basis.P.rows]
            cached = (basis, hit.hermann_mauguin, int(hit.number))
            _REF_COB[hall] = cached
        basis, reference, number = cached
        cell = _unit_volume(apply_to_cell(tuple(float(x) for x in conv), basis))
        deposited = row[1]
        matched = True
    a, b, c, alpha, beta, gamma = cell
    return {
        "deposited": deposited,
        "reference": reference,
        "number": number,
        "matched": matched,
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def selling_parameters(conv, sg_hm: str) -> dict:
    """Volume-1 Selling-reduced primitive cell.

    The deposited conventional cell is reduced to primitive by its centring
    letter, then to one obtuse Selling basis. Edges are divided by the cube
    root of that primitive cell's volume, so the drawn cell has volume 1.
    The three angles are at least 90°.
    """
    from agentsg.cell.celldb import _primitive_for_roots
    from agentsg.cell.rootform import selling_reduced_cell

    deposited = str(sg_hm).strip()
    prim = _primitive_for_roots(tuple(float(x) for x in conv), deposited)
    cell = _unit_volume(selling_reduced_cell(prim))
    a, b, c, alpha, beta, gamma = cell
    return {
        "deposited": deposited,
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def deposited_conventional(pdb: Path, pdb_id: str):
    """Stored conventional cell and Hermann–Mauguin symbol, or None."""
    import duckdb

    db = duckdb.connect(str(pdb), read_only=True)
    row = db.execute(
        "SELECT a, b, c, alpha, beta, gamma, sg_hm FROM cells WHERE pdb_id = ?",
        [pdb_id],
    ).fetchone()
    db.close()
    return row


def _cells_for(pdb: Path, ids: list[str]) -> dict[str, tuple]:
    """Unit-volume primitive cell for each pdb id."""
    import duckdb
    from agentsg.cell.celldb import _primitive_for_roots

    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    db = duckdb.connect(str(pdb), read_only=True)
    rows = db.execute(
        "SELECT pdb_id, a, b, c, alpha, beta, gamma, sg_hm FROM cells "
        f"WHERE pdb_id IN ({marks})",
        ids,
    ).fetchall()
    db.close()
    out = {}
    for row in rows:
        conv = tuple(row[1:7])
        prim = _primitive_for_roots(conv, row[7])
        out[row[0]] = _unit_volume(prim)
    return out


def query(pdb_id: str, *, radius: float, closure_k: int, pdb: Path, cache: Path):
    from scipy.spatial import cKDTree
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.rootform import sorted_linear_distance

    S, meta = load_features(pdb, cache)
    ids = np.asarray(meta["pdb_id"])
    wanted = pdb_id.strip().upper()
    hits = np.flatnonzero(np.char.upper(ids.astype(str)) == wanted)
    if hits.size == 0:
        raise SystemExit(f"{pdb_id} is not in {cache}")
    q = int(hits[0])
    qid = str(ids[q])

    tree = cKDTree(S)
    nbr = tree.query_ball_point(S[q], r=radius)
    nbr = [int(j) for j in nbr if int(j) != q]
    if not nbr:
        print(f"{qid}  {meta['sg_hm'][q]}  no cells within {radius}")
        return

    ball_ids = [str(ids[j]) for j in nbr]
    packed = _cells_for(pdb, [qid, *ball_ids])
    if qid not in packed:
        raise SystemExit(f"{qid} has no cell row in {pdb}")
    cell_q = packed[qid]

    ranked = []
    for j in nbr:
        jid = str(ids[j])
        cell_j = packed.get(jid)
        if cell_j is None:
            continue
        ranked.append((sorted_linear_distance(cell_q, cell_j), j, jid, cell_j))
    ranked.sort()

    print(
        f"{qid}  {meta['sg_hm'][q]}  "
        f"{len(ranked)} cells within {radius} on the volume-normalised key"
    )
    print(f"{'pdb':<6} {'sg':<14} {'sorted':>8} {'closure':>8} {'tweak':>8}")
    for n, (sorted_d, j, jid, cell_j) in enumerate(ranked):
        closure_s = ""
        tweak_s = ""
        if n < closure_k:
            _P, aligned, _resid = closure_distance(cell_q, cell_j, boundary_rel=0.0)
            closure_s = f"{aligned:.4f}"
            tweak_s = f"{aligned - sorted_d:+.4f}"
        sg = str(meta["sg_hm"][j])
        print(
            f"{jid:<6} {sg:<14} {sorted_d:8.4f} {closure_s:>8} {tweak_s:>8}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb_id", help="archive id, for example 1ENX")
    ap.add_argument("--radius", type=float, default=RADIUS,
                    help="metric-net radius on the volume-normalised key")
    ap.add_argument("--closure", type=int, default=CLOSURE_K,
                    help="how many nearest hits get a Selling-closure distance")
    ap.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    query(
        args.pdb_id,
        radius=args.radius,
        closure_k=args.closure,
        pdb=args.pdb,
        cache=args.cache,
    )


if __name__ == "__main__":
    main()
