"""Le Page's 3 degrees on every cell of the PDB snapshot (zenodo 10.5281/zenodo.22986222,
data/pdb_cells.duckdb), and the strain price of the pseudo-symmetries it finds.

For each entry: the reported Bravais type (from the space group); the highest lattice
symmetry that the cctbx lattice-symmetry analysis (Le Page, maximal obliquity 3 deg) finds
for the primitive Niggli cell, with its obliquity; the proper operators of that group that
are not symmetries of the reported space group, nor of the reported lattice's own exact
holohedry (a merohedral space group's lattice symmetries outside its point group price at 0
and are not pseudo-symmetries; they are counted as "merohedral"), each priced in the
strain norm after the scale fit at the deposited metric, by rotation order; and the Delone
sort of the idealised lattice of the top group. Writes pdb_lepage.jsonl (one line per entry)
and prints the tables.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/pdb_lepage.py [--table]
"""
from __future__ import annotations

from cctbx import crystal                                           # noqa: E402 (before numpy users)
from cctbx import sgtbx                                             # noqa: E402
from cctbx.sgtbx import bravais_types, lattice_symmetry             # noqa: E402
import collections                                                  # noqa: E402
import itertools                                                    # noqa: E402
import json                                                         # noqa: E402
import os                                                           # noqa: E402
import sys                                                          # noqa: E402
from concurrent.futures import ProcessPoolExecutor                  # noqa: E402

import numpy as np                                                  # noqa: E402

from agentsg.cell import strain_closure as sc                       # noqa: E402
from agentsg.cell.metric import params_from_metric                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SELLINGPROOFS_DATA", os.path.join(HERE, "..", "..", "..", "data"))
DB = os.path.join(DATA, "pdb_cells.duckdb")
OUTDIR = os.environ.get("SELLINGPROOFS_OUT", HERE)
OUT = os.path.join(OUTDIR, "pdb_lepage.jsonl")
CCTBX = {"oC": "oS"}
ORDER = ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]
HOLO = {"aP": 2, "mP": 4, "mC": 4, "oP": 8, "oS": 8, "oI": 8, "oF": 8, "tP": 16, "tI": 16, "hR": 12, "hP": 24,
        "cP": 48, "cI": 48, "cF": 48}
HOLOSYM = {"aP": "P -1", "mP": "P 1 2/m 1", "mC": "C 1 2/m 1", "oP": "P m m m", "oS": "C m m m", "oI": "I m m m",
           "oF": "F m m m", "tP": "P 4/m m m", "tI": "I 4/m m m", "hR": "R -3 m :H", "hP": "P 6/m m m",
           "cP": "P m -3 m", "cI": "I m -3 m", "cF": "F m -3 m"}
PLACEHOLDER_EDGE = 2.0          # deposited cells with an edge below 2 A are placeholders, not crystals
SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SLOT_FROM_S6 = (3, 4, 5, 2, 1, 0)
PATTERNS = {name: "".join(pat[SLOT_FROM_S6[k]] for k in range(6)) for name, pat in sc._SORT_PATTERNS.items()}
VPERM = []
for pi in itertools.permutations(range(4)):
    VPERM.append([SL.index(tuple(sorted((pi[i], pi[j])))) for i, j in SL])


def btype(group):
    """Bravais type from a space group in its standard setting."""
    s = str(bravais_types.bravais_lattice(group=group))
    if s[0] == "m" and s[1] != "P":
        return "mC"
    if s[0] == "o" and s[1] in "ABC":
        return "oS"
    return CCTBX.get(s, s)


def G_of(uc):
    m = uc.metrical_matrix()
    return np.array([[m[0], m[3], m[4]], [m[3], m[1], m[5]], [m[4], m[5], m[2]]])


def _match(p, pat, tol):
    vals = {}
    for k, ch in enumerate(pat):
        if ch == "0":
            if abs(p[k]) > tol:
                return False
        else:
            if p[k] <= tol:
                return False
            if ch in vals and abs(vals[ch] - p[k]) > tol:
                return False
            vals.setdefault(ch, p[k])
    return True


def classify(G):
    """the Delone sort of an exact metric: match the conorms of each obtuse superbase of its
    closure, under the 24 relabellings of the vertices, against the 24 patterns."""
    cell = params_from_metric(G.tolist())
    red = sc.reduce_cell(cell)
    tol = 1e-5 * float(np.trace(red.G))
    sups = [np.array(sc._S0, float)]
    p0 = sc.conorms(sups[0], red.G)
    if (np.abs(p0) <= tol).any():                       # other superbases of the closure
        try:
            sups = [np.array(m.superbase, float) for m in sc.widened_closure(cell, eps=1e-6, certify=False).members]
            Gc = np.array(sc.reduce_cell(cell).G)
        except Exception:
            Gc = red.G
    else:
        Gc = red.G
    best = None
    for S in sups:
        p = sc.conorms(S, Gc)
        for perm in VPERM:
            q = [p[perm[k]] for k in range(6)]
            for name, pat in PATTERNS.items():
                if _match(q, pat, tol):
                    key = (len(set(pat) - {"0"}), name)
                    if best is None or key < best:
                        best = key
    return best[1] if best else None


def one(row):
    pid, a, b, c, al, be, ga, sg = row
    try:
        if min(a, b, c) < PLACEHOLDER_EDGE:
            return dict(id=pid, sg=sg, error="placeholder")
        cs = crystal.symmetry(unit_cell=(a, b, c, al, be, ga), space_group_symbol=sg)
        rep = btype(cs.space_group())
        cb1 = cs.change_of_basis_op_to_primitive_setting()
        prim = cs.change_basis(cb1)
        cb2 = prim.change_of_basis_op_to_niggli_cell()
        nig = prim.change_basis(cb2)
        rep_rot = {tuple(int(round(x)) for x in op.r().as_double()) for op in nig.space_group().all_ops()}
        # the reported lattice's own holohedry, from its Bravais type: its operators outside the
        # reported point group are merohedral, not pseudo-symmetries
        cbr = cs.space_group_info().change_of_basis_op_to_reference_setting()
        hol = sgtbx.space_group_info(HOLOSYM[rep]).group().change_basis(cbr.inverse()).change_basis(cb1).change_basis(cb2)
        hol_rot = {tuple(int(round(x)) for x in op.r().as_double()) for op in hol.all_ops()}
        if not (rep_rot <= hol_rot and len(hol_rot) == HOLO[rep]):
            return dict(id=pid, sg=sg, error="holohedry")
        mero = hol_rot - rep_rot
        rep_rot |= mero
        p1 = crystal.symmetry(unit_cell=nig.unit_cell(), space_group_symbol="P1")
        groups = lattice_symmetry.metric_subgroups(input_symmetry=p1, max_delta=3.0).result_groups
        top = groups[0]
        tb = btype(top["best_subsym"].space_group())     # type from the standard setting; ops from subsym
        G = G_of(nig.unit_cell())
        inp = top["best_subsym"].change_basis(top["cb_op_inp_best"].inverse())   # idealised, in the Niggli basis
        out = dict(id=pid, sg=sg, reported=rep, lepage=tb, delta=round(top["max_angular_difference"], 4))
        if mero:
            out["merohedral"] = len(mero)
        if HOLO[tb] > HOLO[rep]:
            prices = collections.defaultdict(list)
            for op in inp.space_group().all_ops():
                R = np.array(op.r().as_double()).reshape(3, 3)
                key = tuple(int(round(x)) for x in R.ravel())
                if np.linalg.det(R) < 0 or key in rep_rot:
                    continue
                n = next(k for k in range(1, 7) if np.allclose(np.linalg.matrix_power(R, k), np.eye(3)))
                prices[str(n)].append(sc._price(G, R.T @ G @ R)[0])
            out["extra"] = {k: [round(x, 5) for x in sorted(v)] for k, v in prices.items()}
            out["sort"] = classify(G_of(inp.unit_cell()))
        return out
    except Exception as e:                                           # noqa: BLE001
        return dict(id=pid, sg=sg, error=type(e).__name__)


def run():
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells").fetchall()
    with ProcessPoolExecutor(max(1, os.cpu_count() or 1)) as ex, open(OUT, "w") as f:
        for i, r in enumerate(ex.map(one, rows, chunksize=200)):
            f.write(json.dumps(r) + "\n")
            if i % 20000 == 0:
                print(i, flush=True)
    table()


def table():
    R = [json.loads(l) for l in open(OUT)]
    ok = [r for r in R if "error" not in r]
    print(f"{len(R)} entries, {len(R) - len(ok)} not analysed")
    conf = collections.Counter((r["reported"], r["lepage"]) for r in ok)
    print("\nreported Bravais type (rows) against the highest type Le Page finds at 3 deg (columns)")
    cols = [t for t in ORDER if any(k[1] == t for k in conf)]
    print(f"{'':5s}" + "".join(f"{t:>7s}" for t in cols) + f"{'n':>8s}{'higher':>8s}")
    for t in ORDER:
        n = sum(v for k, v in conf.items() if k[0] == t)
        if n:
            hi = sum(v for k, v in conf.items() if k[0] == t and HOLO[k[1]] > HOLO[t])
            print(f"{t:5s}" + "".join(f"{conf.get((t, c), 0):7d}" for c in cols) + f"{n:8d}{hi:8d}")
    ps = [r for r in ok if "extra" in r and r["extra"]]
    print(f"\npseudo-symmetric entries (Le Page higher than reported): {len(ps)}")
    print("by the Delone sort of the Le Page lattice: entries; obliquity (deg) and the largest and smallest")
    print("price (%) of the pseudo-symmetry operators, quantiles 10/50/90; median largest price by order")
    q = lambda xs: "/".join(f"{v:.2f}" for v in np.percentile(xs, [10, 50, 90])) if xs else "-"
    for name in sc._SORT_PATTERNS:
        s = [r for r in ps if r.get("sort") == name]
        if not s:
            continue
        mx = [100 * max(x for v in r["extra"].values() for x in v) for r in s]
        mn = [100 * min(x for v in r["extra"].values() for x in v) for r in s]
        by = {k: np.median([100 * max(r["extra"][k]) for r in s if k in r["extra"]]) for k in ("2", "3", "4", "6")
              if any(k in r["extra"] for r in s)}
        print(f"{name:5s} {sc.BRAVAIS[name]:4s} {len(s):6d}  delta {q([r['delta'] for r in s]):>16s}  "
              f"largest {q(mx):>18s}  smallest {q(mn):>18s}  " + ", ".join(f"{k}: {v:.2f}" for k, v in by.items()))
    unk = sum(1 for r in ps if not r.get("sort"))
    if unk:
        print(f"(sort not classified: {unk})")


def redo_zero():
    """Recompute, with the current one(), every entry whose cheapest extra price is 0 (the
    only entries the exclusion of the reported holohedry can change), and rewrite OUT."""
    import duckdb
    R = [json.loads(l) for l in open(OUT)]
    ids = {r["id"] for r in R if r.get("extra") and min(x for v in r["extra"].values() for x in v) < 1e-4}
    con = duckdb.connect(DB, read_only=True)
    rows = [r for r in con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells").fetchall()
            if r[0] in ids]
    with ProcessPoolExecutor(max(1, os.cpu_count() or 1)) as ex:
        new = {r["id"]: r for r in ex.map(one, rows, chunksize=20)}
    with open(OUT, "w") as f:
        for r in R:
            f.write(json.dumps(new.get(r["id"], r)) + "\n")
    print(f"recomputed {len(new)} entries")
    table()


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    elif "--redo-zero" in sys.argv:
        redo_zero()
    else:
        run()
