"""Controls on the non-isomorphism band of pdb_nonisomorphism.py, and how often the cell
alone ranks a wrong indexing first.

1. The band against the choices behind it: the crystal-form cut (on the unfitted strain,
   5 / 10 / 20 / 50 %), dropping pairs of identical cells, weighting each deposition block
   (a run of id-adjacent pairs) equally, and per reported Bravais type.
2. Shares of PDB entries whose cheapest pseudo-symmetric alternative (pdb_lepage.jsonl)
   lies within the band, with and without the exact cases, for one band and for the band
   of the entry's own Bravais type.
3. Misranking: for a random sample of same-form pairs (cut 20 %), the second cell is
   reindexed against the first with indexing_classes (band 7.4 %, the first cell's space
   group); counted are pairs where a wrong class is strictly cheaper than the identity's
   (split by margin, and by whether that class contains a relabelling of axes, which a
   different setting convention for the two deposits would explain)
   (the cell ranks a wrong indexing first), pairs with a wrong class at the identity's
   price (an exact lattice symmetry outside the Laue group: merohedry, which no cell can
   resolve), and pairs with a priced wrong class within the band (the cell alone does not
   exclude it at that band).

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/pdb_band_controls.py [n_rank]
"""
from __future__ import annotations

import collections
import json
import os
import sys

import duckdb
import numpy as np

from agentsg.cell.metric import metric_tensor
from agentsg.cell import strain_closure as sc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SELLINGPROOFS_DATA", os.path.join(HERE, "..", "..", "..", "data"))
DB = os.path.join(DATA, "pdb_cells.duckdb")


def _signed_permutation(o):
    M = np.array([[float(x) for x in row] for row in o.rows])
    return bool(np.all(np.isin(M, (-1, 0, 1))) and np.all(np.abs(M).sum(0) == 1) and np.all(np.abs(M).sum(1) == 1))


def best_price(c1, c2):
    """the cheapest proper operator between two cells: the identity, or the lookup of U
    between the two Selling-reduced cells (exact below r_1, Remark A8)."""
    U = sc.universal_operators().astype(float)
    r1, r2 = sc.reduce_cell(c1), sc.reduce_cell(c2)
    B1 = np.array([[float(x) for x in r] for r in r1.basis.rows]); B2 = np.array([[float(x) for x in r] for r in r2.basis.rows])
    sgn = np.sign(np.linalg.det(B1) * np.linalg.det(B2))
    Ws = U[np.round(np.linalg.det(U)) == sgn]
    Ps = np.einsum("ij,njk,kl->nil", B1, Ws, np.linalg.inv(B2))
    G1 = np.array(metric_tensor(tuple(c1)), float); G2 = np.array(metric_tensor(tuple(c2)), float)
    w, V = np.linalg.eigh(G2); Gm = V @ np.diag(w ** -0.5) @ V.T
    Y = np.einsum("ij,nkj,kl,nlm,mp->nip", Gm, Ps, G1, Ps, Gm)
    lv = np.log(np.linalg.eigvalsh(Y))
    return float(np.linalg.norm(lv - lv.mean(1, keepdims=True), axis=1).min())


def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(np.array(q) / 100, c, x)


def pairs():
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells order by pdb_id").fetchall()
    rep = {}
    for l in open(os.path.join(HERE, "pdb_lepage.jsonl")):
        r = json.loads(l); rep[r["id"]] = r.get("reported")
    out, block, prev = [], 0, None
    for r1, r2 in zip(rows, rows[1:]):
        if r1[0][:3] != r2[0][:3] or r1[7] != r2[7]:
            prev = None
            continue
        if prev != r1[0]:
            block += 1
        prev = r2[0]
        G1 = np.array(metric_tensor(tuple(r1[1:7])), float)
        G2 = np.array(metric_tensor(tuple(r2[1:7])), float)
        out.append(dict(c1=r1[1:7], c2=r2[1:7], sg=r1[7], bt=rep.get(r1[0]), block=block,
                        fit=sc._price(G2, G1)[0], raw=sc._price(G2, G1, fit_scale=False)[0],
                        same=all(abs(x - y) < 1e-9 for x, y in zip(r1[1:7], r2[1:7]))))
    return out


def main(n_rank=3000):
    P = pairs()
    fit = np.array([p["fit"] for p in P]); raw = np.array([p["raw"] for p in P])
    ident = np.array([p["same"] for p in P]); blk = np.array([p["block"] for p in P])
    print(f"{len(P)} id-adjacent pairs with the same space group\n")
    print("1. band (fitted price, %) against the choices: 50th / 90th / 95th percentile")
    for cut in (0.05, 0.1, 0.2, 0.5):
        m = raw < cut
        print(f"  cut {100 * cut:4.0f} %  n={m.sum():6d}  " + "/".join(f"{100 * x:.2f}" for x in np.percentile(fit[m], [50, 90, 95])))
    m = (raw < 0.2) & ~ident
    print(f"  cut 20 %, identical cells dropped  n={m.sum():6d}  " + "/".join(f"{100 * x:.2f}" for x in np.percentile(fit[m], [50, 90, 95])))
    m = raw < 0.2
    size = collections.Counter(blk[m])
    w = np.array([1 / size[b] for b in blk[m]])
    print(f"  cut 20 %, each block weighted equally ({len(size)} blocks, largest {max(size.values())} pairs)  "
          + "/".join(f"{100 * x:.2f}" for x in wq(fit[m], w, [50, 90, 95])))
    print("  by reported Bravais type (cut 20 %):")
    band_bt = {}
    for bt in ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]:
        mb = m & np.array([p["bt"] == bt for p in P])
        if mb.sum() < 30:
            continue
        q = np.percentile(fit[mb], [50, 90, 95]); band_bt[bt] = q[1:]
        print(f"    {bt}  n={mb.sum():6d}  " + "/".join(f"{100 * x:.2f}" for x in q))
    print("  with the cheapest operator instead of the identity (lookup of U), by reported type (cut 20 %):")
    for bt in ["aP", "mP", "mC", "oP"]:
        mb = np.where(m & np.array([p["bt"] == bt for p in P]))[0]
        best = np.array([min(fit[i], best_price(P[i]["c1"], P[i]["c2"])) for i in mb])
        print(f"    {bt}  n={len(mb):6d}  " + "/".join(f"{100 * x:.2f}" for x in np.percentile(best, [50, 90, 95]))
              + f"   (identity: " + "/".join(f"{100 * x:.2f}" for x in np.percentile(fit[mb], [50, 90, 95])) + ")")
    N90, N95 = np.percentile(fit[m], [90, 95])

    print("\n2. entries whose cheapest pseudo-symmetric alternative lies within the band")
    R = [json.loads(l) for l in open(os.path.join(HERE, "pdb_lepage.jsonl"))]
    R = [r for r in R if "error" not in r]
    tot = len(R)
    mn = {r["id"]: min(x for v in r["extra"].values() for x in v) for r in R if r.get("extra")}
    ex = sum(1 for v in mn.values() if v < 1e-4)
    pr = np.array([v for v in mn.values() if v >= 1e-4])
    for lab, b in (("90th", N90), ("95th", N95)):
        print(f"  one band, {lab} percentile {100 * b:.1f} %: priced {100 * (pr <= b).sum() / tot:.1f} %, "
              f"with the {ex} exact cases {100 * ((pr <= b).sum() + ex) / tot:.1f} % of {tot}")
    for k, lab in ((0, "90th"), (1, "95th")):
        hit = sum(1 for r in R if r["id"] in mn and r["reported"] in band_bt and mn[r["id"]] <= band_bt[r["reported"]][k])
        print(f"  band of the entry's own Bravais type, {lab} percentile: {100 * hit / tot:.1f} % (exact cases included)")

    print(f"\n3. misranking on {n_rank} same-form pairs (cut 20 %), band {100 * sc.DEFAULT_ACCEPT:.1f} %")
    rng = np.random.default_rng(4)
    idx = rng.choice(np.where(m)[0], min(n_rank, m.sum()), replace=False)
    c = collections.Counter()
    rec = []                                     # (block, strictly cheaper, cheaper otherwise) per scored pair
    eye = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for i in idx:
        p = P[i]
        try:
            W = sc.widened_closure(p["c1"], p["sg"])
            res = sc.indexing_classes(W, p["c2"], p["sg"])
        except (ValueError, KeyError, RuntimeError):
            c["refused"] += 1
            continue
        c["n"] += 1
        rec.append([P[i]["block"], 0, 0])
        key = lambda o: tuple(tuple(int(x) for x in row) for row in o.rows)      # noqa: E731
        idc = [k for k in res.classes if any(key(o) == eye for o in k.operators)]
        if not idc:
            c["identity beyond the band"] += 1
            if res.classes:
                c["  of which a wrong class within the band"] += 1
            continue
        p_id = min(x.price for x in [res.classes[res.classes.index(idc[0])].representative])
        others = [k.representative.price for k in res.classes if k is not idc[0]]
        if others and min(others) < p_id - 1e-6:
            c["wrong class strictly cheaper"] += 1
            rec[-1][1] = 1
            k0 = min((k for k in res.classes if k is not idc[0]), key=lambda k: k.representative.price)
            if p_id - k0.representative.price <= 5e-4:
                c["  by at most 0.05 points (below the precision of a deposited cell)"] += 1
            elif any(_signed_permutation(o) for o in k0.operators):
                c["  by more, through a relabelling of axes (a setting convention)"] += 1
            else:
                c["  by more, otherwise"] += 1
                rec[-1][2] = 1
        if any(abs(x - p_id) <= 1e-6 for x in others):
            c["a wrong class at the same price (exact lattice symmetry)"] += 1
        if any(x > p_id + 1e-6 for x in others):
            c["a priced wrong class within the band"] += 1
    n = c["n"]
    for k in ("wrong class strictly cheaper", "  by at most 0.05 points (below the precision of a deposited cell)",
              "  by more, through a relabelling of axes (a setting convention)", "  by more, otherwise",
              "a wrong class at the same price (exact lattice symmetry)",
              "a priced wrong class within the band", "identity beyond the band",
              "  of which a wrong class within the band"):
        print(f"  {k}: {c[k]} ({100 * c[k] / n:.2f} %)")
    print(f"  pairs scored {n}, refused {c['refused']}")
    blocks = collections.defaultdict(list)
    for b, x, y in rec:
        blocks[b].append((x, y))
    keys = list(blocks)
    brng = np.random.default_rng(9)
    boots = []
    for _ in range(2000):                        # resample deposition blocks
        pick = brng.choice(len(keys), len(keys))
        v = [t for k in pick for t in blocks[keys[k]]]
        boots.append((np.mean([t[0] for t in v]), np.mean([t[1] for t in v])))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"  block-bootstrap 95 % intervals ({len(keys)} blocks): strictly cheaper {100 * lo[0]:.2f}-{100 * hi[0]:.2f} %, "
          f"by more and not a relabelling {100 * lo[1]:.2f}-{100 * hi[1]:.2f} %")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
