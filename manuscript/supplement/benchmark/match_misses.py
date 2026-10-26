"""Per-case anatomy of nearest-reference matching (compare_s6.py): which method picks the
wrong reference, in which family, and how far the wrongly chosen reference is from the
true one.

For each measured cell: the reference chosen by CS6Dist, NCDist (Niggli-reduced cells)
and the closure (cheapest price over the 10 CS6Dist-nearest references); the price of
the true operator to the true reference (p_true) and of the chosen reference's
cheapest candidate (p_chosen); the case's own error strain; and, for each wrong choice,
the strain distance between the chosen and the true reference (cheapest operator
between them, in the strain norm at the true reference). Writes match_misses.jsonl and
prints a summary.

    NCDIST_DIR=/path/to/ncdist PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/match_misses.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_s6 as c6                                            # noqa: E402 (imports cctbx first)
import numpy as np                                                 # noqa: E402

from agentsg.cell.strain_closure import band_for, reindex, widened_closure   # noqa: E402


def err_strain(r):
    G = np.array(c6.metric_tensor(tuple(r["reference_cell"])), float)
    GN = np.array(c6.metric_tensor(tuple(r["measured_cell"])), float)
    M = np.array(r["true_operator"], float); Mi = np.linalg.inv(M)
    w, V = np.linalg.eigh(G); Gm = V @ np.diag(w ** -0.5) @ V.T
    return float(np.linalg.norm(Gm @ (Mi.T @ GN @ Mi - G) @ Gm))


def main(band=0.074):
    cs6 = c6.load("libcs6dist.so", "CS6Dist_in_G6")
    ncd = c6.load("libncdist.so", "NCDist")
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    refs, key = [], {}
    for r in rows:
        k = tuple(round(x, 6) for x in r["reference_cell"])
        if k not in key:
            key[k] = len(refs); refs.append(r["reference_cell"])
    G6r = [c6.g6(c) for c in refs]
    N6r = [c6.g6_niggli(c) for c in refs]
    W = {}

    def wid(i):
        if i not in W:
            W[i] = widened_closure(refs[i], eps=band_for(band))
        return W[i]

    W2 = {}

    def ref_dist(i, j):
        """cheapest operator price between reference j and reference i (i widened for 10 %)."""
        if i not in W2:
            W2[i] = widened_closure(refs[i], eps=band_for(0.1))
        c, _ = reindex(W2[i], refs[j], accept=0.1)
        return c[0].price if c else None

    out = []
    for r in rows:
        t = key[tuple(round(x, 6) for x in r["reference_cell"])]
        gN, nN = c6.g6(r["measured_cell"]), c6.g6_niggli(r["measured_cell"])
        dc = np.array([cs6(gN, g) for g in G6r]); dn = np.array([ncd(nN, g) for g in N6r])
        near = list(np.argsort(dc)[:10])
        prices = {}
        for i in near:
            c, _ = reindex(wid(i), r["measured_cell"], accept=band)
            prices[int(i)] = c[0].price if c else None
        ok = {i: p for i, p in prices.items() if p is not None}
        pick = {"CS6Dist": int(np.argmin(dc)), "NCDist": int(np.argmin(dn)),
                "closure": min(ok, key=ok.get) if ok else -1}
        rec = dict(id=r["id"], family=r["family"], sort=r["sort"], err=err_strain(r), true=t,
                   true_rank_cs6=int(np.where(np.argsort(dc) == t)[0][0]) + 1,
                   p_true=prices.get(t), n_accepted=len(ok), pick=pick)
        rec["dist_to_true"] = {m: (0.0 if j == t else (ref_dist(t, j) if j >= 0 else None)) for m, j in pick.items()}
        rec["p_chosen"] = {m: prices.get(j) for m, j in pick.items()}
        out.append(rec)
    with open(os.path.join(HERE, "match_misses.jsonl"), "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    summarise(out)


def summarise(out):
    meths = ["CS6Dist", "NCDist", "closure"]
    print(f"{len(out)} cases\n\nmisses by family")
    fams = sorted({o["family"] for o in out})
    print(f"{'family':16s} {'n':>4s} " + " ".join(f"{m:>8s}" for m in meths))
    for f in fams + ["all"]:
        s = [o for o in out if f in ("all", o["family"])]
        print(f"{f:16s} {len(s):4d} " + " ".join(f"{sum(o['pick'][m] != o['true'] for o in s):8d}" for m in meths))
    print("\noverlap of misses (C = CS6Dist, N = NCDist, X = closure)")
    pat = collections.Counter("".join(l if o["pick"][m] != o["true"] else "." for l, m in zip("CNX", meths)) for o in out)
    for k, v in sorted(pat.items(), key=lambda kv: -kv[1]):
        print(f"  {k}  {v}")
    print("\nfor each method's misses: strain distance between the chosen and the true reference")
    for m in meths:
        d = [o["dist_to_true"][m] for o in out if o["pick"][m] != o["true"]]
        dd = np.array([x for x in d if x is not None])
        far = sum(x is None for x in d)
        if len(dd):
            q = np.percentile(dd, [50, 90, 100])
            print(f"  {m:8s} {len(d):4d} misses; distance median {100*q[0]:.3f}%, 90th {100*q[1]:.3f}%, max {100*q[2]:.3f}%;"
                  f" within the case's own error: {sum(o['dist_to_true'][m] is not None and o['dist_to_true'][m] <= 2 * o['err'] for o in out if o['pick'][m] != o['true'])};"
                  f" no operator within 10%: {far}")
    xs = [o for o in out if o["pick"]["closure"] != o["true"]]
    if xs:
        r = np.array([o["p_true"] - o["p_chosen"]["closure"] for o in xs if o["p_true"] is not None])
        e = np.array([o["err"] for o in xs if o["p_true"] is not None])
        print(f"\nclosure misses: true reference accepted in {sum(o['p_true'] is not None for o in xs)}/{len(xs)};"
              f" price excess of the true reference over the chosen one: median {100*np.median(r):.3f}%,"
              f" max {100*r.max():.3f}%; median case error {100*np.median(e):.3f}%")
        print("closure misses by sort:", collections.Counter(o["sort"] for o in xs).most_common(10))
    cs = [o for o in out if o["pick"]["CS6Dist"] != o["true"]]
    print("CS6Dist misses: rank of the true reference by CS6Dist:",
          collections.Counter(min(o["true_rank_cs6"], 11) for o in cs).most_common())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarise([json.loads(l) for l in open(os.path.join(HERE, "match_misses.jsonl"))])
    else:
        main()
