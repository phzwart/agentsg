"""Score the strain-norm procedure (agentsg.cell.strain_closure) on benchmark.jsonl,
against a small-matrix baseline.

For each case the reference is widened at band_for(accept) in the strain norm, the
measured cell is reduced once and paired with the members (reindex); success when the
true operator, made proper (-M when det M = -1, which differs by the inversion), is
itself among the proper candidates. Two acceptance bands: 0.074
(one cell angle changed by 3 degrees) and 3 x the strain of the case's own metric
error (eps_err = || G^-1/2 E G^-1/2 ||_F, E in the reference's reduced frame), floored
at 1e-4 and skipped above 0.2.

Baseline ("small"): Selling-reduce both cells, try every unimodular matrix with entries
in {-1, 0, 1} between the two reduced bases (6960 of them), price each at the measured
metric after the best scale, keep those within the band. This is what a search over
small integer matrices between reduced cells, as data-processing programs do, sees.

Also reported: the adjacency layer, quantiles of the candidate count, whether the
band's self-maps of the reference form a group (closed), and the number of distinct
classes of the reference's lattice point group among the candidates.

    PYTHONPATH=agentsg/src python manuscript/benchmark/evaluate_strain.py [benchmark.jsonl]
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import sys
import time

import numpy as np

from agentsg.cell.metric import metric_tensor
from agentsg.cell.strain_closure import (
    LEPAGE_STRAIN, _strain, band_for, reduce_cell, reindex, widened_closure,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def G_of(cell):
    return np.array(metric_tensor(tuple(cell)), float)


def strain(G, D):
    w, V = np.linalg.eigh(G)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    return float(np.linalg.norm(Gm @ D @ Gm))


SMALL = [np.array(m).reshape(3, 3) for m in itertools.product((-1, 0, 1), repeat=9)]
SMALL = [m for m in SMALL if abs(round(np.linalg.det(m))) == 1]          # 6960 matrices


def small_baseline(ref_cell, new_cell, accept):
    """candidate operators (reference input basis -> new input basis) from the 6960
    small matrices between the two reduced bases, priced at the new metric."""
    r, n = reduce_cell(ref_cell), reduce_cell(new_cell)
    GR = np.array([[float(x) for x in row] for row in r.basis.rows])
    BN = np.array([[float(x) for x in row] for row in n.basis.rows])
    G_R_in, G_N_in = G_of(ref_cell), G_of(new_cell)
    out = []
    BNi = np.linalg.inv(BN)
    for m in SMALL:
        P = GR @ m @ BNi
        price, _ = _strain(G_N_in, P.T @ G_R_in @ P)
        if price <= accept:
            P = np.round(P).astype(int)
            out.append(P if round(np.linalg.det(P)) == 1 else -P)
    return out


def has(ops, M):
    return any(np.array_equal(o, M) for o in ops)


def main(path):
    rows = collections.OrderedDict()
    t0 = time.time()
    for line in open(path):
        c = json.loads(line)
        M = np.array(c["true_operator"], int)
        M = -M if round(np.linalg.det(M)) == -1 else M
        GR, GN = G_of(c["reference_cell"]), G_of(c["measured_cell"])
        Mi = np.linalg.inv(M)
        E = Mi.T @ GN @ Mi - GR
        e_err = strain(GR, E)
        res = {}
        for name, acc in (("3deg", LEPAGE_STRAIN), ("3err", max(3 * e_err, 1e-4))):
            if acc >= 0.2:
                res[name] = None
                continue
            ref = widened_closure(c["reference_cell"], eps=band_for(acc))
            cands, layer = reindex(ref, c["measured_cell"], accept=acc)
            ops = [np.array([[int(x) for x in row] for row in k.operator.rows]) for k in cands]
            ok = has(ops, M)
            small = small_baseline(c["reference_cell"], c["measured_cell"], acc)
            res[name] = dict(ok=ok, layer=layer, n=len(cands), members=len(ref), small_ok=has(small, M),
                             small_n=len(small))
        rows.setdefault(c["family"], []).append((c["id"], e_err, res))
    hdr = f"{'family':15s} {'n':>4s} | {'closure 3deg':>12s} {'small 3deg':>10s} | {'closure 3err':>12s} {'small 3err':>10s} | {'layer 1/2/3':>11s} {'cands q10/50/90':>16s} {'median err':>10s}"
    print(hdr)
    tot = collections.Counter()
    misses = []
    for fam, rs in rows.items():
        n = len(rs)
        a = [r[2]["3deg"] for r in rs]
        b = [r[2]["3err"] for r in rs if r[2]["3err"] is not None]
        ok1, s1 = sum(x["ok"] for x in a), sum(x["small_ok"] for x in a)
        ok2, s2 = sum(x["ok"] for x in b), sum(x["small_ok"] for x in b)
        lay = collections.Counter(x["layer"] for x in a)
        q = np.percentile([x["n"] for x in a], [10, 50, 90]).astype(int)
        me = float(np.median([r[1] for r in rs]))
        tot.update(n=n, ok1=ok1, s1=s1, ok2=ok2, s2=s2, n2=len(b))
        print(f"{fam:15s} {n:4d} | {ok1:5d}/{n:<6d} {s1:4d}/{n:<5d} | {ok2:5d}/{len(b):<6d} {s2:4d}/{len(b):<5d} | "
              f"{lay[1]:3d}/{lay[2]:d}/{lay[3]:<4d} {q[0]:4d}/{q[1]:d}/{q[2]:<7d} {me:10.2e}")
        for r in rs:
            for k in ("3deg", "3err"):
                x = r[2][k]
                if x is not None and not (x["ok"] and x["small_ok"]):
                    misses.append((r[0], k, x))
    print(f"{'all':15s} {tot['n']:4d} | {tot['ok1']:5d}/{tot['n']:<6d} {tot['s1']:4d}/{tot['n']:<5d} | "
          f"{tot['ok2']:5d}/{tot['n2']:<6d} {tot['s2']:4d}/{tot['n2']:<5d}   ({time.time() - t0:.0f} s)")
    for m in misses:
        print("   miss", m)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "benchmark.jsonl"))
