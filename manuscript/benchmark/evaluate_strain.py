"""Score the strain-norm procedure (agentsg.cell.strain_closure) on benchmark.jsonl.

For each case the reference is widened at a band eps (strain norm), the measured cell
is reduced once and paired with the members (reindex); success when the true
operator, or its negative, is among the candidates. Two bands: Le Page's 3 degrees
(one angle, strain sqrt2 sin 3deg = 0.074) and 3 x the strain of the case's own
metric error (eps_err = || G^-1/2 E G^-1/2 ||_F, E from the true operator), floored
at 1e-4. Also reported: the tier, the number of candidates and of distinct lattice
classes at the Le Page band.

    PYTHONPATH=agentsg/src python manuscript/benchmark/evaluate_strain.py [benchmark.jsonl]
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np

from agentsg.cell.strain_closure import LEPAGE_STRAIN, reindex, widened_closure
from agentsg.cell.metric import metric_tensor

HERE = os.path.dirname(os.path.abspath(__file__))


def G_of(cell):
    return np.array(metric_tensor(tuple(cell)), float)


def strain(G, D):
    w, V = np.linalg.eigh(G)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    return float(np.linalg.norm(Gm @ D @ Gm))


def main(path):
    rows = collections.OrderedDict()
    t0 = time.time()
    for line in open(path):
        c = json.loads(line)
        M = np.array(c["true_operator"], float)
        GR, GN = G_of(c["reference_cell"]), G_of(c["measured_cell"])
        Mi = np.linalg.inv(M)
        E = Mi.T @ GN @ Mi - GR
        e_err = strain(GR, E)
        res = {}
        for name, eps in (("lepage", LEPAGE_STRAIN), ("3err", max(3 * e_err, 1e-4))):
            if eps >= 0.25:                                   # outside the proved range (Theorem A7 needs eps < 1/4)
                res[name] = (None, None, 0, 0)
                continue
            ref = widened_closure(c["reference_cell"], eps=eps)
            cands, tier = reindex(ref, c["measured_cell"], eps=eps)
            ok = any(np.allclose(np.array([[float(x) for x in r] for r in k.operator.rows]), s * M)
                     for k in cands for s in (1, -1))
            res[name] = (ok, tier, len(cands), len(ref))
        rows.setdefault(c["family"], []).append((c["id"], e_err, res))
    print(f"{'family':16s} {'n':>4s} {'ok 3deg':>9s} {'ok 3err':>9s} {'tier1/2/3':>11s} {'median cands':>13s} {'median eps_err':>15s}")
    tot = collections.Counter()
    for fam, rs in rows.items():
        n = len(rs)
        ok1 = sum(bool(r[2]["lepage"][0]) for r in rs); ok2 = sum(bool(r[2]["3err"][0]) for r in rs)
        big = sum(r[2]["3err"][0] is None for r in rs)
        tiers = collections.Counter(r[2]["lepage"][1] for r in rs)
        mc = int(np.median([r[2]["lepage"][2] for r in rs]))
        me = float(np.median([r[1] for r in rs]))
        tot.update(n=n, ok1=ok1, ok2=ok2)
        print(f"{fam:16s} {n:4d} {ok1:4d}/{n:<4d} {ok2:4d}/{n - big:<4d} {tiers[1]:3d}/{tiers[2]:d}/{tiers[3]:d}  {mc:13d} {me:15.2e}" + (f"  ({big} with 3 eps_err >= 1/4)" if big else ""))
        for r in rs:
            if r[2]["lepage"][0] is False or r[2]["3err"][0] is False:
                print("   miss", r[0], r[2])
    print(f"{'all':16s} {tot['n']:4d} {tot['ok1']:4d}/{tot['n']:<4d} {tot['ok2']:4d}/{tot['n']:<4d}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "benchmark.jsonl"))
