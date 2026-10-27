"""Which indexing does the cheapest candidate give? For the benchmark (exact references)
and the stress family (errors on both cells, about the exact lattice the reference was
drawn from), the candidates of the closure are grouped by the exact lattice symmetries of
the underlying lattice: two operators are in the same lattice coset when they differ by
an exact symmetry h (h^T G h = G). Reported: how often the cheapest candidate lies in the
true operator's coset (the metric chooses the right lattice coset), and the rank of the
true coset among the cosets ordered by their cheapest member.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/coset_rank.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm                                       # noqa: E402
import numpy as np                                                 # noqa: E402

from agentsg.cell.metric import params_from_metric                 # noqa: E402


def coset_rank(G, Ps, ps, M):
    """rank (1 = cheapest) of the coset of M under the exact symmetries of G."""

    def same(P, Q):
        h = P @ np.linalg.inv(Q)
        return np.allclose(h.T @ G @ h, G, atol=1e-6 * np.abs(G).max())
    reps = []                                   # (cheapest price, representative)
    for P, p in sorted(zip(Ps, ps), key=lambda t: t[1]):
        for r in reps:
            if same(P.astype(float), r[1]):
                break
        else:
            reps.append((p, P.astype(float)))
    for i, (_, r) in enumerate(reps):
        if same(M.astype(float), r):
            return i + 1, len(reps)
    return None, len(reps)


def stress_with_base(rows, seed=11, n_ref=60, ts=(0.5, 0.9)):
    rng = np.random.default_rng(seed)
    refs = [rows[i]["reference_cell"] for i in rng.choice(len(rows), n_ref, replace=False)]
    out = []
    for t in (0.5, 0.9, 1.1, 1.5):                        # same draws as compare_methods.stress_cases
        for cell in refs:
            G = cm.G_of(cell)
            w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
            A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
            XA, XB = (A + A.T) / 2, (B + B.T) / 2
            XA -= np.trace(XA) / 3 * np.eye(3); XB -= np.trace(XB) / 3 * np.eye(3)
            k = t * cm.BAND / np.linalg.norm(XA - XB)
            GR, GN0 = Gh @ (np.eye(3) + k * XA) @ Gh, Gh @ (np.eye(3) + k * XB) @ Gh
            while True:
                U = rng.integers(-2, 3, (3, 3))
                if round(np.linalg.det(U)) == 1:
                    break
            if t in ts:
                out.append({"family": f"stress t={t}", "base": G, "reference_cell": params_from_metric(GR.tolist()),
                            "measured_cell": params_from_metric((U.T @ GN0 @ U).tolist()), "true_operator": U.tolist()})
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    for r in rows:
        r["base"] = cm.G_of(r["reference_cell"])
    cases = rows + stress_with_base(rows)
    agg = collections.defaultdict(lambda: collections.Counter())
    for c in cases:
        M = np.array(c["true_operator"], int)
        if round(np.linalg.det(M)) == -1:
            M = -M
        Ps, ps = cm.closure(c["reference_cell"], c["measured_cell"], cm.BAND)
        rank, n = coset_rank(c["base"], Ps, ps, M)
        fam = c["family"] if c["family"].startswith("stress") else "benchmark " + c["family"]
        for f in (fam, "benchmark all" if not fam.startswith("stress") else "stress all"):
            a = agg[f]
            a["n"] += 1; a["found"] += rank is not None; a["rank1"] += rank == 1
            a["cosets"] += n; a["multi"] += n > 1
    print(f"{'family':28s} {'n':>4s} {'found':>6s} {'true coset cheapest':>20s} {'>1 coset':>9s} {'mean cosets':>11s}")
    for f, a in agg.items():
        print(f"{f:28s} {a['n']:4d} {a['found']:6d} {a['rank1']:20d} {a['multi']:9d} {a['cosets'] / a['n']:11.2f}")


if __name__ == "__main__":
    main()
