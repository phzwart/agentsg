"""The reindexing workflow of the supporting information (Section S10): a worked example
step by step, and the comparison with the direct form on the benchmark and on centred
references.

The direct form builds the exact operator P = B_R W Q B_N^-1 for every member W of the
reference's widened set and every relabelling Q, and prices each.  The workflow tabulates
the members once as metrics, compares the new cell with them in its own reduced frame
(where the input bases drop out), uses one scale for all candidates, discards orderings by
a necessary condition on the dot products of the new reduced superbase (Lemma A7(i) with
Proposition A4), prices the rest in one batch and builds the exact operator only for the
candidates within the band.  Both return the same set of operators; the prices agree to
rounding.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/reindex_workflow.py

Writes results_reindex_workflow.txt.  Lines containing 'seconds' report wall-clock times
and are not reproducible.
"""
import json
import os
import time

import numpy as np

from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.selling_cob import primitive_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SELLINGPROOFS_OUT", HERE)

WORKED = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)
P_TRUE = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])          # the new basis (a, b, a + c)
CENTRED = [((50.0, 30.0, 40.0, 90, 105, 90), "C 1 2 1"), ((30.0, 40.0, 50.0, 90, 90, 90), "I 2 2 2"),
           ((60.0, 60.0, 90.0, 90, 90, 120), "R 3 :H"), ((80.0, 80.0, 80.0, 90, 90, 90), "F 2 3"),
           ((40.0, 40.0, 70.0, 90, 90, 90), "I 41"), ((30.0, 50.0, 80.0, 90, 90, 90), "F d d d")]


def strained(G, eta, rng):
    w, V = np.linalg.eigh(G)
    Gh = V @ np.diag(w ** .5) @ V.T
    X = rng.normal(size=(3, 3)); X = X + X.T; X *= eta / np.linalg.norm(X)
    return Gh @ (np.eye(3) + X) @ Gh


def fmt(M):
    return "(" + ", ".join("(" + ", ".join(str(x) for x in row) + ")" for row in M) + ")"


def same(a, b, tol=1e-9):
    A = {tuple(c.operator.rows): c.price for c in a}
    B = {tuple(c.operator.rows): c.price for c in b}
    return set(A) == set(B) and all(abs(A[k] - B[k]) <= tol for k in A), \
        max([abs(A[k] - B[k]) for k in A if k in B] or [0.0])


def worked_example(lines):
    rng = np.random.default_rng(1)
    ref = sc.widened_closure(WORKED)
    G = np.array(UnitCell(*WORKED).metric_tensor(), float)
    new = params_from_metric((P_TRUE.T @ strained(G, 0.003, rng) @ P_TRUE).tolist())
    wf = sc.reindex_workflow(ref, new)
    lines += ["worked example",
              f"  reference cell {WORKED}",
              f"  new cell {tuple(round(x, 4) for x in new)}  (the reference in the basis (a, b, a+c) with a strain of 0.3 %)",
              f"  band {wf.accept:.4f}  eta {wf.eta:.6f}  layer {wf.layer}",
              f"  B_R {fmt(wf.basis_reference.rows)}  det {wf.basis_reference.det()}",
              f"  B_N {fmt(wf.basis_new.rows)}  det {wf.basis_new.det()}",
              "  reference conorms (da db dc ab ac bc) " + " ".join(f"{x:.2f}" for x in wf.reference_conorms),
              "  new conorms (da db dc ab ac bc) " + " ".join(f"{x:.2f}" for x in wf.new_conorms),
              f"  members {wf.n_members}  ordered bases {wf.n_orderings}  with the proper sign {wf.n_signed}",
              f"  common scale s {wf.scale:.6f}  edge scale {wf.edge_scale:.6f}  within the limit {wf.scale_ok}",
              "  bounds eta |u_i||u_j| (rows d a b c) " + "; ".join(" ".join(f"{x:.1f}" for x in r) for r in wf.bounds),
              f"  pass the necessary condition {wf.n_prefilter} of {wf.n_orderings}",
              f"  priced {wf.n_priced}  within the band {wf.n_within}  distinct operators {wf.n_candidates}",
              "  priced orderings (member, ordering, sign, price %): "
              + "; ".join(f"{k} {o} {c:+d} {100 * p:.3f}" for k, o, c, p in wf.priced)]
    for c in wf.candidates:
        lines.append(f"  candidate P {fmt(c.operator.rows)}  price {100 * c.price:.3f} %  scale {c.scale:.6f}  "
                     f"member depth {100 * c.depth:.3f} %")
    direct, _ = sc._reindex_direct(ref, new)
    ok, dmax = same(direct, wf.candidates)
    lines.append(f"  direct form: {len(direct)} candidates, the same set {ok}, largest price difference {dmax:.1e}")
    lines.append(f"  direct form builds {48 * wf.n_members} exact operators; the workflow {wf.n_within}")
    lines.append("  seconds per step " + " ".join(f"{k} {v:.5f}" for k, v in wf.seconds.items()))
    # every member's conorms in its first ordering, for the table of the supporting information
    tab = ref._table["workflow"]
    for k, m in enumerate(ref.members):
        p = sc.conorms(np.array(sc._S0, float), tab["G"][k, 0])
        lines.append(f"  member {k:2d} depth {100 * m.depth:.3f} % conorms " + " ".join(f"{x:.1f}" for x in p))
    lines.append("")


def benchmark(lines):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    n_same = 0; dmax = 0.0; t_direct = t_flow = 0.0
    orderings, passed, within, built, members, distinct = [], [], [], [], [], []
    for r in rows:
        ref = sc.widened_closure(r["reference_cell"])
        t = time.perf_counter(); a, _ = sc._reindex_direct(ref, r["measured_cell"]); t_direct += time.perf_counter() - t
        t = time.perf_counter(); b, _ = sc.reindex(ref, r["measured_cell"]); t_flow += time.perf_counter() - t
        wf = sc.reindex_workflow(ref, r["measured_cell"])
        ok, d = same(a, b)
        n_same += ok; dmax = max(dmax, d)
        orderings.append(wf.n_orderings); passed.append(wf.n_prefilter); within.append(wf.n_within)
        built.append(48 * wf.n_members); members.append(wf.n_members); distinct.append(wf.n_candidates)
    o, p, w = np.array(orderings), np.array(passed), np.array(within)
    lines += ["benchmark (780 cases)",
              f"  the same set of operators in {n_same} of {len(rows)} cases; largest price difference {dmax:.1e}",
              f"  members: median {np.median(members):.0f}, max {max(members)}",
              f"  ordered bases per case: median {np.median(o):.0f}, max {o.max()}",
              f"  passing the necessary condition: median {np.median(p):.0f}, max {p.max()}; "
              f"{100 * p.sum() / o.sum():.2f} % of all ordered bases",
              f"  priced within the band: median {np.median(w):.0f}, max {w.max()}",
              f"  exact operators built: direct form {sum(built)}, workflow {int(w.sum())}; "
              f"distinct operators returned {sum(distinct)}",
              f"  seconds: direct form {t_direct:.2f}, workflow {t_flow:.2f} (factor {t_direct / t_flow:.0f})", ""]


def centred(lines):
    rng = np.random.default_rng(7)
    n = n_same = 0
    for cell, sg in CENTRED:
        ref = sc.widened_closure(cell, sg)
        Gc = np.array(UnitCell(*cell).metric_tensor(), float)
        C = np.array([[float(x) for x in row] for row in primitive_matrix(sg).rows])
        for _ in range(4):
            while True:
                U = rng.integers(-2, 3, (3, 3))
                if round(abs(np.linalg.det(U))) == 1:
                    break
            P = C @ U
            new = params_from_metric(strained(P.T @ Gc @ P, rng.uniform(0.002, 0.03), rng).tolist())
            a, _ = sc._reindex_direct(ref, new, None)
            b, _ = sc.reindex(ref, new, None)
            n += 1; n_same += same(a, b)[0]
    lines += ["centred references (C 1 2 1, I 2 2 2, R 3 :H, F 2 3, I 41, F d d d; 4 new cells each)",
              f"  the same set of operators in {n_same} of {n} cases", ""]


if __name__ == "__main__":
    lines = []
    worked_example(lines)
    centred(lines)
    benchmark(lines)
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "results_reindex_workflow.txt"), "w").write(txt + "\n")
