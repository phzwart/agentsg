"""Compare the widened closure with the reindexing methods data-processing programs use,
on benchmark.jsonl and on a stress family with errors near and above the band.

Methods (each returns proper candidate operators P, reference input basis -> new input
basis, G_N ~ P^T G_R P, priced at G_N after the best scale; accepted at price <= band):

  closure     agentsg.cell.strain_closure.reindex (reference widened at band/(1-band))
  small1      Selling-reduce both cells; every unimodular matrix with entries in {-1,0,1}
              between the reduced bases (6960)
  small2      the same with entries in {-2,...,2}
  cctbx_sim1  cctbx: Niggli-reduce both cells, unit_cell.similarity_transformations with
              unimodular_generator_range=1 (relative length 0.05, angle 3 deg)
  cctbx_sim2  the same with unimodular_generator_range=2
  pipeline    cctbx lattice_symmetry (Le Page, max delta 3 deg) on both cells; take the
              reference's highest-symmetry lattice group, the same group type for the new
              cell, and every operation of that group between the two standard settings
              (the symmetry-then-coset route of standard pipelines)

Scored per case: whether the true operator (made proper: -M when det M = -1) is among
the candidates, the number of candidates, the rank of the true operator by price (1 =
cheapest), and whether a wrong operator is strictly cheaper.

    PYTHONPATH=agentsg/src python manuscript/benchmark/compare_methods.py [--stress]
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import sys
import time

from cctbx import crystal, uctbx                                  # noqa: E402 (before numpy users)
from cctbx.sgtbx import lattice_symmetry                          # noqa: E402
import numpy as np                                                # noqa: E402

from agentsg.cell.metric import metric_tensor, params_from_metric   # noqa: E402
from agentsg.cell.strain_closure import band_for, reduce_cell, reindex, widened_closure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BAND = 0.074


def G_of(cell):
    return np.array(metric_tensor(tuple(cell)), float)


def _whiten(GN):
    w, V = np.linalg.eigh(GN)
    return V @ np.diag(w ** -0.5) @ V.T


def prices(Ps, GR, GN):
    """strain price after the best scale, for a stack of operators (n,3,3)."""
    Gm = _whiten(GN)
    Y = np.einsum("ij,njk,kl,nlm,mp->nip", Gm, np.transpose(Ps, (0, 2, 1)), GR, Ps, Gm)
    tr = np.trace(Y, axis1=1, axis2=2)
    yy = np.einsum("nij,nij->n", Y, Y)
    s = tr / yy
    R = s[:, None, None] * Y - np.eye(3)[None]
    return np.sqrt(np.einsum("nij,nij->n", R, R))


def _unimodular(k):
    rng = range(-k, k + 1)
    out = []
    for m in itertools.product(rng, repeat=9):
        A = np.array(m).reshape(3, 3)
        if round(np.linalg.det(A)) == 1:
            out.append(A)
    return np.array(out, float)


SMALL = {}


def small_search(ref_cell, new_cell, k, band):
    if k not in SMALL:
        SMALL[k] = _unimodular(k)
    r, n = reduce_cell(ref_cell), reduce_cell(new_cell)
    BR = np.array([[float(x) for x in row] for row in r.basis.rows])
    BN = np.array([[float(x) for x in row] for row in n.basis.rows])
    Ps = np.einsum("ij,njk,kl->nil", BR, SMALL[k], np.linalg.inv(BN))
    if round(np.linalg.det(BR) * np.linalg.det(BN)) < 0:               # keep the products proper
        Ps = -Ps
    p = prices(Ps, G_of(ref_cell), G_of(new_cell))
    keep = p <= band
    return np.round(Ps[keep]).astype(int), p[keep]


def _cb_matrix(cb):
    """basis transform X of a cctbx change-of-basis op: G_new = X^T G X."""
    return np.linalg.inv(np.array(cb.c().r().as_double()).reshape(3, 3))


def cctbx_similarity(ref_cell, new_cell, k, band):
    out = []
    csR = crystal.symmetry(unit_cell=tuple(ref_cell), space_group_symbol="P1")
    csN = crystal.symmetry(unit_cell=tuple(new_cell), space_group_symbol="P1")
    cbR, cbN = csR.change_of_basis_op_to_niggli_cell(), csN.change_of_basis_op_to_niggli_cell()
    uR, uN = csR.change_basis(cbR).unit_cell(), csN.change_basis(cbN).unit_cell()
    XR, XN = _cb_matrix(cbR), _cb_matrix(cbN)
    for m in uR.similarity_transformations(other=uN, relative_length_tolerance=0.05,
                                           absolute_angle_tolerance=3.0, unimodular_generator_range=k):
        A = np.array(m, float).reshape(3, 3)
        for S in (A, np.linalg.inv(A)):                                 # convention-free: keep whichever fits
            P = XR @ S @ np.linalg.inv(XN)
            if round(np.linalg.det(P)) == -1:
                P = -P
            out.append(P)
    if not out:
        return np.zeros((0, 3, 3), int), np.zeros(0)
    Ps = np.array(out)
    p = prices(Ps, G_of(ref_cell), G_of(new_cell))
    keep = p <= band
    return np.round(Ps[keep]).astype(int), p[keep]


def _niggli(cell):
    cs = crystal.symmetry(unit_cell=tuple(cell), space_group_symbol="P1")
    cb = cs.change_of_basis_op_to_niggli_cell()
    return cs.change_basis(cb), _cb_matrix(cb)


def pipeline(ref_cell, new_cell, band):
    csR, X0R = _niggli(ref_cell)
    csN, X0N = _niggli(new_cell)
    gR = lattice_symmetry.metric_subgroups(input_symmetry=csR, max_delta=3.0).result_groups
    gN = lattice_symmetry.metric_subgroups(input_symmetry=csN, max_delta=3.0).result_groups
    top = gR[0]
    sym = str(top["ref_subsym"].space_group_info())
    match = [g for g in gN if str(g["ref_subsym"].space_group_info()) == sym]
    if not match:
        return np.zeros((0, 3, 3), int), np.zeros(0)
    XR = X0R @ _cb_matrix(top["cb_op_inp_best"])
    XN = X0N @ _cb_matrix(match[0]["cb_op_inp_best"])
    out = []
    for op in top["best_subsym"].space_group().all_ops():
        W = np.array(op.r().as_double()).reshape(3, 3)
        for S in (W, np.linalg.inv(W).T):
            P = XR @ S @ np.linalg.inv(XN)
            if round(np.linalg.det(P)) == -1:
                P = -P
            out.append(P)
    Ps = np.array(out)
    p = prices(Ps, G_of(ref_cell), G_of(new_cell))
    keep = p <= band
    return np.round(Ps[keep]).astype(int), p[keep]


def closure(ref_cell, new_cell, band):
    ref = widened_closure(ref_cell, eps=band_for(band))
    cands, _ = reindex(ref, new_cell, accept=band)
    Ps = np.array([[[int(x) for x in row] for row in c.operator.rows] for c in cands]).reshape(-1, 3, 3)
    return Ps, np.array([c.price for c in cands])


METHODS = {
    "closure": closure,
    "small1": lambda r, n, b: small_search(r, n, 1, b),
    "small2": lambda r, n, b: small_search(r, n, 2, b),
    "cctbx_sim1": lambda r, n, b: cctbx_similarity(r, n, 1, b),
    "cctbx_sim2": lambda r, n, b: cctbx_similarity(r, n, 2, b),
    "pipeline": pipeline,
}


def score(Ps, ps, M):
    uniq = {}
    for P, p in zip(Ps, ps):
        k = tuple(P.ravel())
        uniq[k] = min(p, uniq.get(k, 9))
    keys = sorted(uniq, key=uniq.get)
    t = tuple(M.ravel())
    if t not in uniq:
        return False, len(keys), None, None
    rank = 1 + sum(uniq[k] < uniq[t] - 1e-12 for k in keys)
    return True, len(keys), rank, rank > 1


def run(cases, band, label):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    timing = collections.Counter()
    for c in cases:
        M = np.array(c["true_operator"], int)
        if round(np.linalg.det(M)) == -1:
            M = -M
        for name, f in METHODS.items():
            t0 = time.time()
            try:
                Ps, ps = f(c["reference_cell"], c["measured_cell"], band)
            except RuntimeError:                                         # cctbx refuses the cell
                Ps, ps = np.zeros((0, 3, 3), int), np.zeros(0)
                agg[(c.get("family", label), name)]["error"].append(1)
            timing[name] += time.time() - t0
            ok, n, rank, wrong = score(Ps, ps, M)
            a = agg[(c.get("family", label), name)]
            a["ok"].append(ok); a["n"].append(n)
            if ok:
                a["rank1"].append(rank == 1); a["wrong"].append(wrong)
    fams = sorted({k[0] for k in agg})
    print(f"\n{label}: band {band}, {len(cases)} cases")
    print(f"{'family':16s} {'method':11s} {'found':>9s} {'cands q50/q90':>14s} {'true cheapest':>14s}")
    for fam in fams + ["all"]:
        for name in METHODS:
            if fam == "all":
                a = collections.defaultdict(list)
                for (f, m), v in agg.items():
                    if m == name:
                        for kk, vv in v.items():
                            a[kk] += vv
            else:
                a = agg[(fam, name)]
            n = len(a["ok"])
            q = np.percentile(a["n"], [50, 90]).astype(int)
            r1 = sum(a["rank1"])
            print(f"{fam:16s} {name:11s} {sum(a['ok']):4d}/{n:<4d} {q[0]:6d}/{q[1]:<7d} {r1:6d}/{sum(a['ok']):<6d}")
    print("seconds per case: " + ", ".join(f"{k} {v / len(cases):.2f}" for k, v in timing.items()))


def stress_cases(rows, seed=11, n_ref=60, ts=(0.5, 0.9, 1.1, 1.5)):
    """errors on both cells, scaled so the true operator's price is about t x band."""
    rng = np.random.default_rng(seed)
    refs = [rows[i]["reference_cell"] for i in rng.choice(len(rows), n_ref, replace=False)]
    out = []
    for t in ts:
        for cell in refs:
            G = G_of(cell)
            w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
            A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
            XA, XB = (A + A.T) / 2, (B + B.T) / 2
            XA -= np.trace(XA) / 3 * np.eye(3); XB -= np.trace(XB) / 3 * np.eye(3)
            k = t * BAND / np.linalg.norm(XA - XB)
            GR, GN0 = Gh @ (np.eye(3) + k * XA) @ Gh, Gh @ (np.eye(3) + k * XB) @ Gh
            while True:
                U = rng.integers(-2, 3, (3, 3))
                if round(np.linalg.det(U)) == 1:
                    break
            out.append({"family": f"t={t}", "reference_cell": params_from_metric(GR.tolist()),
                        "measured_cell": params_from_metric((U.T @ GN0 @ U).tolist()), "true_operator": U.tolist()})
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(HERE, "benchmark.jsonl"))]
    if "--stress" in sys.argv:
        run(stress_cases(rows), BAND, "stress: errors on both cells, true price ~ t x band")
    else:
        run(rows, BAND, "benchmark")
