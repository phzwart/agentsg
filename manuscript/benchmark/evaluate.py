"""Score methods on benchmark.jsonl.

Reindexing task: from the measured cell alone (one Selling reduction) and
the reference, propose integer operators; success when the true operator
(or its negative) is among them. Set task: the widened closure of the
reference at tau, compared with the brute-force count stored in the case.

Methods
  symmetry   pair the measured reduced cell with the reference's reduced cell
             in its 48 labellings (what a symmetry-only reindexing can see)
  exact      pair with every member of the reference's exact closure
  walk-tau   members found by flipping wherever |p| <= tau from the reference's
             reduced cell, filtered at -tau
  walk-2tau  the same with |p| <= 2 tau
  closure    idealise at 4 tau, enumerate, filter (the paper's procedure)
  slot       the same with a band per slot, sigma |u_i||u_j|, for the cases
             that carry an angular error (the anisotropy family); otherwise
             identical to closure

    python evaluate.py [benchmark.jsonl]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "appendix_z"))
import test_z44_widened_complete as t  # noqa: E402

PERM4 = list(itertools.permutations(range(4)))


def walk(p0, tau, band):
    G = t._gram(p0)
    s0 = np.array(t.S0, float)
    seen, front = {t._key(s0): s0}, [s0]
    while front:
        S = front.pop()
        p = t._cn(S, G)
        for n, (i, j) in enumerate(t.SL):
            if abs(p[n]) <= band:
                U = t._flip(S, i, j)
                k = t._key(U)
                if k not in seen:
                    seen[k] = U
                    front.append(U)
    return [S for S in seen.values() if t._cn(S, G).min() >= -tau - 1e-9]


def members_by(method, p0, tau, sigma=None):
    s0 = np.array(t.S0, float)
    if method == "slot":
        if sigma is None:
            return members_by("closure", p0, tau)
        from test_z49_anisotropy_and_shapes import _band
        G = t._gram(p0)
        p = t._cn(s0, G)
        ps = np.where(p <= 4.0 * _band(s0, G, sigma), 0.0, p)
        Gs = t._gram(ps)
        if np.linalg.eigvalsh(Gs).min() <= 1e-9 * np.abs(G).max():
            return None
        tol = 1e-9 * np.abs(G).max()
        seen, front = {t._key(s0): s0}, [s0]
        while front:
            S = front.pop()
            q = t._cn(S, Gs)
            for n, (i, j) in enumerate(t.SL):
                if abs(q[n]) < tol:
                    U = t._flip(S, i, j)
                    k = t._key(U)
                    if k not in seen and t._cn(U, Gs).min() >= -tol:
                        seen[k] = U
                        front.append(U)
        return [S for S in seen.values() if (t._cn(S, G) >= -_band(S, G, sigma) - tol).all()]
    if method == "symmetry":
        return [s0]
    if method == "exact":
        _, mem = t._generate(p0, 0.0, 1e18)
        return list(mem.values())
    if method == "walk-tau":
        return walk(p0, tau, tau)
    if method == "walk-2tau":
        return walk(p0, tau, 2 * tau)
    if method == "closure":
        s0p = t._cn(s0, t._gram(p0))
        ps = np.where(s0p <= 4 * tau, 0.0, s0p)
        if np.linalg.eigvalsh(t._gram(ps)).min() <= 1e-9 * s0p.max():
            return None                                   # hypothesis violated: singular idealised lattice
        _, mem = t._generate(p0, 4 * tau, tau)
        G = t._gram(p0)
        return [S for S in mem.values() if t._cn(S, G).min() >= -tau - 1e-9]
    raise ValueError(method)


def proposed(cache, SN):
    BinvT = np.linalg.inv(SN[1:4].T)
    out = set()
    for S in cache:
        for pm in PERM4:
            for sg in (1.0, -1.0):
                P = np.round((sg * S[list(pm)])[1:4].T @ BinvT)
                out.add(tuple(P.astype(int).ravel()))
    return out


def evaluate(path):
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell, params_from_metric
    methods = ["symmetry", "exact", "walk-tau", "walk-2tau", "closure", "slot"]
    score = {}
    for line in open(path):
        c = json.loads(line)
        p0 = np.array(c["reference_conorms"])
        GR = t._gram(p0)
        tau = c["tau"]
        M = np.array(c["true_operator"], float)
        # measured cell in the reference's reduced frame: the case stores the
        # measured cell; its reduced superbase in its own coordinates is SN
        GN = np.array(UnitCell(*c["measured_cell"]).metric_tensor())
        SN = np.array(canonical_superbase(tuple(params_from_metric(GN)))[0], float)
        true = tuple(np.round(M).astype(int).ravel())
        neg = tuple((-np.round(M)).astype(int).ravel())
        fam = c["family"]
        for m in methods:
            cache = members_by(m, p0, tau, c["expected"].get("per_slot_sigma_rad"))
            rec = score.setdefault((fam, m), {"n": 0, "reindex": 0, "set": 0, "set_n": 0, "flagged": 0})
            rec["n"] += 1
            if cache is None:
                rec["flagged"] += 1
                continue
            cand = proposed(cache, SN)
            rec["reindex"] += (true in cand) or (neg in cand)
            if m in ("walk-tau", "walk-2tau", "closure") or (m == "slot" and fam != "anisotropy"):
                rec["set_n"] += 1
                rec["set"] += len(cache) == c["expected"]["widened_members"]
    return score


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "benchmark.jsonl")
    score = evaluate(path)
    fams = sorted({f for f, m in score})
    print(f"{'family':16s} {'method':10s} {'n':>5s} {'reindex ok':>11s} {'set ok':>8s} {'flagged':>8s}")
    for f in fams:
        for m in ["symmetry", "exact", "walk-tau", "walk-2tau", "closure", "slot"]:
            r = score[(f, m)]
            s = f"{r['set']}/{r['set_n']}" if r["set_n"] else "-"
            print(f"{f:16s} {m:10s} {r['n']:5d} {r['reindex']:5d}/{r['n']:<5d} {s:>8s} {r['flagged']:8d}")


if __name__ == "__main__":
    main()
