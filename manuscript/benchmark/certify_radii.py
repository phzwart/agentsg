"""Certified adjacency radii (Table 8, Remark A8) in exact rational arithmetic.

The radius r_k(G) is the smallest strain from G to a metric for which a superbase outside
the layer L_k is obtuse, searched up to the cap 1/4.  Everything that decides the radius is
checked here with Python integers and fractions; floating point only proposes the dual
multipliers and the primal perturbations, and a bad proposal can only loosen a bound,
never make it wrong.

  1. The lattice.  Each cell is given by the six conorms of its reduced superbase, as the
     floating-point numbers the paper's computation used, read as exact rationals.  F0 is
     obtuse for this metric by construction.
  2. The layers.  L_1 and L_2 are recomputed from the deposited table of U
     (universal_operators_columns.json), which the Lean development proves complete.
  3. The candidates.  Every superbase within strain 1/4 of obtuse has its four vectors in
     the balls of Lemma A6 and every conorm at least -|u||v|/4 (Proposition A4); the balls
     and the conorm test are evaluated exactly, so no superbase within the cap is missed.
  4. The bounds.  The strain needed to make a superbase F obtuse is min ||r||_M over conorm
     changes r >= -q, with q the conorms of F and M the strain norm in F's frame
     (tr(G^-1 dG G^-1 dG) = r^T M r).  For any multipliers lam >= 0 with c = -lam.q > 0,
         depth(F)^2 >= c^2 / (lam^T M^-1 lam)            (weak duality),
     and any r with q + r >= 0 whose metric is positive definite gives
         depth(F)^2 <= r^T M r                            (a feasible point).
     Both are exact rational inequalities.

    PYTHONPATH=agentsg/src python manuscript/benchmark/certify_radii.py
writes results_certified_radii.txt and certified_radii.json next to this script.
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "appendix_z"))

SL = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))       # da, db, dc, ab, ac, bc
S0 = ((-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1))
CAP = Fr(1, 4)


# ---- exact linear algebra --------------------------------------------------------------
def gram(p):
    """metric of (a, b, c) from the six conorms (exact)."""
    return [[p[0] + p[3] + p[4], -p[3], -p[4]],
            [-p[3], p[1] + p[3] + p[5], -p[5]],
            [-p[4], -p[5], p[2] + p[4] + p[5]]]


def dot(u, G, v):
    return sum(u[i] * G[i][j] * v[j] for i in range(3) for j in range(3))


def inv(A):
    n = len(A)
    M = [[Fr(x) for x in row] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        f = M[c][c]
        M[c] = [x / f for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                g = M[r][c]
                M[r] = [x - g * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def pos_def3(G):
    m1 = G[0][0]
    m2 = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    m3 = (G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
          - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
          + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]))
    return m1 > 0 and m2 > 0 and m3 > 0


def det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def conorms(S, G):
    return [-dot(S[i], G, S[j]) for i, j in SL]


E = [gram([Fr(int(k == l)) for l in range(6)]) for k in range(6)]


def strain_form(GF):
    """M with r^T M r = tr(GF^-1 dG GF^-1 dG), dG = sum_k r_k E_k (exact)."""
    Gi = inv(GF)
    A = [matmul(Gi, Ek) for Ek in E]
    return [[sum(A[k][i][j] * A[l][j][i] for i in range(3) for j in range(3)) for l in range(6)]
            for k in range(6)]


def key(S):
    a = tuple(sorted(tuple(int(x) for x in v) for v in S))
    b = tuple(sorted(tuple(-int(x) for x in v) for v in S))
    return min(a, b)


def isqrt_frac_ceil(x):
    """an integer >= sqrt(x) for a non-negative fraction x."""
    return math.isqrt(math.ceil(x)) + 1


def sqrt_floor(x, digits=6):
    """a decimal <= sqrt(x), exact."""
    s = 10 ** digits
    return Fr(math.isqrt(math.floor(x * s * s)), s)


def sqrt_ceil(x, digits=6):
    s = 10 ** digits
    n = math.isqrt(math.ceil(x * s * s))
    if n * n < x * s * s:
        n += 1
    return Fr(n, s)


# ---- layers from the deposited table ----------------------------------------------------
def layers():
    W = json.load(open(os.path.join(HERE, "universal_operators_columns.json")))["W"]
    Ws = [np.array(w, dtype=np.int64) for w in W]

    def sb(M):
        return [tuple(-M.sum(1))] + [tuple(M[:, j]) for j in range(3)]

    L1 = {}
    for w in Ws:
        L1.setdefault(key(sb(w)), w)
    L2 = set()
    for w1 in L1.values():
        for w in Ws:
            L2.add(key(sb(w1 @ w)))
    return set(L1), L2


# ---- the candidates within the cap ---------------------------------------------------------
def ball_vectors(G, eps):
    """all lattice vectors u (not in 2L) with |u|^2 <= (1+eps)/(1-eps) mu_c, exactly (Lemma A6)."""
    f = (1 + eps) / (1 - eps)
    mu = {}
    for s in itertools.product((0, 1), repeat=4):
        if 0 < sum(s) < 4 and s[0] == 0:
            v = [sum(s[i] * S0[i][j] for i in range(4)) for j in range(3)]
            mu[tuple(x % 2 for x in v)] = dot(v, G, v)
    R2 = f * max(mu.values())
    Gi = inv(G)
    bnd = [isqrt_frac_ceil(R2 * Gi[i][i]) for i in range(3)]       # |x_i|^2 <= R2 (G^-1)_ii
    V = []
    for x in itertools.product(*[range(-b, b + 1) for b in bnd]):
        c = tuple(t % 2 for t in x)
        if c == (0, 0, 0):
            continue
        if dot(x, G, x) <= f * mu[c]:
            V.append(x)
    return V, bnd


def candidates(G, eps):
    """every superbase whose four vectors lie in the balls and whose six conorms pass the
    exact test conorm >= -eps |u||v| (Proposition A4): a superset of those within eps."""
    V, bnd = ball_vectors(G, eps)
    n2 = [dot(v, G, v) for v in V]
    idx = {v: i for i, v in enumerate(V)}
    e2 = eps * eps

    def ok(i, j):
        g = dot(V[i], G, V[j])                 # conorm is -g
        return g <= 0 or g * g <= e2 * n2[i] * n2[j]

    nV = len(V)
    okm = [[False] * nV for _ in range(nV)]
    for i in range(nV):
        for j in range(i + 1, nV):
            okm[i][j] = okm[j][i] = ok(i, j)
    seen = {}
    for a in range(nV):
        bs = [b for b in range(a + 1, nV) if okm[a][b]]
        for bi, b in enumerate(bs):
            for c in bs[bi + 1:]:
                if not okm[b][c]:
                    continue
                d = tuple(-(V[a][t] + V[b][t] + V[c][t]) for t in range(3))
                j = idx.get(d)
                if j is None or not (okm[j][a] and okm[j][b] and okm[j][c]):
                    continue
                if abs(det3(V[a], V[b], V[c])) != 1:
                    continue
                S = (d, V[a], V[b], V[c])
                seen.setdefault(key(S), S)
    return seen, len(V), bnd


# ---- certificates for one superbase -------------------------------------------------------
def certify_superbase(G, S, want_upper):
    q = conorms(S, G)
    if min(q) >= 0:
        return Fr(0), Fr(0), {"lam": None, "r": [0] * 6}
    B = S[1:]
    GF = [[dot(B[i], G, B[j]) for j in range(3)] for i in range(3)]
    M = strain_form(GF)
    # floating-point proposal (as in strain_closure.distance_to_obtuse)
    from scipy.optimize import nnls
    Mf = np.array([[float(x) for x in row] for row in M])
    qf = np.array([float(x) for x in q])
    R = np.linalg.cholesky(Mf).T
    x, _ = nnls(R, R @ qf)
    rf = x - qf
    lamf = Mf @ rf
    # lower bound: weak duality, exact
    lam = [max(Fr(v).limit_denominator(10 ** 12), Fr(0)) for v in lamf]
    c = -sum(l * qq for l, qq in zip(lam, q))
    Mi = inv(M)
    den = sum(lam[k] * Mi[k][l] * lam[l] for k in range(6) for l in range(6))
    lower_sq = c * c / den if c > 0 and den > 0 else Fr(0)
    cert = {"lam": [str(l) for l in lam]}
    upper_sq = None
    if want_upper:
        r = [Fr(v).limit_denominator(10 ** 12) for v in rf]
        r = [max(rk, -qk) for rk, qk in zip(r, q)]                  # exactly feasible
        p_new = [qk + rk for qk, rk in zip(q, r)]
        assert min(p_new) >= 0
        assert pos_def3(gram(p_new)), "primal point not positive definite"
        upper_sq = sum(r[k] * M[k][l] * r[l] for k in range(6) for l in range(6))
        cert["r"] = [str(x) for x in r]
        assert lower_sq <= upper_sq
    return lower_sq, upper_sq, cert


def certify_cell(name, p, L1, L2):
    G = gram(p)
    assert min(conorms(S0, G)) >= 0 and pos_def3(G)
    cands, nV, bnd = candidates(G, CAP)
    out = {"cell": name, "conorms": [str(x) for x in p], "ball_vectors": nV, "box": bnd,
           "candidates": len(cands)}
    for lname, L in (("r1", L1), ("r2", L2)):
        outside = [(k, S) for k, S in cands.items() if k not in L]
        best = None
        lows = []
        for k, S in outside:
            lo, _, _ = certify_superbase(G, S, False)
            lows.append((lo, k, S))
        lows.sort()
        res = {"outside_candidates": len(outside)}
        if not lows or lows[0][0] >= CAP * CAP:
            res["lower"] = str(CAP)
            res["upper"] = None
            res["text"] = ">25"
        else:
            lo_sq, k, S = lows[0]
            lo2, up2, cert = certify_superbase(G, S, True)
            # the minimum over all outside candidates is at least min(lower bounds)
            lower = sqrt_floor(min(lo_sq, CAP * CAP))
            upper = sqrt_ceil(up2)
            res.update({"lower": str(lower), "upper": str(upper), "argmin": [list(v) for v in S],
                        "certificate": cert,
                        "text": f"{100 * float(lower):.4f}-{100 * float(upper):.4f}"})
        out[lname] = res
    return out


def cells():
    import test_z58_projection_closure as t58
    from agentsg.cell.metric import metric_tensor
    from agentsg.cell.selling_cob import selling_matrix
    defs = [
        ("worked cell 30.02x39.97x50.03, 90.25/92.05/89.80", (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)),
        ("needle 10x10x30", (10.0, 10.0, 30.0, 90, 90, 90)),
        ("needle 10x10x100", (10.0, 10.0, 100.0, 90, 90, 90)),
        ("plate 100x100.1x5, 90.03/89.98/90.05", (100.0, 100.1, 5.0, 90.03, 89.98, 90.05)),
        ("plate 150x120x4.8, beta 90.4", (150.0, 120.0, 4.8, 90.0, 90.4, 90.0)),
    ]
    out = [("needle 5.58x5.21x132 (conorms of test z57)",
            [Fr(x) for x in t58.NEEDLE.tolist()])]
    for name, cell in defs:
        Gc = np.array(metric_tensor(cell), float)
        P = np.array([[float(x) for x in row] for row in selling_matrix(cell).rows], float)
        S = np.array(S0, float)
        Gr = P.T @ Gc @ P
        p0 = [-(S[i] @ Gr @ S[j]) for i, j in SL]
        out.append((name, [Fr(max(x, 0.0)) for x in p0]))
    return out


def main():
    L1, L2 = layers()
    assert (len(L1), len(L2)) == (127, 2755)
    res = []
    lines = ["Certified adjacency radii (strain, %), exact rational arithmetic; '>25' = no superbase",
             "outside the layer within the cap 1/4.  lower-upper: proved bounds on the radius.", ""]
    for name, p in cells():
        r = certify_cell(name, p, L1, L2)
        res.append(r)
        lines.append(f"{name}: ball {r['ball_vectors']} vectors, {r['candidates']} candidate superbases; "
                     f"r1 {r['r1']['text']}  r2 {r['r2']['text']}")
        print(lines[-1], flush=True)
    open(os.path.join(HERE, "results_certified_radii.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(HERE, "certified_radii.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
