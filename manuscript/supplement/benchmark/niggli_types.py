"""From the Niggli cell to the Selling base: where do the near-zero conorms land?

Near-mP lattices (twofold along the shortest, middle or longest edge; beta 95-125 deg),
measured with a small random metric error (strain 0.01-0.3 %); the Niggli cell from cctbx;
Selling reduction started from the Niggli basis, flipping at the most negative conorm.
Recorded: the Niggli type (I: three acute angles, II: none), the flips made, and whether
the two smallest conorms of the Selling base sit on a slot of d (da, db, dc). The same is
done with agentsg.cell.strain_closure.reduce_cell.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/niggli_types.py
"""
from cctbx import crystal                                           # noqa: E402 (before numpy)
import collections                                                  # noqa: E402

import numpy as np                                                  # noqa: E402

from agentsg.cell.metric import metric_tensor, params_from_metric   # noqa: E402
from agentsg.cell.strain_closure import reduce_cell                 # noqa: E402

NAMES = ["da", "db", "dc", "ab", "ac", "bc"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def conorms(G):
    a, b, c = np.eye(3); V = [-a - b - c, a, b, c]
    return np.array([-V[i] @ G @ V[j] for i, j in PAIRS])


def flip(B, k):
    a, b, c = B; V = [-a - b - c, a, b, c]
    i, j = PAIRS[k]; V2 = list(V); V2[i] = -V[i]
    for m in range(4):
        if m not in (i, j):
            V2[m] = V[m] + V[i]
    return np.array(V2[1:4])


def selling_from(B, G):
    flips = []
    for _ in range(200):
        p = conorms(B @ G @ B.T); k = int(np.argmin(p))
        if p[k] >= 0:
            return B, flips
        flips.append(NAMES[k]); B = flip(B, k)
    raise RuntimeError


def main(n=400, seed=1):
    rng = np.random.default_rng(seed)
    t = collections.Counter(); words = collections.Counter(); lib = collections.Counter()
    for k in range(3 * n):
        L = [(20, 40, 55), (30, 40, 55), (45, 30, 40)][k % 3]
        G = np.array(metric_tensor((L[0], L[1], L[2], 90, 90, rng.uniform(95, 125))), float)
        w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
        X = rng.normal(size=(3, 3)); X = (X + X.T) / 2; X *= rng.uniform(1e-4, 3e-3) / np.linalg.norm(X)
        cs = crystal.symmetry(unit_cell=params_from_metric((Gh @ (np.eye(3) + X) @ Gh).tolist()), space_group_symbol="P1")
        nig = cs.niggli_cell().unit_cell(); g = nig.metrical_matrix()
        Gn = np.array([[g[0], g[3], g[4]], [g[3], g[1], g[5]], [g[4], g[5], g[2]]])
        ntype = "I" if min(g[3], g[4], g[5]) > 0 else "II"
        B, fl = selling_from(np.eye(3), Gn)
        small = sorted(np.argsort(conorms(B @ Gn @ B.T))[:2])
        t[(ntype, any(s < 3 for s in small))] += 1
        words[(ntype, ".".join(fl) or "none", tuple(NAMES[s] for s in small))] += 1
        r = reduce_cell(nig.parameters()); Bm = np.array([[float(x) for x in row] for row in r.basis.rows])
        small = sorted(np.argsort(conorms(Bm.T @ Gn @ Bm))[:2])
        lib[(ntype, any(s < 3 for s in small))] += 1
    print("(Niggli type, near-zero conorms on a d slot): count -- Selling reduction from the Niggli basis")
    for k in sorted(t): print(" ", k, t[k])
    print("flip words and the slots of the two smallest conorms")
    for k, v in words.most_common(): print(" ", k, v)
    print("the same with agentsg reduce_cell")
    for k in sorted(lib): print(" ", k, lib[k])


if __name__ == "__main__":
    main()
