"""Near a three-zero face (oP), where does Selling reduction put the near-zero conorms when it
starts from an arbitrary basis rather than from the Niggli cell? 150 measured near-oP
metrics (strain 0.1-2 %), each presented in its Niggli form and in six random proper
bases; counted by the lattice's Niggli type (sign of (a.b)(a.c)(b.c) in the near-orthogonal
cell) and by whether a near-zero conorm of the reduced superbase sits on a slot of d.

    PYTHONPATH=agentsg/src python manuscript/benchmark/niggli_presentation.py
"""
import collections
import itertools

import numpy as np

from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import params_from_metric


def on_d(H):
    p = sc.conorms(np.array(sc._S0, float), sc.reduce_cell(params_from_metric(H.tolist())).G)
    return any(i < 3 for i in np.argsort(p)[:3])


def main(n=150):
    rng = np.random.default_rng(3)
    c = collections.Counter()
    for _ in range(n):
        G = np.diag(rng.uniform(20, 80, 3) ** 2)
        w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
        X = rng.normal(size=(3, 3)); X = (X + X.T) / 2; X *= rng.uniform(.001, .02) / np.linalg.norm(X)
        Gm = Gh @ (np.eye(3) + X) @ Gh
        t = "I" if Gm[0, 1] * Gm[0, 2] * Gm[1, 2] > 0 else "II"
        for sg in itertools.product((1, -1), repeat=3):              # the Niggli sign convention
            S = np.diag(sg); H = S @ Gm @ S
            off = [H[0, 1], H[0, 2], H[1, 2]]
            if (t == "I" and min(off) > 0) or (t == "II" and max(off) <= 0):
                break
        c[(t, "Niggli form", on_d(H))] += 1
        for _ in range(6):
            while True:
                U = rng.integers(-2, 3, (3, 3))
                if round(np.linalg.det(U)) == 1:
                    break
            c[(t, "random basis", on_d(U.T @ Gm @ U))] += 1
    print("Niggli type, presentation: reductions with a near-zero conorm on d / all")
    for t in ("I", "II"):
        for pres in ("Niggli form", "random basis"):
            k, n_ = c[(t, pres, True)], c[(t, pres, True)] + c[(t, pres, False)]
            print(f"  type {t:2s} {pres:13s} {k:4d} / {n_}")


if __name__ == "__main__":
    main()
