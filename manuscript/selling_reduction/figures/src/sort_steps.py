"""Sort-to-sort step sizes (strain norm) for a cell: python sort_steps.py a b c al be ga out.json

row 0 = the measured cell, row S = the measured cell idealised onto sort S (its projection),
column S' = the smallest strain that puts that lattice in sort S'."""
import sys, json, numpy as np
import os
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "supplement", "src")]
from sellingproofs.claims.test_z58_projection_closure import L_ALL, metric_M, _sort_patterns, _reduced, t

PAT = _sort_patterns()
NAMES = list(PAT)

def spectrum(G, want_argmin=False):
    p0 = t._cn(np.array(t.S0, float), G)
    Q = L_ALL @ p0                                            # (n,6) conorms in every frame
    M = metric_M(np.stack([t._gram(q) for q in Q]))           # (n,6,6)
    out, arg = {}, {}
    for name, (zeros, classes) in PAT.items():
        nc = len(classes)
        A = np.zeros((6, nc))
        for c, cl in enumerate(classes):
            A[cl, c] = 1
        AMA = np.einsum("ka,nkl,lb->nab", A, M, A)
        AMq = np.einsum("ka,nkl,nl->na", A, M, Q)
        y = np.linalg.solve(AMA, AMq[..., None])[..., 0] if nc else np.zeros((len(Q), 0))
        X = y @ A.T
        bad = (y < -1e-12).any(1) if nc else np.zeros(len(Q), bool)
        for k in np.where(bad)[0]:                            # active set where a class goes negative
            active = list(range(nc))
            while True:
                Aa = A[:, active]
                yy = np.linalg.solve(Aa.T @ M[k] @ Aa, Aa.T @ M[k] @ Q[k]) if active else np.zeros(0)
                if (yy >= -1e-12).all():
                    break
                active.pop(int(np.argmin(yy)))
            X[k] = Aa @ yy if active else 0.0
        R = X - Q
        d = np.sqrt(np.maximum(np.einsum("nk,nkl,nl->n", R, M, R), 0))
        k = int(np.argmin(d))
        out[name] = float(d[k])
        arg[name] = X[k]
    return (out, arg) if want_argmin else out

cell = tuple(map(float, sys.argv[1:7])); OUT = sys.argv[7]
G = _reduced(cell)
d0, proj = spectrum(G, want_argmin=True)
order = sorted(NAMES, key=lambda n: d0[n])
rows = {"measured": d0}
for S in order:
    x = proj[S]
    GS = t._gram(np.maximum(x, 0))
    rows[S] = spectrum(GS)
    print(S, round(100 * d0[S], 3), flush=True)
json.dump({"order": order, "rows": rows, "cell": cell}, open(OUT, "w"))
