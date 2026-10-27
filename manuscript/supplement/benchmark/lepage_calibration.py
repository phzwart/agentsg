"""Le Page's 3 degrees, conditioned on the Delone sort.

For every non-triclinic sort: draw an exact lattice of the sort (its pattern's letters
uniform in [400, 3000] A^2), perturb its metric in a random traceless direction, and scale
the perturbation so that the cctbx lattice-symmetry analysis (Le Page) gives the sort's
Bravais lattice group a maximal obliquity of exactly 3 degrees. At that boundary record:
  - the strain distance from the perturbed cell to the sort (minimum over the frames of U),
  - the price (strain, after the scale fit) of every proper symmetry of the exact lattice,
    grouped by rotation order, and the largest of them.
The quantiles over directions give, for each sort, the strain distance and the operator
prices that correspond to Le Page's 3 degrees. Writes lepage_calibration.jsonl and a table.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/lepage_calibration.py [n_per_sort] [delta]
"""
from __future__ import annotations

from cctbx import crystal                                           # noqa: E402 (before numpy users)
from cctbx.sgtbx import bravais_types, lattice_symmetry             # noqa: E402
import collections                                                  # noqa: E402
import json                                                         # noqa: E402
import os                                                           # noqa: E402
import sys                                                          # noqa: E402

import numpy as np                                                  # noqa: E402

from agentsg.cell import strain_closure as sc                       # noqa: E402
from agentsg.cell.metric import params_from_metric                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SELLINGPROOFS_OUT", HERE)          # output directory (default: here)
SLOT_FROM_S6 = (3, 4, 5, 2, 1, 0)        # our slot k takes S6 position SLOT_FROM_S6[k]


def exact_cell(pattern, rng):
    letters = sorted({ch for ch in pattern if ch != "0"})
    for _ in range(100):
        v = dict(zip(letters, rng.uniform(400, 3000, len(letters))))
        s6 = [0.0 if ch == "0" else v[ch] for ch in pattern]
        p = np.array([s6[SLOT_FROM_S6[k]] for k in range(6)])
        G = sc._gram(p)
        if np.linalg.eigvalsh(G).min() > 1.0:
            return G
    raise RuntimeError(pattern)


def sort_distance(G, name):
    """exact constrained projection: the NNLS optimum is the smallest residual over the
    active sets whose unconstrained solution is feasible, vectorised over the frames of U."""
    import itertools
    red = sc.reduce_cell(params_from_metric(G.tolist()))
    Q = sc._L_all() @ sc.conorms(np.array(sc._S0, float), red.G)
    M = sc._metric_M(np.stack([sc._gram(q) for q in Q]))
    classes = sc._classes(sc._SORT_PATTERNS[name])
    A = np.zeros((6, len(classes)))
    for c, cl in enumerate(classes):
        A[cl, c] = 1
    best = np.sqrt(np.einsum("nk,nkl,nl->n", Q, M, Q))                 # empty active set: X = 0
    for r in range(1, len(classes) + 1):
        for act in itertools.combinations(range(len(classes)), r):
            Aa = A[:, list(act)]
            lhs = np.einsum("ka,nkl,lb->nab", Aa, M, Aa)
            rhs = np.einsum("ka,nkl,nl->na", Aa, M, Q)
            y = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            ok = (y >= -1e-12).all(1)
            R = y @ Aa.T - Q
            res = np.sqrt(np.maximum(np.einsum("nk,nkl,nl->n", R, M, R), 0))
            best = np.where(ok, np.minimum(best, res), best)
    return float(best.min())


def lepage_delta(G, btype):
    cs = crystal.symmetry(unit_cell=params_from_metric(G.tolist()), space_group_symbol="P1")
    cs = cs.niggli_cell()
    out = None
    for g in lattice_symmetry.metric_subgroups(input_symmetry=cs, max_delta=12.0).result_groups:
        bt = str(bravais_types.bravais_lattice(group=g["ref_subsym"].space_group()))
        if bt == {"oS": "oC"}.get(btype, btype):          # cctbx writes oC for oS
            d = g["max_angular_difference"]
            out = d if out is None else min(out, d)
    return out


def symmetries(G):
    cell = params_from_metric(G.tolist())
    ref = sc.widened_closure(cell, eps=0.002)
    cands, _ = sc.reindex(ref, cell, accept=0.0015)
    ops = []
    for c in cands:
        P = np.array([[float(x) for x in row] for row in c.operator.rows])
        if np.allclose(P.T @ G @ P, G, rtol=0, atol=1e-6 * np.abs(G).max()):
            ops.append(P)
    return ops


def order(P):
    M = np.eye(3)
    for n in range(1, 7):
        M = M @ P
        if np.allclose(M, np.eye(3)):
            return n
    return 0


def run(n_per=20, target=3.0, seed=3, only=None):
    rng = np.random.default_rng(seed)
    rows = []
    for name, pat in sc._SORT_PATTERNS.items():
        bt = sc.BRAVAIS[name]
        if bt == "aP" or (only and name not in only):
            continue
        for _ in range(n_per):
            G = exact_cell(pat, rng)
            ops = [P for P in symmetries(G) if order(P) > 1]
            w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
            X = rng.normal(size=(3, 3)); X = (X + X.T) / 2; X -= np.trace(X) / 3 * np.eye(3); X /= np.linalg.norm(X)
            lo, hi = 0.0, 0.4
            if (lepage_delta(Gh @ (np.eye(3) + hi * X) @ Gh, bt) or 99) < target:
                continue
            for _ in range(30):
                t = (lo + hi) / 2
                d = lepage_delta(Gh @ (np.eye(3) + t * X) @ Gh, bt)
                if d is not None and d <= target:
                    lo = t
                else:
                    hi = t
            Gp = Gh @ (np.eye(3) + lo * X) @ Gh
            prices = collections.defaultdict(list)
            for P in ops:
                prices[order(P)].append(sc._price(Gp, P.T @ Gp @ P)[0])
            rows.append(dict(sort=name, bravais=bt, delta=lepage_delta(Gp, bt), dist=sort_distance(Gp, name),
                             prices={str(k): max(v) for k, v in prices.items()},
                             max_price=max(max(v) for v in prices.values()) if prices else None, n_ops=len(ops)))
        print(name, len([r for r in rows if r["sort"] == name]), flush=True)
    path = os.path.join(OUT, "lepage_calibration.jsonl")
    if only and os.path.exists(path):                 # replace the rows of the sorts rerun
        rows = [r for r in map(json.loads, open(path)) if r["sort"] not in only] + rows
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    table(rows, target)


def table(rows, target=3.0):
    print(f"\nLe Page obliquity {target} deg on the sort's Bravais group; strain in %, quantiles 10/50/90 over directions")
    print(f"{'sort':5s} {'type':4s} {'n':>3s} {'|G+|':>4s} {'distance to sort':>20s} {'largest price':>20s}  largest price by order (median)")
    for name in sc._SORT_PATTERNS:
        rs = [r for r in rows if r["sort"] == name]
        if not rs:
            continue
        q = lambda xs: "/".join(f"{100 * v:.1f}" for v in np.percentile(xs, [10, 50, 90]))
        by = {k: np.median([r["prices"][k] for r in rs if k in r["prices"]]) for k in ("2", "3", "4", "6")
              if any(k in r["prices"] for r in rs)}
        print(f"{name:5s} {rs[0]['bravais']:4s} {len(rs):3d} {rs[0]['n_ops'] + 1:4d} {q([r['dist'] for r in rs]):>20s} "
              f"{q([r['max_price'] for r in rs]):>20s}  " + ", ".join(f"{k}: {100 * v:.1f}" for k, v in by.items()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        table([json.loads(l) for l in open(os.path.join(HERE, "lepage_calibration.jsonl"))])
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 20, float(sys.argv[2]) if len(sys.argv) > 2 else 3.0,
            only=sys.argv[3].split(",") if len(sys.argv) > 3 else None)
