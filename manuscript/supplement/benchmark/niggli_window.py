"""Is the {-1,0,1} search between two Niggli cells complete?

For lattices on each of the 38 faces (random conorms on the face, isotropic and
anisotropic), a reference and a new measurement are drawn with independent strain errors
of a given size, the new one in a random basis.  Both are Niggli-reduced (cctbx), and the
true operator between the two Niggli cells is computed directly (no search):
X_R^-1 Q X_N, with Q the random basis and X the transforms to the Niggli cells.  Reported
is the largest entry, by face kind and error size; an entry of 2 or more is a case the
{-1,0,1} window misses.  The same is done for the Selling-reduced cells (agentsg), whose
largest entries the paper states exactly (Table 4).

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/niggli_window.py [n_per_face]
"""
import collections
import itertools
import os
import sys

from cctbx import crystal                                          # noqa: E402 (before numpy)
import numpy as np                                                  # noqa: E402

from agentsg.cell.metric import params_from_metric                 # noqa: E402
from agentsg.cell import strain_closure as sc                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SELLINGPROOFS_OUT", HERE)          # output directory (default: here)
SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def conn(Z):
    par = list(range(4))

    def f(x):
        while par[x] != x:
            x = par[x]
        return x
    for k in range(6):
        if k not in Z:
            i, j = SL[k]
            par[f(i)] = f(j)
    return len({f(i) for i in range(4)}) == 1


FACES = [Z for r in range(4) for Z in itertools.combinations(range(6), r) if conn(Z)]


def kind(Z):
    E = [SL[k] for k in Z]
    if len(E) < 2:
        return f"{len(E)} zero"
    deg = collections.Counter(x for e in E for x in e)
    if len(E) == 2:
        return "2 disjoint" if len(deg) == 4 else "2 sharing"
    return "3 triangle" if len(deg) == 3 else "3 path"


def gram(p):
    return np.array([[p[0] + p[3] + p[4], -p[3], -p[4]], [-p[3], p[1] + p[3] + p[5], -p[5]],
                     [-p[4], -p[5], p[2] + p[4] + p[5]]])


def strain(G, eta, rng):
    X = rng.normal(size=(3, 3)); X = X + X.T; X *= eta / np.linalg.norm(X)
    w, V = np.linalg.eigh(G); R = V @ np.diag(np.sqrt(w)) @ V.T
    return R @ (np.eye(3) + X) @ R


def random_basis(rng, steps=6):
    Q = np.eye(3, dtype=int)
    for _ in range(steps):
        i, j = rng.choice(3, 2, replace=False)
        E = np.eye(3, dtype=int); E[i, j] = rng.choice([-1, 1])
        Q = Q @ E
    P = np.eye(3, dtype=int)[rng.permutation(3)]
    return Q @ P


def niggli_X(G):
    cs = crystal.symmetry(unit_cell=params_from_metric(G.tolist()), space_group_symbol="P1")
    cb = cs.change_of_basis_op_to_niggli_cell()
    return np.linalg.inv(np.array(cb.c().r().as_double()).reshape(3, 3))


def selling_X(G):
    r = sc.reduce_cell(params_from_metric(G.tolist()))
    return np.array([[float(x) for x in row] for row in r.basis.rows])


UOPS = sc.universal_operators().astype(float)


def point_group(G0):
    """the exact lattice symmetries of G0 (reduced in its own frame): the elements of U
    that fix it."""
    Y = np.einsum("nji,jk,nkl->nil", UOPS, G0, UOPS)
    ok = np.abs(Y - G0).max((1, 2)) <= 1e-9 * np.abs(G0).max()
    return UOPS[ok]


def lattices(n, rng):
    """(family, label, conorms): n random lattices on each of the 38 faces (half of them
    anisotropic) and on each of Delone's 24 sorts (equal conorms, so the Niggli boundaries
    of equal lengths and special angles are included)."""
    out = []
    for Z in FACES:
        for t in range(n):
            scale = np.exp(rng.uniform(0, np.log(30), 4)) if t % 2 else np.ones(4)
            p = rng.uniform(0.3, 1.0, 6) * 100
            p = np.array([p[k] * scale[SL[k][0]] * scale[SL[k][1]] for k in range(6)])
            p[list(Z)] = 0.0
            out.append(("face", kind(Z), p))
    for name, pat in sc._SORT_PATTERNS.items():
        for t in range(n):
            p = np.zeros(6)
            for cl in sc._classes(pat):
                p[cl] = 100 * np.exp(rng.uniform(0, np.log(30)))
            out.append(("sort", f"{name} ({sc.BRAVAIS[name]})", p))
    return out


ETAS = (0.005, 0.02, 0.05, 0.077)


def main(n=60):
    rng = np.random.default_rng(23)
    stats = collections.defaultdict(collections.Counter)
    examples = []
    for fam, lab, p in lattices(n, rng):
        G0 = gram(p)
        PG = point_group(G0)
        for eta in ETAS:
            GR = strain(G0, eta / 2, rng)
            Q = random_basis(rng)
            GN = Q.T @ strain(G0, eta / 2, rng) @ Q
            for red, f in (("Niggli", niggli_X), ("Selling", selling_X)):
                try:
                    XR, XN = f(GR), f(GN)
                except Exception:
                    stats[(red, fam, lab, eta)]["fail"] += 1
                    continue
                T = np.linalg.inv(XR) @ Q @ XN
                stats[(red, fam, lab, eta)][int(round(np.abs(T).max()))] += 1
                Ts = np.einsum("ij,njk,kl->nil", np.linalg.inv(XR), PG, Q @ XN)
                stats[(red + " mod point group", fam, lab, eta)][int(round(np.abs(Ts).max((1, 2)).min()))] += 1
                if red == "Niggli" and int(round(np.abs(T).max())) >= 2 and len(examples) < 8:
                    examples.append((lab, eta, np.round(T).astype(int).tolist(), [round(x, 3) for x in params_from_metric(GR.tolist())]))
    labels = collections.OrderedDict()
    for k in stats:
        labels[(k[1], k[2])] = 1
    lines = [f"largest entry of the true operator between the two reduced cells: {n} lattices per face or sort,",
             "error eta = total strain (half on each cell), new cell in a random basis.",
             "Each entry: cases with an entry > 1 / cases (largest entry); columns eta = " + ", ".join(f"{100 * e:.1f} %" for e in ETAS), ""]
    for red in ("Niggli", "Niggli mod point group", "Selling", "Selling mod point group"):
        lines.append(red)
        tot_big = tot_all = 0
        for fam, lab in labels:
            row = []
            for eta in ETAS:
                c = stats[(red, fam, lab, eta)]
                tot = sum(v for kk, v in c.items() if kk != "fail")
                big = sum(v for kk, v in c.items() if kk != "fail" and kk >= 2)
                tot_big += big; tot_all += tot
                row.append(f"{big:3d}/{tot:3d} ({max([kk for kk in c if kk != 'fail'] or [0])})")
            lines.append(f"  {fam:4s} {lab:12s} " + "  ".join(row))
        lines.append(f"  total: {tot_big} of {tot_all} cases need an entry > 1")
    lines += ["", "examples (lattice, eta, operator between Niggli cells, reference cell):"]
    lines += [f"  {e}" for e in examples]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "results_niggli_window.txt"), "w").write(txt + "\n")


def thin(n=40):
    """The same test on strongly anisotropic lattices: vertex scales up to 60 on each of the
    38 faces, total strain 2 and 7.7 %, Niggli cells only."""
    rng = np.random.default_rng(5)
    cnt = collections.Counter()
    ratios = []
    for Z in FACES:
        for t in range(n):
            scale = np.exp(rng.uniform(0, np.log(60), 4))
            p = rng.uniform(0.3, 1.0, 6) * 100
            p = np.array([p[k] * scale[SL[k][0]] * scale[SL[k][1]] for k in range(6)])
            p[list(Z)] = 0.0
            G0 = gram(p)
            w = np.linalg.eigvalsh(G0)
            ratios.append(np.sqrt(w.max() / w.min()))
            for eta in (0.02, 0.077):
                GR = strain(G0, eta / 2, rng); Q = random_basis(rng); GN = Q.T @ strain(G0, eta / 2, rng) @ Q
                XR, XN = niggli_X(GR), niggli_X(GN)
                cnt[int(round(np.abs(np.linalg.inv(XR) @ Q @ XN).max()))] += 1
    txt = (f"anisotropic lattices, {n} per face, total strain 2 and 7.7 %: {sum(cnt.values())} cases; "
           f"largest entry between Niggli cells: {dict(sorted(cnt.items()))}; "
           f"ratio of largest to smallest edge scale (sqrt of the metric's eigenvalue ratio): median {np.median(ratios):.0f}, max {max(ratios):.0f}")
    print(txt)
    open(os.path.join(OUT, "results_niggli_window_thin.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["thin"]:
        thin(*(int(x) for x in sys.argv[2:]))
    else:
        main(*(int(x) for x in sys.argv[1:]))
