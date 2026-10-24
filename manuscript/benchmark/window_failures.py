"""Where fixed search windows fail, and whether the 6096-matrix lookup and the closure do.

Two families, each a reference lattice (exact) and a new cell (the same lattice in another
proper basis with a metric error drawn at the size of non-isomorphism, 0.1-2 % strain):

  typeI   lattices on the faces with two zeros sharing a vector or three zeros (sorts M4,
          O4, H4, O5, T5, C5), random parameters; the new cell's Niggli type (I: three
          acute angles) is recorded, since a type I Niggli cell puts the near-zero conorms
          of the Selling base on d slots and can need a matrix entry of 2 between the two
          reduced cells in their fixed labellings.
  entry3  lattices on random two- and three-zero faces whose closure has a member needing
          an entry of 3 in the basis of the reduced cell; the new cell is that member.

Methods (compare_methods.py): the closure (certified), the lookup of the 6096 matrices
between the two Selling-reduced cells (the first layer, uncertified, Remark A7), the
{-1,0,1} and {-2,...,2} searches, cctbx similarity_transformations (ranges 1, 2), and the
lattice-symmetry pipeline; plus LatticeRepLib CmdLM (Andrews & Bernstein's lattice
matcher, LM_BIN), scored by whether the cell it returns lies within the band of the
reference (it returns a cell, not an operator). Band 7.4 %.

    LM_BIN=/path/to/CmdLM PYTHONPATH=agentsg/src python manuscript/benchmark/window_failures.py [n]
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm                                       # noqa: E402 (imports cctbx first)
from cctbx import crystal                                          # noqa: E402
import numpy as np                                                 # noqa: E402

from agentsg.cell import strain_closure as sc                      # noqa: E402
from agentsg.cell.metric import params_from_metric                 # noqa: E402

BAND = cm.BAND
SLOT_FROM_S6 = (3, 4, 5, 2, 1, 0)


def lookup(ref_cell, new_cell, band):
    ref = sc.widened_closure(ref_cell, eps=sc.band_for(band), certify=False)
    cands, _ = sc.reindex(ref, new_cell, accept=band)
    Ps = np.array([[[int(x) for x in row] for row in c.operator.rows] for c in cands]).reshape(-1, 3, 3)
    return Ps, np.array([c.price for c in cands])


METHODS = dict(cm.METHODS, lookup=lookup)


def niggli_type(cell):
    m = crystal.symmetry(unit_cell=tuple(cell), space_group_symbol="P1").niggli_cell().unit_cell().metrical_matrix()
    return "I" if min(m[3], m[4], m[5]) > 0 else "II"


def perturb(G, rng):
    w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
    X = rng.normal(size=(3, 3)); X = (X + X.T) / 2; X *= rng.uniform(0.001, 0.02) / np.linalg.norm(X)
    return Gh @ (np.eye(3) + X) @ Gh


def random_proper(rng):
    while True:
        U = rng.integers(-2, 3, (3, 3))
        if round(np.linalg.det(U)) == 1:
            return U


def typeI_cases(n, rng):
    out = []
    sorts = ["M4", "O4", "H4", "O5", "T5", "C5"]
    for k in range(n):
        name = sorts[k % len(sorts)]
        pat = sc._SORT_PATTERNS[name]
        letters = sorted({ch for ch in pat if ch != "0"})
        while True:
            v = dict(zip(letters, rng.uniform(400, 3000, len(letters))))
            s6 = [0.0 if ch == "0" else v[ch] for ch in pat]
            G = sc._gram(np.array([s6[SLOT_FROM_S6[i]] for i in range(6)]))
            if np.linalg.eigvalsh(G).min() > 1:
                break
        U = random_proper(rng)
        out.append(dict(family="typeI", sort=name, reference_cell=params_from_metric(G.tolist()),
                        measured_cell=params_from_metric(perturb(U.T @ G @ U, rng).tolist()), true_operator=U.tolist()))
    return out


def entry3_cases(n, rng):
    U, *_ = sc._data()
    big = [W.T for W in U if np.abs(W).max() == 3]
    faces = [z for k in (2, 3) for z in itertools.combinations(range(6), k)]
    out = []
    while len(out) < n:
        z = faces[rng.integers(len(faces))]
        p = rng.uniform(300, 2500, 6); p[list(z)] = 0
        G = sc._gram(p)
        if np.linalg.eigvalsh(G).min() < 1:
            continue
        for W in big[rng.integers(len(big)):]:
            S = np.vstack([-W.T.sum(0), W.T])
            if (sc.conorms(S, G) >= -1e-6).all():
                M = np.round(W).astype(int)
                M = M if round(np.linalg.det(M)) == 1 else -M
                out.append(dict(family="entry3", sort=str(z), reference_cell=params_from_metric(G.tolist()),
                                measured_cell=params_from_metric(perturb(M.T @ G @ M, rng).tolist()),
                                true_operator=M.tolist()))
                break
    return out


def cmdlm(ref, new, binary):
    inp = "P " + " ".join(f"{x:.6f}" for x in ref) + "\nP " + " ".join(f"{x:.6f}" for x in new) + "\nend\n"
    out = subprocess.run([binary], input=inp, capture_output=True, text=True, timeout=600).stdout
    lines = [l for l in out.splitlines() if l.startswith("P ")]
    if len(lines) < 2:
        return None
    cell = [float(x) for x in lines[1].split()[1:7]]
    return sc._price(cm.G_of(ref), cm.G_of(cell))[0]


def main(n=60):
    rng = np.random.default_rng(2026)
    cases = typeI_cases(n, rng) + entry3_cases(n // 2, rng)
    lm = os.environ.get("LM_BIN")
    agg = collections.defaultdict(lambda: collections.Counter())
    for c in cases:
        M = np.array(c["true_operator"], int)
        key = (c["family"], niggli_type(c["measured_cell"]) if c["family"] == "typeI" else "-")
        for k in (key, (c["family"], "all")):
            agg[k]["n"] += 1
        for name, f in METHODS.items():
            try:
                Ps, ps = f(c["reference_cell"], c["measured_cell"], BAND)
            except (RuntimeError, ValueError):
                Ps, ps = np.zeros((0, 3, 3), int), np.zeros(0)
            ok = cm.score(Ps, ps, M)[0]
            for k in (key, (c["family"], "all")):
                agg[k][name] += ok
        if lm:
            p = cmdlm(c["reference_cell"], c["measured_cell"], lm)
            for k in (key, (c["family"], "all")):
                agg[k]["CmdLM"] += p is not None and p <= BAND
    names = list(METHODS) + (["CmdLM"] if lm else [])
    print(f"band {BAND}; errors 0.1-2 % strain on the new cell; true operator found (CmdLM: cell within band)")
    print(f"{'family':8s} {'Niggli':6s} {'n':>4s} " + " ".join(f"{m:>10s}" for m in names))
    for k in sorted(agg):
        a = agg[k]
        print(f"{k[0]:8s} {k[1]:6s} {a['n']:4d} " + " ".join(f"{a[m]:10d}" for m in names))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
