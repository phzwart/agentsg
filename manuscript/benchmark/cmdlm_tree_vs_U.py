"""Does LatticeRepLib's lattice matcher (CmdLM, default recursion depth 2) contain the
6096 operators of U?  cmdlm_dumptree.cpp, compiled against the LatticeRepLib sources
(g++ -std=gnu++20 -I<LatticeRepLib>), prints the matcher's MatS6 tree, one 6x6 matrix per
line (row j col i at position 6j+i):

    ./dumptree 2 > lm_tree.txt
    PYTHONPATH=agentsg/src python manuscript/benchmark/cmdlm_tree_vs_U.py lm_tree.txt [--layer2]

Each W of U is mapped to its action on S6 = (b.c, a.c, a.b, a.d, b.d, c.d); the script
counts the distinct actions and how many are in the tree, with their inverses.
"""
import sys

import numpy as np

from agentsg.cell import strain_closure as sc


def s6_of_G(G):
    a, b, c = np.eye(3); d = -a - b - c
    V = [a, b, c, d]
    dot = lambda i, j: V[i] @ G @ V[j]                       # noqa: E731
    return np.array([dot(1, 2), dot(0, 2), dot(0, 1), dot(0, 3), dot(1, 3), dot(2, 3)])


def G_of_s6(s):
    return sc._gram(-np.array([s[3], s[4], s[5], s[2], s[1], s[0]]))


def action(W):
    cols = [s6_of_G(W.T @ G_of_s6(np.eye(6)[k]) @ W) for k in range(6)]
    return tuple(np.round(np.array(cols).T).astype(int).ravel())


if __name__ == "__main__":
    T = np.loadtxt(sys.argv[1]).reshape(-1, 6, 6)
    tree = {tuple(np.round(m).astype(int).ravel()) for m in T}
    acts = {action(W) for W in sc.universal_operators()}
    inv = {tuple(np.round(np.linalg.inv(np.array(k).reshape(6, 6))).astype(int).ravel()) for k in acts}
    print(f"matcher tree: {len(tree)} matrices; U: {len(acts)} distinct S6 actions; "
          f"in the tree: {len(acts & tree)}; inverses in the tree: {len(inv & tree)}")
    if "--layer2" in sys.argv:                        # the products U.U (second layer)
        U = sc.universal_operators().astype(np.int64)
        seen = set()
        for i in range(0, len(U), 200):
            Pm = np.unique(np.einsum("aij,bjk->abik", U[i:i + 200], U).reshape(-1, 9), axis=0)
            seen.update(map(bytes, Pm.astype(np.int64)))
        acts2 = {action(np.frombuffer(b, dtype=np.int64).reshape(3, 3)) for b in seen}
        print(f"U.U: {len(acts2)} distinct S6 actions; in the tree: {len(acts2 & tree)} "
              f"({100 * len(acts2 & tree) / len(acts2):.0f} %)")
