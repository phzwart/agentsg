"""The largest entry of the operators out of a reduced cell F0, for each face (zero pattern
of the six conorms, read in F0's frame): over every member of the exact closure and every
labelling of the member, with F0 as reduction returns it ("as reduced"), and with F0
relabelled in the best of its 48 labellings ("best labelling"). Paths of three zeros touch
every vector, so no labelling keeps the zeros off d, and the best labelling still needs 2.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/face_labellings.py
"""
import itertools, collections
import numpy as np
from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import params_from_metric
NAMES = ["da", "db", "dc", "ab", "ac", "bc"]
rng = np.random.default_rng(1)
S0 = np.array(sc._S0, float)
Q0 = list(sc._relabellings(S0))
out = collections.OrderedDict()
for k in range(0, 4):
    for z in itertools.combinations(range(6), k):
        for _ in range(20):
            p = rng.uniform(300, 2500, 6); p[list(z)] = 0
            G = sc._gram(p)
            if np.linalg.eigvalsh(G).min() > 1:
                break
        else:
            continue
        W = sc.widened_closure(params_from_metric(G.tolist()), eps=1e-6, max_ball=None)
        # W.reduced may relabel; recompute zeros in its frame
        pz = sc.conorms(S0, W.reduced.G)
        zs = tuple(NAMES[i] for i in range(6) if abs(pz[i]) < 1e-6 * pz.max())
        allB = [np.asarray(B, float) for m in W.members for B in sc._relabellings(np.asarray(m.superbase, float))]
        fixed = max(np.abs(B).max() for B in allB)                       # source as reduced, every target labelling
        best = min(max(np.abs(np.round(np.linalg.inv(np.asarray(Q, float).T) @ B.T)).max() for B in allB) for Q in Q0)
        out[zs] = (len(W.members), int(round(fixed)), int(best))
for zs, v in out.items():
    print(f"{len(zs)} zeros  {','.join(zs) or '-':10s} members {v[0]:2d}  largest entry: as reduced {v[1]}, best labelling {v[2]}")
