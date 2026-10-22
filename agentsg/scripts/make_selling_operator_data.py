"""Write agentsg/cell/data/selling_operators.json: the universal Selling operator set U
(6096 integer matrices, Appendix D of the Selling-closure manuscript) and the first two
adjacency layers L1 = U.F0 and L2 = U.L1 as superbases (four integer vectors in the
coordinates of the reduced cell, up to overall sign and order).

Run from the repository root:  python agentsg/scripts/make_selling_operator_data.py
It needs numpy and the manuscript checks in manuscript/appendix_z (for U)."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "manuscript", "appendix_z")]
from test_z52_universal_operators import universal_set  # noqa: E402

def key(S):
    a = tuple(sorted(tuple(int(round(x)) for x in v) for v in S))
    b = tuple(sorted(tuple(-int(round(x)) for x in v) for v in S))
    return min(a, b)

W, _, _, _ = universal_set()
W = [w.astype(int) for w in W]
assert len(W) == 6096
sup = lambda B: np.concatenate([-(B.sum(0, keepdims=True)), B])
L1 = {key(sup(w)): sup(w) for w in W}
seen = dict(L1)
for S in list(L1.values()):
    B = S[1:]
    for w in W:
        S2 = sup(w @ B)
        seen.setdefault(key(S2), S2)
L2 = seen
assert len(L1) == 127 and len(L2) == 2755, (len(L1), len(L2))
out = {
    "description": "Universal Selling operator set U (rows: new basis in reduced-cell coordinates) "
                   "and adjacency layers L1 = U.F0, L2 = U.L1 as superbases; see the Selling-closure manuscript, "
                   "Appendix D and Theorem A12.",
    "U": [w.tolist() for w in W],
    "L1": [list(map(list, k)) for k in sorted(L1)],
    "L2": [list(map(list, k)) for k in sorted(L2)],
}
path = os.path.join(ROOT, "agentsg", "src", "agentsg", "cell", "data", "selling_operators.json")
with open(path, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(path, os.path.getsize(path), "bytes")
