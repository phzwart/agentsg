"""z:universal — the operators alone, with no lattice.

An operator W (rows: the new basis in the reduced cell's coordinates) acts on
conorm vectors by a fixed integer 6x6 matrix L_W. W carries a reduced cell
of a lattice with zero pattern Z to a reduced cell exactly when L_W maps the
face {p >= 0, p_Z = 0} of the orthant into the orthant, i.e. when the columns
of L_W on the free slots are non-negative: a test on the integer matrix, with
no lattice. Enumerating integer matrices gives: the operator set of every
face (48 x 1, 2, 3, 6, 16), their union U of 6096 matrices = 48 x 127 with
entries in {-3..3}, nine kinds of operator up to relabelling at both ends,
flip length 0 to 4, and every operator between members of every one of the
24 sorts' closures lies in U.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytestmark = [pytest.mark.zcheck]

import test_z44_widened_complete as t  # noqa: E402
from test_z46_selling_groupoid import SORTS  # noqa: E402

SL = t.SL
PERM4 = list(itertools.permutations(range(4)))
UNIT_GRAM = [t._gram(np.eye(6)[m]) for m in range(6)]


def L_of(B):
    """L_W for each basis-row matrix in B (n,3,3): conorms(W F0) = L_W p."""
    U = np.concatenate([-(B.sum(1))[:, None, :], B], 1)
    L = np.zeros((len(B), 6, 6))
    for m in range(6):
        P = -np.einsum("nia,ab,njb->nij", U, UNIT_GRAM[m], U)
        L[:, :, m] = np.stack([P[:, i, j] for i, j in SL], 1)
    return np.round(L).astype(int)


def face_ok(L, Z):
    free = [m for m in range(6) if m not in Z]
    return (L[:, :, free] >= 0).all((1, 2))


def valid_face(Z):
    free = [m for m in range(6) if m not in Z]
    adj = {v: set() for v in range(4)}
    for a, b in (SL[m] for m in free):
        adj[a].add(b); adj[b].add(a)
    seen, st = {0}, [0]
    while st:
        x = st.pop()
        for y in adj[x] - seen:
            seen.add(y); st.append(y)
    return len(seen) == 4


def universal_set():
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    L = L_of(B)
    faces = [Z for k in range(4) for Z in map(list, itertools.combinations(range(6), k)) if valid_face(Z)]
    keep = np.zeros(len(B), bool)
    for Z in faces:
        keep |= face_ok(L, Z)
    sw = B[face_ok(L, [])].astype(int)
    S = {tuple(w.ravel()): w for w in B[keep].astype(int)}
    front = list(S.values())
    while front:                       # close under relabelling at both ends (window [-2,2] clips entries of 3)
        new = []
        for w in front:
            for s in sw:
                for x in (s @ w, w @ s):
                    k = tuple(x.ravel())
                    if k not in S:
                        S[k] = x; new.append(x)
        front = new
    W = np.array(list(S.values()))
    return W, L_of(W.astype(float)), sw, faces


def test_universal_operator_set():
    W, L, sw, faces = universal_set()
    assert len(faces) == 38 and len(sw) == 48
    keep = np.zeros(len(W), bool)
    for Z in faces:
        keep |= face_ok(L, Z)
    assert keep.all() and len(W) == 6096 == 48 * 127 and np.abs(W).max() == 3
    # the operator set of each face type, counted up to relabelling of the target
    for Z, n in (([], 1), ([3], 2), ([3, 2], 3), ([3, 5], 6), ([3, 4, 5], 16), ([3, 4, 2], 16)):
        assert face_ok(L, Z).sum() == 48 * n, Z
    # nesting of faces: one < path < triangle; disjoint is not inside the triangle
    one, path, disj, tri = (face_ok(L, Z) for Z in ([3], [3, 5], [3, 2], [3, 4, 5]))
    assert (one <= path).all() and (path <= tri).all() and not (disj <= tri).all()
    # nine kinds of operator: double cosets under relabelling at both ends
    keys = {tuple(w.ravel()): i for i, w in enumerate(W)}
    coset = -np.ones(len(W), int); nc = 0
    for i, w in enumerate(W):
        if coset[i] >= 0:
            continue
        st = [w]; coset[i] = nc
        while st:
            x = st.pop()
            for s in sw:
                for y in (s @ x, x @ s):
                    j = keys[tuple(y.ravel())]
                    if coset[j] < 0:
                        coset[j] = nc; st.append(y)
        nc += 1
    assert nc == 9
    # flip length: breadth-first over flips inside U, relabelling free
    dist = -np.ones(len(W), int)
    s0 = np.array(t.S0, float)

    def mark(S, d):
        for pm in PERM4:
            for sg in (1, -1):
                j = keys.get(tuple(np.round((sg * S[list(pm)])[1:4]).astype(int).ravel()))
                if j is not None and dist[j] < 0:
                    dist[j] = d

    front, seen = [s0], {t._key(s0)}
    mark(s0, 0); d = 0
    while front:
        d += 1; new = []
        for S in front:
            for i, j in SL:
                S2 = t._flip(S, i, j); k = t._key(S2)
                if k in seen or keys.get(tuple(np.round(S2[1:4]).astype(int).ravel())) is None:
                    continue
                seen.add(k); mark(S2, d); new.append(S2)
        front = new
    assert (dist >= 0).all() and len(seen) == 127
    kinds = sorted((int(dist[coset == c].min()), int((coset == c).sum())) for c in range(nc))
    assert kinds == [(0, 48), (1, 288), (2, 1152), (3, 576), (3, 1152), (3, 1152), (4, 576), (4, 576), (4, 576)]
    assert all(len(set(dist[coset == c])) == 1 for c in range(nc))
    # which kinds occur in which face (for the fixed placements)
    occurs = {}
    for name, Z in (("one", [3]), ("disjoint", [3, 2]), ("path", [3, 5]), ("triangle", [3, 4, 5])):
        ok = face_ok(L, Z)
        occurs[name] = sorted(set(int(dist[i]) for i in np.where(ok)[0]))
    assert occurs == {"one": [0, 1], "disjoint": [0, 1], "path": [0, 1, 2, 3], "triangle": [0, 1, 2, 3, 4]}
    # U is closed under inversion, and every element has a labelling whose three
    # basis vectors have entries in [-2,2] (the brute-force window of the paper)
    assert all(tuple(np.round(np.linalg.inv(w)).astype(int).ravel()) in keys for w in W)
    assert all(any(np.abs(s @ w).max() <= 2 or np.abs(w @ s).max() <= 2 for s in sw) for w in W)
    # universality: every operator between members of every sort's closure is in U
    rnd = np.random.default_rng(52)
    for name, bt, pat in SORTS:
        letters = [ch for ch in "rstuvw" if ch in pat]
        vals = dict(zip(letters, rnd.uniform(400, 3000, len(letters))))
        s = [0.0 if ch == "0" else vals[ch] for ch in pat]
        p = np.array([s[3], s[4], s[5], s[2], s[1], s[0]])
        _, mem = t._generate(p, 0.0, 1e18)
        for S in mem.values():
            for pm in PERM4:
                for sg in (1, -1):
                    assert tuple(np.round((sg * S[list(pm)])[1:4]).astype(int).ravel()) in keys, name
