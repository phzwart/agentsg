"""z:window — the obtuse superbases of every zero pattern enumerated inside the
window that Lemma A2 forces, with no input from Kurlin's lists and no
assumed entry bound.

Let F0 be obtuse for a lattice whose non-zero conorms connect the four
vectors, and F any obtuse superbase of the same lattice. Every partial sum
of F is a shortest vector of its parity class, so its excess over F0's
partial sum in that class is zero; by the class excess identity that excess
is 4 sum p0_ij gamma_ij with every term >= 0, so gamma_ij = 0 on every
non-zero slot, i.e. |k_i - k_j| <= 1 along a connected graph on four
vertices, and with k_0 = 0 every |k_i| <= 3. The vectors of F are therefore
partial sums of F0 with superbase coordinates s_i + 2 k_i, s in {0,1}^4,
|k_i| <= 3: a finite window. Enumerating all superbases of such vectors
whose six pairwise products are non-positive on the whole face (the sign
test of the supporting information (Section S3)) gives 1, 2, 3, 6 and 16 members for the five kinds of
zero pattern. Over all 38 faces (every set of zero slots whose complement connects the
four vectors) the members have basis entries in {-1,0,1} when the zeros avoid d, and up
to 2 or 3 in the basis (a, b, c) of F0 for the faces whose zeros touch d -- the path
placement of three zeros needs 3; e.g. conorms (0,0,3,2,0,5) have the obtuse basis
(-3a-2b-c, a, a+b). Each member has a labelling whose basis lies in [-2,2] (z52).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytestmark = [pytest.mark.zcheck]

import test_z44_widened_complete as t  # noqa: E402

SL = t.SL
UNIT = [t._gram(np.eye(6)[m]) for m in range(6)]
FACES = {"none": [], "one": [3], "disjoint": [3, 2], "path": [3, 5], "triangle": [3, 4, 5]}


def window_vectors(kmax=3):
    cands = set()
    for s in itertools.product((0, 1), repeat=4):
        if sum(s) in (0, 4):
            continue
        for k in itertools.product(range(-kmax, kmax + 1), repeat=3):
            c = [s[0]] + [s[i + 1] + 2 * k[i] for i in range(3)]
            v = (c[1] - c[0], c[2] - c[0], c[3] - c[0])          # basis coordinates
            if v != (0, 0, 0):
                cands.add(v)
    return np.array(sorted(cands))


def members_in_window(Z, cands):
    free = [m for m in range(6) if m not in Z]
    n = len(cands)
    F = np.stack([cands @ UNIT[m] @ cands.T for m in range(6)], -1)      # coefficients of u.v in the conorms
    okpair = (-F[:, :, free] >= 0).all(-1)                                 # -u.v >= 0 on the face
    idx = {tuple(v): i for i, v in enumerate(cands)}
    members = set()
    for a in range(n):
        bs = np.where(okpair[a, :])[0]
        bs = bs[bs > a]
        for b in bs:
            for c in bs[(bs > b) & okpair[b, bs]]:
                d = tuple(-(cands[a] + cands[b] + cands[c]))
                di = idx.get(d)
                if di is None or not (okpair[di, a] and okpair[di, b] and okpair[di, c]):
                    continue
                B = np.array([cands[a], cands[b], cands[c]])
                if abs(round(np.linalg.det(B))) != 1:
                    continue
                rows = sorted(map(tuple, np.vstack([B, [d]])))
                neg = sorted(tuple(-np.array(r)) for r in rows)
                members.add(min(tuple(rows), tuple(neg)))
    return members


def test_counts_from_the_window_lemma():
    cands = window_vectors(3)
    assert len(cands) == 2948
    counts = {}
    for name, Z in FACES.items():
        mem = members_in_window(Z, cands)
        counts[name] = len(mem)
        assert max(abs(x) for m in mem for r in m for x in r) == 1       # the window is far from tight
    assert counts == {"none": 1, "one": 2, "disjoint": 3, "path": 6, "triangle": 16}


def _connected(free):
    adj = {v: set() for v in range(4)}
    for m in free:
        a, b = SL[m]
        adj[a].add(b)
        adj[b].add(a)
    seen, st = {0}, [0]
    while st:
        x = st.pop()
        for y in adj[x] - seen:
            seen.add(y)
            st.append(y)
    return len(seen) == 4


def test_all_38_faces():
    faces = [Z for r in range(4) for Z in itertools.combinations(range(6), r)
             if _connected([m for m in range(6) if m not in Z])]
    assert len(faces) == 38
    cands = window_vectors(3)
    by_kind = {}
    for Z in faces:
        verts = {v for m in Z for v in SL[m]}
        kind = {0: "none", 1: "one"}.get(len(Z)) or (
            ("share" if len(verts) == 3 else "disjoint") if len(Z) == 2 else ("triangle" if len(verts) == 3 else "path"))
        mem = members_in_window(list(Z), cands)
        mx = max(abs(x) for m in mem for r in m for x in r)
        by_kind.setdefault(kind, set()).add((len(mem), mx))
        if 0 not in verts:                                   # zeros avoid d: entries in {-1, 0, 1}
            assert mx == 1, Z
    assert {k: {n for n, _ in v} for k, v in by_kind.items()} == {
        "none": {1}, "one": {2}, "disjoint": {3}, "share": {6}, "triangle": {16}, "path": {16}}
    assert max(mx for v in by_kind.values() for _, mx in v) == 3
    # the referee's example: conorms (0,0,3,2,0,5), basis (-3a-2b-c, a, a+b) is obtuse
    G = t._gram(np.array([0, 0, 3, 2, 0, 5], float))
    B = np.array([[-3, -2, -1], [1, 0, 0], [1, 1, 0]], float)
    S = np.vstack([-(B.sum(0)), B])
    assert (-(S @ G @ S.T))[np.triu_indices(4, 1)].min() >= -1e-12 and abs(round(np.linalg.det(B))) == 1


def test_window_members_lie_in_the_universal_set():
    """Completeness of U (z52) from Lemma A1 alone: every member of every one of the 38 faces,
    found in the full window |k_i| <= 3 with no entry bound, is in U in all 48 labellings."""
    import test_z52_universal_operators as u
    W, *_ = u.universal_set()
    keys = {tuple(w.ravel()) for w in W}
    faces = [Z for r in range(4) for Z in itertools.combinations(range(6), r)
             if _connected([m for m in range(6) if m not in Z])]
    cands = window_vectors(3)
    n = 0
    for Z in faces:
        for mem in members_in_window(list(Z), cands):
            S = np.array(mem)
            for pm in itertools.permutations(range(4)):
                for sg in (1, -1):
                    assert tuple((sg * S[list(pm)])[1:4].ravel()) in keys, (Z, mem)
            n += 1
    assert n == sum(len(members_in_window(list(Z), cands)) for Z in faces)
