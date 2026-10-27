"""z:fixed-subspaces — the 24 Delone sorts as the closed subspaces of a
Galois connection, with no lattice and no trial values.

For a face Z of the orthant (a zero pattern) the operators O_Z are the
integer matrices whose conorm action L_W maps the face into the orthant
(test_z52). A lattice with conorm vector p on the face has W as a symmetry
exactly when L_W p = p. So the symmetries of every lattice on a subspace V of
the face are Sym(V) = {W : L_W = id on V}, and the lattices whose symmetry
is exactly Sym(V) fill the fixed subspace Fix(Sym(V)) = intersection of
Fix(W) over W in Sym(V). A Delone sort is a closed subspace, V = Fix(Sym(V)),
that meets the interior of the face. The closed subspaces are the
intersections of the subspaces Fix(W) = {p : L_W p = p}, so they are
enumerated by taking all intersections of the finitely many Fix(W), keeping
those with an interior point, and identifying placements of the face under
the 48 swaps. The result is 8, 5, 5, 3, 3 sorts on the five face types, 24
in all, each a pattern of equalities, with Delone's holohedries as |Sym(V)|.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402
from .test_z46_selling_groupoid import SORTS  # noqa: E402
from .test_z52_universal_operators import face_ok, universal_set  # noqa: E402

SL = t.SL
FACES = {"none": [], "one": [3], "disjoint": [3, 2], "path": [3, 5], "triangle": [3, 4, 5]}


def _rref_basis(A, tol=1e-9):
    """orthonormal basis of the row space of A (rows), as a canonical array."""
    if A.shape[0] == 0:
        return np.zeros((0, A.shape[1]))
    u, s, vt = np.linalg.svd(A)
    r = int((s > tol).sum())
    return vt[:r]


def _key(Bs):
    """canonical key of a subspace given by an orthonormal basis: the projector, rounded."""
    P = Bs.T @ Bs
    return tuple(np.round(P, 6).ravel())


def _intersect(B1, B2):
    """intersection of two subspaces given by orthonormal bases (rows)."""
    k = B1.shape[1]
    P1, P2 = B1.T @ B1, B2.T @ B2
    # intersection = null space of (I-P1) stacked with (I-P2)
    A = np.vstack([np.eye(k) - P1, np.eye(k) - P2])
    u, s, vt = np.linalg.svd(A)
    r = int((s > 1e-9).sum())
    return vt[r:]                       # orthonormal basis of the null space


def fixed_subspaces(face):
    W, L, sw, _ = universal_set()
    Z = FACES[face]
    free = [m for m in range(6) if m not in Z]
    k = len(free)
    ops = np.where(face_ok(L, Z))[0]
    # Fix(W): p on the face with L_W p = p (also on the zero slots: L_W p must vanish there)
    subs = {}
    for i in ops:
        M = L[i][:, free].astype(float)
        A = np.vstack([M[free] - np.eye(k), M[Z]]) if Z else M[free] - np.eye(k)
        u, s, vt = np.linalg.svd(A)
        r = int((s > 1e-9).sum())
        B = vt[r:]
        subs[_key(B)] = B
    # close under intersection
    changed = True
    while changed:
        changed = False
        items = list(subs.values())
        for B1, B2 in itertools.combinations(items, 2):
            B = _intersect(B1, B2)
            kk = _key(B)
            if kk not in subs:
                subs[kk] = B; changed = True
    # keep those with a strictly positive point (the interior of the face)
    from scipy.optimize import linprog
    out = []
    for B in subs.values():
        if B.shape[0] == 0:
            continue
        N = B.T                                           # k x dim, p = N c
        res = linprog(np.zeros(N.shape[1]), A_ub=-N, b_ub=-np.ones(k),
                      bounds=[(None, None)] * N.shape[1], method="highs")
        if res.status == 0:
            out.append(B)
    return out, free, W, L, sw


def tie_pattern(B, free):
    """which free slots are equal on the subspace, as a partition; and whether
    the subspace is exactly the equality subspace of that partition."""
    k = len(free)
    P = B.T @ B
    classes = []
    for a in range(k):
        for cl in classes:
            b = cl[0]
            e = np.zeros(k); e[a] = 1; e[b] = -1
            if np.linalg.norm(P @ e) < 1e-9:             # p_a - p_b vanishes on V
                cl.append(a); break
        else:
            classes.append([a])
    return [tuple(c) for c in classes], (len(classes) == B.shape[0])


@claim("z53", "main Definition A5", "SI §S5", kind="exhaustive")
def test_sorts_are_the_closed_subspaces():
    r"""The closed subspaces Fix(Sym(V)) number 8, 5, 3, 5 and 3 for no zero, one zero, two sharing, two disjoint and three zeros, with the stated symmetry orders; their zero counts and equality patterns are those of Delone's 24 sorts as SELLA lists them (the sort-by-sort match is z54)."""
    counts = {}
    total = 0
    holohedries = {}
    signatures = []
    for face in FACES:
        subs, free, W, L, sw = fixed_subspaces(face)
        Z = FACES[face]
        # identify subspaces that are images of one another under operators of
        # the face (a sort seen from another of its members or placements)
        ops = np.where(face_ok(L, Z))[0]
        maps = []
        for i in ops:
            M = L[i][:, free].astype(float)
            if Z and np.abs(M[Z]).max() > 1e-9:
                continue                                   # leaves the face
            maps.append(M[free])
        seen = {}
        for B in subs:
            pat, exact = tie_pattern(B, free)
            assert exact, (face, pat)                      # every closed subspace is an equality pattern
            P = B.T @ B
            images = [_key(_rref_basis(B @ M.T)) for M in maps]
            if any(k in seen for k in images):
                continue
            # the symmetry group: operators of the face fixing the subspace pointwise
            n_sym = 0
            for i in ops:
                M = L[i][:, free].astype(float)
                A = np.vstack([M[free] - np.eye(len(free)), M[Z]]) if Z else M[free] - np.eye(len(free))
                n_sym += np.linalg.norm(A @ P) < 1e-9
            seen[_key(B)] = B
            holohedries.setdefault(face, []).append(n_sym)
            signatures.append((len(Z), tuple(sorted(len(c) for c in pat))))
        counts[face] = len(seen)
        holohedries[face] = sorted(holohedries[face])
        total += len(seen)
    assert counts == {"none": 8, "one": 5, "disjoint": 5, "path": 3, "triangle": 3} and total == 24
    # the symmetry orders are Delone's holohedries, face by face
    assert holohedries["none"] == [2, 4, 4, 8, 8, 12, 16, 48]          # A1 M1A M1B O1A O1B R1 T1 C1
    assert holohedries["one"] == [2, 4, 4, 8, 16]                      # A2 M2A M2B O2 T2
    assert holohedries["disjoint"] == [2, 4, 8, 12, 48]                # A3 M3 O3 R3 C3
    assert holohedries["path"] == [4, 8, 24]                           # M4 O4 H4
    assert holohedries["triangle"] == [8, 16, 48]                      # O5 T5 C5
    # and they are the SELLA patterns: same multiset of (zeros, equality-class sizes, |group|)
    sella = sorted((pat.count("0"), tuple(sorted(pat.count(ch) for ch in set(pat) - {"0"})))
                   for name, bt, pat in SORTS)
    assert len(sella) == 24 and sorted(signatures) == sella, (sorted(signatures), sella)
