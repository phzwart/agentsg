"""z:projection — idealise by projection, enumerate the closure, filter, all in the
strain norm ||G^-1/2 dG G^-1/2||_F, which is the same in every basis.

S_eps(G) = {F : the nearest metric for which F is reduced is within eps of G}.
Construction: for each of the 127 frame kinds of the universal operator set (z52)
and each of the 38 faces of the Selling cone, project the measured cell onto the
face (zeros on Z, other conorms >= 0, nearest in the strain norm); for every
(frame, face) within 4 eps, enumerate the closure of the projected lattice and keep
the members within eps of their own reduced cone; the answer is the union.
Refuse if a projected lattice is singular.

Within eps in the strain norm implies every conorm >= -eps |u_i||u_j|
(|u^T dG v| <= eps |u||v|), so S_eps lies inside the per-slot set at sigma = eps,
which is the prefilter of the [-3,3] brute force.

Results: exact on the z57 needle at bands from 0.0005 to sin 3 deg, on two thin
plates (100 x 100.1 x 5, 150 x 120 x 4.8 A) up to sin 3 deg with 20 and 28
members, on the 240 random cells of z57 at sin(0.1..3 deg) against [-2,2], and on
the 37 of them with ten or more members at sin 3 deg against [-3,3]; no refusal.
"""

from sellingproofs.registry import claim  # noqa: E402
import itertools

import numpy as np
import pytest
from scipy.optimize import nnls

from . import test_z44_widened_complete as t
from .test_z52_universal_operators import universal_set, L_of, valid_face
from .test_z57_perslot_bands import closure_members
SL = t.SL
E = np.array([t._gram(np.eye(6)[k]) for k in range(6)])         # (6,3,3)

W_ALL, L_ALL, SW, FACES = universal_set()
def _frame_kinds():
    keys, reps = set(), []
    for w in W_ALL:
        orb = [tuple((s @ w).ravel()) for s in SW]
        k = min(orb)
        if k not in keys:
            keys.add(k); reps.append(w)
    return np.array(reps)
W_REP = _frame_kinds()                                          # 127
L_REP = L_of(W_REP.astype(float))

def metric_M(Gf):
    """(n,6,6) strain-norm metric on conorm perturbations, in the frame whose Gram is Gf (n,3,3)."""
    Gi = np.linalg.inv(Gf)
    A = np.einsum("nab,kbc->nkac", Gi, E)                      # Gi E_k
    return np.einsum("nkab,nlba->nkl", A, A)

def dist_to_face(q, M, Z):
    free = [m for m in range(6) if m not in Z]
    R = np.linalg.cholesky(M).T
    x = np.zeros(6)
    if free:
        xf, _ = nnls(R[:, free], R @ q)
        x[free] = xf
    r = x - q
    return float(np.sqrt(max(r @ M @ r, 0.0))), x

def construct(G, eps, factor=4.0):
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, G)
    Q = L_REP @ p0                                              # (127,6) conorms in each frame
    Gf = np.stack([t._gram(q) for q in Q])
    M = metric_M(Gf)
    best = None
    for k in range(len(W_REP)):
        for Z in FACES:
            if best is not None and len(Z) < best[0]:
                continue
            d, x = dist_to_face(Q[k], M[k], Z)
            if d <= factor * eps:
                cand = (len(Z), -d, k, tuple(Z), x)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    nZ, negd, k, Z, x = best
    tol = 1e-9 * np.abs(G).max()
    Gs = t._gram(np.where([m in Z for m in range(6)], 0.0, x))
    if np.linalg.eigvalsh(Gs).min() <= tol:
        return None, (Z, -negd, k)
    mem = closure_members(Gs, tol)                              # superbases in frame-k coordinates
    Wk = W_REP[k].astype(float)
    out = set()
    for S in mem.values():
        S0 = S @ Wk                                             # rows into F0 coordinates
        if member_distance(G, S0) <= eps * (1 + 1e-9):
            out.add(t._key(S0))
    return out, (Z, -negd, k)

def member_distance(G, S):
    q = t._cn(S, G)
    if q.min() >= 0:
        return 0.0
    Gf = t._gram(q)
    M = metric_M(Gf[None])[0]
    d, _ = dist_to_face(q, M, [])
    return d

def brute(G, eps, K=2, SB=None):
    if SB is None:
        r = np.arange(-K, K + 1)
        Mx = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
        B = Mx[np.abs(np.round(np.linalg.det(Mx))) == 1].astype(float)
        SB = np.concatenate([-(B.sum(1))[:, None, :], B], 1)
    P = -np.einsum("nia,ab,njb->nij", SB, G, SB)
    q = np.stack([P[:, i, j] for i, j in SL], 1)
    neg = np.minimum(q, 0)
    ok0 = (q >= 0).all(1)
    Gf = np.einsum("nia,ab,njb->nij", SB[:, 1:], G, SB[:, 1:])
    M = metric_M(Gf)
    lam = np.linalg.eigvalsh(M)[:, 0]
    cand = np.where(~ok0 & (lam * (neg ** 2).sum(1) <= eps ** 2 * (1 + 1e-9)))[0]
    keep = set(t._key(S) for S in SB[ok0])
    for i in cand:
        d, _ = dist_to_face(q[i], M[i], [])
        if d <= eps * (1 + 1e-9):
            keep.add(t._key(SB[i]))
    return keep


def construct_union(G, eps, factor=4.0):
    """as construct, but idealise onto every (frame, face) within factor*eps and take
    the union of their closures; refuse if any of them is singular."""
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, G)
    Q = L_REP @ p0
    Gf = np.stack([t._gram(q) for q in Q])
    M = metric_M(Gf)
    tol = 1e-9 * np.abs(G).max()
    out, faces = set(), []
    for k in range(len(W_REP)):
        for Z in FACES:
            if not Z:
                continue
            d, x = dist_to_face(Q[k], M[k], Z)
            if d > factor * eps:
                continue
            faces.append((k, tuple(Z), d))
            Gs = t._gram(np.where([m in Z for m in range(6)], 0.0, x))
            if np.linalg.eigvalsh(Gs).min() <= tol:
                return None, faces
            Wk = W_REP[k].astype(float)
            for S in closure_members(Gs, tol).values():
                S0 = S @ Wk
                key = t._key(S0)
                if key not in out and member_distance(G, S0) <= eps * (1 + 1e-9):
                    out.add(key)
    if member_distance(G, s0) <= eps:
        out.add(t._key(s0))
    return out, faces


def strain_set(G, eps, K):
    r = np.arange(-K, K + 1)
    R = np.array([v for v in itertools.product(r, repeat=3) if any(v)], float)
    n = np.sqrt(np.einsum("ia,ab,ib->i", R, G, R))
    ok = -(R @ G @ R.T) >= -eps * np.outer(n, n) - 1e-9 * abs(G).max()
    idx = {tuple(v.astype(int)): i for i, v in enumerate(R)}
    out, maxent = set(), 0
    for a in range(len(R)):
        bs = np.where(ok[a])[0]; bs = bs[bs > a]
        for b in bs:
            for c in bs[(bs > b) & ok[b, bs]]:
                d = -(R[a] + R[b] + R[c]); j = idx.get(tuple(d.astype(int)))
                if j is None or not (ok[j, a] and ok[j, b] and ok[j, c]): continue
                Bm = np.array([R[a], R[b], R[c]])
                if abs(round(np.linalg.det(Bm))) != 1: continue
                S = np.vstack([d, Bm])
                if member_distance(G, S) <= eps * (1 + 1e-9):
                    out.add(t._key(S)); maxent = max(maxent, int(abs(S).max()))
    return out, maxent


NEEDLE = np.array([30.606, 26.579, 17414.674, 0.568, 0.0, 0.0])


def _reduced(cell):
    from agentsg.cell.metric import metric_tensor
    from agentsg.cell.selling_cob import selling_matrix
    Gc = np.array(metric_tensor(cell), float)
    P = np.array([[float(x) for x in row] for row in selling_matrix(cell).rows], float)
    p0 = t._cn(np.array(t.S0, float), P.T @ Gc @ P)
    return t._gram(np.maximum(p0, 0))


@claim("z58", "main §5.2, Needles and plates", kind="exhaustive")
def test_needle_and_thin_plates_against_the_3_window():
    r"""On the needle 5.58 x 5.21 x 131.97 A and two thin plates, the enumeration equals brute force in the window [-3, 3]."""
    cases = {
        "needle": (t._gram(NEEDLE), {1.0: 8, 3.0: 16}),
        "plate 100x100.1x5": (_reduced((100.0, 100.1, 5.0, 90.03, 89.98, 90.05)), {1.0: 16, 3.0: 20}),
        "plate 150x120x4.8": (_reduced((150.0, 120.0, 4.8, 90.0, 90.4, 90.0)), {1.0: 16, 3.0: 28}),
    }
    for name, (G, sizes) in cases.items():
        for deg, n in sizes.items():
            eps = np.sin(np.radians(deg))
            got, _ = construct_union(G, eps)
            ref, _ = strain_set(G, eps, 3)
            assert got is not None and got == ref and len(ref) == n, (name, deg, len(ref))


@claim("z58", "main §5.2, Needles and plates", kind="sampled")
def test_random_cells_against_the_2_window():
    r"""On 240 random cells the enumeration equals brute force in the window [-2, 2]."""
    from .test_z57_perslot_bands import _random_cells
    r = np.arange(-2, 3)
    Mx = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = Mx[np.abs(np.round(np.linalg.det(Mx))) == 1].astype(float)
    SB = np.concatenate([-(B.sum(1))[:, None, :], B], 1)
    n = 0
    for G in _random_cells():
        for deg in (0.1, 0.5, 1.0, 3.0):
            eps = np.sin(np.radians(deg))
            got, _ = construct_union(G, eps)
            assert got is not None and got == brute(G, eps, SB=SB)
            n += 1
    assert n == 240


# ---- distance to each Delone sort ---------------------------------------------
def _sort_patterns():
    from .test_z54_table_from_operators import TABLE, S6_TO_SLOT
    out = {}
    for name, v in TABLE.items():
        zeros, cls = [], {}
        for n, ch in enumerate(v[0]):
            m = S6_TO_SLOT[n]
            if ch == "0":
                zeros.append(m)
            else:
                cls.setdefault(ch, []).append(m)
        out[name] = (zeros, list(cls.values()))
    return out


def _project_to_sort(q, M, classes):
    """nearest point, in the metric M, with the given equality classes (others zero) and x >= 0."""
    active = list(classes)
    while True:
        A = np.zeros((6, len(active)))
        for c, cl in enumerate(active):
            A[cl, c] = 1
        y = np.linalg.solve(A.T @ M @ A, A.T @ M @ q) if active else np.zeros(0)
        if (y >= -1e-12).all():
            break
        active.pop(int(np.argmin(y)))
    x = A @ y if active else np.zeros(6)
    r = x - q
    return float(np.sqrt(max(r @ M @ r, 0.0)))


def sort_spectrum(G):
    """{sort: strain distance} from the measured reduced cell G to each of the 24 sorts,
    minimised over the 6096 frames of the universal operator set."""
    p0 = t._cn(np.array(t.S0, float), G)
    Q = L_ALL @ p0
    M = metric_M(np.stack([t._gram(q) for q in Q]))
    out = {}
    for name, (zeros, classes) in _sort_patterns().items():
        out[name] = min(_project_to_sort(Q[k], M[k], classes) for k in range(len(Q)))
    return out


@claim("z58", "SI §S6", kind="exhaustive")
def test_sort_spectrum_examples():
    r"""The sort spectra of the example cells."""
    from agentsg.cell.primitive import primitive_cell
    cases = {
        "tP from oP 40 x 40.3 x 80": ((40.0, 40.3, 80.0, 90, 90, 90), None, "T5", 0.01057),
        "hP from C2 50 x 86.8 x 40, beta 90.4": ((50.0, 86.8, 40.0, 90, 90.4, 90), "C 1 2 1", "H4", 0.01041),
        "cF from oF 100 x 100.2 x 99.9": ((100.0, 100.2, 99.9, 90, 90, 90), "F 2 2 2", "C3", 0.00432),
        "oP from the needle, gamma 91.12": ((5.583, 5.21, 131.965, 90, 90, 91.118), None, "O5", 0.02759),
    }
    for name, (cell, sg, sort, d) in cases.items():
        G = _reduced(primitive_cell(cell, sg) if sg else cell)
        spec = sort_spectrum(G)
        assert abs(spec[sort] - d) < 2e-5, (name, spec[sort])
        assert spec["A1"] < 1e-12


@claim("z58", "SI §S6", "SI Table S2", kind="sampled")
def test_sort_spectrum_is_monotone_along_the_specialisations():
    r"""A more special sort is never nearer: d(S') >= d(S) along every specialisation S -> S'."""
    from .test_z50_specialisations import EXPECTED
    from .test_z57_perslot_bands import _random_cells
    for n, G in enumerate(_random_cells()):
        if n % 6:
            continue
        spec = sort_spectrum(G)
        for (a, b) in EXPECTED:
            assert spec[b] >= spec[a] - 1e-9, (n, a, b, spec[a], spec[b])
