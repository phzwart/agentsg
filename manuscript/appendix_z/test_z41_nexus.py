"""z:nexus — the Bravais skeleton from stacked conorm constraints.

A Bravais family is a list of linear constraints on the conorms (zero
slots and tied slots), read here off generic reduced cells of the family
and taken on whole S4 relabelling orbits. Where two families meet is the
solution set of the stacked lists inside the obtuse cone: a nexus is a
common ray with a positive-definite metric, a seam is a common ray only on
the singular boundary (infinite axial ratio), and an exact edge is a
common face of dimension two. Recovers: cP from tP and hR; cI and cF,
exactly two, from tI and hR; the hP/hR and tP/tI seams; the oC/tP,
oI/tI and oF/tI edges; and the family dimensions 1/2/3.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.zcheck]

def _lib():
    import itertools, random
    import numpy as np
    from scipy.optimize import linprog
    from agentsg.cell.canonical import _closure_for_match
    from agentsg.cell.metric import UnitCell, params_from_metric
    SLOTS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    
    def conorms(cell):
        G = np.array(UnitCell(*cell).metric_tensor())
        M = np.array(_closure_for_match(cell, boundary_rel=0.0)[0], float)
        return np.array([-M[i]@G@M[j] for i,j in SLOTS])
    
    def pattern_rows(p, tol=1e-6):
        s = max(p); rows = []
        zeros = [k for k in range(6) if abs(p[k]) < tol*s]
        for k in zeros:
            r = np.zeros(6); r[k] = 1; rows.append(r)
        classes = []
        for k in range(6):
            if k in zeros: continue
            for c in classes:
                if abs(p[c[0]]-p[k]) < tol*s: c.append(k); break
            else: classes.append([k])
        for c in classes:
            for k in c[1:]:
                r = np.zeros(6); r[c[0]] = 1; r[k] = -1; rows.append(r)
        return np.array(rows) if rows else np.zeros((0,6))
    
    def _slot_perm(g):
        idx = {s:i for i,s in enumerate(SLOTS)}
        return [idx[tuple(sorted((g[i],g[j])))] for i,j in SLOTS]
    PERMS = [_slot_perm(g) for g in itertools.permutations(range(4))]
    
    def relabel(rows, sp):
        out = np.zeros_like(rows)
        for k in range(6): out[:, sp[k]] = rows[:, k]
        return out
    
    def canon(rows):
        cands = []
        for sp in PERMS:
            rr = []
            for row in relabel(rows, sp):
                if row[np.nonzero(row)[0][0]] < 0: row = -row
                rr.append(tuple(np.round(row, 6)))
            cands.append(tuple(sorted(rr)))
        return min(cands)
    
    def face(rows):
        """generators of {p: rows p = 0, p >= 0}; returns (dim, list of extreme-ish rays)"""
        if len(rows) == 0: N = np.eye(6)
        else:
            u, s, vt = np.linalg.svd(rows); r = int((s > 1e-9).sum()); N = vt[r:].T
        if N.shape[1] == 0: return 0, []
        rays = []
        for k in range(6):
            res = linprog(-N[k], A_ub=-N, b_ub=np.zeros(6), bounds=[(-1e3,1e3)]*N.shape[1], method="highs")
            if res.status == 0:
                v = N @ res.x
                if v[k] > 1e-7: rays.append(v / v.sum())
        if not rays: return 0, []
        return int(np.linalg.matrix_rank(np.array(rays), tol=1e-6)), rays
    
    def gram(p):
        return np.array([[p[0]+p[3]+p[4], -p[3], -p[4]],
                         [-p[3], p[1]+p[3]+p[5], -p[5]],
                         [-p[4], -p[5], p[2]+p[4]+p[5]]])
    
    def shape_key(p):
        V = np.sqrt(max(np.linalg.det(gram(p)), 0.0))
        return np.sort(p / np.sqrt(p.sum())) / V ** (1/3)
    
    def prim(cell, centring):
        from math import radians, cos, sin
        a,b,c,al,be,ga = cell
        A = np.array([[a,0,0],[b*cos(radians(ga)), b*sin(radians(ga)), 0], [0,0,0]])
        cx = c*cos(radians(be)); cy = c*(cos(radians(al))-cos(radians(be))*cos(radians(ga)))/sin(radians(ga))
        A[2] = [cx, cy, np.sqrt(max(c*c-cx*cx-cy*cy, 1e-12))]
        T = {"P":np.eye(3), "I":np.array([[-.5,.5,.5],[.5,-.5,.5],[.5,.5,-.5]]),
             "F":np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]]), "C":np.array([[.5,.5,0],[-.5,.5,0],[0,0,1]]),
             "R":np.array([[2/3,1/3,1/3],[-1/3,1/3,1/3],[-1/3,-2/3,1/3]])}[centring]
        V = T @ A
        return tuple(params_from_metric(V @ V.T))
    
    def families(seed=3):
        rnd = random.Random(seed); U = rnd.uniform
        return {
         "cP": lambda: prim((30,30,30,90,90,90),"P"),
         "cI": lambda: prim((30,30,30,90,90,90),"I"),
         "cF": lambda: prim((30,30,30,90,90,90),"F"),
         "tP": lambda: prim((30,30,U(10,90),90,90,90),"P"),
         "tI": lambda: prim((30,30,U(10,90),90,90,90),"I"),
         "hP": lambda: prim((30,30,U(10,90),90,90,120),"P"),
         "hR": lambda: prim((30,30,U(5,150),90,90,120),"R"),
         "oP": lambda: prim((30,U(31,60),U(61,99),90,90,90),"P"),
         "oI": lambda: prim((30,U(31,60),U(61,99),90,90,90),"I"),
         "oF": lambda: prim((30,U(31,60),U(61,99),90,90,90),"F"),
         "oC": lambda: prim((30,U(31,60),U(61,99),90,90,90),"C"),
        }
    
    def strata(n=60, seed=3):
        out = {}
        for name, f in families(seed).items():
            pats = {}
            for _ in range(n):
                rows = pattern_rows(conorms(f()))
                pats.setdefault(canon(rows), rows)
            out[name] = list(pats.values())
        return out
    
    def meet(S, A, B):
        """all 1-dim meets (rays) of strata A and B over patterns and labellings,
        classified as nexus (positive-definite metric) or seam (singular)."""
        nexus, seam, maxdim = [], [], 0
        for ra in S[A]:
            for rb in S[B]:
                for sp in PERMS:
                    rows = np.vstack([ra, relabel(rb, sp)])
                    d, rays = face(rows)
                    maxdim = max(maxdim, d)
                    if d == 1:
                        p = rays[0]
                        if np.linalg.det(gram(p)) > 1e-9:
                            k = np.round(shape_key(p), 3)
                            if not any(np.allclose(k, q) for q in nexus): nexus.append(k)
                        else:
                            z = tuple(np.round(np.sort(p), 3))
                            if z not in seam: seam.append(z)
        return maxdim, nexus, seam
    return dict(strata=strata, meet=meet, face=face)


_S = {}


def _strata():
    if "S" not in _S:
        L = _lib()
        _S["L"] = L
        _S["S"] = L["strata"](n=30, seed=3)
    return _S["L"], _S["S"]


def test_family_dimensions():
    """cubic families are rays, tetragonal/hexagonal/rhombohedral 2-D,
    orthorhombic 3-D -- one constraint list per reduced pattern."""
    L, S = _strata()
    want = {"cP": 1, "cI": 1, "cF": 1, "tP": 2, "tI": 2, "hP": 2,
            "hR": 2, "oP": 3, "oI": 3, "oF": 3, "oC": 3}
    for name, d in want.items():
        assert all(L["face"](rows)[0] == d for rows in S[name]), name


def _close(k, ref):
    return all(abs(a - b) < 2e-3 for a, b in zip(k, ref))


CP = (0, 0, 0, 0.577, 0.577, 0.577)
CI = (0.257,) * 6
CF = (0, 0, 0.397, 0.397, 0.397, 0.397)


def test_nexus_points_are_positive_definite_meets():
    L, S = _strata()
    d, nexus, _ = L["meet"](S, "tP", "hR")
    assert d == 1 and len(nexus) == 1 and _close(nexus[0], CP)
    d, nexus, _ = L["meet"](S, "tI", "hR")
    assert d == 1 and len(nexus) == 2          # exactly two nexus
    assert any(_close(k, CI) for k in nexus)
    assert any(_close(k, CF) for k in nexus)


def test_seams_meet_only_on_the_singular_boundary():
    L, S = _strata()
    for A, B in (("hP", "hR"), ("tP", "tI")):
        d, nexus, seam = L["meet"](S, A, B)
        assert nexus == [] and len(seam) >= 1, (A, B)


def test_exact_edges_are_common_faces():
    L, S = _strata()
    for A, B in (("oC", "tP"), ("oI", "tI"), ("oF", "tI")):
        d, _, _ = L["meet"](S, A, B)
        assert d == 2, (A, B)
