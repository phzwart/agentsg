"""z:specialisations — which sorts a lattice of each sort can turn into, through
which shape, under which linear condition on its parameters.

Each member of a closure, in each labelling, has conorms that are integer
combinations of the reduced cell's conorms, hence linear in the sort's free
parameters. Imposing another sort's pattern (its zeros and equalities) on
that member is a homogeneous linear system in the parameters. When the
system has a solution with every parameter positive, a lattice of sort X
becomes a lattice of sort Y along that shape. The conditions of codimension
one are listed and compared with the classical cell conditions: tI to cF at
c/a = sqrt 2, oC to hP at b = sqrt 3 a, mP to oC at a = c, and so on.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402
from .test_z46_selling_groupoid import SORTS, _orbits_and_shapes  # noqa: E402
from .test_z48_delone_derived import closure_data, sella  # noqa: E402

SL = t.SL
PERM4 = list(itertools.permutations(range(4)))
LETTERS = "rstuvw"


def _param_matrix(pat):
    """p (slots da,db,dc,ab,ac,bc) = A theta, theta the sort's free parameters."""
    letters = [ch for ch in LETTERS if ch in pat]
    s6 = [[1.0 if ch == L else 0.0 for L in letters] if ch != "0" else [0.0] * len(letters) for ch in pat]
    # S6 order bc,ac,ab,ad,bd,cd -> slot order da,db,dc,ab,ac,bc
    A = np.array([s6[3], s6[4], s6[5], s6[2], s6[1], s6[0]])
    return A, letters


def _member_maps(pat):
    """for each member and labelling: the 6x6 integer matrix L with
    conorms(member, labelling) = L p, and the member's shape id."""
    A, letters = _param_matrix(pat)
    theta = np.array([900.0, 1600.0, 2500.0, 400.0, 1200.0, 700.0][: len(letters)])
    p = A @ theta
    G = t._gram(p)
    mem, orbit_of, shape_of, _ = _orbits_and_shapes(p, G)
    out = []
    for k, S in mem.items():
        for pm in PERM4:
            U = S[list(pm)]
            L = np.zeros((6, 6))
            for n, (i, j) in enumerate(SL):
                # conorm_ij = -u_i G u_j, G = Gram(p) linear in p: differentiate per slot
                for m in range(6):
                    e = np.zeros(6); e[m] = 1.0
                    L[n, m] = -U[i] @ t._gram(e) @ U[j]
            out.append((shape_of[k], np.round(L).astype(int)))
    return A, letters, out


def _constraints(pat_y, LA):
    rows = []
    s6 = [LA[5], LA[4], LA[3], LA[0], LA[1], LA[2]]      # back to S6 order to match pat_y
    first = {}
    for n, ch in enumerate(pat_y):
        if ch == "0":
            rows.append(s6[n])
        elif ch in first:
            rows.append(s6[n] - s6[first[ch]])
        else:
            first[ch] = n
    # non-zero letters must stay non-zero and distinct letters distinct: checked after solving
    return np.array(rows) if rows else np.zeros((0, LA.shape[1]))


def _positive_nullspace_point(C, k, rnd):
    """a generic point of the null space of C with every parameter >= 0,
    strictly positive wherever the null space allows it, or None."""
    from scipy.optimize import linprog
    if C.shape[0] == 0:
        return rnd.uniform(1, 2, k)
    u, s, vt = np.linalg.svd(C)
    rank = int((s > 1e-9).sum())
    N = vt[rank:].T
    if N.shape[1] == 0:
        return None
    m = N.shape[1]
    A_eq = (N.sum(0))[None, :]
    free = []
    for i in range(k):
        res = linprog(-N[i], A_ub=-N, b_ub=np.zeros(k), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(None, None)] * m, method="highs")
        if res.status != 0:
            return None
        if -res.fun > 1e-9:
            free.append(i)
    if not free:
        return None
    # maximise the smallest free coordinate, then perturb inside the face
    cvec = np.zeros(m + 1); cvec[-1] = -1.0
    A_ub = np.hstack([-N, np.array([[1.0 if i in free else 0.0] for i in range(k)])])
    res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(k), A_eq=np.hstack([A_eq, [[0.0]]]), b_eq=[1.0],
                  bounds=[(None, None)] * m + [(0, None)], method="highs")
    if res.status != 0 or res.x[-1] <= 1e-9:
        return None
    c0 = res.x[:m]
    x0 = N @ c0
    for _ in range(100):
        c = c0 + rnd.normal(scale=0.2, size=m) * np.linalg.norm(c0)
        x = N @ c
        if (x[free] > 0.3 * x0[free]).all() and (x >= -1e-12).all():
            return x
    return x0


def _canonical_sort_table():
    rnd = np.random.default_rng(3)
    table = {}
    for name, bt, pat in SORTS:
        A, letters = _param_matrix(pat)
        theta = rnd.uniform(400, 3000, len(letters))
        best, *_ = closure_data(A @ theta)
        table[best] = (name, bt)
    assert len(table) == 24
    return table


def _classify(p, table):
    best, *_ = closure_data(p)
    return table.get(best, ("?", best))


def _condition_text(row, letters):
    terms = []
    for c, L in zip(row, letters):
        if abs(c) < 1e-9:
            continue
        c = int(round(c))
        terms.append(("" if c == 1 else "-" if c == -1 else f"{c}") + L)
    return " + ".join(terms).replace("+ -", "- ") + " = 0"


def specialisations(codim_max=1):
    """returns {(X, Y): [(condition text, shape, codim), ...]} over all sorts."""
    rnd = np.random.default_rng(50)
    table = _canonical_sort_table()
    out = {}
    for name, bt, pat in SORTS:
        A, letters, maps = _member_maps(pat)
        k = len(letters)
        if k == 0:
            continue
        seen = set()
        for shape, L in maps:
            LA = L @ A
            for yname, ybt, ypat in SORTS:
                if yname == name:
                    continue
                C = _constraints(ypat, LA)
                if C.shape[0] == 0:
                    continue
                # independent rows
                u, s, vt = np.linalg.svd(C) if C.size else (None, np.zeros(0), None)
                rank = int((s > 1e-9).sum())
                if rank == 0 or rank > codim_max:
                    continue
                x = _positive_nullspace_point(C, k, rnd)
                if x is None:
                    continue
                p = A @ x
                if np.linalg.eigvalsh(t._gram(p)).min() <= 1e-9 * p.max():
                    continue
                got = _classify(p, table)[0]
                if got != yname:
                    continue                       # the condition lands in a deeper sort
                # canonical row (codim 1): integer, positive leading coefficient
                r = vt[0] if rank == 1 else None
                r = r / np.abs(r[np.abs(r) > 1e-9]).min()
                r = np.round(r).astype(int)
                if r[np.nonzero(r)[0][0]] < 0:
                    r = -r
                key = (yname, tuple(r))
                if key in seen:
                    continue
                seen.add(key)
                out.setdefault((name, yname), []).append((_condition_text(r, letters), shape, rank))
    return out


# every codimension-one specialisation between sorts, as (X, Y): conditions on
# the letters of X's pattern (Table 'sorts'), with the number of distinct cell
# shapes of X through which they are reached
EXPECTED = {
    ("A1", "A2"): ({"r = 0", "s = 0", "t = 0", "u = 0", "v = 0", "w = 0"}, 1),
    ("A2", "A3"): ({"v = 0"}, 1), ("A2", "M2A"): ({"s - t = 0", "r - u = 0"}, 1),
    ("A2", "M4"): ({"r = 0", "s = 0", "t = 0", "u = 0"}, 1),
    ("A3", "M3"): ({"r - s = 0", "r - t = 0", "r - u = 0", "s - t = 0", "s - u = 0", "t - u = 0"}, 3),
    ("M1A", "O1A"): ({"r - t = 0"}, 1), ("M1A", "O2"): ({"u = 0"}, 1), ("M1A", "O4"): ({"r = 0", "t = 0"}, 1),
    ("M1B", "M2B"): ({"t = 0", "u = 0"}, 1), ("M1B", "M3"): ({"r = 0", "s = 0"}, 1),
    ("M1B", "O1A"): ({"r - s = 0"}, 1), ("M1B", "O1B"): ({"t - u = 0"}, 1),
    ("M2A", "O2"): ({"r - t = 0"}, 1),
    ("M2B", "O3"): ({"t = 0"}, 1), ("M2B", "T2"): ({"r - s = 0"}, 1), ("M2B", "T5"): ({"r = 0", "s = 0"}, 1),
    ("M3", "O3"): ({"r - t = 0"}, 1), ("M3", "R3"): ({"r - s = 0", "s - t = 0"}, 1),
    ("M4", "O4"): ({"r - s = 0", "r - t = 0", "s - t = 0"}, 3),
    ("M4", "O5"): ({"r = 0", "s = 0", "t = 0"}, 3),
    ("O1A", "T1"): ({"s - t = 0"}, 1), ("O1A", "T2"): ({"s = 0", "t = 0"}, 1),
    ("O1B", "O3"): ({"r = 0", "s = 0", "t = 0"}, 1), ("O1B", "T1"): ({"r - s = 0", "r - t = 0", "s - t = 0"}, 1),
    ("O2", "T2"): ({"r - s = 0"}, 1), ("O3", "C3"): ({"r - s = 0"}, 1),
    ("O4", "H4"): ({"r - s = 0"}, 1), ("O4", "T5"): ({"r = 0"}, 1),
    ("O5", "T5"): ({"r - s = 0", "r - t = 0", "s - t = 0"}, 1),
    ("R1", "C1"): ({"r - s = 0"}, 1), ("R1", "C5"): ({"r = 0"}, 1), ("R3", "C3"): ({"r - s = 0"}, 1),
    ("T1", "C1"): ({"r - s = 0"}, 1), ("T1", "C3"): ({"s = 0"}, 1), ("T2", "C3"): ({"s = 0"}, 1),
    ("T5", "C5"): ({"r - s = 0"}, 1),
}


@claim("z50", "SI Table S2", kind="exhaustive")
def test_codimension_one_specialisations():
    r"""SI Table S2: the 36 codimension-one specialisations between sorts, with their conditions and shape counts."""
    sp = specialisations(1)
    got = {k: ({c for c, _, _ in v}, len({sh for _, sh, _ in v})) for k, v in sp.items()}
    assert got == EXPECTED, {k: got.get(k) for k in set(got) | set(EXPECTED) if got.get(k) != EXPECTED.get(k)}
    # the three monoclinic conditions for oC and for oP are carried by the
    # three cell shapes (cell choices), one each; everywhere else by one shape
    assert got[("M4", "O4")][1] == 3 and got[("M4", "O5")][1] == 3 and got[("A3", "M3")][1] == 3
    # no specialisation lowers the number of symmetries: every target has a
    # larger or equal holohedry
    order = {name: {"cI": 48, "cF": 48, "cP": 48, "hP": 24, "tI": 16, "tP": 16, "hR": 12, "oF": 8, "oI": 8,
                    "oS": 8, "oP": 8, "mC": 4, "mP": 4, "aP": 2}[bt] for name, bt, pat in SORTS}
    assert all(order[y] >= order[x] for x, y in got)
