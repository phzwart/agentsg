"""z:table-from-operators — Table 'sorts' computed exactly from the operators,
with no lattice and no numbers.

For a zero pattern Z: the operator set O_Z is the set of integer matrices
whose conorm action maps the face into the orthant (test_z52); its members
are O_Z modulo the 48 swaps; a sort is a closed set of equalities on the
face (test_z53), i.e. a subspace V with V = Fix(Sym(V)); its point group is
Sym(V); the symmetries visible in the reduced cell are Sym(V) intersected
with the swaps; two members have the same shape when their conorm vectors,
as linear forms on V, agree up to a swap. Every column of the table follows
by exact linear algebra on the matrices, and the result is the table, sort
by sort: members 1,2,3,6,16; |group| the holohedry; visible and shapes as
printed; shapes = orbits of Sym(V) on the members in every sort.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402
from .test_z52_universal_operators import face_ok, universal_set  # noqa: E402
from .test_z53_sorts_as_fixed_subspaces import FACES, _key, _rref_basis, fixed_subspaces, tie_pattern  # noqa: E402

# Table 'sorts' of the paper: (zeros, members, |group|, visible, shapes) by sort,
# with the sort's SELLA pattern to identify it by its equality classes
TABLE = {
    "C1": ("rrrrrr", 0, 1, 48, 48, 1), "C3": ("rr0rr0", 2, 3, 48, 16, 1), "C5": ("000rrr", 3, 16, 48, 12, 2),
    "H4": ("00rrrs", 2, 6, 24, 4, 1), "T1": ("rrsrrs", 0, 1, 16, 16, 1), "T2": ("rr0rrs", 1, 2, 16, 8, 1),
    "T5": ("000rrs", 3, 16, 16, 4, 3), "R1": ("rrrsss", 0, 1, 12, 12, 1), "R3": ("rr0sr0", 2, 3, 12, 4, 1),
    "O1A": ("rrsrrt", 0, 1, 8, 8, 1), "O1B": ("rstrst", 0, 1, 8, 8, 1), "O2": ("rs0srt", 1, 2, 8, 4, 1),
    "O3": ("rs0rs0", 2, 3, 8, 8, 2), "O4": ("00rsst", 2, 6, 8, 4, 2), "O5": ("000rst", 3, 16, 8, 2, 4),
    "M1A": ("rrsttu", 0, 1, 4, 4, 1), "M1B": ("rstrsu", 0, 1, 4, 4, 1), "M2A": ("rs0stu", 1, 2, 4, 2, 1),
    "M2B": ("rs0rst", 1, 2, 4, 4, 2), "M3": ("rs0ts0", 2, 3, 4, 4, 2), "M4": ("00rstu", 2, 6, 4, 2, 3),
    "A1": ("rstuvw", 0, 1, 2, 2, 1), "A2": ("rs0tuv", 1, 2, 2, 2, 2), "A3": ("rs0tu0", 2, 3, 2, 2, 3),
}
S6_TO_SLOT = [5, 4, 3, 0, 1, 2]       # S6 order bc,ac,ab,ad,bd,cd -> slot order da,db,dc,ab,ac,bc


def _sella_partition(pat):
    """equality classes of a SELLA pattern as frozensets of slot indices (zeros dropped)."""
    cls = {}
    for n, ch in enumerate(pat):
        if ch != "0":
            cls.setdefault(ch, set()).add(S6_TO_SLOT[n])
    return frozenset(frozenset(v) for v in cls.values()), frozenset(S6_TO_SLOT[n] for n, ch in enumerate(pat) if ch == "0")


def table_from_operators():
    out = {}
    W, L, sw, _ = universal_set()
    swap_idx = {tuple(s.ravel()) for s in sw}
    for face in FACES:
        subs, free, W, L, sw = fixed_subspaces(face)
        Z = FACES[face]
        ops = np.where(face_ok(L, Z))[0]
        # members: operators modulo swaps on the target (W -> Q W in row convention)
        keys = {tuple(W[i].ravel()): i for i in ops}
        member_of = {}
        m = 0
        for i in ops:
            if i in member_of:
                continue
            for s in sw:
                j = keys.get(tuple((s @ W[i]).ravel()))
                if j is not None:
                    member_of[j] = m
            m += 1
        members = m
        # the operators that keep the face (so that L_W acts on V)
        keep = [i for i in ops if (not Z) or np.abs(L[i][:, free][Z]).max() < 1e-9]
        seen = {}
        for B in subs:
            pat, _ = tie_pattern(B, free)
            P = B.T @ B
            images = [_key(_rref_basis(B @ L[i][:, free][free].astype(float).T)) for i in keep]
            if any(k in seen for k in images):
                continue
            # symmetry group: operators fixing V pointwise
            sym = []
            for i in ops:
                M = L[i][:, free].astype(float)
                A = np.vstack([M[free] - np.eye(len(free)), M[Z]]) if Z else M[free] - np.eye(len(free))
                if np.linalg.norm(A @ P) < 1e-9:
                    sym.append(i)
            visible = sum(tuple(W[i].ravel()) in swap_idx for i in sym)
            # orbits of Sym(V) on members: W -> g W
            orbit = {}
            for i in ops:
                mi = member_of[i]
                if mi in orbit:
                    continue
                for g in sym:
                    j = keys.get(tuple((W[i] @ W[g]).ravel()))      # rows: vectors in F0 coords, v -> v g
                    if j is not None:
                        orbit[member_of[j]] = mi
            n_orbits = len(set(orbit.values()))
            # shapes: members whose conorm forms on V agree up to a swap
            forms = {}
            rep_of_member = {}
            for i in ops:
                rep_of_member.setdefault(member_of[i], i)
            shape_keys = {}
            perms = []
            for sidx in [keys[tuple(s_.ravel())] for s_ in sw if tuple(s_.ravel()) in keys]:
                perms.append(L[sidx].astype(float))                # the 48 slot permutations (6x6)
            for mi, i in rep_of_member.items():
                Mf = L[i][:, free].astype(float) @ P                # the member's six conorms as forms on V
                best = None
                for Ls in perms:
                    kk = tuple(np.round(Ls @ Mf, 6).ravel())
                    best = kk if best is None or kk < best else best
                shape_keys[mi] = best
            n_shapes = len(set(shape_keys.values()))
            # stabiliser order of each member (symmetries that relabel it), keyed by the
            # (zeros, equalities) placement of each of its labellings, so that the table's
            # 'visible' can be read for the pattern as SELLA writes it
            stab = {}
            for mi, i in rep_of_member.items():
                n = 0
                for g in sym:
                    j = keys.get(tuple((W[i] @ W[g]).ravel()))
                    n += j is not None and member_of[j] == mi
                stab[mi] = n
            vis_by_pattern = {}
            for i in ops:
                Mf = np.round(L[i][:, free].astype(float) @ P, 9)
                zs = frozenset(r for r in range(6) if np.abs(Mf[r]).max() < 1e-9)
                cls = []
                for r in range(6):
                    if r in zs:
                        continue
                    for cl in cls:
                        if np.allclose(Mf[r], Mf[cl[0]]):
                            cl.append(r); break
                    else:
                        cls.append([r])
                vis_by_pattern[(zs, frozenset(frozenset(c) for c in cls))] = stab[member_of[i]]
            seen[_key(B)] = dict(zeros=len(Z), members=members, group=len(sym), visible=visible,
                                 shapes=n_shapes, orbits=n_orbits, vis_by_pattern=vis_by_pattern,
                                 partition=frozenset(frozenset(free[a] for a in cl) for cl in pat))
        out[face] = list(seen.values())
    return out


@claim("z54", "main §3.1", "SI Table S10", "SI Table S7", kind="exhaustive")
def test_table_sorts_follows_from_the_operators():
    r"""SI Table S10 follows from the operators: for each sort its zeros, superbases, group order, visible symmetries and shapes."""
    computed = table_from_operators()
    matched = {}
    for name, (pat, zeros, mem, grp, vis, shp) in TABLE.items():
        part, zs = _sella_partition(pat)
        hits = [c for face, sorts in computed.items() for c in sorts if (zs, part) in c["vis_by_pattern"]]
        assert len(hits) == 1, (name, len(hits))
        c = hits[0]
        assert (c["zeros"], c["members"], c["group"], c["shapes"]) == (zeros, mem, grp, shp), (name, c)
        assert c["vis_by_pattern"][(zs, part)] == vis, (name, c["vis_by_pattern"][(zs, part)], vis)
        assert c["orbits"] == c["shapes"], name                     # shapes are the symmetry orbits
        matched[name] = c
    assert len(matched) == 24
    # the sort placements found exhaust the computed sorts: 24 closed subspaces in all
    assert sum(len(v) for v in computed.values()) == 24
