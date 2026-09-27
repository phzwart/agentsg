"""Translationengleiche (t) and klassengleiche (k) subgroups, derived from ops.

t (type I): same translation lattice, a point-group subgroup.
k (type II): same point group, fewer translations.
  IIa — drop a G-invariant subset of the conventional centring (same cell).
  IIb — G-invariant finite-index sublattice of Z³ (enlarged cell).

Isomorphic k-series (arbitrary 2a, 3c, …) are infinite; this module returns
the small-index ones that come out of the operator list, not ITA A1 tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Fr
from functools import lru_cache
from itertools import combinations, product
from typing import Iterable, Literal

from .change_of_basis import ChangeOfBasis
from .group import centering_translations, close_group, point_group
from .identify import identify_space_group
from .linalg import IDENTITY3, ZERO3, Matrix3, Vector3
from .setting import _is_crystallographic_W, _lattice_coset_ops, format_cob
from .space_groups import SpaceGroup, space_group
from .symmetry_op import SymmetryOp

# F/I conventional cells of the same lattice (4-fold along c, b, or a).
_CONTRACTION_P: tuple[Matrix3, ...] = (
    Matrix3([[Fr(1, 2), Fr(-1, 2), 0], [Fr(1, 2), Fr(1, 2), 0], [0, 0, 1]]),
    Matrix3([[Fr(1, 2), 0, Fr(-1, 2)], [0, 1, 0], [Fr(1, 2), 0, Fr(1, 2)]]),
    Matrix3([[1, 0, 0], [0, Fr(1, 2), Fr(-1, 2)], [0, Fr(1, 2), Fr(1, 2)]]),
)

# 45° in a coordinate plane: diagonal 2-folds → axis-aligned ITA settings.
_DIAG_P: tuple[Matrix3, ...] = (
    Matrix3([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
    Matrix3([[1, 0, 1], [0, 1, 0], [-1, 0, 1]]),
    Matrix3([[1, 0, 0], [0, 1, 1], [0, -1, 1]]),
)


@dataclass(frozen=True)
class SubgroupEdge:
    """One derived group → subgroup relation."""

    parent: int
    parent_hm: str
    child: int
    child_hm: str
    type: Literal["t", "k"]
    kind: str  # I, IIa, IIb
    index: int
    isomorphic: bool
    cob: str
    embeddings: int = 1


def _mat_close(seed: Iterable[Matrix3]) -> frozenset[Matrix3]:
    s = set(seed)
    s.add(IDENTITY3)
    grew = True
    while grew:
        grew = False
        extra = {a @ b for a in s for b in s}
        if not extra <= s:
            s |= extra
            grew = True
        if len(s) > 48:
            return frozenset()
    return frozenset(s)


def _f2_nullspace(rows: list[list[int]]) -> list[list[int]]:
    """Basis of the nullspace of a matrix over F2 (row lists of 0/1)."""
    if not rows:
        return []
    n = len(rows[0])
    A = [row[:] for row in rows]
    pivots = []
    r = 0
    for col in range(n):
        piv = None
        for i in range(r, len(A)):
            if A[i][col]:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(len(A)):
            if i != r and A[i][col]:
                A[i] = [(A[i][j] ^ A[r][j]) for j in range(n)]
        pivots.append((r, col))
        r += 1
    pivot_cols = {c for _, c in pivots}
    free = [j for j in range(n) if j not in pivot_cols]
    basis = []
    for f in free:
        vec = [0] * n
        vec[f] = 1
        for pr, pc in pivots:
            if A[pr][f]:
                vec[pc] = 1
        basis.append(vec)
    return basis


def _index2_subgroups(Ws: frozenset[Matrix3]) -> list[frozenset[Matrix3]]:
    """Index-2 subgroups = kernels of nontrivial homomorphisms G → C2."""
    elems = [IDENTITY3] + [W for W in Ws if W != IDENTITY3]
    idx = {W: i for i, W in enumerate(elems)}
    n = len(elems)
    rows = [[0] * n]
    rows[0][0] = 1  # χ(I) = 0 in F2
    for a in elems:
        ia = idx[a]
        for b in elems:
            row = [0] * n
            row[ia] ^= 1
            row[idx[b]] ^= 1
            row[idx[a @ b]] ^= 1
            rows.append(row)
    out = []
    seen: set[frozenset[Matrix3]] = set()
    rots = frozenset(W for W in Ws if W.det() == 1)
    if len(rots) * 2 == len(Ws):
        seen.add(rots)
        out.append(rots)
    for vec in _f2_nullspace(rows):
        if not any(vec):
            continue
        H = frozenset(elems[i] for i, bit in enumerate(vec) if bit == 0)
        if 0 < len(H) < n and H not in seen:
            seen.add(H)
            out.append(H)
    return out


def point_group_subgroups(Ws: frozenset[Matrix3]) -> list[frozenset[Matrix3]]:
    """Subgroups of a crystallographic point group.

    Small groups (|P|≤16): cyclic plus 2-generated. Larger groups (cubic):
    cyclic plus index-2 kernels — enough for maximal t-subgroups, avoids
    closing every pair in m-3m.
    """
    found = {frozenset([IDENTITY3])}
    elems = [e for e in Ws if e != IDENTITY3]
    for e in elems:
        c = _mat_close({e})
        if c:
            found.add(c)
    if len(Ws) <= 16:
        for a, b in combinations(elems, 2):
            c = _mat_close({a, b})
            if c and c <= Ws:
                found.add(c)
    else:
        found.update(_index2_subgroups(Ws))
    if Ws:
        found.add(Ws)
    return [s for s in found if s]


def _maximal_among(subs: list[frozenset[Matrix3]], whole: frozenset[Matrix3]):
    out = []
    for h in subs:
        if h == whole:
            continue
        if any(h < k < whole for k in subs):
            continue
        out.append(h)
    return out


def _identify_hit(ops) -> tuple[int, str, str] | None:
    """Origin + unimodular CoB only — full |det|≤4 search is too slow on misses."""
    hit = identify_space_group(ops, allow_cob=True, max_det=1, signed_perm=True)
    if hit is not None:
        return hit.number, hit.hermann_mauguin, format_cob(hit.change_of_basis)
    return None


def _identify_diag(ops) -> tuple[int, str, str] | None:
    """Rotate diagonal 2-folds onto the crystal axes (det ±2, no expansion)."""
    src = list(ops)
    for P in _DIAG_P:
        cob = ChangeOfBasis(P, ZERO3)
        try:
            transformed = [cob.apply_to_op(op) for op in src]
        except Exception:
            continue
        if any(not _is_crystallographic_W(op.W) for op in transformed):
            continue
        hit = _identify_hit(list(frozenset(transformed)))
        if hit is not None:
            return hit
    return None


def _identify_contracted(ops) -> tuple[int, str, str] | None:
    """F-cell 4/mmm etc. → I-cell standard (same lattice, smaller conventional cell)."""
    for P in _CONTRACTION_P:
        cob = ChangeOfBasis(P, ZERO3)
        try:
            transformed = [cob.apply_to_op(op) for op in ops]
        except Exception:
            continue
        if any(not _is_crystallographic_W(op.W) for op in transformed):
            continue
        uniq = list(frozenset(transformed))
        if len(uniq) >= len(list(ops)):
            continue
        hit = _identify_hit(uniq)
        if hit is not None:
            return hit
    return None


def _identify(ops) -> tuple[int, str, str] | None:
    hit = _identify_hit(ops)
    if hit is not None:
        return hit
    hit = _identify_diag(ops)
    if hit is not None:
        return hit
    return _identify_contracted(ops)


def _t_edges(rec: SpaceGroup, *, maximal: bool) -> list[SubgroupEdge]:
    ops = list(rec.operations())
    pgrp = point_group(ops)
    if len(pgrp) > 16:
        # Cubic: the incomplete subgroup lattice would mark stray cyclics as maximal.
        chosen = _index2_subgroups(pgrp)
    else:
        subs = point_group_subgroups(pgrp)
        chosen = (
            _maximal_among(subs, pgrp)
            if maximal
            else [h for h in subs if h != pgrp]
        )
    buckets: dict[int, list[SubgroupEdge]] = {}
    for h in chosen:
        sub_ops = [op for op in ops if op.W in h]
        if not sub_ops or len(sub_ops) == rec.order():
            continue
        hit = _identify(sub_ops)
        if hit is None:
            continue
        num, hm, cob = hit
        idx = rec.order() // len(sub_ops)
        edge = SubgroupEdge(
            rec.number, rec.hermann_mauguin, num, hm,
            "t", "I", idx, num == rec.number, cob,
        )
        buckets.setdefault(num, []).append(edge)
    out = []
    for edges in buckets.values():
        e0 = edges[0]
        out.append(SubgroupEdge(
            e0.parent, e0.parent_hm, e0.child, e0.child_hm,
            e0.type, e0.kind, e0.index, e0.isomorphic, e0.cob,
            embeddings=len(edges),
        ))
    out.sort(key=lambda e: (e.index, e.child))
    return out


def _plus_mod1(a: Vector3, b: Vector3) -> Vector3:
    return (a + b).mod1()


def _close_T(seed: Iterable[Vector3], universe: frozenset[Vector3]) -> frozenset[Vector3] | None:
    s = {ZERO3.mod1() if t == ZERO3 else t.mod1() for t in seed}
    s.add(ZERO3)
    grew = True
    while grew:
        grew = False
        extra = {_plus_mod1(a, b) for a in s for b in s}
        if not extra <= universe:
            return None
        if not extra <= s:
            s |= extra
            grew = True
    return frozenset(s)


def _centering_subgroups(T: frozenset[Vector3]) -> list[frozenset[Vector3]]:
    nonzero = [t for t in T if t != ZERO3]
    found = {frozenset([ZERO3])}
    for r in range(1, len(nonzero) + 1):
        for combo in combinations(nonzero, r):
            closed = _close_T(combo, T)
            if closed is not None:
                found.add(closed)
    return [s for s in found if s != T]


def _is_T_invariant(Tprime: frozenset[Vector3], Ws: frozenset[Matrix3]) -> bool:
    for W in Ws:
        for t in Tprime:
            img = (W @ t).mod1()
            if img not in Tprime:
                return False
    return True


def _ops_closed(ops: list[SymmetryOp]) -> bool:
    s = set(ops)
    return all((a * b) in s for a in ops for b in ops)


def _closed_primitive_sections(ops: list[SymmetryOp]) -> list[list[SymmetryOp]]:
    """Primitive (one op per W) closed sections of a centred copy of G.

    Lexicographic drop of extra centring is not always a subgroup: a screw
    may square to the dropped vector. Enumerate sections when |P| is small.
    """
    by_W: dict[Matrix3, list[SymmetryOp]] = {}
    for op in ops:
        by_W.setdefault(op.W, []).append(op)
    if IDENTITY3 not in by_W:
        return []
    id_ops = [op for op in by_W[IDENTITY3] if op.w == ZERO3]
    if not id_ops:
        return []
    others = [W for W in by_W if W != IDENTITY3]
    if not others:
        return []
    if len(others) > 8:
        dropped = _drop_centering(ops, frozenset([ZERO3]))
        return [dropped] if dropped and _ops_closed(dropped) else []
    found = []
    seen: set[frozenset[SymmetryOp]] = set()
    for combo in product(*(by_W[W] for W in others)):
        cand = [id_ops[0], *combo]
        if not _ops_closed(cand):
            continue
        key = frozenset(cand)
        if key in seen:
            continue
        seen.add(key)
        found.append(cand)
    return found


def _drop_centering(ops: list[SymmetryOp], Tprime: frozenset[Vector3]) -> list[SymmetryOp]:
    by_W: dict[Matrix3, list[SymmetryOp]] = {}
    for op in ops:
        by_W.setdefault(op.W, []).append(op)
    kept = []
    for olist in by_W.values():
        wp = min((op.w for op in olist), key=lambda v: tuple(float(x) for x in v.v))
        for op in olist:
            if (op.w - wp).mod1() in Tprime:
                kept.append(op)
    return kept


def _k_iia_edges(rec: SpaceGroup) -> list[SubgroupEdge]:
    ops = list(rec.operations())
    T = centering_translations(ops)
    if len(T) <= 1:
        return []
    pgrp = point_group(ops)
    out = []
    seen = set()
    for Tp in _centering_subgroups(T):
        if not _is_T_invariant(Tp, pgrp):
            continue
        sub = _drop_centering(ops, Tp)
        if len(sub) >= rec.order() or not sub:
            continue
        hit = _identify(sub)
        if hit is None:
            continue
        num, hm, cob = hit
        if num in seen:
            continue
        seen.add(num)
        idx = rec.order() // len(sub)
        out.append(SubgroupEdge(
            rec.number, rec.hermann_mauguin, num, hm,
            "k", "IIa", idx, num == rec.number, cob,
        ))
    out.sort(key=lambda e: (e.index, e.child))
    return out


def _kernel_basis(n: tuple[int, int, int], p: int) -> list[list[int]] | None:
    """Columns of a Z-basis of {x ∈ Z³ : n·x ≡ 0 (mod p)}; det ±p."""
    for k in range(3):
        nk = n[k] % p
        if nk == 0:
            continue
        try:
            inv = pow(nk, -1, p)
        except ValueError:
            continue
        cols = []
        for j in range(3):
            e = [0, 0, 0]
            if j == k:
                e[k] = p
            else:
                e[j] = 1
                e[k] = (-(n[j] % p) * inv) % p
            cols.append(e)
        det = (
            cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
            - cols[0][1] * (cols[1][0] * cols[2][2] - cols[1][2] * cols[2][0])
            + cols[0][2] * (cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0])
        )
        if abs(det) != p:
            continue
        return cols
    return None


def _normals(p: int) -> list[tuple[int, int, int]]:
    found = []
    seen = set()
    for a in range(p):
        for b in range(p):
            for c in range(p):
                n = (a, b, c)
                if n == (0, 0, 0):
                    continue
                # unique kernels: n and λn, λ≠0, give the same kernel
                canon = min(((n[0] * u) % p, (n[1] * u) % p, (n[2] * u) % p)
                            for u in range(1, p))
                if canon in seen:
                    continue
                seen.add(canon)
                found.append(n)
    return found


def _W_int(W: Matrix3) -> list[list[int]] | None:
    rows = []
    for row in W.rows:
        if any(x.denominator != 1 for x in row):
            return None
        rows.append([int(x) for x in row])
    return rows


def _invariant_mod_p(Ws: frozenset[Matrix3], n: tuple[int, int, int], p: int) -> bool:
    for W in Ws:
        Wi = _W_int(W)
        if Wi is None:
            return False
        # (W^T n) ≡ λ n (mod p)
        Wt_n = tuple(sum(Wi[i][j] * n[i] for i in range(3)) % p for j in range(3))
        if all(x == 0 for x in Wt_n):
            if n != (0, 0, 0):
                return False
            continue
        ok = False
        for lam in range(1, p):
            if Wt_n == tuple((lam * n[k]) % p for k in range(3)):
                ok = True
                break
        if not ok:
            return False
    return True


def _k_iib_edges(rec: SpaceGroup) -> list[SubgroupEdge]:
    """Primitive-cell IIb: invariant index-2 and index-3 sublattices of Z³."""
    ops = list(rec.operations())
    T = centering_translations(ops)
    if len(T) > 1:
        return []  # IIa already covers losing conventional centring
    pgrp = point_group(ops)
    out = []
    seen = set()
    for p in (2, 3):
        for n in _normals(p):
            if not _invariant_mod_p(pgrp, n, p):
                continue
            cols = _kernel_basis(n, p)
            if cols is None:
                continue
            P = Matrix3([[cols[j][i] for j in range(3)] for i in range(3)])
            cob = ChangeOfBasis(P, ZERO3)
            try:
                transformed = [cob.apply_to_op(op) for op in ops]
            except Exception:
                continue
            if any(not _is_crystallographic_W(op.W) for op in transformed):
                continue
            seeds = list(transformed) + _lattice_coset_ops(cob)
            try:
                closed = close_group(seeds, max_order=max(192, rec.order() * p))
            except RuntimeError:
                continue
            Tnew = centering_translations(closed)
            if len(Tnew) <= 1:
                continue
            for sub in _closed_primitive_sections(list(closed)):
                hit = _identify_hit(sub)
                if hit is None:
                    hit = _identify_diag(sub)
                if hit is None:
                    continue
                num, hm, cob_s = hit
                key = (num, p)
                if key in seen:
                    continue
                seen.add(key)
                out.append(SubgroupEdge(
                    rec.number, rec.hermann_mauguin, num, hm,
                    "k", "IIb", p, num == rec.number, cob_s,
                ))
                break
    out.sort(key=lambda e: (e.index, e.child))
    return out


@lru_cache(maxsize=256)
def _t_cached(number: int, maximal: bool) -> tuple[SubgroupEdge, ...]:
    return tuple(_t_edges(space_group(number), maximal=maximal))


@lru_cache(maxsize=256)
def _k_cached(number: int) -> tuple[SubgroupEdge, ...]:
    rec = space_group(number)
    return tuple(_k_iia_edges(rec) + _k_iib_edges(rec))


def subgroup_edges(
    sg,
    *,
    kind: str = "both",
    maximal: bool = True,
) -> list[SubgroupEdge]:
    """Derived maximal t and/or k subgroup edges of ``sg``."""
    rec = sg if isinstance(sg, SpaceGroup) else space_group(sg)
    if kind not in ("t", "k", "both"):
        raise ValueError("kind: must be t, k, or both")
    edges: list[SubgroupEdge] = []
    if kind in ("t", "both"):
        edges.extend(_t_cached(rec.number, maximal))
    if kind in ("k", "both"):
        edges.extend(_k_cached(rec.number))
    return edges


def subgroup_graph(sg, *, kind: str = "both", maximal: bool = True) -> dict:
    """JSON-ready graph: parent node plus derived t/k child edges."""
    rec = sg if isinstance(sg, SpaceGroup) else space_group(sg)
    edges = subgroup_edges(rec, kind=kind, maximal=maximal)
    nodes = {
        rec.number: {
            "sg_number": rec.number,
            "sg_hm": rec.hermann_mauguin,
            "order": rec.order(),
            "role": "parent",
        }
    }
    for e in edges:
        child = space_group(e.child)
        nodes.setdefault(e.child, {
            "sg_number": e.child,
            "sg_hm": e.child_hm,
            "order": child.order(),
            "role": "subgroup",
        })
    return {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "order": rec.order(),
        "note": (
            "Derived from operators. t = translationengleiche (type I, same lattice). "
            "k = klassengleiche (type II): IIa drops conventional centring; "
            "IIb is a G-invariant index-2 or index-3 sublattice of Z³. "
            "Infinite isomorphic k-series are not enumerated."
        ),
        "kind": kind,
        "maximal": maximal,
        "nodes": list(nodes.values()),
        "edges": [
            {
                "from": e.parent,
                "to": e.child,
                "to_hm": e.child_hm,
                "type": e.type,
                "kind": e.kind,
                "index": e.index,
                "isomorphic": e.isomorphic,
                "embeddings": e.embeddings,
                "cob": e.cob,
            }
            for e in edges
        ],
    }
