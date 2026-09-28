"""Deposited-cell to Selling-reduced change of basis, and reference-orbit match.

The database stores one obtuse cell per PDB row and the rational matrix that
takes the deposited basis to that cell. A query Selling-reduces the reference,
enumerates the typed Selling-superbase closure once, and composes

    P_ref @ S @ P_pdb^{-1}

for each stored row whose reduced metric matches an orbit member. ``S`` runs
over that closure with the S4 x {±1} labeling. Columns of every matrix are the
new basis vectors in the old basis (``P^T G_old P = G_new``).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from ..change_of_basis import ChangeOfBasis
from ..linalg import Matrix3, Vector3
from ..setting import format_cob
from .canonical import canonical_superbase
from .metric import UnitCell, params_from_metric
from .primitive import lattice_letter, primitive_cell
from .selling_closure import selling_superbase_closure

_H = Fraction(1, 2)
_T = Fraction(1, 3)
_TT = Fraction(2, 3)

# Exact copies of primitive._PRIM_P. Float 1/3 must not enter the stored matrix.
_PRIM_EXACT = {
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "A": [[1, 0, 0], [0, _H, -_H], [0, _H, _H]],
    "B": [[_H, 0, -_H], [0, 1, 0], [_H, 0, _H]],
    "C": [[_H, _H, 0], [-_H, _H, 0], [0, 0, 1]],
    "I": [[-_H, _H, _H], [_H, -_H, _H], [_H, _H, -_H]],
    "F": [[0, _H, _H], [_H, 0, _H], [_H, _H, 0]],
    "R": [[_TT, -_T, -_T], [_T, _T, -_TT], [_T, _T, _T]],
}

COB_COLUMNS = tuple(f"cob{i}{j}" for i in range(3) for j in range(3))
_PERMS = tuple(permutations(range(4)))
_ZERO = Vector3((0, 0, 0))


def primitive_matrix(sg_hm) -> Matrix3:
    """Centering matrix as exact rationals. Columns are the primitive basis.

    An empty or unrecognised symbol is treated as primitive, matching
    :func:`agentsg.cell.celldb._primitive_for_roots`.
    """
    letter = "P"
    if sg_hm:
        try:
            letter = lattice_letter(sg_hm)
        except ValueError:
            letter = "P"
    return Matrix3(_PRIM_EXACT[letter])


def _primitive_or_self(cell, sg_hm):
    """Primitive cell, or ``cell`` itself when the symbol does not name a centring."""
    cell = tuple(float(x) for x in cell)
    if not sg_hm:
        return cell
    try:
        return tuple(float(x) for x in primitive_cell(cell, sg_hm))
    except (ValueError, ZeroDivisionError):
        return cell


def selling_matrix(cell) -> Matrix3:
    """Integer unimodular matrix, primitive (or already-P) cell to one obtuse basis."""
    C, _ = canonical_superbase(cell)
    return Matrix3(
        [[Fraction(int(C[1 + j][i])) for j in range(3)] for i in range(3)]
    )


def _metric_of(G, P: Matrix3):
    """Float ``P^T G P``."""
    Pf = [[float(x) for x in row] for row in P.rows]
    return [
        [
            sum(Pf[i][a] * G[i][k] * Pf[k][b] for i in range(3) for k in range(3))
            for b in range(3)
        ]
        for a in range(3)
    ]


def deposited_to_reduced(cell, sg_hm):
    """Return ``(red_params, P)`` from the deposited cell to one Selling-reduced cell.

    ``P = primitive_transform @ M``. ``red_params`` is the cell of ``P^T G P``.
    """
    prim = _primitive_or_self(cell, sg_hm)
    P = primitive_matrix(sg_hm) @ selling_matrix(prim)
    red = params_from_metric(_metric_of(UnitCell(*cell).metric_tensor(), P))
    return red, P


def cob_column_values(P: Matrix3) -> list[str]:
    """Nine ``num/den`` strings, row-major."""
    return [f"{c.numerator}/{c.denominator}" for row in P.rows for c in row]


def parse_cob_columns(values) -> Matrix3:
    """Inverse of :func:`cob_column_values`."""
    rows = []
    vals = list(values)
    if len(vals) != 9:
        raise ValueError("expected 9 change-of-basis entries")
    for i in range(3):
        row = []
        for j in range(3):
            num, den = str(vals[3 * i + j]).split("/")
            row.append(Fraction(int(num), int(den)))
        rows.append(row)
    return Matrix3(rows)


def cob_to_json(P: Matrix3):
    """3×3 lists of ``[numerator, denominator]``."""
    return [[[c.numerator, c.denominator] for c in row] for row in P.rows]


def cob_xyz(P: Matrix3) -> str:
    """Column notation for ``P`` (no origin shift)."""
    return format_cob(ChangeOfBasis(P, _ZERO), "abc")


class ReferenceOrbit:
    """One query: deposited-to-reduced ``P_ref`` and the labeled closure metrics."""

    __slots__ = ("P_ref", "labeled")

    def __init__(self, P_ref: Matrix3, labeled):
        self.P_ref = P_ref
        # list of (S, metric S^T G_red S)
        self.labeled = labeled


def reference_orbit(cell, sg_hm) -> ReferenceOrbit:
    """Selling-reduce ``cell`` and enumerate the typed closure once.

    Each labeled superbase is an integer ``S`` in the reduced basis.
    """
    prim = _primitive_or_self(cell, sg_hm)
    M = selling_matrix(prim)
    P_ref = primitive_matrix(sg_hm) @ M
    G_red = _metric_of(UnitCell(*prim).metric_tensor(), M)
    Minv = M.inverse()
    labeled = []
    seen = set()
    for C in selling_superbase_closure(prim):
        for perm in _PERMS:
            for sign in (1, -1):
                Q = Matrix3(
                    [[Fraction(sign * int(C[perm[1 + j]][i])) for j in range(3)]
                     for i in range(3)]
                )
                if Q.det() not in (1, -1):
                    continue
                S = Minv @ Q
                if S.det() not in (1, -1) or S.rows in seen:
                    continue
                seen.add(S.rows)
                labeled.append((S, _metric_of(G_red, S)))
    return ReferenceOrbit(P_ref, labeled)


def _metrics_close(G, H, verify_rel=1e-6) -> bool:
    scale = abs(H[0][0]) + abs(H[1][1]) + abs(H[2][2])
    tol = verify_rel * max(scale, 1e-12)
    return all(
        abs(G[a][b] - H[a][b]) <= tol for a in range(3) for b in range(3)
    )


_COB_TARGET = "a,b,c"


def _cob_rank(P: Matrix3):
    """Prefer the fewest minus signs, then the spelling closest to ``a,b,c``.

    Closeness is the shared prefix with ``a,b,c``, then the alphabetical
    distance of the first differing character.
    """
    body = cob_xyz(P)[1:-1]
    target = _COB_TARGET
    n = 0
    limit = min(len(body), len(target))
    while n < limit and body[n] == target[n]:
        n += 1
    left = body[n] if n < len(body) else ""
    right = target[n] if n < len(target) else ""
    if left and right:
        mismatch = abs(ord(left) - ord(right))
    elif not left and not right:
        mismatch = 0
    else:
        mismatch = 10_000
    return (body.count("-"), -n, mismatch, body)


def match_operators(orbit: ReferenceOrbit, red_cell, P_pdb: Matrix3, verify_rel=1e-6):
    """Operators from the reference deposited cell to the PDB deposited cell.

    The first operator is the preferred representative: fewest minus signs,
    then the change-of-basis text closest to ``a,b,c``. Empty when the stored
    reduced cell is not an orbit member (sorted-key impostor).
    """
    G_pdb = UnitCell(*red_cell).metric_tensor()
    P_inv = P_pdb.inverse()
    found = {}
    for S, G_s in orbit.labeled:
        if not _metrics_close(G_s, G_pdb, verify_rel=verify_rel):
            continue
        P = orbit.P_ref @ S @ P_inv
        found[P.rows] = P
    return sorted(found.values(), key=_cob_rank)


def annotate_search_hits(db, cell, sg_hm, hits, verify_rel=1e-6):
    """Attach ``cob`` / ``cob_xyz`` / ``cob_coset`` to search hit dicts.

    Databases without ``cob00`` still return hits; ``cob`` is null.
    """
    if not hits:
        return hits
    if not db.has_selling_cob():
        for hit in hits:
            hit["cob"] = None
            hit["cob_xyz"] = None
        return hits
    reductions = db.lookup_reductions([hit["pdb_id"] for hit in hits])
    orbit = None
    if any(rec.get("cob") is not None and rec.get("red") is not None
           for rec in reductions.values()):
        orbit = reference_orbit(cell, sg_hm)
    for hit in hits:
        rec = reductions.get(hit["pdb_id"])
        ops = []
        if orbit is not None and rec and rec.get("cob") is not None and rec.get("red") is not None:
            ops = match_operators(orbit, rec["red"], rec["cob"], verify_rel=verify_rel)
        if not ops:
            hit["cob"] = None
            hit["cob_xyz"] = None
            continue
        hit["cob"] = cob_to_json(ops[0])
        hit["cob_xyz"] = cob_xyz(ops[0])
        if len(ops) > 1:
            hit["cob_coset"] = [
                {"cob": cob_to_json(P), "cob_xyz": cob_xyz(P)} for P in ops
            ]
    return hits
