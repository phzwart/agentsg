"""Two stored changes of basis, combined only by a later step.

Step 1, :func:`to_selling`, takes any cell (a Niggli cell is the usual one)
to one obtuse cell and stores that single change of basis.

Step 2, :func:`closure_cobs`, is the Selling-superbase closure of that obtuse
cell. Every matrix there is written in the reduced basis, not in the basis
that was submitted.

:func:`combine_cobs` is the later step: the submitted-basis operator is
``to_selling @ closure``. Nothing in steps 1 or 2 does that product.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from agentsg.cell.canonical import superbase_variants
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.selling_closure import selling_superbase_closure
from agentsg.cell.selling_cob import selling_matrix
from agentsg.linalg import Matrix3

_PERMS = tuple(permutations(range(4)))
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _as_matrix(cob) -> Matrix3:
    return Matrix3([[Fraction(int(x)) for x in row] for row in cob])


def _as_int(matrix: Matrix3):
    """3×3 integer rows, or None when a denominator remains."""
    rows = []
    for row in matrix.rows:
        out = []
        for value in row:
            if value.denominator != 1:
                return None
            out.append(int(value))
        rows.append(tuple(out))
    return tuple(rows)


def _metric_of(G, cob: Matrix3):
    """Float ``P^T G P``."""
    pf = [[float(x) for x in row] for row in cob.rows]
    return [
        [
            sum(pf[i][a] * G[i][k] * pf[k][b] for i in range(3) for k in range(3))
            for b in range(3)
        ]
        for a in range(3)
    ]


def _members(cell, boundary_rel, angle_sigma):
    """Typed Selling closure, plus near-obtuse flips when ``boundary_rel > 0``."""
    seen = set()
    found = []
    groups = [selling_superbase_closure(cell, angle_sigma=angle_sigma)]
    if boundary_rel > 0:
        groups.append(superbase_variants(cell, boundary_rel=boundary_rel))
    for group in groups:
        for superbase in group:
            key = frozenset(tuple(int(x) for x in vector) for vector in superbase)
            if key in seen:
                continue
            seen.add(key)
            found.append(superbase)
    return found


def to_selling(cell):
    """Step 1. One obtuse cell and the change of basis onto it.

    Parameters
    ----------
    cell :
        Any ``(a, b, c, alpha, beta, gamma)``. A Niggli cell is the usual
        input. It does not need to be Selling-reduced.

    Returns
    -------
    dict
        ``cell`` is the input. ``reduced`` is the obtuse cell.
        ``cob`` is an integer 3×3 matrix whose columns are the Selling basis
        in the input basis: ``cob^T G(cell) cob = G(reduced)``.
    """
    cell_t = tuple(float(x) for x in cell)
    cob = selling_matrix(cell_t)
    reduced = params_from_metric(_metric_of(UnitCell(*cell_t).metric_tensor(), cob))
    return {
        "cell": cell_t,
        "reduced": tuple(float(x) for x in reduced),
        "cob": _as_int(cob),
    }


def closure_cobs(reduced, *, boundary_rel=0.0, angle_sigma=None):
    """Step 2. Every closure change of basis, in the reduced basis.

    ``reduced`` is the obtuse cell from :func:`to_selling`. The matrices are
    not multiplied by that step's change of basis.

    A positive ``boundary_rel`` also keeps near-obtuse Selling flips.
    ``angle_sigma`` (degrees) widens zero detection inside the typed closure.
    """
    reduced_t = tuple(float(x) for x in reduced)
    to_reduced = selling_matrix(reduced_t)
    back = to_reduced.inverse()
    found = {}
    for superbase in _members(reduced_t, float(boundary_rel), angle_sigma):
        for perm in _PERMS:
            for sign in (1, -1):
                columns = []
                for j in range(3):
                    vector = superbase[perm[1 + j]]
                    columns.append([Fraction(sign * int(vector[i])) for i in range(3)])
                # columns are currently stacked as new-basis components;
                # Matrix3 wants rows, so transpose the column list.
                basis = Matrix3(list(zip(*columns)))
                if basis.det() not in (1, -1):
                    continue
                cob = _as_int(back @ basis)
                if cob is None or _as_matrix(cob).det() not in (1, -1):
                    continue
                found.setdefault(cob, None)
    ordered = [_IDENTITY] if _IDENTITY in found else []
    ordered.extend(sorted(cob for cob in found if cob not in ordered))
    return ordered


def combine_cobs(to_selling_cob, closure_cob):
    """Later step. Submitted-basis operator ``to_selling @ closure``.

    ``to_selling_cob`` is step 1. ``closure_cob`` is one matrix from step 2.
    """
    combined = _as_int(_as_matrix(to_selling_cob) @ _as_matrix(closure_cob))
    if combined is None:
        raise ValueError("combined change of basis is not integral")
    return combined


def _residual(metric, cob, target):
    predicted = _metric_of(metric, _as_matrix(cob))
    return max(
        abs(predicted[a][b] - target[a][b]) for a in range(3) for b in range(3)
    )


def combine_coset(query, query_cobs, reference, reference_cobs, *, band_rel):
    """Later step. Coset of ``query`` onto ``reference`` in the submitted basis.

    Each operator is ``combine(query, S_query) @ combine(reference, S_reference)^{-1}``.
    ``query`` and ``reference`` are :func:`to_selling` results. The cob lists
    are :func:`closure_cobs` results and stay in the reduced basis until this
    product. Operators within ``band_rel`` of the best metric residual are kept.
    """
    if not query_cobs or not reference_cobs:
        return []
    left = [_as_matrix(combine_cobs(query["cob"], cob)) for cob in query_cobs]
    right = [_as_matrix(combine_cobs(reference["cob"], cob)).inverse() for cob in reference_cobs]
    metric = UnitCell(*query["cell"]).metric_tensor()
    target = UnitCell(*reference["cell"]).metric_tensor()
    scale = abs(target[0][0]) + abs(target[1][1]) + abs(target[2][2])
    best = {}
    best_resid = float("inf")
    for base in left:
        for back in right:
            cob = _as_int(base @ back)
            if cob is None:
                continue
            resid = _residual(metric, cob, target)
            prev = best.get(cob)
            if prev is None or resid < prev:
                best[cob] = resid
            if resid < best_resid:
                best_resid = resid
    if not best:
        return []
    band = best_resid + float(band_rel) * scale
    kept = [(resid, cob) for cob, resid in best.items() if resid <= band]
    kept.sort(key=lambda item: (item[0], item[1]))
    return [cob for _resid, cob in kept]
