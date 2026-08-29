"""Enumerate the settings of a space group under the Selling reduction group.

Given a cell and a base space-group symbol, this sweeps the full order-48
Selling change-of-basis group (see :mod:`agentsg.cell.selling_group`) over the
group and reports, for each operator, the resulting *setting*: the composed
change of basis (from the original basis), the transformed symmetry operations,
and the cell parameters of that basis.

The space group itself is INVARIANT under every operator -- each is an integer
unimodular change of basis, so it can only reindex the group, never change it.
What changes is the setting: for a monoclinic group, for instance, the unique
axis reorients among a, b, c and the oblique body-diagonal directions, and the
operation triplets change accordingly.  This is exactly the reindexing-ambiguity
orbit of one crystal -- every symmetry-equivalent way of writing the same
lattice.

Everything is exact rational arithmetic; no runtime dependencies.  (A caller can
confirm the space-group invariance against spglib, but that is an external
oracle used only in the tests.)
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Fr

from ..linalg import Matrix3, Vector3
from ..change_of_basis import ChangeOfBasis
from ..setting import SpaceGroupSetting, format_cob
from ..space_groups import space_group
from .metric import UnitCell, params_from_metric
from .canonical import canonical_superbase
from .selling_group import selling_group


@dataclass(frozen=True)
class SellingSetting:
    """One setting of a base group under a Selling operator.

    Attributes
    ----------
    operator_index : int
        Index (0..47) of the Selling operator in :func:`selling_group`.
    det : int
        Determinant of the Selling operator (+1 proper, -1 improper).
    change_of_basis : ChangeOfBasis
        The FULL change of basis from the ORIGINAL input basis to this setting
        (the reduction to the obtuse superbase composed with the operator).
    cob_string : str
        Human-readable extended-HM change-of-basis string, e.g. ``(-a,a-c,-b)``.
    setting : SpaceGroupSetting
        The base group viewed through ``change_of_basis``.
    operations : tuple[str, ...]
        The transformed operation triplets (sorted, ``as_xyz`` form).
    cell : tuple[float, ...]
        The six cell parameters (a, b, c, alpha, beta, gamma) of this basis.
    """
    operator_index: int
    det: int
    change_of_basis: ChangeOfBasis
    cob_string: str
    setting: SpaceGroupSetting
    operations: tuple
    cell: tuple


def _reduction_cob(cell):
    """The integer change of basis taking ``cell`` to its Delaunay/Selling-
    reduced (obtuse-superbase) basis, plus that reduced metric as a list."""
    C, _ = canonical_superbase(cell)
    P = Matrix3([[Fr(int(C[1 + j][i])) for j in range(3)] for i in range(3)])
    cob = ChangeOfBasis(P, Vector3((Fr(0), Fr(0), Fr(0))))
    # reduced metric G' = P^T G P (exact rational)
    G = UnitCell(*cell).metric_tensor()
    Pr = P.rows
    # (P^T G P)[a][b] = sum_{i,k} P[i][a] G[i][k] P[k][b]
    Gr = [[sum(Pr[i][a] * G[i][k] * Pr[k][b] for i in range(3) for k in range(3))
           for b in range(3)] for a in range(3)]
    return cob, Gr


def _metric_after(op_P, Gr):
    """S^T Gr S for a Selling operator matrix ``op_P`` (Matrix3) and reduced
    metric ``Gr`` (list of lists)."""
    S = op_P.rows
    return [[sum(S[i][a] * Gr[i][k] * S[k][b] for i in range(3) for k in range(3))
             for b in range(3)] for a in range(3)]


def selling_settings(cell, base_symbol):
    """All 48 settings of ``base_symbol`` under the Selling reduction group.

    Parameters
    ----------
    cell : sequence of 6 floats
        (a, b, c, alpha, beta, gamma) of the input cell.
    base_symbol : str
        Hermann-Mauguin or Hall symbol of the base space group.

    Returns
    -------
    list[SellingSetting]
        One record per Selling operator (48 total), operator 0 first.  Every
        record carries the same space group (the group is invariant); the
        operations and cell reorient with the operator.
    """
    red_cob, Gr = _reduction_cob(cell)
    records = []
    for k, S in enumerate(selling_group()):
        full = ChangeOfBasis(red_cob.P @ S.P, Vector3((Fr(0), Fr(0), Fr(0))))
        setting = SpaceGroupSetting(base_symbol, full)
        ops = tuple(sorted(o.as_xyz() for o in setting.operations()))
        Gm = _metric_after(S.P, Gr)
        cellp = tuple(round(x, 6) for x in params_from_metric(Gm))
        records.append(SellingSetting(
            operator_index=k,
            det=int(S.P.det()),
            change_of_basis=full,
            cob_string=format_cob(full, "abc"),
            setting=setting,
            operations=ops,
            cell=cellp,
        ))
    return records


def distinct_settings(cell, base_symbol):
    """The DISTINCT settings under the Selling group, keyed by operation set.

    Several Selling operators map to the same operation triplets (the group is
    larger than the number of settings -- the ``+/-I`` metric degeneracy and the
    group's own order collapse them).  Returns a dict mapping the sorted
    operation-triplet tuple to the list of :class:`SellingSetting` records that
    realise it.
    """
    out = {}
    for rec in selling_settings(cell, base_symbol):
        out.setdefault(rec.operations, []).append(rec)
    return out
