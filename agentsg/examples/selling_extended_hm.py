"""Selling-reduce a P2_1 cell, track the change of basis, and re-express the
cell and symmetry in extended Hermann-Mauguin notation ("<base> (<cob>)").

Run against the agentsg package.  The base symbol is kept and the reduction
operator is appended, so a non-standard setting is expressed exactly without
renaming the group.
"""
from fractions import Fraction as Fr
import numpy as np

from agentsg.space_groups import space_group
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.canonical import canonical_superbase
from agentsg.change_of_basis import ChangeOfBasis
from agentsg.linalg import Matrix3, Vector3
from agentsg.setting import SpaceGroupSetting, format_cob


def selling_setting(cell, base_symbol="P 1 21 1"):
    """Return (reduced_params, ChangeOfBasis, SpaceGroupSetting) for ``cell``.

    ``cell`` is a 6-tuple (a,b,c,alpha,beta,gamma).  The change of basis is the
    integer, volume-preserving operator taking the input basis to its
    Selling/Delaunay-reduced basis; the setting object carries the base group
    viewed through that operator.
    """
    C, _ = canonical_superbase(cell)                 # obtuse superbase, int coords
    # reduced cell vectors are C[1], C[2], C[3]; columns of P = new in old coords
    P = Matrix3([[Fr(int(C[1 + j][i])) for j in range(3)] for i in range(3)])
    cob = ChangeOfBasis(P, Vector3((Fr(0), Fr(0), Fr(0))))
    G = np.array(UnitCell(*cell).metric_tensor())
    Pn = np.array([[float(x) for x in r] for r in P.rows])
    red = params_from_metric((Pn.T @ G @ Pn).tolist())
    return red, cob, SpaceGroupSetting(base_symbol, cob)


if __name__ == "__main__":
    cell = (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)        # P2_1, acute beta
    red, cob, setting = selling_setting(cell)
    print("original cell   :", cell)
    print("reduced cell    :", tuple(round(x, 4) for x in red))
    print("change of basis :", format_cob(cob, letters="abc"))
    print("extended HM     :", str(setting))
    print("operations (new setting):")
    for op in sorted(setting.operations(), key=lambda o: o.as_xyz()):
        print("   ", op.as_xyz())
