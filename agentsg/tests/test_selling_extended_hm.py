"""Selling-reduce a cell, track the change of basis, and re-express the cell and
symmetry in extended Hermann-Mauguin notation ("<base> (<cob>)").

Covered for one representative group in every crystal system and every centring
type that has a standard HM symbol (P, C, I, F, R).  The invariants checked are:

  * the tracked change of basis is integer and volume-preserving (det = +1);
  * the reduced cell has the same volume as the input;
  * the transformed operation set has the same order as the base group;
  * applying the inverse change of basis to the transformed operations recovers
    the base operations exactly (round trip);
  * the reduced cell is Delaunay-stationary (re-reducing gives identity);
  * (oracle, if spglib present) the transformed symmetry has the same
    space-group number as the base group -- the operator changes the setting,
    not the group.
"""
from fractions import Fraction as Fr

import pytest

from agentsg.space_groups import space_group
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.canonical import canonical_superbase
from agentsg.change_of_basis import ChangeOfBasis
from agentsg.linalg import Matrix3, Vector3
from agentsg.setting import SpaceGroupSetting

np = pytest.importorskip("numpy")


# (label, HM symbol, cell) -- one group per crystal system x centring.
# Cells are chosen in the conventional setting; triclinic and monoclinic-P use
# an acute angle so the Selling reduction fires non-trivially, the orthogonal
# systems are already Delaunay-reduced (identity change of basis is correct).
CASES = [
    ("triclinic_P",     "P -1",       (7.0, 8.0, 9.0, 85.0, 95.0, 105.0)),
    ("monoclinic_P",    "P 1 21 1",   (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)),
    ("monoclinic_C",    "C 1 2/c 1",  (10.0, 8.0, 12.0, 90.0, 100.0, 90.0)),
    ("orthorhombic_P",  "P 21 21 21", (7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("orthorhombic_C",  "C 2 2 21",   (7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("orthorhombic_I",  "I 2 2 2",    (7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("orthorhombic_F",  "F 2 2 2",    (7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("tetragonal_P",    "P 4",        (8.0, 8.0, 12.0, 90.0, 90.0, 90.0)),
    ("tetragonal_I",    "I 41",       (8.0, 8.0, 12.0, 90.0, 90.0, 90.0)),
    ("trigonal_P",      "P 3",        (8.0, 8.0, 12.0, 90.0, 90.0, 120.0)),
    ("rhombohedral_R",  "R 3",        (8.0, 8.0, 20.0, 90.0, 90.0, 120.0)),
    ("hexagonal_P",     "P 6",        (8.0, 8.0, 12.0, 90.0, 90.0, 120.0)),
    ("cubic_P",         "P 2 3",      (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)),
    ("cubic_I",         "I 2 3",      (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)),
    ("cubic_F",         "F 2 3",      (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)),
]
IDS = [c[0] for c in CASES]


def selling_setting(cell, base_symbol):
    """Return (reduced_params, ChangeOfBasis, SpaceGroupSetting).

    The change of basis is the integer, volume-preserving operator taking the
    input basis to its Selling/Delaunay-reduced basis; the setting carries the
    base group viewed through that operator (extended HM notation).
    """
    C, _ = canonical_superbase(cell)                 # obtuse superbase, int coords
    # reduced cell vectors are C[1], C[2], C[3]; columns of P = new in old coords
    P = Matrix3([[Fr(int(C[1 + j][i])) for j in range(3)] for i in range(3)])
    cob = ChangeOfBasis(P, Vector3((Fr(0), Fr(0), Fr(0))))
    G = np.array(UnitCell(*cell).metric_tensor())
    Pn = np.array([[float(x) for x in r] for r in P.rows])
    red = params_from_metric((Pn.T @ G @ Pn).tolist())
    return red, cob, SpaceGroupSetting(base_symbol, cob)


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_change_of_basis_is_unimodular(label, sym, cell):
    """The tracked Selling change of basis is integer with det = +1."""
    _, cob, _ = selling_setting(cell, sym)
    for row in cob.P.rows:
        for x in row:
            assert x.denominator == 1                # integer entries
    assert cob.P.det() == 1                          # volume-preserving, proper


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_reduced_cell_preserves_volume(label, sym, cell):
    red, _, _ = selling_setting(cell, sym)
    v0 = UnitCell(*cell).volume()
    v1 = UnitCell(*red).volume()
    assert abs(v0 - v1) < 1e-6 * v0


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_setting_preserves_group_order(label, sym, cell):
    _, _, setting = selling_setting(cell, sym)
    assert setting.order() == space_group(sym).order()


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_inverse_cob_round_trips_operations(label, sym, cell):
    """Applying the inverse change of basis to the transformed operations
    recovers the base operation set exactly."""
    _, cob, setting = selling_setting(cell, sym)
    inv = cob.inverse()
    recovered = sorted(inv.apply_to_op(o).as_xyz() for o in setting.operations())
    base = sorted(o.as_xyz() for o in space_group(sym).operations())
    assert recovered == base


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_reduced_cell_is_delaunay_stationary(label, sym, cell):
    """Re-reducing the reduced cell gives the identity change of basis: the
    Selling output is a fixed point of the reduction."""
    red, _, _ = selling_setting(cell, sym)
    C2, _ = canonical_superbase(red)
    P2 = np.array([[float(C2[1 + j][i]) for j in range(3)] for i in range(3)])
    assert np.allclose(P2, np.eye(3))


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_monoclinic_p_moves_screw_axis(label, sym, cell):
    """Sanity anchor: the acute-beta monoclinic-P case must actually move the
    2_1 screw from b to c (non-identity change of basis), proving the reduction
    is exercised and not a no-op for at least one centred/primitive case."""
    if label != "monoclinic_P":
        pytest.skip("axis-move anchor only checked for monoclinic_P")
    _, cob, setting = selling_setting(cell, sym)
    assert cob.P.det() == 1 and not np.allclose(
        np.array([[float(x) for x in r] for r in cob.P.rows]), np.eye(3))
    ops = {o.as_xyz() for o in setting.operations()}
    assert "-x,-y,z+1/2" in ops                      # screw now along new c


def _sg_number_of_setting(setting, red):
    spglib = pytest.importorskip("spglib")
    uc = UnitCell(*red)
    latt = np.array(uc.orthogonalization_matrix()).T
    rot, tr = [], []
    for o in setting.operations():
        rot.append(np.array([[float(x) for x in r] for r in o.W.rows]
                            ).round().astype(int))
        tr.append(np.array([float(x) for x in o.w.v]))
    t = spglib.get_spacegroup_type_from_symmetry(rot, tr, latt, symprec=1e-3)
    return t.number


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_spglib_setting_has_same_group_number(label, sym, cell):
    """Oracle: the transformed symmetry is the SAME space group (number) as the
    base -- the extended-HM operator changes the setting, not the group."""
    base = space_group(sym)
    base_num = getattr(base, "number", None)
    if base_num is None:
        pytest.skip("base group exposes no number")
    red, _, setting = selling_setting(cell, sym)
    assert _sg_number_of_setting(setting, red) == base_num
