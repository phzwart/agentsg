"""Settings of a space group under the Selling reduction group.

Sweeping the order-48 Selling change-of-basis group over a base group produces
48 settings of the SAME space group: every operator is a unimodular change of
basis, so the group is invariant and only the setting (axis direction, operation
triplets, cell) reorients.  The invariance is asserted here against spglib as an
external oracle -- the packaged function itself is dependency-free.
"""
import pytest

from agentsg.space_groups import space_group
from agentsg.cell.selling_settings import (
    selling_settings, distinct_settings, SellingSetting,
)

# representative cases: a group per system whose reduction actually reorients
# (triclinic and monoclinic move; the orthogonal systems stay put but the group
# is still swept and the space group must stay invariant).
CASES = [
    ("P2_monoclinic",    "P 1 2 1",     (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)),
    ("P21_monoclinic",   "P 1 21 1",    (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)),
    ("C2_monoclinic",    "C 1 2 1",     (10.0, 8.0, 12.0, 90.0, 100.0, 90.0)),
    ("P1_triclinic",     "P 1",         (7.0, 8.0, 9.0, 85.0, 95.0, 105.0)),
    ("P222_ortho",       "P 2 2 2",     (7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("P4_tetragonal",    "P 4",         (8.0, 8.0, 12.0, 90.0, 90.0, 90.0)),
]
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_returns_48_records(label, sym, cell):
    recs = selling_settings(cell, sym)
    assert len(recs) == 48
    assert all(isinstance(r, SellingSetting) for r in recs)
    assert recs[0].operator_index == 0


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_all_operators_unimodular(label, sym, cell):
    for r in selling_settings(cell, sym):
        assert r.det in (1, -1)


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_every_setting_preserves_group_order(label, sym, cell):
    base_order = space_group(sym).order()
    for r in selling_settings(cell, sym):
        assert len(r.setting.operations()) == base_order


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_setting_cells_preserve_volume(label, sym, cell):
    from agentsg.cell.metric import UnitCell
    v0 = UnitCell(*cell).volume()
    for r in selling_settings(cell, sym):
        assert abs(UnitCell(*r.cell).volume() - v0) < 1e-4 * v0


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_inverse_cob_round_trips_operations(label, sym, cell):
    """The inverse of each setting's change of basis maps its operations back
    onto the base operation set."""
    base = sorted(o.as_xyz() for o in space_group(sym).operations())
    for r in selling_settings(cell, sym):
        inv = r.change_of_basis.inverse()
        recovered = sorted(inv.apply_to_op(o).as_xyz()
                           for o in r.setting.operations())
        assert recovered == base


def test_distinct_settings_partition_the_48():
    """distinct_settings groups the 48 operators by operation set; the groups
    partition all 48 with no loss."""
    d = distinct_settings((8.0, 6.0, 11.0, 90.0, 70.0, 90.0), "P 1 2 1")
    total = sum(len(v) for v in d.values())
    assert total == 48
    # P2 reorients into 12 distinct settings (4 operators each)
    assert len(d) == 12
    assert all(len(v) == 4 for v in d.values())


def test_p2_reorients_twofold_axis():
    """The P2 2-fold axis appears along a, b, c and the body diagonal across the
    48 settings (the reindexing orbit of the monoclinic axis)."""
    recs = selling_settings((8.0, 6.0, 11.0, 90.0, 70.0, 90.0), "P 1 2 1")
    triplets = {op for r in recs for op in r.operations if op != "x,y,z"}
    # standard b-axis, a-axis and c-axis forms are all present
    assert "-x,y,-z" in triplets       # 2-fold along b
    assert "x,-y,-z" in triplets       # 2-fold along a
    assert "-x,-y,z" in triplets       # 2-fold along c


# --- oracle: the space group is invariant under the whole Selling group -------
def _sg_number(setting, cell):
    np = pytest.importorskip("numpy")
    spglib = pytest.importorskip("spglib")
    from agentsg.cell.metric import UnitCell
    uc = UnitCell(*cell)
    latt = np.array(uc.orthogonalization_matrix()).T
    rot, tr = [], []
    for o in setting.operations():
        rot.append(np.array([[float(x) for x in r] for r in o.W.rows]
                            ).round().astype(int))
        tr.append(np.array([float(x) for x in o.w.v]))
    t = spglib.get_spacegroup_type_from_symmetry(rot, tr, latt, symprec=1e-3)
    return t.number


@pytest.mark.parametrize("label,sym,cell", CASES, ids=IDS)
def test_spglib_space_group_invariant_across_all_48(label, sym, cell):
    """Oracle: every one of the 48 Selling settings is the SAME space group
    number as the base -- the group is invariant, only the setting changes."""
    base = space_group(sym)
    base_num = getattr(base, "number", None)
    if base_num is None:
        pytest.skip("base group exposes no number")
    for r in selling_settings(cell, sym):
        assert _sg_number(r.setting, r.cell) == base_num
