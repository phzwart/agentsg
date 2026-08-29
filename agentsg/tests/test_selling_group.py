"""The Selling reduction group as change-of-basis operators.

The obtuse superbase {v0, v1, v2, v3} (v0 = -(v1+v2+v3)) has symmetry S4 x {+/-I}
of order 48: the 24 relabellings of the four indices and their global negations.
This is a faithful subgroup of GL(3, Z); every element is integer and
unimodular, and every element preserves the lattice.
"""
from fractions import Fraction as Fr

import pytest

from agentsg.linalg import Matrix3
from agentsg.change_of_basis import ChangeOfBasis
from agentsg.cell.selling_group import (
    selling_group, selling_group_S4,
    selling_generators, selling_generators_S4,
    expand_group, permutation_cob, inversion_cob,
)

np = pytest.importorskip("numpy")


def _key(cob):
    return tuple(int(x) for row in cob.P.rows for x in row)


def _npmat(cob):
    return np.array([[float(x) for x in row] for row in cob.P.rows])


# --- group size and structure ------------------------------------------------
def test_full_group_has_order_48():
    assert len(selling_group()) == 48


def test_permutation_subgroup_has_order_24():
    assert len(selling_group_S4()) == 24


def test_s4_is_subgroup_of_full():
    full = {_key(c) for c in selling_group()}
    sub = {_key(c) for c in selling_group_S4()}
    assert sub <= full
    assert len(sub) == 24 and len(full) == 48


def test_all_elements_distinct():
    keys = [_key(c) for c in selling_group()]
    assert len(set(keys)) == 48


def test_identity_is_first():
    ident = selling_group()[0]
    assert _key(ident) == (1, 0, 0, 0, 1, 0, 0, 0, 1)


# --- every element is an integer, unimodular matrix ---------------------------
def test_all_elements_integer():
    for c in selling_group():
        for row in c.P.rows:
            for x in row:
                assert x.denominator == 1


def test_all_elements_unimodular():
    for c in selling_group():
        assert abs(c.P.det()) == 1


def test_proper_improper_split_is_even():
    dets = [int(c.P.det()) for c in selling_group()]
    assert dets.count(1) == 24
    assert dets.count(-1) == 24


# --- closure -----------------------------------------------------------------
def test_group_is_closed_under_multiplication():
    G = selling_group()
    keyset = {_key(c) for c in G}
    mats = [_npmat(c) for c in G]
    for A in mats:
        for B in mats:
            prod = tuple(int(v) for v in (A @ B).round().astype(int).flatten())
            assert prod in keyset


def test_inverse_of_each_element_in_group():
    G = selling_group()
    keyset = {_key(c) for c in G}
    for c in G:
        inv = c.inverse()
        assert tuple(int(x) for row in inv.P.rows for x in row) in keyset


# --- generators expand to the group ------------------------------------------
def test_full_generators_expand_to_order_48():
    G = expand_group(selling_generators())
    assert len(G) == 48
    assert {_key(c) for c in G} == {_key(c) for c in selling_group()}


def test_s4_generators_expand_to_order_24():
    G = expand_group(selling_generators_S4())
    assert len(G) == 24
    assert {_key(c) for c in G} == {_key(c) for c in selling_group_S4()}


def test_inversion_is_the_missing_generator():
    """S4 alone is order 24; adding -I doubles it to 48."""
    s4 = {_key(c) for c in expand_group(selling_generators_S4())}
    assert _key(inversion_cob()) not in s4
    with_inv = {_key(c) for c in expand_group(
        selling_generators_S4() + [inversion_cob()])}
    assert len(with_inv) == 48


def test_expand_group_rejects_nongroup():
    """A single non-torsion generator (an infinite-order shear) blows past
    max_order, so expand_group refuses it rather than looping forever."""
    from agentsg.linalg import Vector3
    shear = ChangeOfBasis(Matrix3([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
                          Vector3((0, 0, 0)))
    with pytest.raises(RuntimeError):
        expand_group([shear], max_order=48)


# --- crystallographic meaning: every element preserves the lattice -----------
_CELLS = [
    ("triclinic",   (7.0, 8.0, 9.0, 85.0, 95.0, 105.0)),
    ("monoclinic",  (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)),
    ("orthorhombic",(7.0, 9.0, 11.0, 90.0, 90.0, 90.0)),
    ("hexagonal",   (8.0, 8.0, 12.0, 90.0, 90.0, 120.0)),
]


def _conorms(params):
    from agentsg.cell.canonical import canonical_superbase
    C, P = canonical_superbase(params)
    return tuple(sorted(round(P[i][j], 5)
                        for i in range(4) for j in range(i + 1, 4)))


@pytest.mark.parametrize("label,cell", _CELLS, ids=[c[0] for c in _CELLS])
def test_all_48_preserve_the_lattice(label, cell):
    """A unimodular change of basis maps a lattice to itself; the conorm
    fingerprint (Delaunay invariant) must be identical for all 48 images."""
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.canonical import canonical_superbase
    C, _ = canonical_superbase(cell)
    Pred = np.array([[float(C[1 + j][i]) for j in range(3)] for i in range(3)])
    Gred = Pred.T @ np.array(UnitCell(*cell).metric_tensor()) @ Pred
    base = _conorms(params_from_metric(Gred.tolist()))
    for c in selling_group():
        M = _npmat(c)
        Gm = M.T @ Gred @ M
        assert _conorms(params_from_metric(Gm.tolist())) == base


def test_inversion_leaves_metric_unchanged():
    """G' = M^T G M is invariant under M -> -M, so the 48 operators realise 24
    distinct metric tensors (a Gram matrix cannot see a global sign)."""
    from agentsg.cell.metric import UnitCell
    Gred = np.array(UnitCell(7.0, 8.0, 9.0, 85.0, 95.0, 105.0).metric_tensor())
    metrics = set()
    for c in selling_group():
        M = _npmat(c)
        metrics.add(tuple(np.round(M.T @ Gred @ M, 6).flatten()))
    assert len(metrics) == 24
